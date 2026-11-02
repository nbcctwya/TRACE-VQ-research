"""CSI800 membership, common preprocessing and entrypoint regressions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm

from trace_vq.data.market import CanonicalSampler, market_feature_config
from trace_vq.data.prepare import load_config, default_jkp_path
from trace_vq.data.prepare_union import robust_parameters, normalize, process_shard, prepare_union_dataset
from trace_vq.data.schema import MARKET_INDICES, dataset_basename
from trace_vq.data.universe import merge_membership, daily_membership_counts, select_membership, configured_instruments


class CSI800Tests(unittest.TestCase):
    def test_backtest_keeps_exit_quotes_and_can_reproduce_historical_union(self):
        import sys
        from types import ModuleType
        from trace_vq.evaluation.backtest import main

        union = {'A': [(pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-05'))]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = pd.MultiIndex.from_product([[pd.Timestamp('2023-01-03')], ['A']],
                                               names=['datetime', 'instrument'])
            prediction = root / '0_best.pkl'
            pd.DataFrame({'score': [1.0], 'label': [0.0]}, index=index).to_pickle(prediction)
            for universe, quote_universe in (('csi300', 'all'), ('csi800', 'all'),
                                             ('csi300', 'historical'), ('csi800', 'historical')):
                strategy_module = ModuleType('qlib.contrib.strategy')
                strategy_module.TopkDropoutStrategy = MagicMock()
                evaluate_module = ModuleType('qlib.contrib.evaluate')
                backtest = evaluate_module.backtest_daily = MagicMock(return_value=(pd.DataFrame(), {}))
                args = ['backtest', '--pred_path', str(prediction), '--universe', universe,
                        '--quote-universe', quote_universe,
                        '--output_dir', str(root)]
                with patch.object(sys, 'argv', args), patch('qlib.init'), \
                     patch.dict(sys.modules, {'qlib.contrib.strategy': strategy_module,
                                              'qlib.contrib.evaluate': evaluate_module}), \
                     patch('trace_vq.data.universe.load_union', return_value=union) as resolve, \
                     patch('trace_vq.evaluation.backtest._make_metric_frame', return_value=pd.DataFrame()), \
                     patch('trace_vq.evaluation.backtest._save_outputs'):
                    main()
                codes = backtest.call_args.kwargs['exchange_kwargs']['codes']
                if universe == 'csi800' and quote_universe == 'historical':
                    self.assertIs(codes, union)
                    resolve.assert_called_once_with(['csi300', 'csi500'], '2023-01-01', '2025-12-31')
                else:
                    self.assertEqual(codes, 'all' if quote_universe == 'all' else 'csi300')
                    resolve.assert_not_called()

    def test_configured_union_never_uses_native_csi800_membership(self):
        cfg = load_config('csi800')
        union = {'A': [(pd.Timestamp('2009-01-01'), pd.Timestamp('2025-12-31'))]}
        with patch('trace_vq.data.universe.load_union', return_value=union) as resolver:
            self.assertIs(configured_instruments(cfg), union)
            resolver.assert_called_once_with(['csi300', 'csi500'], cfg['data_handler_config']['start_time'],
                                             cfg['data_handler_config']['end_time'])
        self.assertEqual(configured_instruments(load_config('csi300')), 'csi300')

    def test_public_stock_and_market_handlers_receive_the_union(self):
        from trace_vq.data.prepare import build_handler
        from trace_vq.data.market import load_market_frame
        cfg = load_config('csi800')
        union = {'A': [(pd.Timestamp('2008-01-01'), pd.Timestamp('2025-12-31'))]}
        dates = pd.bdate_range('2020-01-01', periods=2, name='datetime')
        columns = pd.MultiIndex.from_product([['market'], market_feature_config(MARKET_INDICES['csi800'])[1]])
        market = pd.DataFrame(np.ones((len(dates), 63), dtype=np.float32), index=dates, columns=columns)
        with patch('trace_vq.data.prepare.configured_instruments', return_value=union), \
             patch('trace_vq.data.prepare.D') as data, patch('trace_vq.data.prepare.pd.read_csv'), \
             patch('trace_vq.data.prepare.build_factor_matrix'), \
             patch('trace_vq.data.prepare.load_market_frame', return_value=market), \
             patch('trace_vq.data.prepare.Alpha158WithJKP') as handler:
            data.calendar.return_value = dates
            build_handler(cfg, '/tmp/not-read.csv')
            self.assertIs(handler.call_args.kwargs['instruments'], union)
        fetched = market.copy()
        fetched.index = pd.MultiIndex.from_product([dates, ['A']], names=['datetime', 'instrument'])
        with patch('trace_vq.data.market.configured_instruments', return_value=union), \
             patch('trace_vq.data.market.MarketDataHandler') as handler:
            handler.return_value.fetch.return_value = fetched
            load_market_frame(cfg, 'csi800')
            self.assertIs(handler.call_args.kwargs['instruments'], union)

    def test_point_in_time_union_deduplicates_overlaps_and_keeps_gaps(self):
        pools = [
            {'A': [('2020-01-01', '2020-01-05'), ('2020-01-10', '2020-01-12')]},
            {'A': [('2020-01-03', '2020-01-07')], 'B': [('2020-01-02', '2020-01-03')]},
        ]
        union = merge_membership(pools)
        dates = pd.date_range('2020-01-01', '2020-01-12')
        self.assertEqual(daily_membership_counts(union, dates).tolist(), [1, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1])
        self.assertEqual(len(union['A']), 2)
        restricted = select_membership(union, '2020-01-04', '2020-01-11')
        self.assertNotIn('B', restricted)
        self.assertEqual(restricted['A'][0][0], pd.Timestamp('2020-01-04'))
        self.assertEqual(restricted['A'][1][1], pd.Timestamp('2020-01-11'))

    def test_columnwise_fit_matches_qlib_with_nans_constants_and_float_dtypes(self):
        for dtype in (np.float32, np.float64):
            values = np.random.default_rng(42).normal(size=(101, 8)).astype(dtype)
            values[3:9, 0] = np.nan
            values[:, 1] = 2
            index = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=101), ['A']],
                                              names=['datetime', 'instrument'])
            frame = pd.DataFrame(values, index=index)
            reference = RobustZScoreNorm('2020-01-01', '2020-04-10', clip_outlier=True)
            reference.fit(frame)
            means, scales = robust_parameters(values.T)
            np.testing.assert_array_equal(means, reference.mean_train)
            np.testing.assert_array_equal(scales, reference.std_train)
            expected = reference(frame.copy()).fillna(0).to_numpy(dtype=np.float32)
            np.testing.assert_array_equal(normalize(values, means, scales), expected)

    def test_union_preprocessing_matches_qlib_joint_fit_and_label_ranking(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range('2020-01-01', periods=45, name='datetime')
        index = pd.MultiIndex.from_product([dates, ['A', 'B', 'C']], names=['datetime', 'instrument'])
        names = [f'f{i}' for i in range(158)] + [f'l{i}' for i in range(10)]
        raw = pd.DataFrame(rng.normal(size=(len(index), 168)).astype(np.float32), index=index, columns=names)
        raw.iloc[7, 0] = np.nan
        raw.iloc[35, -1] = np.nan
        factors = pd.DataFrame(rng.normal(size=(len(dates), 13)), index=dates, columns=[f'p{i}' for i in range(13)])
        market_names = market_feature_config(MARKET_INDICES['csi800'])[1]
        daily_market = pd.DataFrame(rng.normal(size=(len(dates), 63)).astype(np.float32), index=dates,
                                    columns=pd.MultiIndex.from_product([['market'], market_names]))
        training = index.get_level_values('datetime') <= dates[29]
        row_dates = index.get_level_values('datetime')
        prior = factors.loc[row_dates].set_axis(index)
        market = daily_market.loc[row_dates].set_axis(index)
        feature = raw.iloc[:, :158].copy()
        feature.columns = pd.MultiIndex.from_product([['feature'], feature.columns])
        prior.columns = pd.MultiIndex.from_product([['prior'], prior.columns])
        label = raw.iloc[:, 158:].copy()
        label.columns = pd.MultiIndex.from_product([['label'], label.columns])
        reference = pd.concat([feature, prior, market, label], axis=1)
        params = {}
        for group in ('feature', 'prior', 'market'):
            processor = RobustZScoreNorm(dates[0], dates[29], fields_group=group)
            processor.fit(reference)
            params[group] = robust_parameters(reference[group].to_numpy()[training].T)
            np.testing.assert_array_equal(params[group][0], processor.mean_train)
            np.testing.assert_array_equal(params[group][1], processor.std_train)
            reference = Fillna(fields_group=group)(processor(reference))
        normalized_market = pd.DataFrame(normalize(daily_market.to_numpy(), *params['market']),
                                         index=dates, columns=daily_market.columns)
        for learning in (False, True):
            expected = reference.copy()
            if learning:
                expected = CSRankNorm(fields_group='label')(DropnaLabel()(expected))
            actual = process_shard(raw, factors, normalized_market, params, learning)
            pd.testing.assert_frame_equal(actual, expected.astype(np.float32), check_exact=True)
            sampler = CanonicalSampler(actual.copy(), normalized_market, 'csi800', dates[20], dates[-1])
            self.assertEqual(sampler[0].shape, (20, 244))
            self.assertEqual(sampler.data_arr.dtype, np.float32)

    def test_complete_sharded_build_crosses_year_boundary_and_preserves_splits(self):
        rng = np.random.default_rng(9)
        dates = pd.bdate_range('2019-11-01', periods=90, name='datetime')
        symbols = ['A', 'B', 'C']
        index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'instrument'])
        raw = pd.DataFrame(rng.normal(size=(len(index), 168)).astype(np.float32), index=index)
        raw.iloc[200, -1] = np.nan
        raw = raw.drop((dates[10], 'C'))
        market = pd.DataFrame(rng.normal(size=(len(dates), 63)).astype(np.float32), index=dates)
        union = {symbol: [(dates[0], dates[-1])] for symbol in symbols}
        cfg = load_config('csi800')
        cfg['data_handler_config'].update(start_time=str(dates[0].date()), end_time=str(dates[-1].date()),
                                         fit_start_time=str(dates[25].date()), fit_end_time=str(dates[59].date()))
        cfg['dataset']['segments'] = {'train': [dates[25], dates[59]], 'valid': [dates[60], dates[74]],
                                      'test': [dates[75], dates[-1]]}

        def features(instruments, fields, start_time, end_time, **kwargs):
            begin, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
            if len(fields) == 168:
                frame = raw.loc[pd.IndexSlice[begin:end, :], :].copy().swaplevel().sort_index()
            else:
                frame = market.loc[begin:end].copy()
                frame.index = pd.MultiIndex.from_product([['SH600000'], frame.index], names=['instrument', 'datetime'])
            frame.columns = fields
            return frame

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = root / 'provider'
            (provider / 'instruments').mkdir(parents=True)
            (provider / 'calendars').mkdir()
            for name in ('csi300', 'csi500'):
                (provider / 'instruments' / f'{name}.txt').write_text('A\t2019-11-01\t2020-12-31\n')
            (provider / 'calendars/day.txt').write_text('\n'.join(str(day.date()) for day in dates))
            cfg['qlib_init']['provider_uri'] = str(provider)
            csv = root / 'jkp.csv'
            pd.DataFrame([{'date': day, 'name': f'p{i}', 'ret': .001 * (i + 1), 'location': 'chn',
                           'weighting': 'vw_cap', 'freq': 'daily'} for day in dates for i in range(13)]).to_csv(csv, index=False)
            with patch('trace_vq.data.prepare_union.qlib.init'), patch('trace_vq.data.prepare_union.load_union', return_value=union), \
                 patch('trace_vq.data.prepare_union.D') as data_provider:
                data_provider.calendar.return_value = dates
                data_provider.features.side_effect = features
                paths = prepare_union_dataset(root / 'output', csv, cfg, kernels=1)
            for name, path in paths.items():
                sampler = pd.read_pickle(path)
                self.assertEqual(sampler.data_arr.dtype, np.float32)
                self.assertEqual(sampler.universe, 'csi800')
                self.assertEqual(sampler[0].shape, (20, 244))
                actual_dates = sampler.get_index().get_level_values('datetime')
                self.assertGreaterEqual(actual_dates.min(), pd.Timestamp(cfg['dataset']['segments'][name][0]))
                self.assertLessEqual(actual_dates.max(), pd.Timestamp(cfg['dataset']['segments'][name][1]))
                if name == 'test':
                    # Test labels stay raw; training/validation use joint CS ranks.
                    date, symbol = sampler.get_index()[0]
                    np.testing.assert_array_equal(sampler[0][-1, -10:], raw.loc[(date, symbol)].iloc[-10:].to_numpy())
            with self.assertRaises(FileExistsError):
                prepare_union_dataset(root / 'output', csv, cfg)

    def test_csi800_config_keeps_csi300_hyperparameters_and_supports_training(self):
        import stage2
        root = Path(__file__).resolve().parents[1]
        first = load_config('csi300')
        second = load_config('csi800')
        self.assertEqual(second['source_universes'], ['csi300', 'csi500'])
        for key in ('infer_processors', 'learn_processors', 'fit_start_time', 'fit_end_time'):
            self.assertEqual(first['data_handler_config'][key], second['data_handler_config'][key])
        self.assertEqual(first['dataset'], second['dataset'])
        self.assertEqual(dataset_basename('csi800'), 'csi800_20_h10')
        self.assertEqual(default_jkp_path('csi800'), default_jkp_path('csi300'))
        cfg = OmegaConf.load(root / 'configs/config.yaml')
        self.assertEqual(OmegaConf.to_container(cfg.stage2_presets.csi300), OmegaConf.to_container(cfg.stage2_presets.csi800))
        self.assertEqual(stage2._resolve_universe('csi800'), ('CN', 'csi800'))

    def test_evaluation_uses_csi800_saved_config_without_universe_flag(self):
        from trace_vq.cli.evaluate import main
        root = Path(__file__).resolve().parents[1]
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            cfg = OmegaConf.load(root / 'configs/config.yaml')
            cfg.data.universe = 'csi800'
            config_file = tmp / 'config.yaml'
            OmegaConf.save(cfg, config_file)
            for name in ('stage1.ckpt', 'stage2.ckpt'):
                (tmp / name).touch()
            args = ['evaluate', '--config', str(config_file), '--data-path', str(tmp),
                    '--checkpoint', str(tmp / 'stage2.ckpt'), '--stage1-checkpoint', str(tmp / 'stage1.ckpt')]
            with patch.object(sys, 'argv', args), self.assertRaises(FileNotFoundError) as error:
                main()
            self.assertEqual(Path(error.exception.args[0]), tmp / 'CN/csi800_20_h10_test.pkl')
