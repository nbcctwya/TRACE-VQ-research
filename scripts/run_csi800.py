"""Build the CSI300/500 union, then train Stage1 seed42 and Stage2 seed0.

Run from the prism-vq environment. Each completed phase is recorded so the
same command can resume without repeating training. Logs and checkpoints are
isolated from CSI300 and imported experiments.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()


def now():
    return datetime.now(timezone.utc).isoformat()


def best_checkpoint(directory):
    checkpoints = [path for path in directory.glob('*.ckpt') if path.name != 'last.ckpt']
    if len(checkpoints) != 1:
        raise RuntimeError(f'Expected one best checkpoint under {directory}, found {checkpoints}')
    return checkpoints[0].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=ROOT / 'dataset/processed')
    parser.add_argument('--artifact-root', type=Path, default=ROOT / 'artifacts/csi800/s1seed42_s2seed0')
    parser.add_argument('--jkp-path', type=Path, default=ROOT.parent / 'HVQ-Stock/dataset/jkpdata/[chn]_[all_themes]_[daily]_[vw_cap].csv')
    args = parser.parse_args()
    output, data_root = args.artifact_root.resolve(), args.data_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / 'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'started': now(), 'phases': {}}
    state.update(pid=os.getpid(), python=sys.executable, artifact_root=str(output), data_path=str(data_root))

    def save():
        state['updated'] = now()
        temporary = state_path.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def run_phase(name, arguments):
        if state['phases'].get(name, {}).get('status') == 'complete':
            print(f'[{now()}] reusing completed phase: {name}', flush=True)
            return
        command = [sys.executable] + arguments
        state['status'], state['current_phase'] = 'running', name
        state.pop('error', None)
        state.pop('traceback', None)
        entry = {'status': 'running', 'started': now(), 'command': command, 'log': str(output / f'{name}.log')}
        state['phases'][name] = entry
        save()
        print(f'[{now()}] starting {name}; log={entry["log"]}', flush=True)
        with Path(entry['log']).open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            save()
            code = process.wait()
        entry.update(finished=now(), returncode=code, status='complete' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(f'{name} exited {code}; see {entry["log"]}')
        print(f'[{now()}] completed {name}', flush=True)

    try:
        splits = [data_root / 'CN' / f'csi800_20_h10_{name}.pkl' for name in ('train', 'valid', 'test')]
        if not all(path.exists() for path in splits):
            run_phase('dataset', ['-m', 'trace_vq.data.prepare', '--universe', 'csi800',
                                  '--output-dir', str(data_root), '--jkp-path', str(args.jkp_path.resolve())])
        else:
            if not (data_root / 'csi800_build_manifest.json').is_file():
                raise RuntimeError('Existing CSI800 splits have no build manifest')
            state['phases'].setdefault('dataset', {'status': 'complete', 'reused': True})
        from omegaconf import OmegaConf
        from hydra import compose, initialize_config_dir
        if not OmegaConf.has_resolver('half'):
            OmegaConf.register_new_resolver('half', lambda value: int(value) // 2)
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
            effective = compose(config_name='config', overrides=['experiment=csi800'])
        base = OmegaConf.to_container(effective, resolve=True, throw_on_missing=False)
        historical = next((ROOT / 'artifacts/034/run/checkpoints').glob('*.config.yaml'), None)
        if historical:
            previous = OmegaConf.to_container(OmegaConf.load(historical), resolve=True)
            for key in ('vqvae', 'predictor'):
                left, right = dict(base[key]), dict(previous[key])
                if key == 'predictor':
                    left.pop('saved_model', None)
                    right.pop('saved_model', None)
                if left != right:
                    raise RuntimeError(f'{key} differs from the previous CSI300 full model')
            state['csi300_hyperparameters_verified_against'] = str(historical)
        stage1_root, stage2_root = output / 'stage1', output / 'stage2'
        base['train']['seed'] = 42
        base['train']['save_dir'] = str(stage1_root / 'checkpoints')
        base['train']['save_res'] = str(stage1_root / 'res')
        base['data']['data_path'] = str(data_root)
        base['artifact_root'] = str(stage1_root)
        OmegaConf.save(OmegaConf.create(base), output / 'stage1.effective.config.yaml')
        run_phase('stage1', ['-m', 'trace_vq.cli.stage1', 'experiment=csi800', 'train.seed=42',
                             f'data.data_path={data_root}', f'artifact_root={stage1_root}'])
        first = best_checkpoint(stage1_root / 'checkpoints')
        state['stage1_checkpoint'] = str(first)
        save()
        run_phase('stage2', ['-m', 'trace_vq.cli.stage2', 'experiment=csi800', 'train.seed=0',
                             f'data.data_path={data_root}', f'artifact_root={stage2_root}',
                             f"predictor.saved_model='{first}'"])
        second = best_checkpoint(stage2_root / 'checkpoints')
        configs = list((stage2_root / 'checkpoints').glob('*.config.yaml'))
        predictions = list((stage2_root / 'res').rglob('0_best.pkl'))
        if len(configs) != 1 or len(predictions) != 1:
            raise RuntimeError('Expected one Stage2 effective config and test prediction file')
        config, prediction = configs[0], predictions[0]
        state.update(stage2_checkpoint=str(second), effective_config=str(config), test_prediction=str(prediction))
        save()
        run_phase('evaluation', ['-m', 'trace_vq.cli.evaluate', '--checkpoint', str(second),
                                 '--stage1-checkpoint', str(first), '--config', str(config),
                                 '--data-path', str(data_root), '--device', 'cuda',
                                 '--output-dir', str(output / 'evaluation')])
        import pandas as pd
        from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
        archived = pd.read_pickle(prediction)
        repeated = pd.read_pickle(output / 'evaluation/test_predictions.pkl')
        pd.testing.assert_frame_equal(archived, repeated, check_exact=True)
        state['standalone_predictions_exact'] = True
        formal = ranking_metrics(archived.score.astype('float64'), archived.label.astype('float64'))
        metadata_path = output / 'backtest/backtest_config.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if metadata.get('quote_universe') != 'all' and 'backtest' in state['phases']:
            state['phases']['backtest']['status'] = 'outdated'
        run_phase('backtest', ['-m', 'trace_vq.evaluation.backtest', '--pred_path', str(prediction),
                               '--universe', 'csi800', '--quote-universe', 'all',
                               '--output_dir', str(output / 'backtest')])
        returns = pd.read_csv(output / 'backtest/portfolio_return.csv', index_col=0, parse_dates=True)['return']
        formal.update(portfolio_metrics(returns))
        pd.Series(formal, name='CSI800_seed0').to_csv(output / 'protocol_metrics.csv')
        state.update(status='complete', current_phase=None, finished=now(), protocol_metrics=formal,
                     backtest_quote_universe='all')
        save()
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
