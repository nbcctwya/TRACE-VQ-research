"""Run the remaining CSI800 Stage2 seeds against the completed seed42 Stage1."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def unique(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one artifact, found {paths}')
    return paths[0].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s not in range(1, 5) for s in args.seeds):
        parser.error('Remaining seeds must be unique values from 1, 2, 3, 4')
    output = ROOT / 'artifacts/csi800'
    source = output / 's1seed42_s2seed0'
    reference = json.loads((source / 'progress.json').read_text())
    if reference['status'] != 'complete':
        raise RuntimeError('The reference seed0 pipeline has not completed')
    stage1 = Path(reference['stage1_checkpoint']).resolve()
    data_root = Path(reference['data_path']).resolve()
    stage1_hash = sha256(stage1)
    manifest = data_root / 'csi800_build_manifest.json'
    for split in ('train', 'valid', 'test'):
        if not (data_root / 'CN' / f'csi800_20_h10_{split}.pkl').is_file():
            raise FileNotFoundError(split)
    lock = (output / 'remaining_seeds.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / 'remaining_seeds.progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'started': now(), 'phases': {}}
    identity = {'stage1_checkpoint': str(stage1), 'stage1_sha256': stage1_hash,
                'data_path': str(data_root), 'dataset_manifest_sha256': sha256(manifest)}
    if any(key in state and state[key] != value for key, value in identity.items()):
        raise RuntimeError('Shared Stage1 or dataset identity has changed')
    state.update(identity, pid=os.getpid(), python=sys.executable, seeds=args.seeds, status='running')
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def save():
        state['updated'] = now()
        temporary = state_path.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2, default=str))
        temporary.replace(state_path)

    def phase(name, arguments, log):
        entry = state['phases'].get(name, {})
        if entry.get('status') == 'complete':
            print(f'[{now()}] reusing {name}', flush=True)
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable] + arguments
        entry = {'status': 'running', 'started': now(), 'command': command, 'log': str(log)}
        state['phases'][name] = entry
        state['current_phase'] = name
        save()
        print(f'[{now()}] starting {name}; log={log}', flush=True)
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            save()
            code = process.wait()
        entry.update(finished=now(), returncode=code, status='complete' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(f'{name} exited {code}: {log}')

    try:
        save()
        for seed in args.seeds:
            run = output / f's1seed42_s2seed{seed}'
            stage2 = run / 'stage2'
            if sha256(stage1) != stage1_hash:
                raise RuntimeError('Shared Stage1 checkpoint changed during the experiment')
            phase(f'seed{seed}.stage2', ['-m', 'trace_vq.cli.stage2', 'experiment=csi800',
                  f'train.seed={seed}', f'data.data_path={data_root}', f'artifact_root={stage2}',
                  f"predictor.saved_model='{stage1}'"], run / 'stage2.log')
            checkpoint = unique(p for p in (stage2 / 'checkpoints').glob('*.ckpt') if p.name != 'last.ckpt')
            config = unique((stage2 / 'checkpoints').glob('*.config.yaml'))
            prediction = unique((stage2 / 'res').rglob(f'{seed}_best.pkl'))
            from omegaconf import OmegaConf
            previous = OmegaConf.to_container(OmegaConf.load(reference['effective_config']), resolve=True)
            current = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
            for value in (previous, current):
                value.pop('artifact_root', None)
                for key in ('seed', 'save_dir', 'save_res'):
                    value['train'].pop(key, None)
            if previous != current:
                raise RuntimeError(f'Seed{seed} effective configuration differs from seed0')
            phase(f'seed{seed}.evaluation', ['-m', 'trace_vq.cli.evaluate', '--checkpoint', str(checkpoint),
                  '--stage1-checkpoint', str(stage1), '--config', str(config), '--data-path', str(data_root),
                  '--device', 'cuda', '--output-dir', str(run / 'evaluation')], run / 'evaluation.log')
            phase(f'seed{seed}.backtest', ['-m', 'trace_vq.evaluation.backtest', '--pred_path', str(prediction),
                  '--universe', 'csi800', '--quote-universe', 'all', '--output_dir', str(run / 'backtest')],
                  run / 'backtest.log')
            import pandas as pd
            from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
            archived = pd.read_pickle(prediction)
            pd.testing.assert_frame_equal(archived, pd.read_pickle(run / 'evaluation/test_predictions.pkl'), check_exact=True)
            metrics = ranking_metrics(archived.score.astype('float64'), archived.label.astype('float64'))
            returns = pd.read_csv(run / 'backtest/portfolio_return.csv', index_col=0, parse_dates=True)['return']
            metrics.update(portfolio_metrics(returns))
            pd.Series(metrics, name=f'CSI800_seed{seed}').to_csv(run / 'protocol_metrics.csv')
            report = dict(identity, seed=seed, status='complete', finished=now(), stage2_checkpoint=str(checkpoint),
                          effective_config=str(config), test_prediction=str(prediction), protocol_metrics=metrics,
                          standalone_predictions_exact=True, seed0_configuration_equal=True, backtest_quote_universe='all')
            (run / 'completion_report.json').write_text(json.dumps(report, indent=2, default=str))
            state.setdefault('completed_seeds', {})[str(seed)] = report
            save()
        rows = []
        for seed in range(5):
            path = output / f's1seed42_s2seed{seed}/protocol_metrics.csv'
            if not path.exists():
                continue
            row = pd.read_csv(path, index_col=0).iloc[:, 0].to_dict()
            rows.append(dict(seed=seed, **row))
        table = pd.DataFrame(rows).set_index('seed')
        table.to_csv(output / 'seed_metrics.csv')
        table.agg(['mean', 'std', 'count']).to_csv(output / 'seed_mean_std.csv')
        if list(table.index) != list(range(5)):
            raise RuntimeError('The five-seed summary is incomplete')
        state.update(status='complete', current_phase=None, finished=now())
        save()
        print(f'[{now()}] TRACE-VQ CSI800 seeds0-4 complete', flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
