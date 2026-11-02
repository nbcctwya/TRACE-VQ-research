"""Sequential six-direction, seed42/seed0 screen with matched controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()
import pandas as pd
from trace_vq.data.frozen import atomic_json, sha256
from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
from trace_vq.models.next_screen import NEXT_VARIANTS, PHASES
from research.run_innovation_screen import PROTOCOL


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    paths=list((ROOT/'trace_vq').rglob('*.py'))+list((ROOT/'configs').rglob('*.yaml'))
    paths += [Path(__file__).resolve(),ROOT/'research/run_innovation_screen.py',ROOT/'research/run.sh',
              ROOT/'tests/test_next_screen.py',ROOT/'AGENTS.md']
    return {str(path.relative_to(ROOT)):sha256(path) for path in sorted(set(paths))}


PRIMARY = {'ranknet':'pairwise_reference','top_ranknet':'ranknet',
           'turnover_control':'pairwise_reference','turnover':'turnover_control',
           'soft_route':'baseline','soft_market':'soft_route',
           'history_mean':'baseline','history_gru':'history_mean'}


class Screen:
    def __init__(self,args):
        self.args=args
        self.output=args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT/'artifacts/research'):
            raise ValueError('Research output must be isolated inside this copy')
        self.output.mkdir(parents=True,exist_ok=True)
        self.lock=(self.output/'screen.lock').open('w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.mutex=threading.RLock()
        self.source=identity()
        protocol=dict(PROTOCOL)
        protocol['historical_behavior']='Default historical path unchanged. Soft routing and research quantizers are explicit variants.'
        protocol['combination_selection']='No adaptive combinations or test-dependent hyperparameter changes in this screen'
        plan={'protocol':protocol,'universes':['csi300','csi800'],'phases':PHASES,'variants':NEXT_VARIANTS,
              'source_sha256':self.source,'primary_comparisons':PRIMARY,
              'new_stage2_runs':22,'new_stage1_runs':6,
              'quantizer_budgets':{'rvq_256x256':{'total_entries':512,'combinations':65536},
                                  'rvq_32x16':{'total_entries':48,'combinations':512},
                                  'fsq':{'levels':[8,8,8],'combinations':512}},
              'retention':'Preserve every completed and failed run, weights, predictions, configs, and logs',
              'selection_rule':'Promising: validation/test RankIC and net Sharpe improve, absolute MDD loss <=.02',
              'interpretation':'Exploratory shared-test, single-trained-seed comparisons; no novelty or stability guarantee'}
        plan=json.loads(json.dumps(plan))
        path=self.output/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            raise RuntimeError('Frozen plan/source changed: do not mix research versions in one run')
        atomic_json(path,plan)
        for name in self.source:
            destination=self.output/'source_snapshot'/name
            if not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name,destination)
        self.state_path=self.output/'progress.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'started':now(),'phases':{},'runs':{},'direction_status':{}}
        self.state.update(status='running',pid=os.getpid(),python=sys.executable)
        self.save()

    def save(self):
        with self.mutex:
            self.state['updated']=now()
            atomic_json(self.state_path,self.state)

    def event(self,message):
        with self.mutex:
            print(f'[{now()}] {message}',flush=True)
            with (self.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':now(),'message':message})+'\n')

    def phase(self,key,command,log,required=()):
        with self.mutex:
            prior=self.state['phases'].get(key,{})
            if prior.get('status')=='complete' and all(Path(p).is_file() for p in required):
                return
            if identity()!=self.source:
                raise RuntimeError('Research source changed during execution')
            entry={'status':'running','started':now(),'command':command,'log':str(log),
                   'previous_attempts':prior.get('previous_attempts',[])+([prior] if prior else [])}
            self.state['phases'][key]=entry
            self.save()
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        self.event(f'start {key}')
        environment=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',WANDB_MODE='disabled')
        start=time.monotonic()
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            with self.mutex:
                entry['pid']=process.pid;self.save()
            code=process.wait()
        with self.mutex:
            entry.update(status='complete' if code==0 else 'failed',returncode=code,
                         finished=now(),elapsed_seconds=time.monotonic()-start)
            self.save()
        self.event(f'{entry["status"]} {key}')
        if code or not all(Path(p).is_file() for p in required):
            raise RuntimeError(f'{key} failed; see {log}')

    def cli(self,command,universe,variant,output,stage1=None):
        quant=NEXT_VARIANTS[variant].get('quantizer_kind')
        cache=self.output/'cache'/(variant if quant else 'legacy')
        result=[sys.executable,'-m','trace_vq.cli.next_screen',command,'--universe',universe,
                '--variant',variant,'--output-dir',str(output),'--cache-root',str(cache),
                '--history-root',str(self.output/'history'),'--num-workers',str(self.args.num_workers),'--device','cuda']
        if stage1:
            result+=['--stage1-checkpoint',str(stage1)]
        return result

    def integrity(self,label):
        self.phase(f'integrity.{label}',[sys.executable,'research/verify_baseline.py','--full'],
                   self.output/f'integrity_{label}.log')
        result=json.loads((ROOT/'research/logs/source-integrity.json').read_text())
        if not result['source_unchanged'] or not result['full_hash_check']:
            raise RuntimeError('Original TRACE-VQ changed')
        atomic_json(self.output/f'integrity_{label}.json',result)

    def backtest(self,universe,run):
        return [sys.executable,'-m','trace_vq.evaluation.backtest','--pred_path',str(run/'evaluation/test_predictions.pkl'),
                '--universe',universe,'--quote-universe','all','--start_time','2023-01-01','--end_time','2025-12-31',
                '--topk','30','--drop','5','--open_cost','0.0005','--close_cost','0.0015','--min_cost','0',
                '--account','100000000','--output_dir',str(run/'backtest')]

    def run_model(self,universe,variant):
        run=self.output/universe/variant
        run.mkdir(parents=True,exist_ok=True)
        key=f'{universe}.{variant}'
        if self.state['runs'].get(key,{}).get('status')=='complete' and (run/'completion.json').exists():
            return
        try:
            stage1=None
            if NEXT_VARIANTS[variant].get('quantizer_kind'):
                first=self.output/'stage1'/universe/variant
                self.phase(f'{key}.stage1',self.cli('train-stage1',universe,variant,first),first/'train.log',
                           [first/'training_result.json'])
                record=json.loads((first/'training_result.json').read_text())
                if record['stage1_seed']!=42:
                    raise RuntimeError('Stage 1 seed changed')
                stage1=record['checkpoint']
                self.phase(f'{key}.cache',self.cli('cache',universe,variant,self.output/'cache'/variant/universe,stage1),
                           self.output/f'cache_{universe}_{variant}.log',
                           [self.output/'cache'/variant/universe/'completion.json'])
            command=self.cli('train',universe,variant,run,stage1)
            if variant in ('baseline','pairwise_reference'):
                old='baseline' if variant=='baseline' else 'pairwise_loss'
                command=self.cli('adopt',universe,variant,run)+['--reference-dir',str(self.args.previous_dir/universe/old)]
            self.phase(f'{key}.train',command,run/'train.log',[run/'training_result.json'])
            self.phase(f'{key}.evaluate',self.cli('evaluate',universe,variant,run,stage1),run/'evaluate.log',
                       [run/'evaluation/completion.json'])
            self.phase(f'{key}.backtest',self.backtest(universe,run),run/'backtest.log',
                       [run/'backtest/portfolio_return.csv',run/'backtest/backtest_config.json'])
            result=json.loads((run/'training_result.json').read_text())
            settings=json.loads((run/'backtest/backtest_config.json').read_text())
            strategy,exchange=settings['strategy'],settings['exchange']
            assert settings['quote_universe']=='all' and strategy['topk']==30 and strategy['n_drop']==5
            assert strategy['risk_degree']==.95 and exchange['open_cost']==.0005 and exchange['close_cost']==.0015
            assert exchange['min_cost']==0 and settings['account']==1e8
            assert (settings['start_time'],settings['end_time'])==('2023-01-01','2025-12-31')
            prediction=pd.read_pickle(run/'evaluation/test_predictions.pkl')
            metrics=ranking_metrics(prediction.score.astype('float64'),prediction.label.astype('float64'))
            returns=pd.read_csv(run/'backtest/portfolio_return.csv',index_col=0,parse_dates=True)['return']
            metrics.update(portfolio_metrics(returns))
            result.update(status='complete',test_metrics=metrics,backtest_config=settings,standalone_exact=True,
                          primary_reference=PRIMARY.get(variant,'baseline'),completed=now())
            atomic_json(run/'completion.json',result)
            with self.mutex:
                self.state['runs'][key]=result;self.save();self.summarize()
            self.event(f'RESULT {key}: valid RankIC={result["validation_metrics"]["RankIC"]:.6f}, '
                       f'test RankIC={metrics["RankIC"]:.6f}, net Sharpe={metrics["Sharpe"]:.6f}')
        except BaseException as error:
            failure={'status':'technical_failure','universe':universe,'variant':variant,
                     'error':str(error),'traceback':traceback.format_exc(),'finished':now()}
            atomic_json(run/'failure.json',failure)
            with self.mutex:
                self.state['runs'][key]=failure;self.save();self.summarize()
            raise

    def summarize(self):
        records=[]
        for result in self.state['runs'].values():
            row={key:result.get(key) for key in ('universe','variant','status','elapsed_seconds','completed_epochs')}
            if row['status']=='complete':
                row.update(seed=0,**result['test_metrics'])
                row.update({f'valid_{key}':value for key,value in result['validation_metrics'].items()})
                epoch=re.search(r'epoch=(\d+)',Path(result['checkpoint']).name)
                row['best_epoch']=int(epoch[1]) if epoch else None
                row['primary_reference']=result['primary_reference']
                for label,reference in [('baseline','baseline'),('primary',result['primary_reference'])]:
                    control=self.state['runs'].get(f'{row["universe"]}.{reference}',{})
                    if control.get('status')=='complete':
                        for metric in ('RankIC','Sharpe','AR','MDD'):
                            row[f'delta_{label}_{metric}']=row[metric]-control['test_metrics'][metric]
                        row[f'delta_{label}_valid_RankIC']=row['valid_RankIC']-control['validation_metrics']['RankIC']
                if row['variant']=='baseline':
                    row['decision']='control'
                elif row.get('delta_baseline_RankIC',-1)>0 and row.get('delta_baseline_Sharpe',-1)>0 and \
                     row.get('delta_baseline_valid_RankIC',-1)>0 and row.get('delta_baseline_MDD',-1)>=-.02:
                    row['decision']='promising'
                else:
                    row['decision']='mixed' if row.get('delta_baseline_Sharpe',-1)>0 or row.get('delta_baseline_RankIC',-1)>0 else 'underperforming'
                row['delta_RankIC']=row.get('delta_baseline_RankIC')
                row['delta_Sharpe']=row.get('delta_baseline_Sharpe')
            records.append(row)
        if records:
            table=pd.DataFrame(records).sort_values(['universe','variant'])
            temporary=self.output/'leaderboard.partial.csv'
            table.to_csv(temporary,index=False);temporary.replace(self.output/'leaderboard.csv')
            atomic_json(self.output/'decisions.json',records)

    def parallel(self,variant):
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures=[pool.submit(self.run_model,universe,variant) for universe in ('csi300','csi800')]
            errors=[]
            for future in futures:
                try:future.result()
                except BaseException as error:errors.append(str(error))
            if errors:
                raise RuntimeError('; '.join(errors))

    def execute(self):
        self.integrity('before')
        for universe in ('csi300','csi800'):
            directory=self.output/'cache/legacy'/universe
            self.phase(f'cache.{universe}',self.cli('cache',universe,'baseline',directory),
                       self.output/f'cache_{universe}.log',[directory/'completion.json'])
        for variant in ('baseline','pairwise_reference'):
            self.parallel(variant)
        for direction,variants in PHASES:
            self.state['current_direction']=direction
            self.state['direction_status'][direction]='running';self.save()
            self.event(f'DIRECTION {direction}')
            if direction=='04_code_history':
                for universe in ('csi300','csi800'):
                    directory=self.output/'history'/universe
                    self.phase(f'history.{universe}',self.cli('history-cache',universe,'baseline',directory),
                               self.output/f'history_{universe}.log',[directory/'completion.json'])
            for variant in variants:
                self.parallel(variant)
            self.state['direction_status'][direction]='complete';self.save()
        self.integrity('after')
        datasets={}
        for universe in ('csi300','csi800'):
            for split in ('train','valid','test'):
                entry=json.loads((self.output/'cache/legacy'/universe/f'{split}.json').read_text())
                datasets[f'{universe}.{split}']=sha256(entry['identity']['source'])==entry['source_sha256']
        atomic_json(self.output/'dataset_integrity_after.json',datasets)
        if not all(datasets.values()):
            raise RuntimeError('A source dataset changed')
        self.state.update(status='complete',finished=now(),failed_runs=[])
        self.save();self.summarize();self.event('screen finished: complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/research/next_directions_20261027')
    parser.add_argument('--previous-dir',type=Path,default=ROOT/'artifacts/research/innovation_screen_20261026')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--num-workers',type=int,default=2)
    args=parser.parse_args()
    screen=Screen(args)
    try:screen.execute()
    except BaseException as error:
        screen.state.update(status='failed',error=str(error),traceback=traceback.format_exc())
        screen.save();screen.event(f'screen failed: {error}');raise


if __name__=='__main__':
    main()
