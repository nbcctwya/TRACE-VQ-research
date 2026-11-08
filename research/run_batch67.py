"""Only batches06/07: serial fits with bounded retries and checkpoint recovery."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
from omegaconf import OmegaConf
from research.run_batch01 import METRICS,result_metrics,validate_backtest
from trace_vq.cli.batch01 import reference_directory
from trace_vq.cli.batch67 import make_config
from trace_vq.cli.screening import now
from trace_vq.data.frozen import atomic_json,sha256
from trace_vq.models.batch67 import VARIANTS

SOURCES=['research/run_batch67.py','research/run.sh','trace_vq/cli/batch67.py',
    'trace_vq/models/batch67.py','trace_vq/data/hopfield.py','trace_vq/data/fsq_history.py','trace_vq/data/schema.py','trace_vq/training/resilient.py',
    'trace_vq/training/research_run.py','trace_vq/models/screening.py','trace_vq/training/prediction.py',
    'trace_vq/models/prediction/loadings.py','trace_vq/models/prediction/temporal_transformer.py',
    'trace_vq/models/experts/fusion.py','trace_vq/models/experts/shared_routed.py','trace_vq/data/frozen.py',
    'trace_vq/models/batch01.py']


def tasks():return [(market,variant) for variant in VARIANTS for market in ('csi300','csi800')]
def key(market,variant):return f'batch{VARIANTS[variant]["batch"]:02d}.{market}.{variant}.seed0' if variant in VARIANTS else f'reference.{market}.{variant}.seed0'
def read_json(path,default=None):return json.loads(path.read_text()) if path.exists() else default


class Batch67:
    def __init__(self,args):
        self.args=args;self.output=args.output_dir.resolve();assert self.output.is_relative_to(ROOT/'artifacts/research')
        self.output.mkdir(parents=True,exist_ok=True)
        self.lock=(self.output/'batch.lock').open('w');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.source={p:sha256(ROOT/p) for p in SOURCES};self.path=self.output/'progress.json'
        self.state=read_json(self.path,dict(status='running',started=now(),runs={},phases={},recoveries=[],planned_new_stage2=22,
                                         planned_new_stage1=0,source_sha256=self.source))
        assert self.source==self.state['source_sha256'],'Training implementation changed; inspect before resuming'
        phase=self.state['phases'].get(self.state.get('current_phase'),{})
        if phase.get('status')=='running' and phase.get('pid'):
            path=Path(f"/proc/{phase['pid']}/cmdline")
            if path.exists() and path.read_bytes().split(b'\0')[:len(phase['command'])]==[s.encode() for s in phase['command']]:
                raise RuntimeError('Previous child still alive; no duplicate GPU work allowed')
        if self.state.get('status') in ('complete','complete_with_failures') and (self.output/'FINISHED').exists():return
        self.state.update(status='running',pid=os.getpid());self.save()

    def save(self):self.state['updated']=now();atomic_json(self.path,self.state)
    def event(self,message):
        line=now()+' '+message;print(line,flush=True)
        with (self.output/'events.log').open('a') as f:f.write(line+'\n')

    def interrupted_fit(self,name,phase):
        if not name.endswith('.train'):return
        folder=Path(phase['log']).parent;heartbeat=read_json(folder/'training_heartbeat.json',{})
        checkpoint=read_json(folder/'checkpoints/latest_checkpoint.json',{})
        lost=max(0,float(heartbeat.get('fit_seconds',0))-float(checkpoint.get('fit_seconds') or 0))
        record=dict(updated=now(),phase=name,unsaved_fit_seconds_lower_bound=lost,
                    checkpoint_epoch=checkpoint.get('epoch'),heartbeat=heartbeat,reason='interrupted or failed attempt')
        self.state['recoveries'].append(record);self.event(f'RECOVERY {name}: unsaved fit at least {lost:.1f}s')

    def snapshot(self):
        phase=self.state.get('current_phase');record=self.state['phases'].get(phase,{})
        models=[r for r in self.state['runs'].values() if r.get('role')=='new_model']
        info=dict(updated=now(),status=self.state['status'],pid=os.getpid(),child_pid=record.get('pid'),current_phase=phase,
            planned_new_training=22,completed_new_training=sum(r['status']=='complete' for r in models),
            failed=[r['universe']+'/'+r['variant'] for r in models if r['status'] in ('failed','blocked')])
        if record.get('log'):
            folder=Path(record['log']).parent;epoch=read_json(folder/'training_progress.json',{})
            heartbeat=read_json(folder/'training_heartbeat.json',{})
            if epoch.get('latest'):info.update(completed_epochs=epoch['latest']['epoch']+1,val_loss=epoch['latest'].get('val_loss'))
            if heartbeat:info['training_heartbeat']=heartbeat
        atomic_json(self.output/'monitoring.json',info)
        text=['# 第六、七批实时进度','',f"更新时间（UTC）：{info['updated']}",'',
              f"状态：{info['status']}；完整结果 {info['completed_new_training']}/22；失败/阻塞 {len(info['failed'])}。",'',
              f"当前阶段：{phase or '无'}。"]
        if 'completed_epochs' in info:text+=['',f"当前配置已完成 {info['completed_epochs']} epochs；验证损失 {info['val_loss']}。"]
        (self.output/'STATUS.md').write_text('\n'.join(text)+'\n')

    def setup(self):
        plan=[]
        for market,variant in tasks():
            batch=VARIANTS[variant]['batch'];batch_root=self.output/f'batch{batch:02d}_20261107'
            folder=batch_root/market/variant/'seed0';config=make_config(market,variant,folder,self.args.num_workers,self.output)
            path=batch_root/'configs'/market/(variant+'.yaml');path.parent.mkdir(parents=True,exist_ok=True)
            OmegaConf.save(OmegaConf.create(config),path)
            r=dict(batch=batch,universe=market,variant=variant,stage1_seed=42,stage2_seed=0,role='new_model',status='pending',new_training=True,
                   reused_reference=False,baseline_variant='baseline_'+VARIANTS[variant]['family'],config=str(path),output_dir=str(folder),runtime_workers=self.args.num_workers)
            self.state['runs'].setdefault(key(market,variant),r);plan.append(r)
        for market in ('csi300','csi800'):
            for family in ('fsq',):
                name=key(market,'baseline_'+family)
                if name in self.state['runs']:continue
                folder=reference_directory(market,family);validate_backtest(folder,market)
                self.state['runs'][name]=dict(universe=market,variant='baseline_'+family,stage1_seed=42,stage2_seed=0,
                    role='baseline',status='complete',new_training=False,reused_reference=True,baseline_variant='',output_dir=str(folder),
                    training_seconds=0,evaluation_backtest_seconds=0,metrics=result_metrics(folder))
        atomic_json(self.output/'plan.json',dict(batches=[6,7],items=list(range(28,39)),planned_new_stage2=22,new_stage1=0,tasks=plan,
            source_sha256=self.source,stage1_seed=42,stage2_seed=0,only_requested_batches=[6,7],serial=True,backend='existing torch; no package installs'))
        self.save();self.summarize()

    def summarize(self):
        rows=[]
        for name,r in self.state['runs'].items():
            row={k:v for k,v in r.items() if k not in ('metrics','validation_metrics','test_metrics','routing_valid','routing_test')}
            row.update(r.get('metrics',{}))
            if r.get('metrics') and r.get('baseline_variant'):
                base=self.state['runs'][key(r['universe'],r['baseline_variant'])]
                for metric in METRICS:row['delta_'+metric]=r['metrics'][metric]-base['metrics'][metric]
            row['unsaved_training_seconds_lower_bound']=sum(p['unsaved_fit_seconds_lower_bound'] for p in self.state['recoveries'] if p['phase']==name+'.train')
            if 'training_seconds' in r:row['training_seconds_including_replay_lower_bound']=r['training_seconds']+row['unsaved_training_seconds_lower_bound']
            rows.append(row)
        frame=pd.DataFrame(rows).sort_values(['universe','role','variant'])
        temporary=self.output/'results.partial.csv';frame.to_csv(temporary,index=False);temporary.replace(self.output/'results.csv')
        frame[frame.role.eq('new_model')].to_csv(self.output/'variant_results.csv',index=False)
        frame[frame.role.eq('baseline')].to_csv(self.output/'baselines.csv',index=False)

    def phase(self,name,command,log,required=(),run=None):
        record=self.state['phases'].setdefault(name,dict(attempts=[]))
        if record.get('status')=='complete' and all(p.exists() for p in required):return record['seconds']
        if any(p.name=='training_result.json' and p.exists() for p in required):
            trained=read_json(next(p for p in required if p.name=='training_result.json'))
            record.update(status='complete',seconds=trained['train_phase_seconds'],adopted=True);self.save();return record['seconds']
        if record.get('status')=='running':
            self.interrupted_fit(name,record)
            record['attempts'].append({k:v for k,v in record.items() if k!='attempts'})
        for attempt in range(len(record['attempts']),3):
            assert self.source=={p:sha256(ROOT/p) for p in SOURCES},'Training source changed'
            if (self.output/'PAUSED').exists():raise RuntimeError('PAUSED marker: do not launch further work')
            actual=list(command)
            if run and '--num-workers' in actual:actual[actual.index('--num-workers')+1]=str(run['runtime_workers'])
            log.parent.mkdir(parents=True,exist_ok=True);self.event(f'START {name} attempt {attempt+1}/3');start=time.perf_counter()
            with log.open('a') as stream:
                process=subprocess.Popen(actual,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                record.update(status='running',started=now(),pid=process.pid,command=actual,log=str(log),active_elapsed_seconds=0)
                self.state['current_phase']=name;self.save()
                while True:
                    record['active_elapsed_seconds']=time.perf_counter()-start;self.save();self.snapshot()
                    try:code=process.wait(timeout=20);break
                    except subprocess.TimeoutExpired:pass
            seconds=time.perf_counter()-start;record.update(returncode=code,seconds=seconds,finished=now())
            if code==0 and all(p.exists() for p in required):
                record['status']='complete';self.save();self.event(f'DONE {name}: {seconds:.1f}s');return seconds
            self.interrupted_fit(name,record)
            record['status']='failed';record['attempts'].append({k:v for k,v in record.items() if k!='attempts'});self.save()
            tail='\n'.join(log.read_text(errors='replace').splitlines()[-100:])
            if run and any(s in tail for s in ('DataLoader','worker exited','BrokenPipe','EOFError')):run['runtime_workers']=0
            self.event(f'RETRY {name}: exit={code}; preserving checkpoint and inputs')
        raise RuntimeError(f'{name} failed after 3 attempts; inspect {log}')

    def model(self,market,variant):
        name=key(market,variant);r=self.state['runs'][name]
        if r['status'] in ('complete','blocked','failed'):return
        folder=Path(r['output_dir']);common=['--universe',market,'--variant',variant,'--output-dir',str(folder),
            '--device','cuda','--cache-root',str(self.output),'--num-workers',str(r['runtime_workers'])];base=[sys.executable,'-u','-m','trace_vq.cli.batch67']
        r['status']='running';self.save()
        try:
            self.phase(name+'.train',base+['train']+common,folder/'train.log',[folder/'training_result.json'],r)
            evaluation=self.phase(name+'.evaluate',base+['evaluate']+common,folder/'evaluate.log',[folder/'evaluation/completion.json'],r)
            command=[sys.executable,'-m','trace_vq.evaluation.backtest','--pred_path',str(folder/'evaluation/test_predictions.pkl'),
                '--universe',market,'--quote-universe','all','--start_time','2023-01-01','--end_time','2025-12-31',
                '--topk','30','--drop','5','--open_cost','0.0005','--close_cost','0.0015','--min_cost','0','--account','100000000','--output_dir',str(folder/'backtest')]
            backtest=self.phase(name+'.backtest',command,folder/'backtest.log',[folder/'backtest/portfolio_return.csv',folder/'backtest/backtest_config.json'])
            validate_backtest(folder,market);trained=read_json(folder/'training_result.json');evaluated=read_json(folder/'evaluation/completion.json')
            assert evaluated['standalone_exact'] and trained['stage1_unchanged'] and not trained['stage1_retrained']
            r.update(trained,status='complete',metrics=result_metrics(folder),standalone_exact=True,evaluation_seconds=evaluation,
                backtest_seconds=backtest,evaluation_backtest_seconds=trained['training_inference_seconds']+evaluation+backtest,completed=now())
            for field in ('gpu_peak_allocated_mib','gpu_peak_reserved_mib'):r[field]=max(trained[field],evaluated[field])
            atomic_json(folder/'completion.json',r);self.event('RESULT '+name+': '+str(r['metrics']))
        except Exception as error:
            r.update(status='failed',error=str(error));self.event(f'FAILED {name}: {error}')
            with (self.output/'failures.log').open('a') as f:traceback.print_exc(file=f)
        self.save();self.summarize();self.snapshot()

    def publish_batch(self,batch):
        folder=self.output/f'batch{batch:02d}_20261107';folder.mkdir(exist_ok=True)
        runs={k:r for k,r in self.state['runs'].items() if r.get('batch')==batch or r.get('role')=='baseline'}
        rows=[r for r in runs.values() if r.get('role')=='new_model']
        status='complete' if all(r['status']=='complete' for r in rows) else 'complete_with_failures'
        substate=dict(self.state,runs=runs,status=status,finished=now(),batch=batch,
            recoveries=[r for r in self.state['recoveries'] if r['phase'].startswith(f'batch{batch:02d}.')])
        atomic_json(folder/'progress.json',substate)
        frame=pd.read_csv(self.output/'results.csv');frame=frame[frame.role.eq('baseline')|frame.batch.eq(batch)]
        frame.to_csv(folder/'results.csv',index=False)
        frame[frame.role.eq('new_model')].to_csv(folder/'variant_results.csv',index=False)
        frame[frame.role.eq('baseline')].to_csv(folder/'baselines.csv',index=False)
        plan=read_json(self.output/'plan.json');plan.update(batch=batch,items=list(range(28,35)) if batch==6 else list(range(35,39)),
            tasks=rows,planned_new_stage2=len(rows),only_requested_batches=[6,7])
        atomic_json(folder/'plan.json',plan)
        validation=read_json(self.output/'validation.json');pre=read_json(self.output/'preflight/gpu_validation.json')
        validation.update(batch=batch,gpu_checks=[r for r in pre['checks'] if VARIANTS[r['variant']]['batch']==batch],
            gpu_failures=[r for r in pre['failures'] if VARIANTS[r['variant']]['batch']==batch])
        atomic_json(folder/'validation.json',validation)
        self.phase(f'original.after.batch{batch:02d}',[sys.executable,'research/verify_baseline.py'],folder/'original_integrity_after.log')
        self.phase(f'batch{batch:02d}.aggregate.report',[sys.executable,'research/report_batch67.py','--output-dir',str(folder),
            '--batch',str(batch)],folder/'report.log',[folder/'report_verification.json'])
        (folder/'FINISHED').write_text(status+'\n');self.state.setdefault('finished_batches',{})[str(batch)]=dict(status=status,finished=now())
        self.save();self.event(f'BATCH{batch:02d} FINISHED: '+status)

    def execute(self):
        if (self.output/'FINISHED').exists():return
        pre=read_json(self.output/'preflight/gpu_validation.json');assert pre and len(pre['checks'])+len(pre['failures'])==22
        assert read_json(self.output/'validation.json')['unit_tests_passed']
        self.setup();passed={(r['universe'],r['variant']) for r in pre['checks']}
        for market,variant in tasks():
            if (market,variant) not in passed:
                reason=next(r['error'] for r in pre['failures'] if r['universe']==market and r['variant']==variant)
                self.state['runs'][key(market,variant)].update(status='blocked',error='GPU preflight: '+reason)
        self.phase('original.before',[sys.executable,'research/verify_baseline.py'],self.output/'original_integrity_before.log')
        for batch in (6,7):
            if (self.output/f'batch{batch:02d}_20261107/FINISHED').exists():continue
            self.state['current_batch']=batch;self.save()
            for market,variant in tasks():
                if VARIANTS[variant]['batch']==batch:self.model(market,variant)
            self.summarize();self.publish_batch(batch)
        rows=[r for r in self.state['runs'].values() if r.get('role')=='new_model']
        self.state.update(status='complete' if all(r['status']=='complete' for r in rows) else 'complete_with_failures',current_phase=None,finished=now())
        self.save();self.summarize();self.snapshot()
        (self.output/'FINISHED').write_text(self.state['status']+'\n');self.event('BATCH0607 FINISHED: '+self.state['status'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--num-workers',type=int,default=2)
    args=p.parse_args();queue=Batch67(args)
    try:queue.execute()
    except BaseException as error:
        queue.state.update(status='controller_failed',controller_error=str(error));queue.save();queue.snapshot();raise


if __name__=='__main__':main()
