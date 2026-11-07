"""Independent batch05 watchdog: GPU telemetry and bounded service recovery."""
import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def read(path,default):return json.loads(path.read_text()) if path.exists() else default
def alive(pid,fragment):
    try:return fragment.encode() in Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except (OSError,TypeError,ValueError):return False


def sample_gpu(folder,phase):
    command=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    stamp=datetime.now(timezone.utc).isoformat()
    try:
        query=subprocess.run([command,'--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=8)
        peaks=read(folder/'resource_peaks.json',{});samples=[]
        for line in query.stdout.splitlines():
            index,utilization,used,total=[float(s.strip()) for s in line.split(',')]
            row=dict(updated=stamp,phase=phase,gpu_index=int(index),utilization_percent=utilization,
                device_memory_used_mib=used,device_memory_total_mib=total)
            p=folder/'resource_samples.csv';header=not p.exists()
            with p.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if header:w.writeheader()
                w.writerow(row)
            peak=peaks.setdefault(phase,dict(device_memory_peak_mib=0,samples=0))
            peak.update(device_memory_peak_mib=max(used,peak['device_memory_peak_mib']),samples=peak['samples']+1,last_sample=stamp)
            samples.append(row)
        atomic_json(folder/'resource_peaks.json',peaks)
        return dict(status='available',command=command,samples=samples)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        return dict(status='unavailable',command=command,error=f'{type(error).__name__}: {error}')


def inspect_health(folder,state):
    controller=alive(state.get('pid'),'research/run_batch05.py');warning=None
    finished=(folder/'FINISHED').exists();paused=(folder/'PAUSED').exists()
    if not finished and not paused and state.get('status')!='preparing':
        if not controller:warning='controller_not_running'
        elif time.time()-(folder/'progress.json').stat().st_mtime>180:warning='controller_heartbeat_stale'
        phase=state.get('current_phase');record=state.get('phases',{}).get(phase,{})
        if record.get('status')=='running' and record.get('log'):
            log=Path(record['log']);heartbeat=log.parent/'training_heartbeat.json'
            newest=max(log.stat().st_mtime if log.exists() else 0,heartbeat.stat().st_mtime if heartbeat.exists() else 0)
            limit=1200 if phase.endswith('.train') else 1800
            if newest and time.time()-newest>limit:warning='child_phase_stalled'
    return dict(controller_alive=controller,warning=warning,finished=finished,paused=paused)


def repair(folder,unit,state,health):
    assert unit.startswith('trace-vq-batch05-') and unit.endswith('.service')
    if health['finished'] or health['paused'] or not health['warning']:return None
    if 'implementation changed' in str(state.get('controller_error','')).lower():return 'source_changed_requires_review'
    history=read(folder/'repair_history.json',[]);now=time.time()
    if history and now-history[-1]['time_epoch']<90:return 'waiting_for_service_recovery'
    if len(history)>=8 or sum(r['time_epoch']>now-1800 for r in history)>=3:return 'automatic_repair_limit_reached'
    result=subprocess.run(['systemctl','--user','restart',unit],capture_output=True,text=True,timeout=40)
    history.append(dict(updated=datetime.now(timezone.utc).isoformat(),time_epoch=now,reason=health['warning'],
        action='restart_only_batch05_controller_service',returncode=result.returncode,error=result.stderr[-1000:]))
    atomic_json(folder/'repair_history.json',history)
    return 'controller_restarted' if result.returncode==0 else 'service_restart_failed'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--controller-unit',default='trace-vq-batch05-20261106.service');p.add_argument('--repair',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args();folder=args.output_dir.resolve();assert folder.is_relative_to(ROOT/'artifacts/research')
    folder.mkdir(parents=True,exist_ok=True);lock=(folder/'watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state=read(folder/'progress.json',dict(status='preparing'));phase=state.get('current_phase') or 'between_phases'
        health=inspect_health(folder,state);gpu=sample_gpu(folder,phase)
        snapshot=dict(updated=datetime.now(timezone.utc).isoformat(),status=state['status'],phase=phase,**health,gpu=gpu,
            measurement='sampled whole-device memory includes desktop and other processes')
        if args.repair and health['warning']:
            try:snapshot['repair']=repair(folder,args.controller_unit,state,health)
            except (OSError,subprocess.SubprocessError) as error:snapshot['repair_error']=str(error)
        atomic_json(folder/'watchdog.json',snapshot)
        if health['warning'] or gpu['status']!='available':
            with (folder/'alerts.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
        if args.once:
            print(json.dumps(snapshot),flush=True)
            if gpu['status']!='available':raise SystemExit(1)
            break
        if health['finished']:
            print(snapshot['updated']+' Batch05 and reports finished',flush=True);break
        time.sleep(20)


if __name__=='__main__':main()
