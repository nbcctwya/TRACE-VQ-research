"""Prepare and optionally install two user services for this batch only."""
import argparse
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def unit_text(output,watchdog=False):
    controller='trace-vq-batch89-20261108.service';uid=os.getuid()
    script='research/monitor_batch89.py' if watchdog else 'research/run_batch89.py'
    extra=f' --controller-unit {controller} --repair' if watchdog else ' --num-workers 2'
    log='watchdog.log' if watchdog else 'batch.log'
    return f'''[Unit]
Description=TRACE-VQ batch89 {'independent watchdog' if watchdog else 'serial experiment queue'}
ConditionPathExists=!{output}/FINISHED
ConditionPathExists=!{output}/PAUSED
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory={ROOT}
Environment=PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=XDG_RUNTIME_DIR=/run/user/{uid}
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{uid}/bus
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
ExecStart={ROOT}/research/run.sh -u {ROOT}/{script} --output-dir {output}{extra}
StandardOutput=append:{output}/{log}
StandardError=inherit
Restart=always
RestartSec=30s
KillMode=control-group
TimeoutStopSec=30s

[Install]
WantedBy=default.target
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--install',action='store_true')
    args=p.parse_args();output=args.output_dir.resolve();assert output.is_relative_to(ROOT/'artifacts/research')
    output.mkdir(parents=True,exist_ok=True);units=[]
    for watchdog in (False,True):
        name='trace-vq-batch89-'+('watchdog-' if watchdog else '')+'20261108.service'
        path=output/name;path.write_text(unit_text(output,watchdog));units.append(path)
    if args.install:
        if (output/'FINISHED').exists():raise RuntimeError('Batch already finished; no service restart required')
        subprocess.run(['systemctl','--user','link',*[str(p) for p in units]],check=True)
        subprocess.run(['systemctl','--user','daemon-reload'],check=True)
        subprocess.run(['systemctl','--user','enable','--now',*[p.name for p in units]],check=True)
        subprocess.run(['systemctl','--user','show',*[p.name for p in units],'--property=Id','--property=ActiveState','--property=SubState','--property=MainPID'],check=True)
    else:
        for unit in units:print(unit)


if __name__=='__main__':main()
