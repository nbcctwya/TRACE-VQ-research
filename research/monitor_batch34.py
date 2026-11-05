"""Independent watchdog and GPU device sampler for the serial batch queue."""
import argparse
import csv
import fcntl
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];folder=args.output_dir.resolve()
    assert folder.is_relative_to(root/"artifacts/research")
    lock=(folder/"watchdog.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=folder/"resource_peaks.json";peaks=json.loads(path.read_text()) if path.exists() else {}
    last_warning=None
    while True:
        state=json.loads((folder/"progress.json").read_text());stamp=datetime.now(timezone.utc).isoformat()
        phase=state.get("current_phase") or "between_phases"
        try:
            query=subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],
                                 capture_output=True,text=True,check=True,timeout=5)
            for line in query.stdout.splitlines():
                index,utilization,used,total=[float(x.strip()) for x in line.split(",")]
                entry=dict(updated=stamp,phase=phase,gpu_index=int(index),utilization_percent=utilization,
                           device_memory_used_mib=used,device_memory_total_mib=total)
                report=folder/"resource_samples.csv";header=not report.exists()
                with report.open("a",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(entry))
                    if header:writer.writeheader()
                    writer.writerow(entry)
                info=peaks.setdefault(phase,dict(device_memory_peak_mib=0,samples=0))
                info.update(device_memory_peak_mib=max(info["device_memory_peak_mib"],used),samples=info["samples"]+1,last_sample=stamp)
            temporary=path.with_suffix(".partial.json");temporary.write_text(json.dumps(peaks,indent=2)+"\n");temporary.replace(path)
        except (subprocess.SubprocessError,FileNotFoundError,ValueError) as error:
            print(f"{stamp} GPU sample unavailable: {type(error).__name__}",flush=True)
        try:
            alive=b"research/run_batch34.py" in Path(f"/proc/{state['pid']}/cmdline").read_bytes()
        except (OSError,KeyError):alive=False
        warning=None
        if state["status"]=="running" and not alive:warning="Serial controller exited unexpectedly; inspect child/checkpoint before recovery"
        elif state["status"] in ("controller_failed","interrupted"):warning=f"Controller {state['status']}: {state.get('controller_error','')}"
        if warning and warning!=last_warning:
            with (folder/"alerts.jsonl").open("a") as f:f.write(json.dumps(dict(updated=stamp,phase=phase,warning=warning))+"\n")
            print(f"{stamp} ALERT {warning}",flush=True)
        last_warning=warning
        health=dict(updated=stamp,batch_status=state["status"],controller_alive=alive,current_phase=phase,warning=warning,
                    measurement="sampled whole-device memory includes desktop and other processes")
        temporary=folder/"watchdog.partial.json";temporary.write_text(json.dumps(health,indent=2)+"\n");temporary.replace(folder/"watchdog.json")
        if state["status"] in ("complete","complete_with_failures") and (folder/"report_verification.json").exists():
            print(stamp+" Both batches and reports complete; watchdog finished",flush=True);break
        time.sleep(20)


if __name__=="__main__":main()
