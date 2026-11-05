"""Independent batch02 heartbeat watchdog and whole-device GPU sampling."""
import argparse
import csv
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    folder=args.output_dir.resolve()
    assert folder.is_relative_to(root/"artifacts/research") and (folder/"progress.json").exists()
    lock=(folder/"watchdog.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    peaks_path=folder/"resource_peaks.json"
    peaks=json.loads(peaks_path.read_text()) if peaks_path.exists() else {}
    last_alert=None
    while True:
        state=json.loads((folder/"progress.json").read_text())
        phase=state.get("current_phase") or "between_phases"
        stamp=datetime.now(timezone.utc).isoformat()
        query=["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"]
        try:
            result=subprocess.run(query,capture_output=True,text=True,check=True,timeout=5)
            for line in result.stdout.splitlines():
                index,utilization,used,total=[float(s.strip()) for s in line.split(",")]
                entry=dict(updated=stamp,phase=phase,gpu_index=int(index),utilization_percent=utilization,
                           device_memory_used_mib=used,device_memory_total_mib=total)
                path=folder/"resource_samples.csv"
                header=not path.exists()
                with path.open("a",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(entry))
                    if header: writer.writeheader()
                    writer.writerow(entry)
                peak=peaks.setdefault(phase,dict(device_memory_peak_mib=0,samples=0))
                peak.update(device_memory_peak_mib=max(peak["device_memory_peak_mib"],used),
                            samples=peak["samples"]+1,last_sample=stamp)
            temporary=peaks_path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(peaks,indent=2)+"\n"); temporary.replace(peaks_path)
        except (subprocess.SubprocessError,FileNotFoundError,ValueError) as error:
            print(f"{stamp} GPU sampling unavailable: {type(error).__name__}",flush=True)
        pid=state.get("pid")
        alive=bool(pid and Path(f"/proc/{pid}/cmdline").exists() and
                   b"research/run_batch02.py" in Path(f"/proc/{pid}/cmdline").read_bytes())
        warning=None
        if state["status"]=="running" and not alive:
            warning="Batch controller exited unexpectedly; inspect checkpoint and child before restarting"
        elif state["status"] in ("controller_failed","interrupted"):
            warning=f"Controller status {state['status']}: {state.get('controller_error','')}"
        if warning and warning!=last_alert:
            with (folder/"alerts.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(updated=stamp,phase=phase,warning=warning))+"\n")
            print(f"{stamp} ALERT {warning}",flush=True)
        last_alert=warning
        health=dict(updated=stamp,controller_alive=alive,batch_status=state["status"],current_phase=phase,
                    warning=warning,measurement="whole-device sampled memory includes desktop/other processes")
        temporary=folder/"watchdog.partial.json"
        temporary.write_text(json.dumps(health,indent=2)+"\n"); temporary.replace(folder/"watchdog.json")
        if state["status"] in ("complete","complete_with_failures") and (folder/"report_verification.json").exists() and not state.get("current_phase"):
            print(f"{stamp} Batch and aggregate report complete; monitoring finished",flush=True)
            break
        time.sleep(20)


if __name__=="__main__":
    main()
