"""Real retry behavior and watchdog boundaries for an unattended batch."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.run_batch67 import Batch67,SOURCES,ROOT,tasks
from research.monitor_batch67 import repair,sample_gpu
from trace_vq.data.frozen import sha256


class RunnerTests(unittest.TestCase):
    def queue(self,folder):
        q=object.__new__(Batch67);q.output=folder;q.args=SimpleNamespace(num_workers=2)
        q.source={p:sha256(ROOT/p) for p in SOURCES};q.path=folder/'progress.json'
        q.state=dict(status='running',runs={},phases={},recoveries=[])
        return q

    def test_exactly_twenty_two_tasks_and_batch06_precedes_batch07(self):
        work=tasks();self.assertEqual(len(work),22);self.assertEqual(len(set(work)),22)
        self.assertEqual({m for m,v in work},{'csi300','csi800'})
        from trace_vq.models.batch67 import VARIANTS
        self.assertEqual([VARIANTS[v]['batch'] for m,v in work],[6]*14+[7]*8)

    def test_failed_child_retries_preserving_state_and_adopts_success(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);q=self.queue(folder);script=folder/'child.py'
            script.write_text("from pathlib import Path\nimport sys\np=Path(sys.argv[1]);n=int(p.read_text()) if p.exists() else 0\np.write_text(str(n+1))\nif n==0:raise SystemExit(9)\nPath(sys.argv[2]).write_text('done')\n")
            counter=folder/'attempts';done=folder/'done'
            q.phase('probe.evaluate',[sys.executable,str(script),str(counter),str(done)],folder/'phase.log',[done])
            self.assertEqual(counter.read_text(),'2');self.assertEqual(q.state['phases']['probe.evaluate']['status'],'complete')
            q.phase('probe.evaluate',[sys.executable,str(script),str(counter),str(done)],folder/'phase.log',[done])
            self.assertEqual(counter.read_text(),'2')

    def test_permanent_failure_is_bounded_at_three_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);q=self.queue(folder)
            with self.assertRaises(RuntimeError):q.phase('probe.evaluate',[sys.executable,'-c','raise SystemExit(7)'],folder/'fail.log',[folder/'missing'])
            self.assertEqual(len(q.state['phases']['probe.evaluate']['attempts']),3)

    def test_watchdog_respects_pause_completion_scope_and_repair_limits(self):
        with tempfile.TemporaryDirectory() as directory,patch('research.monitor_batch67.subprocess.run') as run:
            folder=Path(directory);unit='trace-vq-batch67-test.service';health=dict(finished=False,paused=True,warning='controller_not_running')
            self.assertIsNone(repair(folder,unit,{},health));run.assert_not_called()
            health.update(paused=False,finished=True);self.assertIsNone(repair(folder,unit,{},health));run.assert_not_called()
            health['finished']=False
            with self.assertRaises(AssertionError):repair(folder,'unrelated.service',{},health)
            self.assertEqual(repair(folder,unit,dict(controller_error='Training implementation changed'),health),'source_changed_requires_review')
            import time
            (folder/'repair_history.json').write_text(json.dumps([dict(time_epoch=time.time()-200)]*3))
            self.assertEqual(repair(folder,unit,{},health),'automatic_repair_limit_reached');run.assert_not_called()

    def test_gpu_sampler_works_with_minimal_systemd_path(self):
        with tempfile.TemporaryDirectory() as directory,patch('research.monitor_batch67.shutil.which',return_value=None),patch('research.monitor_batch67.subprocess.run') as run:
            run.return_value=subprocess.CompletedProcess([],0,stdout='0, 35, 4096, 16311\n',stderr='')
            folder=Path(directory);sample=sample_gpu(folder,'test.train')
            self.assertEqual(sample['status'],'available');self.assertEqual(run.call_args.args[0][0],'/usr/lib/wsl/lib/nvidia-smi')
            self.assertEqual(json.loads((folder/'resource_peaks.json').read_text())['test.train']['device_memory_peak_mib'],4096)


if __name__=='__main__':unittest.main()
