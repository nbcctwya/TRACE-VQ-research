"""Exercise end-of-batch publication in a disposable copy, including a failure."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from research import report_batch89 as report
from research.report_batch34 import portable
from research.run_batch01 import result_metrics
from trace_vq.cli.batch01 import reference_directory
from trace_vq.cli.batch89 import make_config
from trace_vq.models.batch89 import VARIANTS

REAL_ROOT=Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def test_two_serial_publications_preserve_failures_and_deduplicate_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shutil.copytree(REAL_ROOT/'experiments',root/'experiments')
            (root/'research').mkdir();(root/'docs').mkdir()
            shutil.copy2(REAL_ROOT/'research/organize_experiment_results.py',root/'research/organize_experiment_results.py')
            for name in ('README.md','docs/analysis-guide.md','docs/github-sync.md'):shutil.copy2(REAL_ROOT/name,root/name)
            original_run=subprocess.run
            def dispatch(command,**kwargs):
                if command[1]=='research/audit_git_scope.py':return subprocess.CompletedProcess(command,0)
                return original_run(command,**kwargs)
            for batch in (8,9):
                folder=root/f'artifacts/research/batch{batch:02d}_20261108';folder.mkdir(parents=True)
                rows=[];runs={}
                for market in ('csi300','csi800'):
                    baseline=result_metrics(reference_directory(market,'fsq'))
                    rows.append(dict(universe=market,variant='baseline_fsq',role='baseline',status='complete',new_training=False,
                        reused_reference=True,stage1_seed=42,stage2_seed=0,training_seconds=0,evaluation_backtest_seconds=0,
                        unsaved_training_seconds_lower_bound=0,**baseline))
                    for variant,settings in VARIANTS.items():
                        if settings['batch']!=batch:continue
                        output=folder/market/variant/'seed0';cfg=make_config(market,variant,output,0)
                        config=folder/'configs'/market/variant/'stage2_seed0.yaml';config.parent.mkdir(parents=True,exist_ok=True)
                        OmegaConf.save(OmegaConf.create(cfg),config)
                        s1_config=config.with_name('stage1_seed42.yaml');s1_cfg=make_config(market,variant,output,0,stage=1)
                        OmegaConf.save(OmegaConf.create(s1_cfg),s1_config)
                        failed=batch==9 and variant=='fsq_contrast' and market=='csi800'
                        row=dict(batch=batch,universe=market,variant=variant,stage1_seed=42,stage2_seed=0,role='new_model',
                            status='failed' if failed else 'complete',new_training=True,reused_reference=False,baseline_variant='baseline_fsq',
                            config=str(config),stage1_config=str(s1_config),stage1_output_dir=str(output/'stage1'),output_dir=str(output),unsaved_training_seconds_lower_bound=0)
                        if failed:row['error']='fixture: missing input; this is not a real experiment'
                        else:
                            row.update(standalone_exact=True,stage1_unchanged=True,stage1_retrained=True,stage1_training_seconds=1.,stage2_training_seconds=1.,training_seconds=2.,
                                evaluation_backtest_seconds=2.,training_seconds_including_replay_lower_bound=1.,completed_epochs=1,
                                gpu_peak_allocated_mib=128.,gpu_peak_reserved_mib=256.,**copy.deepcopy(baseline))
                            for metric in report.METRICS:row['delta_'+metric]=0.
                        rows.append(row);runs[market+'.'+variant]=row
                for name in ('results.csv','variant_results.csv','baselines.csv'):
                    f=pd.DataFrame(rows)
                    if name=='variant_results.csv':f=f[f.role.eq('new_model')]
                    if name=='baselines.csv':f=f[f.role.eq('baseline')]
                    f.to_csv(folder/name,index=False)
                (folder/'progress.json').write_text(json.dumps(dict(status='complete' if batch==8 else 'complete_with_failures',runs=runs,
                    recoveries=[],source_sha256={'fixture':'not scientific evidence'})))
                (folder/'original_integrity_after.log').write_text(json.dumps(dict(source_unchanged=True)))
                (folder/'validation.json').write_text(json.dumps(dict(unit_tests_passed=True,fixture_only=True)))
                (folder/'plan.json').write_text(json.dumps(dict(batch=batch,fixture_only=True)))
                with patch.object(report,'ROOT',root),patch.object(report,'portable',lambda text:portable(text.replace(str(root),str(REAL_ROOT)))),\
                     patch.object(sys,'argv',['report','--output-dir',str(folder),'--batch',str(batch)]),\
                     patch.object(report.subprocess,'run',side_effect=dispatch):
                    report.main()
                verified=json.loads((folder/'report_verification.json').read_text())
                self.assertEqual(verified['new_training_complete'],22 if batch==8 else 15)
            self.assertEqual(len(pd.read_csv(root/'experiments/research_runs.csv')),246)
            self.assertEqual(len(pd.read_csv(root/'experiments/research_mean_std.csv')),182)
            self.assertEqual(len(pd.read_csv(root/'experiments/config_index.csv')),334)
            failed=pd.read_csv(root/'experiments/batch09_20261108/results.csv')
            self.assertEqual(int(failed.status.eq('failed').sum()),1)
            self.assertTrue(failed.loc[failed.status.eq('failed'),'RankIC'].isna().all())
            self.assertNotIn(str(REAL_ROOT),(root/'experiments/batch08_20261108/plan.json').read_text())


if __name__=='__main__':unittest.main()
