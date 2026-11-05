"""Guard the number of new runs and scientifically matched seed comparisons."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.run_batch02 import Batch02, METRICS, key, tasks


class Batch02RunnerTests(unittest.TestCase):
    def test_plan_adds_only_missing_vq_seeds_and_four_fsq_seed0_runs(self):
        planned=tasks()
        self.assertEqual(len(planned),20)
        self.assertEqual(len(set(planned)),20)
        vq=[task for task in planned if not task[1].startswith("fsq")]
        fsq=[task for task in planned if task[1].startswith("fsq")]
        self.assertEqual(len(vq),16)
        self.assertEqual({task[2] for task in vq},{1,2,3,4})
        self.assertEqual(len(fsq),4)
        self.assertEqual({task[2] for task in fsq},{0})

    def test_treatment_deltas_use_same_seed_control_and_ddof1_summaries(self):
        def record(variant,seed,value):
            return dict(universe="csi300",variant=variant,stage1_seed=42,stage2_seed=seed,status="complete",
                        metrics={m:value for m in METRICS})
        runs={}
        for variant,seed,value in [("turnover_control",0,.05),("turnover",0,.1),
                                   ("turnover_control",1,.4),("turnover",1,.6),
                                   ("fsq_turnover_control",0,.7),("fsq_turnover",0,.9)]:
            runs[key("csi300",variant,seed)]=record(variant,seed,value)
        benchmarks={key("csi300","baseline_vq",0):record("baseline_vq",0,.01),
                    key("csi300","baseline_vq",1):record("baseline_vq",1,.2),
                    key("csi300","baseline_fsq",0):record("baseline_fsq",0,.5)}
        with tempfile.TemporaryDirectory() as directory:
            batch=object.__new__(Batch02)
            batch.output=Path(directory)
            batch.state=dict(runs=runs,benchmarks=benchmarks)
            batch.summarize()
            frame=pd.read_csv(batch.output/"results.csv")
            treatment=frame[(frame.variant=="turnover")&(frame.stage2_seed==1)].iloc[0]
            self.assertAlmostEqual(treatment.delta_RankIC,.2)
            self.assertAlmostEqual(treatment.delta_vs_family_RankIC,.4)
            fsq=frame[frame.variant=="fsq_turnover"].iloc[0]
            self.assertAlmostEqual(fsq.delta_RankIC,.2)
            self.assertAlmostEqual(fsq.delta_vs_vq_paired_RankIC,.8)
            summary=pd.read_csv(batch.output/"mean_std.csv")
            group=summary[summary.variant=="turnover"].iloc[0]
            self.assertEqual(group.completed_seeds,2)
            self.assertEqual(group.expected_seeds,5)
            self.assertAlmostEqual(group.RankIC_mean,.35)
            self.assertAlmostEqual(group.RankIC_std,2**.5*.25)


if __name__=="__main__":
    unittest.main()
