"""Atomic latest-epoch recovery, independent of whether validation improves."""
import os
import time
from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from trace_vq.cli.screening import now
from trace_vq.data.frozen import atomic_json
from trace_vq.training.research_run import RecoverableEpochRecord


class BatchHeartbeat(RecoverableEpochRecord):
    def beat(self,trainer,stage):
        atomic_json(self.output/'training_heartbeat.json',dict(updated=now(),stage=stage,epoch=trainer.current_epoch,
            global_step=trainer.global_step,fit_seconds=self.previous_fit_seconds+time.perf_counter()-self.fit_start,
            gpu_peak_allocated_mib=max(self.previous_peak_allocated,torch.cuda.max_memory_allocated()/2**20 if self.device.startswith('cuda') else 0),
            gpu_peak_reserved_mib=max(self.previous_peak_reserved,torch.cuda.max_memory_reserved()/2**20 if self.device.startswith('cuda') else 0)))

    def on_fit_start(self,trainer,pl_module):
        super().on_fit_start(trainer,pl_module);self.beat(trainer,'fit_start')

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if batch_idx%200==0:self.beat(trainer,'training')

    def on_validation_end(self,trainer,pl_module):
        super().on_validation_end(trainer,pl_module)
        if not trainer.sanity_checking:self.beat(trainer,'validated')


class AtomicEpochCheckpoint(ModelCheckpoint):
    def _save_last_checkpoint(self,trainer,monitor_candidates):
        if not self.save_last:return
        directory=Path(self.dirpath);directory.mkdir(parents=True,exist_ok=True)
        last=directory/'last.ckpt';temporary=directory/'last.partial.ckpt';previous=directory/'last.previous.ckpt'
        self.last_model_path=str(last)
        trainer.save_checkpoint(str(temporary))
        with temporary.open('rb') as f:os.fsync(f.fileno())
        if last.exists():
            link=directory/'last.previous.partial.ckpt';link.unlink(missing_ok=True);os.link(last,link);os.replace(link,previous)
        os.replace(temporary,last)
        self._last_global_step_saved=trainer.global_step;self._last_checkpoint_saved=str(last)
        recorder=next((c for c in trainer.callbacks if isinstance(c,BatchHeartbeat)),None)
        seconds=recorder.previous_fit_seconds+time.perf_counter()-recorder.fit_start if recorder else None
        atomic_json(directory/'latest_checkpoint.json',dict(updated=now(),epoch=trainer.current_epoch,
            global_step=trainer.global_step,path=str(last),fit_seconds=seconds))

    def on_train_epoch_end(self,trainer,pl_module):
        super().on_train_epoch_end(trainer,pl_module)
        # Lightning 2.6 only invokes save_last when top-k actually saves.
        # An unchanged validation best must still yield a fresh resume point.
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            if self._last_global_step_saved!=trainer.global_step:
                self._save_last_checkpoint(trainer,self._monitor_candidates(trainer))
