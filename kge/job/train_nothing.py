import time

import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn


class TrainingJobNothing(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
            self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )

        if self.__class__ == TrainingJobNothing:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "triples": self.dataset.split(self.train_split)[batch, :].long()
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
            self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["triples"])

    def _process_subbatch(
            self,
            batch_index,
            batch,
            subbatch_slice,
            result: TrainingJob._ProcessBatchResult,
    ):
        pass
