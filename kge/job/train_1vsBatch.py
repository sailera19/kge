import time

import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn


class TrainingJob1vsBatch(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
            self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsBatch"

        self.self_pred_loss_factor_type = config.get_default(self.type_str + ".self_pred_loss.factor_type")
        self.self_pred_loss_factor = config.get_default(self.type_str + ".self_pred_loss.factor")
        self.self_pred_loss_factor_max_epoch = config.get_default(self.type_str + ".self_pred_loss.max_epoch")
        self.self_pred_loss_smoothing = config.get_default(self.type_str + ".self_pred_loss.smoothing")
        self.self_pred_loss_scale_at_epoch_start = config.get_default(self.type_str + ".self_pred_loss.scale_at_epoch_start")
        self.scaling_factor = 1
        self.scaling_epoch = -1


        if self.__class__ == TrainingJob1vsBatch:
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
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        if self.self_pred_loss_factor_type == "decreasing":
            self_pred_loss_factor = (
                ((self.self_pred_loss_smoothing + self.self_pred_loss_factor_max_epoch + 1 - self.epoch)
                 / self.self_pred_loss_factor_max_epoch + self.self_pred_loss_smoothing)
                * self.self_pred_loss_factor
                if self.epoch < self.self_pred_loss_factor_max_epoch else 0
            )
        else:
            self_pred_loss_factor = self.self_pred_loss_factor
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        result.forward_time -= time.time()
        unique_targets, traceback = triples[:, 2].unique(return_inverse=True)
        scores_sp = self.model.score_sp(
            triples[:, 0],
            triples[:, 1],
            unique_targets,
            ground_truth=triples[:, 2])
        if isinstance(scores_sp, tuple):
            scores_sp, self_pred_loss_sp = scores_sp
            loss_value_sp = self.loss(scores_sp, traceback) / batch_size
            if self.self_pred_loss_scale_at_epoch_start and self.scaling_epoch != self.epoch:
                self.scaling_factor = loss_value_sp.item() / self_pred_loss_sp.item()
                self.scaling_epoch = self.epoch
            self_pred_loss_sp *= self_pred_loss_factor * self.scaling_factor
            loss_value_sp = loss_value_sp + self_pred_loss_sp
            result.avg_loss_self += self_pred_loss_sp.item()
        else:
            loss_value_sp = self.loss(scores_sp, traceback) / batch_size
        result.avg_loss += loss_value_sp.item()
        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss_value_sp.backward()
        result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()

        unique_targets, traceback = triples[:, 0].unique(return_inverse=True)
        scores_po = self.model.score_po(
            triples[:, 1],
            triples[:, 2],
            unique_targets,
            ground_truth=triples[:, 0])
        if isinstance(scores_po, tuple):
            scores_po, self_pred_loss_po = scores_po
            loss_value_po = self.loss(scores_po, traceback) / batch_size
            self_pred_loss_po *= self_pred_loss_factor * self.scaling_factor
            loss_value_po = loss_value_po + self_pred_loss_po
            result.avg_loss_self += self_pred_loss_po.item()
        else:
            loss_value_po = self.loss(scores_po, traceback) / batch_size
        result.avg_loss += loss_value_po.item()
        result.forward_time += time.time()
        result.backward_time -= time.time()
        if not self.is_forward_only:
            loss_value_po.backward()
        result.backward_time += time.time()
