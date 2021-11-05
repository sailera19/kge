import time

import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn


class TrainingJob1vsAllMaskedEntityPrediction(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsAllMaskedEntityPrediction"
        self.scoring_fraction_replaced = \
            config.get("1vsAllMaskedEntityPrediction.scoring_fraction_replaced")
        self.scoring_fraction_unchanged = \
            config.get("1vsAllMaskedEntityPrediction.scoring_fraction_unchanged")
        self.entity_prediction_fraction = config.get("1vsAllMaskedEntityPrediction.entity_prediction_fraction")
        self.entity_prediction_fraction_replaced = \
            config.get("1vsAllMaskedEntityPrediction.entity_prediction_fraction_replaced")
        self.entity_prediction_fraction_unchanged = \
            config.get("1vsAllMaskedEntityPrediction.entity_prediction_fraction_unchanged")

        if self.__class__ == TrainingJob1vsAllMaskedEntityPrediction:
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
        result.prepare_time += time.time()

        # Link Prediction
        # forward/backward pass (sp)
        num_scoring_replaced = round(batch_size * self.scoring_fraction_replaced)
        num_scoring_unchanged = round(batch_size * self.scoring_fraction_unchanged)
        result.forward_time -= time.time()
        scores_sp = self.model.score_sp(
            triples[:, 0],
            triples[:, 1],
            num_replaced=num_scoring_replaced,
            num_unchanged=num_scoring_unchanged)
        loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        result.avg_loss += loss_value_sp.item()
        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss_value_sp.backward()
        result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()
        scores_po = self.model.score_po(
            triples[:, 1],
            triples[:, 2],
            num_replaced=num_scoring_replaced,
            num_unchanged=num_scoring_unchanged)
        loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        result.avg_loss += loss_value_po.item()
        result.forward_time += time.time()
        result.backward_time -= time.time()
        if not self.is_forward_only:
            loss_value_po.backward()
        result.backward_time += time.time()


        # Masked Entity Prediction
        num_entity_prediction = round(batch_size * self.entity_prediction_fraction)
        num_entity_prediction_replaced = round(num_entity_prediction * self.entity_prediction_fraction_replaced)
        num_entity_prediction_unchanged = round(num_entity_prediction * self.entity_prediction_fraction_unchanged)
        # forward/backward pass (sp)
        result.forward_time -= time.time()
        scores_sp = self.model.recover_entity_sp(
            triples[:num_entity_prediction, 0],
            triples[:num_entity_prediction, 1],
            num_replaced=num_entity_prediction_replaced,
            num_unchanged=num_entity_prediction_unchanged,
        )
        loss_value_sp = self.loss(scores_sp, triples[:num_entity_prediction, 0]) / batch_size
        result.avg_loss += loss_value_sp.item()
        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss_value_sp.backward()
        result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()
        scores_po = self.model.recover_entity_po(
            triples[:num_entity_prediction, 1],
            triples[:num_entity_prediction, 2],
            num_replaced=num_entity_prediction_replaced,
            num_unchanged=num_entity_prediction_unchanged,
        )
        loss_value_po = self.loss(scores_po, triples[:num_entity_prediction, 2]) / batch_size
        result.avg_loss += loss_value_po.item()
        result.forward_time += time.time()
        result.backward_time -= time.time()
        if not self.is_forward_only:
            loss_value_po.backward()
        result.backward_time += time.time()
