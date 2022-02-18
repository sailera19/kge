import math

import torch
from datetime import datetime
from kge import Config, Dataset
from kge.job import EvaluationJob, EntityRankingJob


class QualitativeEvaluationJob(EvaluationJob):

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.num_samples = self.config.get("qualitative_eval.num_samples")
        self.top_k = self.config.get("qualitative_eval.top_k")

    def _prepare(self):
        super()._prepare()
        """Construct all indexes needed to run."""

        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            shuffle=False,
            batch_size=1,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    @torch.no_grad()
    def _evaluate(self):
        chunk_size = self.dataset.num_entities()
        done = False

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="qualitative_entity_ranking",
            scope="epoch",
            split=self.eval_split,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
        )

        while not done:
            output_list = []
            try:
                for batch_number, triples in enumerate(self.loader):
                    self.config.print(f"Starting with Batch Number {batch_number}", end="", flush=True,)
                    triples = triples.to(self.device)
                    batch_scores_sp = None
                    batch_scores_po = None
                    for chunk_number in range(math.ceil(self.dataset.num_entities() / chunk_size)):
                        chunk_start = chunk_number * chunk_size
                        chunk_end = (chunk_number + 1) * chunk_size
                        chunk_end = self.dataset.num_entities() if chunk_end > self.dataset.num_entities() else chunk_end

                        chunk_scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1], torch.arange(chunk_start, chunk_end, device=self.device))
                        chunk_scores_po = self.model.score_po(triples[:, 1], triples[:, 2], torch.arange(chunk_start, chunk_end, device=self.device))

                        batch_scores_sp = chunk_scores_sp if batch_scores_sp is None else torch.cat((batch_scores_sp, chunk_scores_sp), dim=1)
                        batch_scores_po = chunk_scores_po if batch_scores_po is None else torch.cat(
                            (batch_scores_po, chunk_scores_po), dim=1)
                    for i in range(len(triples)):
                        output_list.append((
                            triples[i].tolist(),
                            batch_scores_sp[i].sort(dim=0, descending=True)[1][:self.top_k].tolist(),
                            batch_scores_po[i].sort(dim=0, descending=True)[1][:self.top_k].tolist()))
                    if len(output_list) >= self.num_samples:
                        break
                self.config.print("\033[2K\r", end="", flush=True)
                done = True
            except RuntimeError as e:
                chunk_size = math.ceil(chunk_size / 2)
                self.config.log("Ran out of memory. New chunk_size: ", chunk_size)

        file_path = f"{self.config.folder}/quality_eval_{self.eval_split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(file_path, "w") as file:

            for item in output_list:
                file.write(str(item))
                file.write("\n")

        entity_mapping = self.dataset.entity_strings()
        relation_mapping = self.dataset.relation_strings()

        output_list = [((entity_mapping[x[0][0]], relation_mapping[x[0][1]], entity_mapping[x[0][2]]),
                        [entity_mapping[y] for y in x[1]],
                        [entity_mapping[y] for y in x[2]]) for x in output_list]

        file_path = f"{self.config.folder}/quality_eval_names_{self.eval_split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(file_path, "w") as file:

            for item in output_list:
                file.write(str(item))
                file.write("\n")

        self.config.print("Quality Evaluation finished")
        for output in output_list:
            self.config.print(f"Tail for triple {output[0]}: {output[1]}")
            self.config.print(f"Head for triple {output[0]}: {output[2]}")

