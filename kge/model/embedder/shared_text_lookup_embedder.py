from dataclasses import dataclass

import numpy as np
import regex
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import CharDelimiterSplit, BertPreTokenizer
from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder, TextLookupEmbedder
from kge.misc import round_to_points

from typing import List

from kge.model.embedder.text_lookup_embedder import TextLookupEmbedding

SENTENCE_SPLIT_REGEX = "(?<!\\d)\.(?<!\\d)"

class SharedTextLookupEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        self.lookup_vocab_size = vocab_size

        self.shared_embedder: TextLookupEmbedder = None
        self.ids_to_to_token_ids: Tensor = None
        self.attention_mask: Tensor = None

        self.vocab_size: int = None

    def _set_shared_lookup_embedder(self, embedder, dataset):
        self.shared_embedder = embedder
        self.vocab_size = embedder.vocab_size
        include_inverse = dataset.num_relations() * 2 == self.lookup_vocab_size
        ids_to_strings, _ = self.shared_embedder.prepare_texts(
            dataset,
            self.shared_embedder.get_relation_texts(dataset, include_inverse=include_inverse),
            self.embedding_type)

        self.ids_to_token_ids, self.attention_mask = self.shared_embedder.encode_texts(ids_to_strings)

    @property
    def device(self):
        if self.shared_embedder:
            return self.shared_embedder.device

    @property
    def embedding_type(self):
        if "relation_embedder" in self.configuration_key or "relation_text_embedder" in self.configuration_key:
            return "relations"
        else:
            return "entities"

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        job.pre_run_hooks.append(self._pre_run_hook)

        super().prepare_job(job, **kwargs)

    def _pre_run_hook(self, job):
        if self.shared_embedder is None:
            self._set_shared_lookup_embedder(job.model.get_s_embedder(), job.dataset)

    def embed(self, indexes: Tensor) -> TextLookupEmbedding:
        token_ids = self.ids_to_token_ids[indexes.long()].to(self.device)
        return TextLookupEmbedding(self.shared_embedder.embed_tokens(token_ids), self.attention_mask[indexes.long()].to(self.device), token_ids)

    def embed_all(self) -> TextLookupEmbedding:
        all_ids = torch.arange(self.lookup_vocab_size, dtype=torch.long, device=self.device)
        return self.embed(all_ids)

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def embed_tokens(self, indexes: Tensor) -> Tensor:
        return self.shared_embedder.embed_tokens(indexes)

    def embed_all_tokens(self) -> Tensor:
        return self.shared_embedder.embed_all_tokens()

    @property
    def max_token_length(self):
        return self.ids_to_token_ids.shape[1]

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")
