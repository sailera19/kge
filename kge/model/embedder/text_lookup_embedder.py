import itertools
import json
from dataclasses import dataclass
from datetime import datetime

import os
import numpy as np
import regex
import tokenizers
from tokenizers import Tokenizer
from tokenizers import models
from tokenizers.pre_tokenizers import CharDelimiterSplit, BertPreTokenizer
from tokenizers import trainers
from torch import Tensor
import torch.nn
import torch.nn.functional
from transformers import BertModel

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

SENTENCE_SPLIT_REGEX = "(?<!\\d)\.(?<!\\d)"
UNKNOWN_TOKEN = "[UNK]"
INVERSE_TOKEN = "[INV]"

@dataclass
class TextLookupEmbedding:
    embeddings: Tensor
    attention_mask: Tensor
    tokens: Tensor

    def __len__(self):
        return len(self.embeddings)


class TextLookupEmbedder(KgeEmbedder):
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
        self.tokenizer: Tokenizer = None

        self.ids_to_token_ids, self.attention_mask = self.generate_token_mapping(dataset)

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = self.tokenizer.get_vocab_size()

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        self._embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse,
        )
        self._model = None

        if not init_for_load_only:
            # initialize weights
            self.initialize(self._embeddings.weight.data)
            self._normalize_embeddings()

        if self.get_option("from_huggingface_pretrained"):
            self._embeddings.weight.data = BertModel.from_pretrained(self.get_option("from_huggingface_pretrained")).get_input_embeddings().weight.data

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)


    def _normalize_embeddings(self):
        if self.normalize_p > 0:
            with torch.no_grad():
                self._embeddings.weight.data = torch.nn.functional.normalize(
                    self._embeddings.weight.data, p=self.normalize_p, dim=-1
                )

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0 and isinstance(job, TrainingJob):
            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self._normalize_embeddings())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self._normalize_embeddings())


    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: KgeEmbedder) -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        self._embeddings.weight[
            torch.from_numpy(self_intersect_ind)
            .to(self._embeddings.weight.device)
            .long()
        ] = pretrained_embedder.embed(torch.from_numpy(pretrained_intersect_ind)).to(
            self._embeddings.weight.device
        )

    def embed(self, indexes: Tensor) -> TextLookupEmbedding:
        return TextLookupEmbedding(self._postprocess(self._embeddings(self.ids_to_token_ids[indexes.long()].to(indexes.device))),
                                   self.attention_mask[indexes.long()].to(indexes.device), self.ids_to_token_ids[indexes.long()].to(indexes.device))

    def embed_all(self) -> TextLookupEmbedding:
        embeddings = self._embeddings_all()
        return TextLookupEmbedding(self._postprocess(embeddings.embeddings), embeddings.attention_mask, embeddings.tokens)

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> TextLookupEmbedding:
        return TextLookupEmbedding(self._embeddings(self.ids_to_token_ids.to(self._embeddings.weight.data.device)),
                                   self.attention_mask.to(self._embeddings.weight.data.device),
                                   self.ids_to_token_ids.to(self._embeddings.weight.data.device))

    def embed_tokens(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self._embeddings(indexes.long()))

    def embed_all_tokens(self) -> Tensor:
        return self._postprocess(self._embeddings(torch.arange(
                self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            )))

    @property
    def embedding_type(self):
        if "relation_embedder" in self.configuration_key or "relation_text_embedder" in self.configuration_key:
            return "relations"
        else:
            return "entities"

    @property
    def max_token_length(self):
        return self.ids_to_token_ids.shape[1]

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embeddings(unique_indexes)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result

    @property
    def device(self):
        return self._embeddings.weight.data.device

    def generate_token_mapping(self, dataset):
        self.train_tokenizer(dataset)

        ids_to_strings, _ = self.prepare_texts(dataset, self.get_entity_texts(dataset), self.embedding_type)

        entity_ids_to_tokens, attention_mask = self.encode_texts(ids_to_strings)

        return entity_ids_to_tokens, attention_mask

    def train_tokenizer(self, dataset):
        tokenizers_folder = os.path.join(dataset.folder, "tokenizers")
        if not os.path.exists(tokenizers_folder):
            os.mkdir(tokenizers_folder)

        tokenizers_dict_path = os.path.join(tokenizers_folder, "tokenizers.json")
        saved_tokenizers = {}
        if os.path.exists(tokenizers_dict_path):
            with open(tokenizers_dict_path, "r") as file:
                saved_tokenizers = json.load(file)
            for tokenizer_name, tokenizer_config in saved_tokenizers.items():
                if tokenizer_config == self.config.get(self.configuration_key):
                    self.tokenizer = tokenizers.Tokenizer.from_file(os.path.join(tokenizers_folder, tokenizer_name))

        if self.get_option("tokenizer_from_pretrained"):
            self.tokenizer = Tokenizer.from_pretrained(self.get_option("tokenizer_from_pretrained"))
            self.tokenizer.pre_tokenizer = BertPreTokenizer()

        text_bases = {}

        if self.embedding_type == "relations" or self.get_option("include_relation_texts"):
            ids_to_strings = self.get_relation_texts(dataset)
            text_bases["relations"] = {"ids_to_strings": ids_to_strings}

        if self.embedding_type == "entities":
            ids_to_strings = self.get_entity_texts(dataset)
            text_bases["entities"] = {"ids_to_strings": ids_to_strings}

        for text_base, mapping in text_bases.items():
            ids_to_strings, strings_in_train = self.prepare_texts(dataset, mapping["ids_to_strings"], text_base)
            text_bases[text_base] = {"ids_to_strings": ids_to_strings, "strings_in_train": strings_in_train}

        if not self.tokenizer:
            model_class = getattr(models, self.get_option("tokenizer_model"))
            self.tokenizer = Tokenizer(model_class(**self.get_option("tokenizer_model_parameters")))

            self.tokenizer.pre_tokenizer = BertPreTokenizer()

            trainer_class = getattr(trainers, self.get_option("tokenizer_trainer"))

            trainer = trainer_class(special_tokens=[UNKNOWN_TOKEN, INVERSE_TOKEN, "[PAD]"], **self.get_option("tokenizer_trainer_parameters"))

            strings_in_train = list(itertools.chain.from_iterable([x["strings_in_train"] for _, x in text_bases.items()]))

            self.tokenizer.train_from_iterator(strings_in_train, trainer=trainer)

            tokenizer_name = datetime.now().strftime("%Y%m%d%H%M%S")
            self.tokenizer.save(os.path.join(tokenizers_folder, tokenizer_name))

            saved_tokenizers.update({tokenizer_name: self.config.get(self.configuration_key)})

            with open(tokenizers_dict_path, "w") as file:
                json.dump(saved_tokenizers, file)


    def prepare_texts(self, dataset, ids_to_strings, text_base):
        ids_to_strings = [x.replace("_", " ") if isinstance(x, str) else UNKNOWN_TOKEN for x in ids_to_strings]
        max_sentence_count = self.get_option("max_sentence_count")
        if max_sentence_count:
            ids_to_strings = [". ".join(regex.split(SENTENCE_SPLIT_REGEX, x)[:max_sentence_count]) + "." for x in
                              ids_to_strings]
        max_word_count = self.get_option("max_word_count")
        if max_word_count:
            ids_to_strings = [" ".join(x.split(" ")[:max_word_count]) for x in ids_to_strings]
        remove_partial_sentences = self.get_option("remove_partial_sentences")
        if remove_partial_sentences:
            ids_to_strings = [". ".join(regex.split(SENTENCE_SPLIT_REGEX, x)[:-1]) + "." if len(
                regex.split(SENTENCE_SPLIT_REGEX, x)) > 1 else x for x in ids_to_strings]
        ids_to_strings = np.array(ids_to_strings)
        train_triples = dataset.load_triples("train")
        if text_base == "relations":
            strings_in_train = ids_to_strings[train_triples[:, 1].unique()]
        else:
            strings_in_train = ids_to_strings[torch.cat((train_triples[:, 0], train_triples[:, 2])).unique()]
        strings_in_train = strings_in_train[~(strings_in_train == None)]
        return ids_to_strings, strings_in_train

    def get_entity_texts(self, dataset):
        if self.get_option("text_source") == "descriptions":
            ids_to_strings = dataset.entity_descriptions()
        else:
            ids_to_strings = dataset.entity_strings()
        return ids_to_strings

    def get_relation_texts(self, dataset, include_inverse=False):
        ids_to_strings = dataset.relation_strings()
        # add inverse for reciprocals:
        if include_inverse or dataset.num_relations() == len(ids_to_strings) * 2:
            ids_to_strings = [*ids_to_strings, *[INVERSE_TOKEN + " " + x for x in ids_to_strings]]
        return ids_to_strings

    def encode_texts(self, ids_to_strings):
        self.tokenizer.enable_padding()
        output = self.tokenizer.encode_batch([x if x else "" for x in ids_to_strings], add_special_tokens=False)

        entity_ids_to_tokens = torch.tensor([x.ids for x in output], dtype=torch.int64)

        attention_mask = torch.tensor([x.attention_mask for x in output], dtype=torch.bool)
        return entity_ids_to_tokens, attention_mask

