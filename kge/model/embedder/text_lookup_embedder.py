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
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

SENTENCE_SPLIT_REGEX = "(?<!\\d)\.(?<!\\d)"

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

        _, _, vocab_size = self.generate_token_mapping(dataset, config, configuration_key + ".tokenization")

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        self._embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse,
        )

        if not init_for_load_only:
            # initialize weights
            self.initialize(self._embeddings.weight.data)
            self._normalize_embeddings()

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
        tokens, attention_mask, _ = self.dataset.index(self.index_name)
        return TextLookupEmbedding(self._postprocess(self._embeddings(tokens.to(indexes.device)[indexes.long()])),
                                   attention_mask.to(indexes.device)[indexes.long()], tokens[
                                       indexes.long()])

    def embed_all(self) -> TextLookupEmbedding:
        embeddings = self._embeddings_all()
        return TextLookupEmbedding(self._postprocess(embeddings.embeddings), embeddings.attention_mask, embeddings.tokens)

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> TextLookupEmbedding:
        tokens, attention_mask, _ = self.dataset.index(self.index_name)
        all_entities = torch.arange(self.lookup_vocab_size, dtype=torch.long, device=self._embeddings.weight.device)
        return TextLookupEmbedding(self._embeddings(all_entities),
                                   attention_mask.to(self._embeddings.weight.data.device),
                                   tokens.to(self._embeddings.weight.data.device))

    def embed_tokens(self, indexes: Tensor) -> TextLookupEmbedding:
        return self._postprocess(self._embeddings(indexes.long()))

    def embed_all_tokens(self) -> Tensor:
        return self._postprocess(self._embeddings(torch.arange(
                self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            )))

    @property
    def index_name(self):
        if "relation_embedder" in self.configuration_key or "relation_text_embedder" in self.configuration_key:
            return "relation_ids_to_tokens"
        else:
            return "entity_ids_to_tokens"
    @property
    def max_token_length(self):
        tokens, _, _ = self.dataset.index(self.index_name)
        return tokens.shape[1]

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

    def generate_token_mapping(self, dataset, config, configuration_key):
        name = self.index_name

        if not dataset._indexes.get(name):
            unknown_token = "[UNK]"
            special_tokens = [unknown_token]

            if name == "relation_ids_to_tokens":
                ids_to_strings = dataset.relation_strings()
                # add inverse for reciprocals:
                if dataset.num_relations() == len(ids_to_strings) * 2:
                    inverse_token = "[INV]"
                    special_tokens.append(inverse_token)
                    ids_to_strings = [*ids_to_strings, *[inverse_token + " " + x for x in ids_to_strings]]
            else:
                if self.get_option("text_source") == "descriptions":
                    ids_to_strings = dataset.entity_descriptions()
                else:
                    ids_to_strings = dataset.entity_strings()
            ids_to_strings = [x.replace("_", " ") if isinstance(x, str) else "[UNK]" for x in
                                     ids_to_strings]
            max_sentence_count = config.get(configuration_key).get("max_sentence_count", 0)
            if max_sentence_count > 0:
                ids_to_strings = [". ".join(regex.split(SENTENCE_SPLIT_REGEX, x)[:max_sentence_count]) + "." for x in ids_to_strings]
            max_word_count = config.get(configuration_key).get("max_word_count", 0)
            if max_word_count > 0:
                ids_to_strings = [" ".join(x.split(" ")[:max_word_count]) for x in ids_to_strings]
            remove_partial_sentences = config.get(configuration_key).get("remove_partial_sentences", False)
            if remove_partial_sentences:
                ids_to_strings = [". ".join(regex.split(SENTENCE_SPLIT_REGEX, x)[:-1]) + "." if len(regex.split(SENTENCE_SPLIT_REGEX, x)) > 1 else x for x in ids_to_strings]
            ids_to_strings = np.array(ids_to_strings)
            train_triples = dataset.load_triples("train")
            if name == "relation_ids_to_tokens":
                strings_in_train = ids_to_strings[train_triples[:, 1].unique()]
            else:
                strings_in_train = ids_to_strings[torch.cat((train_triples[:, 0], train_triples[:, 2])).unique()]
            strings_in_train = strings_in_train[~(strings_in_train == None)]

            tokenizer = Tokenizer(BPE())

            tokenizer.pre_tokenizer = BertPreTokenizer()

            from tokenizers.trainers import BpeTrainer

            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                                 **config.get(configuration_key).get("trainer"))

            tokenizer.train_from_iterator(strings_in_train, trainer=trainer)

            tokenizer.enable_padding()

            output = tokenizer.encode_batch([x if x else "" for x in ids_to_strings])

            entity_ids_to_tokens = torch.tensor([x.ids for x in output], dtype=torch.int64)

            attention_mask = torch.tensor([x.attention_mask for x in output], dtype=torch.bool)

            dataset._indexes[name] = entity_ids_to_tokens, attention_mask, tokenizer.get_vocab_size()

        return dataset._indexes[name]
