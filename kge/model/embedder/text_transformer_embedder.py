import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import CharDelimiterSplit
from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List


class TextTransformerEmbedder(KgeEmbedder):
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

        self.entity_vocab_size = vocab_size

        _, _, vocab_size = self.generate_token_mapping(dataset, config, configuration_key)

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

        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.dim))
        self.initialize(self.cls_emb)

        self.feedforward_dim = self.get_option("encoder.dim_feedforward")
        if not self.feedforward_dim:
            # set ff dim to 4 times of embeddings dim, as in Vaswani 2017 and Devlin 2019
            self.feedforward_dim = self.dim * 4

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.get_option("encoder.nhead"),
            dim_feedforward=self.feedforward_dim,
            dropout=dropout,
            activation=self.get_option("encoder.activation"),
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.get_option("encoder.num_layers")
        )
        for layer in self.encoder.layers:
            self.initialize(layer.linear1.weight.data)
            self.initialize(layer.linear2.weight.data)
            self.initialize(layer.self_attn.out_proj.weight.data)

            if layer.self_attn._qkv_same_embed_dim:
                self.initialize(layer.self_attn.in_proj_weight)
            else:
                self.initialize(layer.self_attn.q_proj_weight)
                self.initialize(layer.self_attn.k_proj_weight)
                self.initialize(layer.self_attn.v_proj_weight)

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

    def embed(self, indexes: Tensor) -> Tensor:
        tokens, attention_mask, _ = self.dataset.index("entity_ids_to_tokens")

        batch_size = len(indexes)

        text_embeddings = self._embeddings(tokens[indexes.long()].to(indexes.device))

        # transform the sp pairs
        text_embeddings = self.encoder.forward(
            torch.cat(
                (
                    self.cls_emb.repeat((1, batch_size, 1)),
                    text_embeddings.transpose(1, 0)
                ),
                dim=0,
            ),
            src_key_padding_mask=~torch.cat(
                (
                    torch.ones(batch_size, 1, dtype=torch.bool, device=text_embeddings.device),
                    attention_mask[indexes.long()].to(text_embeddings.device)
                ),
                dim=1)
        )[0, :]

        return self._postprocess(text_embeddings)

    def embed_all(self) -> Tensor:
        return self.embed(torch.arange(
                self.entity_vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            ))

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

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
        name = "entity_ids_to_tokens"

        if not dataset._indexes.get(name):
            entity_ids_to_strings = np.array(dataset.entity_strings())
            train_triples = dataset.load_triples("train")
            strings_in_train = entity_ids_to_strings[torch.cat((train_triples[:, 0], train_triples[:, 2])).unique()]
            strings_in_train = strings_in_train[~(strings_in_train == None)]
            tokenizer = Tokenizer(BPE())

            tokenizer.pre_tokenizer = CharDelimiterSplit("_")

            from tokenizers.trainers import BpeTrainer

            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], **self.get_option("tokenization.trainer"))

            tokenizer.train_from_iterator(strings_in_train, trainer=trainer)

            tokenizer.enable_padding()

            output = tokenizer.encode_batch([x if x else "" for x in entity_ids_to_strings])

            entity_ids_to_tokens = torch.tensor([x.ids for x in output], dtype=torch.int64)

            attention_mask = torch.tensor([x.attention_mask for x in output], dtype=torch.bool)

            dataset._indexes[name] = entity_ids_to_tokens, attention_mask, tokenizer.get_vocab_size()

        return dataset._indexes[name]