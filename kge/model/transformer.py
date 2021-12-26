import torch
import torch.nn
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel, KgeEmbedder


class TransformerScorer(RelationalScorer):
    r"""Scorer that uses a plain Transformer encoder.

    Concatenates (1) CLS embedding, (2) subject entity embedding (one per entity) +
    subject type embedding, (3) relation embedding (one per relation) + relation type
    embedding. Then runs transformer encoder and takes dot product with transformed CLS
    emebdding and object entity embedding to produce score.

    Must be used with ReciprocalRelationsModel.

    Based on the "No context" model from:

    HittER: Hierarchical Transformers for Knowledge Graph Embeddings
    Sanxing Chen, Xiaodong Liu, Jianfeng Gao, Jian Jiao, Ruofei Zhang and Yangfeng Ji
    https://arxiv.org/abs/2008.12813

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.emb_dim = self.get_option("entity_embedder.dim")

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.cls_emb)
        # the type embeddings
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_type_emb)
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_type_emb)

        # the type embeddings
        self.sub_text_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_text_type_emb)

        self.feedforward_dim = self.get_option("encoder.dim_feedforward")
        if not self.feedforward_dim:
            # set ff dim to 4 times of embeddings dim, as in Vaswani 2017 and Devlin 2019
            self.feedforward_dim = self.emb_dim * 4

        dropout = self.get_option("encoder.dropout")
        if dropout < 0.0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.encoder.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0.0

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
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

        _, _, num_tokens = dataset.index("entity_ids_to_tokens")

        self._text_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".text_embedder",
                num_tokens)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str, ground_truth_s: Tensor, ground_truth_p: Tensor, ground_truth_o: Tensor, **kwargs):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        tokens, attention_mask, _ = self.dataset.index("entity_ids_to_tokens")

        s_text_embeddings = self._text_embedder.embed(tokens[ground_truth_s.long()].to(ground_truth_s.device))

        # transform the sp pairs
        batch_size = len(s_emb)
        out = self.encoder.forward(
            torch.cat(
                (
                    self.cls_emb.repeat((1, batch_size, 1)),
                    (s_emb + self.sub_type_emb.unsqueeze(0)).unsqueeze(0),
                    (p_emb + self.rel_type_emb.unsqueeze(0)).unsqueeze(0),
                    (s_text_embeddings + self.sub_text_type_emb.unsqueeze(0)).transpose(1, 0)
                ),
                dim=0,
            ),
            src_key_padding_mask=~torch.cat(
                (
                    torch.ones(batch_size, 3, dtype=torch.bool, device=ground_truth_s.device),
                    attention_mask[ground_truth_s.long()].to(ground_truth_s.device)
                ),
                dim=1)
        )  # SxNxE = 3 x batch_size x emb_size

        # pick the transformed CLS embeddings
        out = out[0, ::]

        # now take dot product
        if combine == "sp_":
            out = torch.mm(out, o_emb.transpose(1, 0))
        elif combine == "spo":
            out = (out * o_emb).sum(-1)
        else:
            raise Exception("can't happen")

        # all done
        return out.view(batch_size, -1)


class Transformer(KgeModel):
    r"""Implementation of the Transformer KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransformerScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, **kwargs) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If Transformer is wrapped in a reciprocal relations model, this will always be
        # the case.
        if direction == "o":
            super().score_spo(s, p, o, direction)
        else:
            raise ValueError("Transformer can only score objects")
