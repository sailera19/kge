import torch
import torch.nn
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class CokEScorer(RelationalScorer):
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
        self.s_cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.s_cls_emb)
        self.o_cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_cls_emb)
        # the type embeddings
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_type_emb)
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_type_emb)
        self.obj_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.obj_type_emb)

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

    def score_emb(self, s_emb, p_emb, o_emb, combine: str, direction=None, **kwargs):
        #if combine not in ["sp_", "spo"]:
        #    raise ValueError(
        #        "Combine {} not supported in Transformer's score function".format(
        #            combine
        #        )
        #    )
        if direction is None:
            if combine == "sp_":
                direction = "o"
            elif combine == "_po":
                direction = "s"
            else:
                raise ValueError("Unknown combine or direction.")

        # transform the sp pairs
        batch_size = len(p_emb)
        s_size = len(s_emb)
        o_size = len(o_emb)

        out = self.encoder.forward(
            torch.stack(
                (
                    s_emb + self.sub_type_emb.unsqueeze(0) if direction == "o" else self.s_cls_emb.repeat((batch_size, 1)),
                    p_emb + self.rel_type_emb.unsqueeze(0),
                    self.o_cls_emb.repeat((batch_size, 1)) if direction == "o" else o_emb + self.obj_type_emb.unsqueeze(0),
                ),
                dim=0,
            )
        )  # SxNxE = 3 x batch_size x emb_size

        # pick the transformed CLS embeddings
        out = out[2 if direction == "o" else 0, ::]

        # now take dot product
        if combine == "sp_" or combine == "_po":
            out = torch.mm(out, (o_emb if direction == "o" else s_emb).transpose(1, 0))
        elif combine == "spo":
            out = (out * (o_emb if direction == "o" else s_emb)).sum(-1)
        else:
            raise TypeError()

        # all done
        return out.view(batch_size, -1)


class CokE(KgeModel):
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
            scorer=CokEScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

