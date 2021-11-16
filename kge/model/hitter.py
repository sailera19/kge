import torch
import torch.nn
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_context_model import KgeContextModel
from kge.model.kge_model import RelationalScorer, KgeModel


class HitterScorer(RelationalScorer):
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

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None, embedding_weights=None):
        super().__init__(config, dataset, configuration_key)
        self.emb_dim = self.get_option("entity_embedder.dim")

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.cls_emb)
        self.gcls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.gcls_emb)

        # the type embeddings
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_type_emb)
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_type_emb)
        self.src_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.src_type_emb)
        self.context_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.context_type_emb)

        # masked entity prediction embeddings
        self.mask_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.mask_emb)

        dropout = self.get_option("encoder.dropout")
        if dropout < 0.0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.encoder.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0.0

        self.entity_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.get_option("encoder.nhead"),
            dim_feedforward=self.get_option("encoder.dim_feedforward"),
            dropout=dropout,
            activation=self.get_option("encoder.activation"),
        )
        self.entity_encoder = torch.nn.TransformerEncoder(
            self.entity_encoder_layer, num_layers=self.get_option("encoder.entity_encoder.num_layers")
        )
        for layer in self.entity_encoder.layers:
            self.initialize(layer.linear1.weight.data)
            self.initialize(layer.linear2.weight.data)
            self.initialize(layer.self_attn.out_proj.weight.data)

            if layer.self_attn._qkv_same_embed_dim:
                self.initialize(layer.self_attn.in_proj_weight)
            else:
                self.initialize(layer.self_attn.q_proj_weight)
                self.initialize(layer.self_attn.k_proj_weight)
                self.initialize(layer.self_attn.v_proj_weight)

        self.context_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.get_option("encoder.nhead"),
            dim_feedforward=self.get_option("encoder.dim_feedforward"),
            dropout=dropout,
            activation=self.get_option("encoder.activation"),
        )
        self.context_encoder = torch.nn.TransformerEncoder(
            self.context_encoder_layer, num_layers=self.get_option("encoder.context_encoder.num_layers")
        )
        for layer in self.context_encoder.layers:
            self.initialize(layer.linear1.weight.data)
            self.initialize(layer.linear2.weight.data)
            self.initialize(layer.self_attn.out_proj.weight.data)

            if layer.self_attn._qkv_same_embed_dim:
                self.initialize(layer.self_attn.in_proj_weight)
            else:
                self.initialize(layer.self_attn.q_proj_weight)
                self.initialize(layer.self_attn.k_proj_weight)
                self.initialize(layer.self_attn.v_proj_weight)

        self.entity_prediction_layer = torch.nn.Linear(in_features=self.emb_dim, out_features=dataset.num_entities())

    def set_masked_entity_prediction_weights(self, weights):
        self.entity_prediction_layer.weight = weights

    def score_emb(
            self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, context_s_emb: Tensor, context_p_emb: Tensor,
            attention_mask, combine: str, num_replaced=0, num_unchanged=0
    ):
        """

        Args:
            s_emb:
            p_emb:
            o_emb:
            context_s_emb: context
            context_p_emb: context
            combine:

        Returns:

        """
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        # transform the sp pairs
        batch_size = len(s_emb)
        #entity_prepare_time = -time.time()
        context_size = context_s_emb.shape[1]
        context_s_dim = context_s_emb.shape[2]
        context_p_dim = context_p_emb.shape[2]

        s_emb[num_replaced + num_unchanged:] = self.mask_emb

        attention_mask = torch.cat([attention_mask.new_ones(batch_size).unsqueeze(1), attention_mask], dim=1)
        attention_mask_flattened = attention_mask.view(batch_size * (context_size + 1))

        entity_out = s_emb.new_empty((batch_size * (context_size + 1), context_s_dim))
        #entity_prepare_time += time.time()
        #print(f"entity_prepare_time: {entity_prepare_time}")
        #entity_forward_time = -time.time()
        entity_out[attention_mask_flattened] = self.entity_encoder.forward(
            torch.stack(
                (
                    self.cls_emb.repeat((attention_mask.sum(), 1)),
                    torch.cat([s_emb.view(batch_size, 1, context_s_dim), context_s_emb], dim=1).view((batch_size * (context_size + 1), context_s_dim))[attention_mask_flattened] + self.sub_type_emb.unsqueeze(0),
                    torch.cat([p_emb.view(batch_size, 1, context_p_dim), context_p_emb], dim=1).view((batch_size * (context_size + 1), context_s_dim))[attention_mask_flattened] + self.rel_type_emb.unsqueeze(0),
                ),
                dim=0,
                )
            )[0, :]
        #entity_forward_time += time.time()
        #print(f"entity_forward_time: {entity_forward_time}")
        #context_prepare_time = -time.time()
        entity_out[~attention_mask_flattened] = 0

        entity_out = torch.transpose(entity_out.view((batch_size, context_size + 1, context_s_dim)), 0, 1)

        attention_mask = torch.cat([attention_mask.new_ones(batch_size).unsqueeze(1), attention_mask], dim=1)

        out = self.context_encoder.forward(
            torch.cat([self.gcls_emb.repeat((batch_size, 1)).unsqueeze(0), entity_out]), src_key_padding_mask=~attention_mask
            )

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

    def recover_entity_emb(self, s_emb, p_emb, context_s_emb, context_p_emb, attention_mask, num_replaced=0, num_unchanged=0):
        """
        recovers labels
        Args:
            s_emb:
            p_emb:
            context_s_emb:
            context_p_emb:

        Returns:
            tensor of labels
        """

        batch_size = len(s_emb)
        context_size = context_s_emb.shape[1]
        context_s_dim = context_s_emb.shape[2]
        context_p_dim = context_p_emb.shape[2]

        s_emb[num_replaced + num_unchanged:] = self.mask_emb

        attention_mask = torch.cat([attention_mask.new_ones(batch_size).unsqueeze(1), attention_mask], dim=1)
        attention_mask_flattened = attention_mask.view(batch_size * (context_size + 1))

        entity_out = s_emb.new_empty((batch_size * (context_size + 1), context_s_dim))

        entity_out[attention_mask_flattened] = self.entity_encoder.forward(
            torch.stack(
                (
                    self.cls_emb.repeat((attention_mask.sum(), 1)),
                    torch.cat([s_emb.view(batch_size, 1, context_s_dim), context_s_emb], dim=1).view((batch_size * (context_size + 1), context_s_dim))[attention_mask_flattened] + self.sub_type_emb.unsqueeze(0),
                    torch.cat([p_emb.view(batch_size, 1, context_p_dim), context_p_emb], dim=1).view((batch_size * (context_size + 1), context_s_dim))[attention_mask_flattened] + self.rel_type_emb.unsqueeze(0),
                ),
                dim=0,
                )
            )[0, :]

        entity_out[~attention_mask_flattened] = 0

        entity_out = torch.transpose(entity_out.view((batch_size, context_size + 1, context_s_dim)), 0, 1)

        attention_mask = torch.cat([attention_mask.new_ones(batch_size).unsqueeze(1), attention_mask], dim=1)

        out = self.context_encoder.forward(
            torch.cat([self.gcls_emb.repeat((batch_size, 1)).unsqueeze(0), entity_out]), src_key_padding_mask=~attention_mask
            )

        out = out[1, ::]

        out = self.entity_prediction_layer.forward(out)

        return out


class Hitter(KgeModel):
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
            scorer=HitterScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.get_scorer().set_masked_entity_prediction_weights(self.get_s_embedder()._embeddings.weight)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, **kwargs) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If Transformer is wrapped in a reciprocal relations model, this will always be
        # the case.
        if direction == "o":
            super().score_spo(s, p, o, direction)
        else:
            raise ValueError("Transformer can only score objects")
