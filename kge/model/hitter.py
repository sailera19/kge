import torch
import torch.nn
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_context_model import KgeContextModel
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util import rat, KgeLoss

from pytorch_pretrained_bert.modeling import BertEncoder, BertConfig, BertLayerNorm, BertPreTrainedModel

from functools import partial


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

        self.loss = KgeLoss.create(config)

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
        self.gcls_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.gcls_type_emb)
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

        self.entity_dropout = self.get_option("encoder.entity_encoder.dropout")
        self.context_dropout = self.get_option("encoder.context_encoder.dropout")
        self.output_dropout = self.get_option("output_dropout")
        self.mlm_fraction = self.get_option("mlm_fraction")


        self.entity_layer_norm = torch.nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.context_layer_norm = torch.nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.transformer_impl = self.get_option("implementation")

        if self.transformer_impl == "pytorch":
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
        elif self.transformer_impl == "microsoft":
            self.context_encoder = rat.Encoder(
                lambda: rat.EncoderLayer(
                    self.emb_dim,
                    rat.MultiHeadedAttentionWithRelations(
                        self.get_option("encoder.nhead"),
                        self.emb_dim,
                        self.get_option("encoder.dropout")),
                    rat.PositionwiseFeedForward(
                        self.emb_dim,
                        self.get_option("encoder.dim_feedforward"),
                        self.get_option("encoder.dropout")),
                    num_relation_kinds=0,
                    dropout=self.get_option("encoder.dropout")),
                self.get_option("encoder.context_encoder.num_layers"),
                self.get_option("initialize_args.std"),
                tie_layers=False)

            config = BertConfig(0, hidden_size=self.emb_dim,
                                num_hidden_layers=self.get_option("encoder.entity_encoder.num_layers"),
                                num_attention_heads=self.get_option("encoder.nhead"),
                                intermediate_size=self.get_option("encoder.dim_feedforward"),
                                hidden_act=self.get_option("encoder.activation"),
                                hidden_dropout_prob=self.get_option("encoder.dropout"),
                                attention_probs_dropout_prob=self.get_option("encoder.dropout"),
                                max_position_embeddings=0,  # no effect
                                type_vocab_size=0,  # no effect
                                initializer_range=self.get_option("initialize_args.std"))
            self.entity_encoder = BertEncoder(config)
            self.entity_encoder.config = config
            self.entity_encoder.apply(partial(BertPreTrainedModel.init_bert_weights, self.entity_encoder))

    def score_emb(
            self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, context_s_emb: Tensor, context_p_emb: Tensor,
            attention_mask, combine: str, num_replaced=0, num_masked=0, ground_truth_s: Tensor = None
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
        context_size = context_s_emb.shape[1]
        context_s_dim = context_s_emb.shape[2]
        context_p_dim = context_p_emb.shape[2]

        s_emb[num_replaced : num_replaced + num_masked] = self.mask_emb

        attention_mask = torch.cat([attention_mask.new_ones(batch_size).unsqueeze(1), attention_mask], dim=1)
        attention_mask_flattened = attention_mask.view(batch_size * (context_size + 1))

        entity_in = torch.stack(
            (
                self.cls_emb.repeat((attention_mask.sum(), 1)),
                torch.cat([s_emb.view(batch_size, 1, context_s_dim), context_s_emb], dim=1).view(
                    (batch_size * (context_size + 1), context_s_dim))[
                    attention_mask_flattened] + self.sub_type_emb.unsqueeze(0),
                torch.cat([p_emb.view(batch_size, 1, context_p_dim), context_p_emb], dim=1).view(
                    (batch_size * (context_size + 1), context_s_dim))[
                    attention_mask_flattened] + self.rel_type_emb.unsqueeze(0),
            ),
            dim=0,
        )

        entity_in = torch.nn.functional.dropout(entity_in, p=self.entity_dropout, training=self.training)

        entity_in = self.entity_layer_norm(entity_in)

        entity_out = s_emb.new_empty((batch_size * (context_size + 1), context_s_dim))

        if self.transformer_impl == "pytorch":
            entity_out[attention_mask_flattened] = self.entity_encoder.forward(entity_in)[0, :]
        else:
            entity_out[attention_mask_flattened] = self.entity_encoder.forward(entity_in.transpose(0,1),
                                                                               self.convert_mask(entity_in.new_ones(attention_mask_flattened.sum(), 3, dtype=torch.long)),
                                                                               output_all_encoded_layers=False)[-1][: ,0, :]

        entity_out[~attention_mask_flattened] = 0

        entity_out = torch.transpose(entity_out.view((batch_size, context_size + 1, context_s_dim)), 0, 1)

        entity_out = torch.cat([self.gcls_emb.repeat((batch_size, 1)).unsqueeze(0), entity_out])
        entity_out[:, 1] += self.src_type_emb
        entity_out[:, 2:] += self.context_type_emb

        entity_out = torch.nn.functional.dropout(entity_out, p=self.context_dropout, training=self.training)
        entity_out = self.context_layer_norm(entity_out)

        attention_mask = torch.cat([attention_mask.new_ones(batch_size).unsqueeze(1), attention_mask], dim=1)

        if self.transformer_impl == "pytorch":
            out = self.context_encoder.forward(entity_out, src_key_padding_mask=~attention_mask)
        else:
            out = self.context_encoder.forward(entity_out.transpose(0, 1), None, self.convert_mask_rat(attention_mask))[-1].transpose(0,1)

        out = out[:2, ::]

        o_emb = torch.nn.functional.dropout(o_emb, self.output_dropout, training=self.training)

        if self.training and self.mlm_fraction > 0.0:
            num_mlm = round(num_masked * self.mlm_fraction)

            mlm_scores = torch.mm(out[1, ::][num_replaced : num_replaced + num_mlm], o_emb.transpose(1, 0))
            self_pred_loss = self.loss(mlm_scores, ground_truth_s[num_replaced : num_replaced + num_mlm]) / num_mlm
        else:
            self_pred_loss = 0

        # now take dot product
        if combine == "sp_":
            out = torch.mm(out[0, ::], o_emb.transpose(1, 0))
        elif combine == "spo":
            out = (out[0, ::] * o_emb).sum(-1)
        else:
            raise Exception("can't happen")

        # all done
        if self.training and self.mlm_fraction > 0.0:
            return out.view(batch_size, -1), self_pred_loss
        else:
            return out.view(batch_size, -1)

    def convert_mask_rat(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def convert_mask(self, attention_mask):
        # extend mask to Transformer format
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

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
