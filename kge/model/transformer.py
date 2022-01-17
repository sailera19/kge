import numpy as np
import torch
import torch.nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import CharDelimiterSplit, Sequence, Digits, Whitespace, BertPreTokenizer
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.embedder.text_lookup_embedder import TextLookupEmbedding, TextLookupEmbedder
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
        self.s_embedder: KgeEmbedder = None
        self.p_embedder: KgeEmbedder = None
        self.o_embedder: KgeEmbedder = None

        self.enable_text = self.get_option("enable_text")
        self.text_only = self.get_option("text_only")
        self.built_in_text_embedder = self.get_option("built_in_text_embedder")
        self.self_pred_loss_weighing = self.get_option("self_pred_loss_weighing")

        if self.enable_text and self.built_in_text_embedder:
            self._text_embedder: TextLookupEmbedder = KgeEmbedder.create(
                    config,
                    dataset,
                    self.configuration_key + ".text_embedder",
                    dataset.num_entities())
            self.text_pos_embeddings = torch.nn.Parameter(torch.zeros((self._text_embedder.max_token_length, self.emb_dim)))
            self.initialize(self.text_pos_embeddings)

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.cls_emb)
        # the type embeddings
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_type_emb)
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_type_emb)

        # text embeddings
        self.sub_text_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_text_type_emb)
        self.obj_text_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.obj_text_type_emb)
        self.any_rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.any_rel_type_emb)
        self.o_cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_cls_emb)
        self.o_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_type_emb)

        # masks
        self.s_mask = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.s_mask)
        self.o_mask = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_mask)

        self.mlm_mask_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.mlm_mask_emb)

        self.s_dropout = self.get_option("s_dropout")
        self.o_dropout = self.get_option("o_dropout")
        self.s_dropout_masked = self.get_option("s_dropout_masked")
        self.o_dropout_masked = self.get_option("o_dropout_masked")
        self.s_dropout_replaced = self.get_option("s_dropout_replaced")
        self.o_dropout_replaced = self.get_option("o_dropout_replaced")

        self.s_text_dropout = self.get_option("s_text_dropout")
        self.o_text_dropout = self.get_option("o_text_dropout")
        self.s_text_dropout_masked = self.get_option("s_text_dropout_masked")
        self.o_text_dropout_masked = self.get_option("o_text_dropout_masked")
        self.s_text_dropout_replaced = self.get_option("s_text_dropout_replaced")
        self.o_text_dropout_replaced = self.get_option("o_text_dropout_replaced")

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

    def set_embedders(self, s_embedder, p_embedder, o_embedder):
        self.s_embedder = s_embedder
        self.p_embedder = p_embedder
        self.o_embedder = o_embedder
        if isinstance(self.s_embedder, TextLookupEmbedder):
            self._text_embedder = s_embedder
            self.text_pos_embeddings = torch.nn.Parameter(torch.zeros((self._text_embedder.max_token_length, self.emb_dim)))
            self.initialize(self.text_pos_embeddings)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str, ground_truth_s: Tensor, ground_truth_p: Tensor, ground_truth_o: Tensor, targets_o: Tensor=None, **kwargs):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        batch_size = len(s_emb)

        self_pred_loss_dropout, self_pred_loss_text_dropout = 0, 0

        if self.enable_text:
            if not isinstance(s_emb, TextLookupEmbedding):
                s_text_embeddings = self._text_embedder.embed(ground_truth_s)

            else:
                s_text_embeddings = s_emb

            s_tokens, s_attention_mask, s_text_embeddings = s_text_embeddings.tokens, s_text_embeddings.attention_mask, s_text_embeddings.embeddings

            if self.training:
                if not self.text_only:
                    s_dropout = torch.zeros(len(s_emb), dtype=torch.bool).bernoulli_(self.s_dropout)
                    s_masked = torch.zeros(len(s_emb), dtype=torch.bool).bernoulli_(self.s_dropout_masked) & s_dropout
                    s_replaced = torch.zeros(len(s_emb), dtype=torch.bool).bernoulli_(self.s_dropout_replaced) & s_dropout & ~s_masked
                    s_emb[s_masked] = self.s_mask
                    s_emb[s_replaced] = self.s_embedder.embed(torch.randint(low=0, high=self.dataset.num_entities(), size=(s_replaced.sum(),), device=s_emb.device))
                else:
                    s_dropout = torch.zeros(1)
                s_text_embeddings_dropout = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                        device=s_text_embeddings.device).bernoulli_(
                    self.s_text_dropout) & s_attention_mask
                s_text_embeddings_masked = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                       device=s_text_embeddings.device).bernoulli_(
                    self.s_text_dropout_masked) & s_text_embeddings_dropout
                s_text_embeddings_replaced = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                         device=s_text_embeddings.device).bernoulli_(
                    self.s_text_dropout_replaced) & s_text_embeddings_dropout & ~s_text_embeddings_masked
                s_text_embeddings[s_text_embeddings_masked] = self.mlm_mask_emb
                s_text_embeddings[s_text_embeddings_replaced] = self._text_embedder._embeddings(
                    torch.randint(low=0, high=self._text_embedder.vocab_size,
                                  size=(s_text_embeddings_replaced.sum(),), device=s_text_embeddings.device))

            if not isinstance(o_emb, TextLookupEmbedding):
                if combine == "spo":
                    targets_o = ground_truth_o
                    o_text_embeddings = self._text_embedder.embed(targets_o)
                else:
                    if targets_o is None:
                        o_text_embeddings = self._text_embedder.embed_all()
                    else:
                        o_text_embeddings = self._text_embedder.embed(targets_o)
            else:
                o_text_embeddings = o_emb

            o_tokens, o_attention_mask, o_text_embeddings = o_text_embeddings.tokens, o_text_embeddings.attention_mask, o_text_embeddings.embeddings
            num_o_embeddings = len(o_tokens)

            if self.training:
                if not self.text_only:
                    o_dropout = torch.zeros(len(o_emb), dtype=torch.bool).bernoulli_(self.o_dropout)
                    o_masked = torch.zeros(len(o_emb), dtype=torch.bool).bernoulli_(self.o_dropout_masked) & o_dropout
                    o_replaced = torch.zeros(len(o_emb), dtype=torch.bool).bernoulli_(self.o_dropout_replaced) & o_dropout & ~o_masked
                    o_emb[o_masked] = self.o_mask
                    o_emb[o_replaced] = self.o_embedder.embed(torch.randint(low=0, high=self.dataset.num_entities(), size=(o_replaced.sum(),), device=o_emb.device))
                else:
                    o_dropout = torch.zeros(1)
                o_text_embeddings_dropout = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool, device=o_text_embeddings.device).bernoulli_(
                    self.o_text_dropout) & o_attention_mask
                o_text_embeddings_masked = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool, device=o_text_embeddings.device).bernoulli_(
                    self.o_text_dropout_masked) & o_text_embeddings_dropout
                o_text_embeddings_replaced = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool, device=o_text_embeddings.device).bernoulli_(
                    self.o_text_dropout_replaced) & o_text_embeddings_dropout & ~o_text_embeddings_masked
                o_text_embeddings[o_text_embeddings_masked] = self.mlm_mask_emb
                o_text_embeddings[o_text_embeddings_replaced] = self._text_embedder.embed_tokens(
                    torch.randint(low=0, high=self._text_embedder.vocab_size,
                                  size=(o_text_embeddings_replaced.sum(),), device=o_text_embeddings.device))

                if not self.text_only:
                    del s_masked, s_replaced, o_masked, o_replaced
                del s_text_embeddings_masked, s_text_embeddings_replaced, o_text_embeddings_masked, o_text_embeddings_replaced

            # transform the sp pairs
            out = self.encoder.forward(
                torch.cat((
                    torch.cat(
                        [x for x in
                            (
                                self.cls_emb.repeat((1, batch_size, 1)),
                                (s_emb + self.sub_type_emb.unsqueeze(0)).unsqueeze(0) if not self.text_only else None,
                                (p_emb + self.rel_type_emb.unsqueeze(0)).unsqueeze(0),
                                (s_text_embeddings + self.text_pos_embeddings + self.sub_text_type_emb.unsqueeze(0)).transpose(1, 0)
                            )
                         if x is not None],
                        dim=0,
                    ),
                    torch.cat(
                        [x for x in
                            (
                                self.o_cls_emb.repeat((1, num_o_embeddings, 1)),
                                (o_emb + self.o_type_emb.unsqueeze(0)).unsqueeze(0) if not self.text_only else None,
                                (self.any_rel_type_emb.repeat(1, num_o_embeddings, 1) + self.rel_type_emb),
                                (o_text_embeddings + self.text_pos_embeddings + self.sub_text_type_emb.unsqueeze(0)).transpose(1, 0),
                            )
                         if x is not None],
                        dim=0,
                    )
                ), dim=1),
                src_key_padding_mask=torch.cat((
                    ~torch.cat((
                        torch.ones(batch_size, 3 - self.text_only, dtype=torch.bool, device=ground_truth_s.device),
                        s_attention_mask.to(ground_truth_s.device)
                    ), dim=1),
                    ~torch.cat((
                        torch.ones(num_o_embeddings, 3 - self.text_only, dtype=torch.bool,
                                   device=ground_truth_o.device),
                        o_attention_mask
                    ), dim=1)
                ), dim=0)
            )  # SxNxE = 3 x batch_size x emb_size


            if self.training:
                if not self.text_only and (s_dropout.any() or o_dropout.any()):
                    self_pred_loss_dropout = torch.nn.functional.cross_entropy(
                        torch.mm(out[1][torch.cat((s_dropout, o_dropout))], self.s_embedder.embed_all().transpose(1, 0)),
                        torch.cat((ground_truth_s[s_dropout], targets_o[o_dropout])),
                    )
                if s_text_embeddings_dropout.any() or o_text_embeddings_dropout.any():
                    self_pred_loss_text_dropout = torch.nn.functional.cross_entropy(
                        torch.mm(out[3 - self.text_only:][torch.cat(
                            (s_text_embeddings_dropout.transpose(1, 0), o_text_embeddings_dropout.transpose(1, 0)),
                            dim=1)], self._text_embedder.embed_all_tokens().transpose(1, 0)),
                        torch.cat((s_tokens[s_text_embeddings_dropout].to(out.device),
                                   o_tokens[o_text_embeddings_dropout].to(out.device))),
                    )

            o_emb = out[:, batch_size:]
            out = out[:, :batch_size]

            o_emb = o_emb[0, ::]

        else:
            out = self.encoder.forward(
                torch.stack(
                    (
                        self.cls_emb.repeat((batch_size, 1)),
                        s_emb + self.sub_type_emb.unsqueeze(0),
                        p_emb + self.rel_type_emb.unsqueeze(0),
                    ),
                    dim=0,
                )
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
        if self.training and self.enable_text:
            if self.self_pred_loss_weighing == "by_count":
                self_pred_loss = (
                    self_pred_loss_dropout * (s_dropout.sum() + o_dropout.sum())
                    + self_pred_loss_text_dropout * (s_text_embeddings_dropout.sum() + o_text_embeddings_dropout.sum())
                                 ) / (
                    s_dropout.sum()
                    + o_dropout.sum()
                    + s_text_embeddings_dropout.sum()
                    + o_text_embeddings_dropout.sum()
                )
            else:
                self_pred_loss = (self_pred_loss_dropout + self_pred_loss_text_dropout) / (
                        (self_pred_loss_dropout > 0) + (self_pred_loss_text_dropout > 0))

            return out.view(batch_size, -1), self_pred_loss
        else:
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
        self.get_scorer().set_embedders(self.get_s_embedder(), self.get_p_embedder(), self.get_o_embedder())

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, **kwargs) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If Transformer is wrapped in a reciprocal relations model, this will always be
        # the case.
        if direction == "o":
            super().score_spo(s, p, o, direction)
        else:
            raise ValueError("Transformer can only score objects")
