import numpy as np
import torch
import torch.nn
from torch import Tensor
import math

from transformers import BertTokenizer, BertModel

from kge import Config, Dataset
from kge.model import SharedTextLookupEmbedder
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

        self.pretrained_name = self.get_option("pretrained_name")

        self.embedding_dropout = self.get_option("embedding_dropout")

        self.enable_text = self.get_option("enable_text")
        self.enable_entity_text = self.get_option("enable_entity_text")
        self.enable_entity_structure = self.get_option("enable_entity_structure")
        self.entity_built_in_text_embedder = self.get_option("entity_built_in_text_embedder")
        self.enable_relation_text = self.get_option("enable_relation_text")
        self.enable_relation_structure = self.get_option("enable_relation_structure")
        self.relation_built_in_text_embedder = self.get_option("relation_built_in_text_embedder")
        self.self_pred_loss_weighing = self.get_option("self_pred_loss_weighing")

        if self.enable_text:
            if self.entity_built_in_text_embedder:
                self._text_embedder: TextLookupEmbedder = KgeEmbedder.create(
                        config,
                        dataset,
                        self.configuration_key + ".text_embedder",
                        dataset.num_entities())
                self.text_pos_embeddings = torch.nn.Parameter(torch.zeros((self._text_embedder.max_token_length, self.emb_dim)))
                self.initialize(self.text_pos_embeddings)

            if self.relation_built_in_text_embedder:
                self._relation_text_embedder: TextLookupEmbedder = KgeEmbedder.create(
                        config,
                        dataset,
                        self.configuration_key + ".relation_text_embedder",
                        dataset.num_relations())
                if isinstance(self._relation_text_embedder, SharedTextLookupEmbedder) and self.entity_built_in_text_embedder:
                    self._relation_text_embedder._set_shared_lookup_embedder(self._text_embedder, dataset)
                self.rel_text_pos_embeddings = torch.nn.Parameter(torch.zeros((self._relation_text_embedder.max_token_length, self.emb_dim)))
                self.initialize(self.rel_text_pos_embeddings)

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
        self.rel_text_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_text_type_emb)
        self.any_rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_text_type_emb)
        self.any_rel_text_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.any_rel_text_type_emb)
        self.o_cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_cls_emb)
        self.o_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_type_emb)

        self.seperator_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.seperator_emb)

        # masks
        self.s_mask = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.s_mask)
        self.p_mask = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.s_mask)
        self.o_mask = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.o_mask)

        self.mlm_mask_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.mlm_mask_emb)

        self.all_text_mask_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.all_text_mask_emb)

        self.s_dropout = self.get_option("s_dropout")
        self.o_dropout = self.get_option("o_dropout")
        self.p_dropout = self.get_option("p_dropout")
        self.s_dropout_loss = self.get_option("s_dropout_loss")
        self.o_dropout_loss = self.get_option("o_dropout_loss")
        self.p_dropout_loss = self.get_option("p_dropout_loss")
        self.s_dropout_masked = self.get_option("s_dropout_masked")
        self.o_dropout_masked = self.get_option("o_dropout_masked")
        self.p_dropout_masked = self.get_option("p_dropout_masked")
        self.s_dropout_replaced = self.get_option("s_dropout_replaced")
        self.o_dropout_replaced = self.get_option("o_dropout_replaced")
        self.p_dropout_replaced = self.get_option("p_dropout_replaced")

        self.s_text_dropout = self.get_option("s_text_dropout")
        self.o_text_dropout = self.get_option("o_text_dropout")
        self.p_text_dropout = self.get_option("p_text_dropout")
        self.s_text_dropout_loss = self.get_option("s_text_dropout_loss")
        self.o_text_dropout_loss = self.get_option("o_text_dropout_loss")
        self.p_text_dropout_loss = self.get_option("p_text_dropout_loss")
        self.s_text_dropout_masked = self.get_option("s_text_dropout_masked")
        self.o_text_dropout_masked = self.get_option("o_text_dropout_masked")
        self.p_text_dropout_masked = self.get_option("p_text_dropout_masked")
        self.s_text_dropout_replaced = self.get_option("s_text_dropout_replaced")
        self.o_text_dropout_replaced = self.get_option("o_text_dropout_replaced")
        self.p_text_dropout_replaced = self.get_option("p_text_dropout_replaced")

        self.mask_all_text = self.get_option("mask_all_text")

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

        self.encoder = BertModel.from_pretrained(self.pretrained_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_name)
        self.seperator_emb = torch.nn.parameter.Parameter(self.encoder.embeddings.word_embeddings(torch.LongTensor([self.tokenizer.sep_token_id])))
        self.mlm_mask_emb = torch.nn.parameter.Parameter(self.encoder.embeddings.word_embeddings(torch.LongTensor([self.tokenizer.mask_token_id])))
        print("done")

    def set_embedders(self, s_embedder, p_embedder, o_embedder):
        self.s_embedder = s_embedder
        self.p_embedder = p_embedder
        self.o_embedder = o_embedder

    def _initialize_text_embedders(self):
        if isinstance(self.s_embedder, TextLookupEmbedder):
            self._text_embedder = self.s_embedder
            self.text_pos_embeddings = torch.nn.Parameter(self.encoder.embeddings.position_embeddings.weight[:self._text_embedder.max_token_length])
            self.initialize(self.text_pos_embeddings)
        if isinstance(self.p_embedder, TextLookupEmbedder) or isinstance(self.p_embedder, SharedTextLookupEmbedder):
            self._relation_text_embedder = self.p_embedder
            self.rel_text_pos_embeddings = torch.nn.Parameter(
                self.encoder.embeddings.position_embeddings.weight[:self._relation_text_embedder.max_token_length])
            self.initialize(self.rel_text_pos_embeddings)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str, ground_truth_s: Tensor, ground_truth_p: Tensor, ground_truth_o: Tensor, targets_o: Tensor=None, **kwargs):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        if isinstance(s_emb, TextLookupEmbedding):
            device = s_emb.embeddings.device
        else:
            device = s_emb.device

        batch_size = len(s_emb)

        self_pred_loss_dropout, self_pred_loss_text_dropout = 0, 0
        self_pred_loss_p_dropout, self_pred_loss_p_text_dropout = 0, 0

        if self.enable_text:
            if not isinstance(s_emb, TextLookupEmbedding) and self.enable_entity_text:
                s_text_embeddings = self._text_embedder.embed(ground_truth_s)

            else:
                s_text_embeddings = s_emb

            s_tokens, s_attention_mask, s_text_embeddings = s_text_embeddings.tokens, s_text_embeddings.attention_mask, s_text_embeddings.embeddings

            s_text_length = s_tokens.shape[1]

            if self.training:
                if self.mask_all_text > 0.0:
                    s_mask_all_text = torch.zeros(batch_size, dtype=torch.bool,
                                                  device=device).bernoulli_(self.mask_all_text)
                    s_text_embeddings[s_mask_all_text] = self.all_text_mask_emb
                else:
                    s_mask_all_text = torch.zeros(batch_size, dtype=torch.bool, device=device)

                if self.enable_entity_structure:
                    s_dropout = torch.zeros(len(s_emb), dtype=torch.bool, device=device).bernoulli_(self.s_dropout) & ~s_mask_all_text
                    s_masked = torch.zeros(len(s_emb), dtype=torch.bool, device=device).bernoulli_(self.s_dropout_masked) & s_dropout
                    s_replaced = torch.zeros(len(s_emb), dtype=torch.bool, device=device).bernoulli_(self.s_dropout_replaced) & s_dropout & ~s_masked
                    s_emb[s_masked] = self.s_mask
                    s_emb[s_replaced] = self.s_embedder.embed(torch.randint(low=0, high=self.dataset.num_entities(), size=(s_replaced.sum(),), device=device))
                    s_dropout = torch.zeros(len(s_emb), dtype=torch.bool, device=device).bernoulli_(self.s_dropout_loss) & s_dropout
                    del s_masked, s_replaced
                else:
                    s_dropout = torch.zeros(1)
                s_text_embeddings_dropout = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                        device=device).bernoulli_(
                    self.s_text_dropout) & s_attention_mask & ~s_mask_all_text.unsqueeze(1).repeat(1, s_text_length)
                s_text_embeddings_masked = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                       device=device).bernoulli_(
                    self.s_text_dropout_masked) & s_text_embeddings_dropout
                s_text_embeddings_replaced = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                         device=device).bernoulli_(
                    self.s_text_dropout_replaced) & s_text_embeddings_dropout & ~s_text_embeddings_masked
                s_text_embeddings[s_text_embeddings_masked] = self.mlm_mask_emb
                s_text_embeddings[s_text_embeddings_replaced] = self._text_embedder._embeddings(
                    torch.randint(low=0, high=self._text_embedder.vocab_size,
                                  size=(s_text_embeddings_replaced.sum(),), device=device))
                s_text_embeddings_dropout = torch.zeros(s_text_embeddings.shape[:2], dtype=torch.bool,
                                                        device=device).bernoulli_(
                    self.s_text_dropout_loss) & s_text_embeddings_dropout
                del s_text_embeddings_masked, s_text_embeddings_replaced

            if not isinstance(o_emb, TextLookupEmbedding) and self.enable_entity_text:
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
                if self.mask_all_text > 0.0:
                    o_mask_all_text = torch.zeros(num_o_embeddings, dtype=torch.bool,
                                                  device=device).bernoulli_(self.mask_all_text)
                    o_text_embeddings[o_mask_all_text] = self.all_text_mask_emb
                else:
                    o_mask_all_text = torch.zeros(num_o_embeddings, dtype=torch.bool, device=device)

                if self.enable_entity_structure:
                    o_dropout = torch.zeros(len(o_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.o_dropout) & ~o_mask_all_text
                    o_masked = torch.zeros(len(o_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.o_dropout_masked) & o_dropout
                    o_replaced = torch.zeros(len(o_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.o_dropout_replaced) & o_dropout & ~o_masked
                    o_emb[o_masked] = self.o_mask
                    o_emb[o_replaced] = self.o_embedder.embed(torch.randint(low=0, high=self.dataset.num_entities(), size=(o_replaced.sum(),), device=device))
                    o_dropout = torch.zeros(len(o_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.o_dropout_loss) & o_dropout
                    del o_masked, o_replaced
                else:
                    o_dropout = torch.zeros(1)

                if self.enable_entity_text:
                    o_text_length = o_tokens.shape[1]
                    o_text_embeddings_dropout = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool, device=device).bernoulli_(
                        self.o_text_dropout) & o_attention_mask & ~o_mask_all_text.unsqueeze(1).repeat(1, o_text_length)
                    o_text_embeddings_masked = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool, device=device).bernoulli_(
                        self.o_text_dropout_masked) & o_text_embeddings_dropout
                    o_text_embeddings_replaced = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool, device=device).bernoulli_(
                        self.o_text_dropout_replaced) & o_text_embeddings_dropout & ~o_text_embeddings_masked
                    o_text_embeddings[o_text_embeddings_masked] = self.mlm_mask_emb
                    o_text_embeddings[o_text_embeddings_replaced] = self._text_embedder.embed_tokens(
                        torch.randint(low=0, high=self._text_embedder.vocab_size,
                                      size=(o_text_embeddings_replaced.sum(),), device=device))
                    o_text_embeddings_dropout = torch.zeros(o_text_embeddings.shape[:2], dtype=torch.bool,
                                                            device=device).bernoulli_(
                        self.o_text_dropout_loss) & o_text_embeddings_dropout
                    del o_text_embeddings_masked, o_text_embeddings_replaced

            if self.enable_relation_text:
                if not isinstance(p_emb, TextLookupEmbedding):
                    p_text_embeddings = self._relation_text_embedder.embed(ground_truth_p)
                else:
                    p_text_embeddings = p_emb

                p_tokens, p_attention_mask, p_text_embeddings = p_text_embeddings.tokens, p_text_embeddings.attention_mask, p_text_embeddings.embeddings

                p_text_length = p_tokens.shape[1]

            if self.training:
                if self.mask_all_text > 0.0:
                    p_mask_all_text = torch.zeros(batch_size, dtype=torch.bool,
                                                  device=device).bernoulli_(self.mask_all_text)
                    p_text_embeddings[p_mask_all_text] = self.all_text_mask_emb
                else:
                    p_mask_all_text = torch.zeros(batch_size, dtype=torch.bool, device=device)

                if self.enable_relation_structure:
                    p_dropout = torch.zeros(len(p_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.p_dropout) & ~p_mask_all_text
                    p_masked = torch.zeros(len(p_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.p_dropout_masked) & p_dropout
                    p_replaced = torch.zeros(len(p_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.p_dropout_replaced) & p_dropout & ~p_masked
                    p_emb[p_masked] = self.p_mask
                    p_emb[p_replaced] = self.p_embedder.embed(torch.randint(low=0, high=self.dataset.num_relations(), size=(p_replaced.sum(),), device=device))
                    p_dropout = torch.zeros(len(p_emb), dtype=torch.bool, device=device).bernoulli_(
                        self.p_dropout_loss) & p_dropout
                else:
                    p_dropout = torch.zeros(1)

                if self.enable_relation_text:
                    p_text_embeddings_dropout = torch.zeros(p_text_embeddings.shape[:2], dtype=torch.bool,
                                                            device=device).bernoulli_(
                        self.p_text_dropout) & p_attention_mask & ~p_mask_all_text.unsqueeze(1).repeat(1, p_text_length)
                    p_text_embeddings_masked = torch.zeros(p_text_embeddings.shape[:2], dtype=torch.bool,
                                                           device=device).bernoulli_(
                        self.p_text_dropout_masked) & p_text_embeddings_dropout
                    p_text_embeddings_replaced = torch.zeros(p_text_embeddings.shape[:2], dtype=torch.bool,
                                                             device=device).bernoulli_(
                        self.p_text_dropout_replaced) & p_text_embeddings_dropout & ~p_text_embeddings_masked
                    p_text_embeddings[p_text_embeddings_masked] = self.mlm_mask_emb
                    p_text_embeddings[p_text_embeddings_replaced] = self._relation_text_embedder.embed_tokens(
                        torch.randint(low=0, high=self._relation_text_embedder.vocab_size,
                                      size=(p_text_embeddings_replaced.sum(),), device=device))
                    p_text_embeddings_dropout = torch.zeros(p_text_embeddings.shape[:2], dtype=torch.bool,
                                                            device=device).bernoulli_(
                        self.p_text_dropout_loss) & p_text_embeddings_dropout
                else:
                    p_text_embeddings_dropout = torch.zeros(1)

            entity_text_length = s_text_embeddings.shape[1] if self.enable_entity_text else 0
            relation_text_length = p_text_embeddings.shape[1] if self.enable_relation_text else 0

            s_structure_position = int(self.enable_entity_structure)
            s_text_position = s_structure_position + self.enable_entity_text
            seperator_position = s_text_position + entity_text_length
            p_structure_position = seperator_position + self.enable_relation_structure
            p_text_position = p_structure_position + self.enable_relation_text

            # transform the sp pairs
            out = self.encoder.forward(inputs_embeds=torch.nn.functional.dropout(
                torch.cat((
                    torch.cat(
                        [x for x in
                         (
                             self.cls_emb.repeat((1, batch_size, 1)),
                             (s_emb + self.sub_type_emb.unsqueeze(0)).unsqueeze(
                                 0) if self.enable_entity_structure else None,
                             (s_text_embeddings + self.text_pos_embeddings + self.sub_text_type_emb.unsqueeze(
                                 0)).transpose(1, 0) if self.enable_entity_text else None,
                             self.seperator_emb.repeat((1, batch_size, 1)),
                             (p_emb + self.rel_type_emb.unsqueeze(0)).unsqueeze(
                                 0) if self.enable_relation_structure else None,
                             (p_text_embeddings + self.rel_text_pos_embeddings + self.rel_text_type_emb.unsqueeze(
                                 0)).transpose(1, 0) if self.enable_relation_text else None,
                         )
                         if x is not None],
                        dim=0,
                    ),
                    torch.cat(
                        [x for x in
                         (
                             self.o_cls_emb.repeat((1, num_o_embeddings, 1)),
                             (o_emb + self.o_type_emb.unsqueeze(0)).unsqueeze(0)
                             if self.enable_entity_structure else None,
                             (o_text_embeddings + self.text_pos_embeddings + self.sub_text_type_emb.unsqueeze(0))
                                 .transpose(1, 0)
                             if self.enable_entity_text else None,
                             self.seperator_emb.repeat((1, num_o_embeddings, 1)),
                             (self.any_rel_type_emb.repeat(1, num_o_embeddings, 1) + self.rel_type_emb)
                             if self.enable_relation_structure else None,
                             (self.any_rel_text_type_emb.repeat(1, num_o_embeddings, 1) + self.rel_text_type_emb)
                             if self.enable_relation_text else None,
                             torch.zeros(p_attention_mask.shape[1] - 1, num_o_embeddings, self.emb_dim,
                                         device=device)
                             if self.enable_relation_text else None
                         )
                         if x is not None],
                        dim=0,
                    )
                ), dim=1), self.embedding_dropout, training=self.training).transpose(0, 1),
                attention_mask=torch.cat((
                    ~torch.cat([x for x in (
                        torch.ones(batch_size, 1 + self.enable_entity_structure, dtype=torch.bool, device=device),
                        s_attention_mask.to(device) if self.enable_entity_text else None,
                        torch.ones(batch_size, 1 + self.enable_relation_structure, dtype=torch.bool, device=device),
                        p_attention_mask.to(device) if self.enable_relation_text else None,
                    ) if x is not None], dim=1),
                    ~torch.cat([x for x in (
                        torch.ones(num_o_embeddings, 1 + self.enable_entity_structure, dtype=torch.bool, device=device),
                        o_attention_mask if self.enable_entity_text else None,
                        torch.ones(num_o_embeddings, 1 + self.enable_relation_structure + self.enable_relation_text, dtype=torch.bool, device=device),
                        torch.zeros(num_o_embeddings, p_attention_mask.shape[1] - 1, dtype=torch.bool, device=device)
                        if self.enable_relation_text else None,
                    ) if x is not None], dim=1)
                ), dim=0)
            )  # SxNxE = 3 x batch_size x emb_size

            out = out[0].transpose(1, 0)

            if self.training:
                if self.enable_entity_structure and (s_dropout.any() or o_dropout.any()):
                    self_pred_loss_dropout = torch.nn.functional.cross_entropy(
                        torch.mm(out[s_structure_position][torch.cat((s_dropout, o_dropout))], self.s_embedder.embed_all().transpose(1, 0)),
                        torch.cat((ground_truth_s[s_dropout], targets_o[o_dropout])),
                    )
                if self.enable_entity_text and (s_text_embeddings_dropout.any() or o_text_embeddings_dropout.any()):
                    self_pred_loss_text_dropout = torch.nn.functional.cross_entropy(
                        torch.mm(out[s_text_position:s_text_position + entity_text_length][torch.cat(
                            (s_text_embeddings_dropout.transpose(1, 0), o_text_embeddings_dropout.transpose(1, 0)),
                            dim=1)], self._text_embedder.embed_all_tokens().transpose(1, 0)),
                        torch.cat((s_tokens[s_text_embeddings_dropout].to(device),
                                   o_tokens[o_text_embeddings_dropout].to(device))),
                    )

                if self.enable_relation_structure and p_dropout.any():
                    self_pred_loss_p_dropout = torch.nn.functional.cross_entropy(
                        torch.mm(out[p_structure_position][:batch_size][p_dropout],
                                 self.p_embedder.embed_all().transpose(1, 0)),
                        ground_truth_p[p_dropout],
                    )
                if self.enable_relation_text and p_text_embeddings_dropout.any():
                    self_pred_loss_p_text_dropout = torch.nn.functional.cross_entropy(
                        torch.mm(out[p_text_position:p_text_position + relation_text_length]
                                 [:, :batch_size][p_text_embeddings_dropout.transpose(1, 0)], self._relation_text_embedder.embed_all_tokens().transpose(1, 0)),
                        p_tokens[p_text_embeddings_dropout].to(device)
                    )

            o_emb = out[:, batch_size:]
            out = out[:, :batch_size]

            o_emb = o_emb[0, ::]

        else:
            out = self.encoder.forward(torch.nn.functional.dropout(
                torch.stack(
                    (
                        self.cls_emb.repeat((batch_size, 1)),
                        s_emb + self.sub_type_emb.unsqueeze(0),
                        p_emb + self.rel_type_emb.unsqueeze(0),
                    ),
                    dim=0,
                ), self.embedding_dropout, training=self.training)
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
                    + self_pred_loss_p_dropout * p_dropout.sum()
                    + self_pred_loss_p_text_dropout * p_text_embeddings_dropout.sum()
                                 ) / (
                    s_dropout.sum()
                    + o_dropout.sum()
                    + s_text_embeddings_dropout.sum()
                    + o_text_embeddings_dropout.sum()
                    + p_dropout.sum()
                    + p_text_embeddings_dropout.sum()
                )
            else:
                self_pred_loss = (
                     self_pred_loss_dropout
                     + self_pred_loss_text_dropout
                     + self_pred_loss_p_dropout
                     + self_pred_loss_p_text_dropout) / (
                     (self_pred_loss_dropout > 0)
                     + (self_pred_loss_text_dropout > 0)
                     + (self_pred_loss_p_dropout > 0)
                     + (self_pred_loss_p_text_dropout > 0))

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
        if isinstance(self.get_p_embedder(), SharedTextLookupEmbedder):
            self.get_p_embedder()._set_shared_lookup_embedder(self.get_s_embedder(), dataset)
        self.get_scorer()._initialize_text_embedders()


    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)

        job.pre_run_hooks.append(self._pre_run_hook)

    def _pre_run_hook(self, job):
        self.get_scorer()._initialize_text_embedders()

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, **kwargs) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If Transformer is wrapped in a reciprocal relations model, this will always be
        # the case.
        if direction == "o":
            super().score_spo(s, p, o, direction)
        else:
            raise ValueError("Transformer can only score objects")
