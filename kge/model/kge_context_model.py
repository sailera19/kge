import time

import torch
from typing import Union

from torch import Tensor

from kge import Config, Dataset
from kge.model import KgeModel
from kge.model.kge_model import RelationalScorer
from kge.util import sc


class KgeContextModel(KgeModel):
    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            scorer: Union[RelationalScorer, type],
            create_embedders=True,
            configuration_key=None,
            init_for_load_only=False
    ):
        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=scorer,
            create_embedders=create_embedders,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

        if self.has_option("drop_neighborhood_fraction"):
            self.drop_neighborhood_fraction = self.get_option("drop_neighborhood_fraction")
        else:
            self.drop_neighborhood_fraction = 0

        if self.has_option("neighborhood_size"):
            self.neighborhood_size = self.get_option("neighborhood_size")

        self.context_implementation = self.get_option("context_implementation")



    def score_sp(self,
                 s: Tensor,
                 p: Tensor,
                 o: Tensor = None,
                 ground_truth: Tensor = None,
                 **kwargs,
            ) -> Tensor:
        r"""Compute scores for triples formed from a set of sp-pairs and all (or a subset of the) objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        If `o` is not None, it is a vector holding the indexes of the objects to score.

        """
        s_embedder = self.get_s_embedder()

        context_s, context_p, attention_mask = self.embed_context(s, p, ground_truth, drop_neighborhood_fraction=self.drop_neighborhood_fraction)

        s_emb = s.clone()

        s_emb = s_embedder.embed(s)
        p = self.get_p_embedder().embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(s_emb, p, o, context_s, context_p, attention_mask, combine="sp_", ground_truth_s=s)



    def score_po(self, p, o, s=None, ground_truth=None, **kwargs):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)

        o_embedder = self.get_o_embedder()

        context_o, context_p, attention_mask = self.embed_context(o, p, ground_truth, s_embedder=o_embedder, drop_neighborhood_fraction=self.drop_neighborhood_fraction)

        o_emb = o.clone()

        p = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o_emb)
        return self._scorer.score_emb(o_emb, p, s, context_o, context_p, attention_mask, combine="sp_", ground_truth_s=o)


    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, ground_truth=None, **kwargs) -> Tensor:
        r"""Compute scores for a set of triples.

        `s`, `p`, and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        `direction` may influence how scores are computed. For most models, this setting
        has no meaning. For reciprocal relations, direction must be either `"s"` or
        `"o"` (depending on what is predicted).

        Returns a vector of size :math:`n`, in which the :math:`i`-th entry holds the
        score of triple :math:`(s_i, p_i, o_i)`.

        """
        batch_size = len(s)

        context_s, context_p, attention_mask = self.embed_context(s, p, ground_truth)

        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s, p, o, context_s, context_p, attention_mask, combine="spo").view(-1)


    def embed_context(self, s, p, ground_truth, s_embedder=None, p_embedder=None, drop_neighborhood_fraction=0.):
        if not s_embedder:
            s_embedder = self.get_s_embedder()
        if not p_embedder:
            p_embedder = self.get_p_embedder()

        if not s.dtype == torch.int64:
            s = s.long()

        if not p.dtype == torch.int64:
            p = p.long()

        device = s.device
        batch_size = len(s)

        ctx_list, ctx_size = self.dataset.index('neighbor')
        ctx_ids = ctx_list[s].to(device).transpose(1, 2)
        ctx_size = ctx_size[s].to(device)

        # sample neighbors unifromly during training
        if self.training:
            perm_vector = sc.get_randperm_from_lengths(ctx_size, ctx_ids.size(1))
            ctx_ids = torch.gather(ctx_ids, 1, perm_vector.unsqueeze(-1).expand_as(ctx_ids))

        # [bs, length, 2]
        ctx_ids = ctx_ids[:, :self.neighborhood_size]
        ctx_size[ctx_size > self.neighborhood_size] = self.neighborhood_size

        # [bs, max_ctx_size]
        entity_ids = ctx_ids[...,0]
        relation_ids = ctx_ids[...,1]

        attention_mask = sc.get_mask_from_sequence_lengths(ctx_size, self.neighborhood_size)

        if self.training:
            # mask out ground truth during training to avoid overfitting
            # else is filtering out relations to the entity itself as well.
            if self.context_implementation == "hitter":
                gt_mask = ((entity_ids != ground_truth.view(batch_size, 1)) | (
                            ((relation_ids - self.dataset.num_relations()) != p.view(batch_size, 1)) &
                            ((relation_ids + self.dataset.num_relations()) != p.view(batch_size, 1))
                            ))
            else:
                gt_mask = ((entity_ids != ground_truth.view(batch_size, 1)) |
                           (
                            (relation_ids != p.view(batch_size, 1)) &
                            ((relation_ids - self.dataset.num_relations()) != p.view(batch_size, 1)) &
                            ((relation_ids + self.dataset.num_relations()) != p.view(batch_size, 1))
                            )
                           )
            ctx_random_mask = (attention_mask
                               .new_ones((batch_size, self.neighborhood_size))
                               .bernoulli_(1 - drop_neighborhood_fraction))
            attention_mask = attention_mask & ctx_random_mask & gt_mask


        context_s = torch.empty((batch_size * self.neighborhood_size, s_embedder.dim), device=device)
        context_p = torch.empty((batch_size * self.neighborhood_size, p_embedder.dim), device=device)
        context_s[attention_mask.view(batch_size * self.neighborhood_size)] = s_embedder.embed(entity_ids[attention_mask])
        context_p[attention_mask.view(batch_size * self.neighborhood_size)] = p_embedder.embed(relation_ids[attention_mask])

        context_s[~attention_mask.view(batch_size * self.neighborhood_size)] = 0
        context_p[~attention_mask.view(batch_size * self.neighborhood_size)] = 0

        context_s = context_s.view(batch_size, self.neighborhood_size, s_embedder.dim)
        context_p = context_p.view(batch_size, self.neighborhood_size, p_embedder.dim)


        return context_s, context_p, attention_mask


