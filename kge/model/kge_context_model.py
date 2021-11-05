import torch
from typing import Union

from torch import Tensor

from kge import Config, Dataset
from kge.model import KgeModel
from kge.model.kge_model import RelationalScorer


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

        if self.has_option("scoring_drop_neighborhood_fraction"):
            self.scoring_drop_neighborhood_fraction = self.get_option("scoring_drop_neighborhood_fraction")
        else:
            self.scoring_drop_neighborhood_fraction = 0

        if self.has_option("recover_entity_drop_neighborhood_fraction"):
            self.recover_entity_drop_neighborhood_fraction = self.get_option(
                "recover_entity_drop_neighborhood_fraction")
        else:
            self.recover_entity_drop_neighborhood_fraction = 0

        # initialize context
        if self.has_option("neighborhood_size"):
            context_map = dataset.load_context_map("train")
            n_neighbors = torch.Tensor([len(x) for x in context_map])
            self.neighborhood_size = self.get_option("neighborhood_size")

            combined_context_tensor = torch.cat(context_map)

            lookup_tensor = torch.div(torch.rand(len(context_map), self.neighborhood_size), (1 / n_neighbors.unsqueeze(1)), rounding_mode="floor")

            starting_position = n_neighbors.cumsum(0)
            starting_position = torch.nn.functional.pad(starting_position, (1, -1), "constant", 0).unsqueeze(1)

            combined_position = (lookup_tensor + starting_position).long()

            combined_position[n_neighbors == 0] = 0

            lookup = combined_context_tensor[combined_position.view(-1)]

            context = lookup.reshape((lookup_tensor.shape[0], lookup_tensor.shape[1], 2))

            context[n_neighbors == 0] = -1

            self._context = context.to(config.get("job").get("device"))


    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None, num_replaced=0, num_unchanged=0) -> Tensor:
        r"""Compute scores for triples formed from a set of sp-pairs and all (or a subset of the) objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        If `o` is not None, it is a vector holding the indexes of the objects to score.

        """
        context = self._context[s]

        s_embedder = self.get_s_embedder()

        context_s, context_p = self.embed_context(s, p, drop_neighborhood_fraction=self.scoring_drop_neighborhood_fraction)

        if num_replaced > 0:
            s[:num_replaced] = torch.randint(low=0, high=s_embedder.vocab_size, size=(num_replaced,))

        s = s_embedder.embed(s)
        p = self.get_p_embedder().embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(s, p, o, context_s, context_p, combine="sp_", num_replaced=num_replaced, num_unchanged=num_unchanged)

    def score_po(self, p, o, s=None, num_replaced=0, num_unchanged=0):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)


        o_embedder = self.get_o_embedder()

        context_o, context_p = self.embed_context(o, p, s_embedder=o_embedder, drop_neighborhood_fraction=self.scoring_drop_neighborhood_fraction)


        if num_replaced > 0:
            o[:num_replaced] = torch.randint(low=0, high=o_embedder.vocab_size, size=(num_replaced,))

        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(o, p, s, context_o, context_p, combine="sp_", num_replaced=num_replaced, num_unchanged=num_unchanged)


    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
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

        context_s, context_p = self.embed_context(s, p)

        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s, p, o, context_s, context_p, combine="spo", num_unchanged=batch_size).view(-1)

    def recover_entity_sp(self, s, p, num_replaced=0, num_unchanged=0):
        """

        Args:
            s:
            p:

        Returns:

        """
        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()

        context_s, context_p = self.embed_context(s, p, drop_neighborhood_fraction=self.recover_entity_drop_neighborhood_fraction)

        if num_replaced > 0:
            s[:num_replaced] = torch.randint(low=0, high=s_embedder.vocab_size, size=(num_replaced,))

        s = s_embedder.embed(s)
        p = p_embedder.embed(p)
        return self._scorer.recover_entity_emb(s, p, context_s, context_p, num_replaced, num_unchanged)

    def recover_entity_po(self, p, o, num_replaced=0, num_unchanged=0):
        context = self._context[o]
        context_shape = context.shape

        o_embedder = self.get_o_embedder()

        context_o, context_p = self.embed_context(o, p, s_embedder=o_embedder, drop_neighborhood_fraction=self.recover_entity_drop_neighborhood_fraction)

        if num_replaced > 0:
            o[:num_replaced] = torch.randint(low=0, high=o_embedder.vocab_size, size=(num_replaced,))

        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.recover_entity_emb(o, p, context_o, context_p, num_replaced, num_unchanged)

    def embed_context(self, s, p, s_embedder=None, p_embedder=None, drop_neighborhood_fraction=0.):
        if not s_embedder:
            s_embedder = self.get_s_embedder()
        if not p_embedder:
            p_embedder = self.get_p_embedder()

        if not s.dtype == torch.int64:
            s = s.long()

        if not p.dtype == torch.int64:
            p = p.long()

        context = self._context[s]
        batch_size = len(s)

        new_neighborhood_size = round(self.neighborhood_size * (1 - drop_neighborhood_fraction))

        context = context[:, torch.randperm(self.neighborhood_size)[:new_neighborhood_size]]

        context = context.view((batch_size * new_neighborhood_size, 2))
        context_s = torch.empty((batch_size * new_neighborhood_size, s_embedder.dim), device=context.device)
        context_p = torch.empty((batch_size * new_neighborhood_size, p_embedder.dim), device=context.device)
        unknown_mask = context[:, 0] == -1

        context_s[~unknown_mask] = s_embedder.embed(context[:, 0][~unknown_mask])
        context_p[~unknown_mask] = p_embedder.embed(context[:, 1][~unknown_mask])

        context_s[unknown_mask] = 0
        context_p[unknown_mask] = 0

        context_s = context_s.view(batch_size, new_neighborhood_size, s_embedder.dim)
        context_p = context_p.view(batch_size, new_neighborhood_size, p_embedder.dim)

        return context_s, context_p
