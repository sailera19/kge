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

        # initialize context
        if config.get(configuration_key).get("neighborhood_size"):
            context_map = dataset.load_context_map("train")
            n_neighbors = torch.Tensor([len(x) for x in context_map])
            neighborhood_size = config.get(configuration_key).get("neighborhood_size")

            combined_context_tensor = torch.cat(context_map)

            lookup_tensor = (torch.rand(len(context_map), neighborhood_size) // (1 / n_neighbors.unsqueeze(1)))

            starting_position = n_neighbors.cumsum(0)
            starting_position = torch.nn.functional.pad(starting_position, (1, -1), "constant", 0).unsqueeze(1)

            combined_position = (lookup_tensor + starting_position).long()

            combined_position[n_neighbors == 0] = 0

            lookup = combined_context_tensor[combined_position.view(-1)]

            context = lookup.reshape((lookup_tensor.shape[0], lookup_tensor.shape[1], 2))

            context[n_neighbors == 0] = -1

            self._context = context.to(config.get("job").get("device"))


    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        r"""Compute scores for triples formed from a set of sp-pairs and all (or a subset of the) objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        If `o` is not None, it is a vector holding the indexes of the objects to score.

        """
        context = self._context[s]
        context_shape = context.shape

        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()

        context = context.view((context_shape[0] * context_shape[1], context_shape[2]))
        context_s = s_embedder.embed(context[:, 0]).view(context_shape[0], context_shape[1], s_embedder.dim)
        context_p = p_embedder.embed(context[:, 1]).view(context_shape[0], context_shape[1], p_embedder.dim)

        s = s_embedder.embed(s)
        p = p_embedder.embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(s, p, o, context_s, context_p, combine="sp_")


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
        context = self._context[s.long()]
        context_shape = context.shape

        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()

        context = context.view((context_shape[0] * context_shape[1], context_shape[2]))
        context_s = torch.zeros((context_shape[0] * context_shape[1], s_embedder.dim), device=context.device)
        context_p = torch.zeros((context_shape[0] * context_shape[1], p_embedder.dim), device=context.device)
        unknown_mask = context[:, 0] == -1

        context_s[~unknown_mask] = s_embedder.embed(context[:, 0][~unknown_mask])
        context_p[~unknown_mask] = p_embedder.embed(context[:, 1][~unknown_mask])

        context_s = context_s.view(context_shape[0], context_shape[1], s_embedder.dim)
        context_p = context_p.view(context_shape[0], context_shape[1], p_embedder.dim)



        s = s_embedder.embed(s)
        p = p_embedder.embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s, p, o, context_s, context_p, combine="spo").view(-1)