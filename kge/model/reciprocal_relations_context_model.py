import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_context_model import KgeContextModel
from kge.model.kge_model import KgeModel


class ReciprocalRelationsContextModel(KgeContextModel):
    """Modifies a base model to use different relation embeddings for predicting subject and object.

    This implements the reciprocal relations training procedure of [TODO cite ConvE].
    Note that this model cannot be used to score a single triple, but only to rank sp_
    or _po questions.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        # Using a dataset with twice the number of relations to initialize base model
        alt_dataset = dataset.shallow_copy()
        alt_dataset._num_relations = dataset.num_relations() * 2
        alt_dataset._meta = dataset._meta.copy()
        alt_dataset._meta["relation_ids"] = dataset._meta["relation_ids"].copy()
        alt_dataset._meta["relation_ids"].extend([
            rel_id + "_reciprocal" for rel_id in dataset.relation_ids()
        ])
        base_model = KgeModel.create(
            config=config,
            dataset=alt_dataset,
            configuration_key=self.configuration_key + ".base_model",
            init_for_load_only=init_for_load_only,
        )

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=base_model.get_scorer(),
            create_embedders=False,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self._base_model = base_model
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

        print("halt")



    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return self._base_model.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        if direction == "o":
            return super().score_spo(s, p, o, "o")
        elif direction == "s":
            return super().score_spo(o, p + self.dataset.num_relations(), s, "o")
        else:
            raise Exception(
                "The reciprocal relations model cannot compute "
                "undirected spo scores."
            )

    def score_po(self, p, o, s=None):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)

        context = self._context[o]
        context_shape = context.shape

        o_embedder = self.get_o_embedder()
        p_embedder = self.get_p_embedder()

        context = context.view((context_shape[0] * context_shape[1], context_shape[2]))
        context_o = torch.zeros((context_shape[0] * context_shape[1], o_embedder.dim), device=context.device)
        context_p = torch.zeros((context_shape[0] * context_shape[1], p_embedder.dim), device=context.device)
        unknown_mask = context[:, 0] == -1

        context_o[~unknown_mask] = o_embedder.embed(context[:, 0][~unknown_mask])
        context_p[~unknown_mask] = p_embedder.embed(context[:, 1][~unknown_mask])

        context_o = context_o.view(context_shape[0], context_shape[1], o_embedder.dim)
        context_p = context_p.view(context_shape[0], context_shape[1], p_embedder.dim)

        p = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(o, p, s, context_o, context_p, combine="sp_")

    def score_so(self, s, o, p=None):
        raise Exception("The reciprocal relations model cannot score relations.")

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
    ) -> torch.Tensor:
        context = self._context[s.long()]
        context_shape = context.shape

        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()
        o_embedder = self.get_o_embedder()

        context = context.view((context_shape[0] * context_shape[1], context_shape[2]))
        context_s = torch.zeros((context_shape[0] * context_shape[1], s_embedder.dim), device=context.device)
        context_p = torch.zeros((context_shape[0] * context_shape[1], p_embedder.dim), device=context.device)
        unknown_mask = context[:, 0] == -1

        context_s[~unknown_mask] = s_embedder.embed(context[:, 0][~unknown_mask])
        context_p[~unknown_mask] = p_embedder.embed(context[:, 1][~unknown_mask])

        context_s = context_s.view(context_shape[0], context_shape[1], s_embedder.dim)
        context_p = context_p.view(context_shape[0], context_shape[1], p_embedder.dim)


        context = self._context[o.long()]
        context_shape = context.shape
        context = context.view((context_shape[0] * context_shape[1], context_shape[2]))
        context_o = torch.zeros((context_shape[0] * context_shape[1], s_embedder.dim), device=context.device)
        context_p_inv = torch.zeros((context_shape[0] * context_shape[1], p_embedder.dim), device=context.device)
        unknown_mask = context[:, 0] == -1

        context_o[~unknown_mask] = o_embedder.embed(context[:, 0][~unknown_mask])
        context_p_inv[~unknown_mask] = p_embedder.embed(context[:, 1][~unknown_mask])

        context_o = context_o.view(context_shape[0], context_shape[1], o_embedder.dim)
        context_p_inv = context_p_inv.view(context_shape[0], context_shape[1], p_embedder.dim)

        s = s_embedder.embed(s)
        p_inv = p_embedder.embed(p + self.dataset.num_relations())
        p = p_embedder.embed(p)
        o = o_embedder.embed(o)
        if s_embedder is o_embedder:
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()

            sp_scores = self._scorer.score_emb(s, p, all_entities, context_s, context_p, combine="sp_")
            po_scores = self._scorer.score_emb(o, p_inv, all_entities, context_o, context_p_inv, combine="sp_")
        else:
            if entity_subset is not None:
                all_objects = o_embedder.embed(entity_subset)
                all_subjects = s_embedder.embed(entity_subset)
            else:
                all_objects = o_embedder.embed_all()
                all_subjects = s_embedder.embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, context_s, context_p, combine="sp_")
            po_scores = self._scorer.score_emb(o, p_inv, all_subjects, context_o, context_p_inv, combine="sp_")
        return torch.cat((sp_scores, po_scores), dim=1)
