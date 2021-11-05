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

    def score_po(self, p, o, s=None, num_replaced=0, num_unchanged=0):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)

        o_embedder = self.get_o_embedder()

        context_o, context_p = self.embed_context(o, p, s_embedder=o_embedder, drop_neighborhood_fraction=self.scoring_drop_neighborhood_fraction)

        if num_replaced > 0:
            o[:num_replaced] = torch.randint(low=0, high=o_embedder.vocab_size, size=(num_replaced,))

        p = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(o, p, s, context_o, context_p, combine="sp_", num_replaced=num_replaced, num_unchanged=num_unchanged)

    def score_so(self, s, o, p=None):
        raise Exception("The reciprocal relations model cannot score relations.")

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = len(s)

        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()
        o_embedder = self.get_o_embedder()

        context_s, context_p = self.embed_context(s, p)

        context_o, context_p_inv = self.embed_context(o, p, s_embedder=o_embedder)

        s = s_embedder.embed(s)
        p_inv = p_embedder.embed(p + self.dataset.num_relations())
        p = p_embedder.embed(p)
        o = o_embedder.embed(o)
        if s_embedder is o_embedder:
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()

            sp_scores = self._scorer.score_emb(s, p, all_entities, context_s, context_p, combine="sp_", num_unchanged=batch_size)
            po_scores = self._scorer.score_emb(o, p_inv, all_entities, context_o, context_p_inv, combine="sp_", num_unchanged=batch_size)
        else:
            if entity_subset is not None:
                all_objects = o_embedder.embed(entity_subset)
                all_subjects = s_embedder.embed(entity_subset)
            else:
                all_objects = o_embedder.embed_all()
                all_subjects = s_embedder.embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, context_s, context_p, combine="sp_", num_unchanged=batch_size)
            po_scores = self._scorer.score_emb(o, p_inv, all_subjects, context_o, context_p_inv, combine="sp_", num_unchanged=batch_size)
        return torch.cat((sp_scores, po_scores), dim=1)

    def recover_entity_po(self, p, o, num_replaced=0, num_unchanged=0):
        o_embedder = self.get_o_embedder()

        context_o, context_p = self.embed_context(o, p, s_embedder=o_embedder, drop_neighborhood_fraction=self.recover_entity_drop_neighborhood_fraction)

        if num_replaced > 0:
            o[:num_replaced] = torch.randint(low=0, high=o_embedder.vocab_size, size=(num_replaced,))

        p = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o = self.get_o_embedder().embed(o)
        return self._scorer.recover_entity_emb(o, p, context_o, context_p, num_replaced, num_unchanged)
