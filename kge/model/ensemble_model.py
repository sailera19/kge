import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel


class EnsembleModel(KgeModel):
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
        base_model = KgeModel.create(
            config=config,
            dataset=dataset,
            configuration_key=self.configuration_key + ".base_model",
            init_for_load_only=init_for_load_only,
        )

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=base_model.get_scorer(),
            create_embedders=False,
            init_for_load_only=init_for_load_only,
        )
        self._base_model = base_model
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return self._base_model.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, **kwargs) -> Tensor:
        return self._base_model.get_scorer().score_spo(s, p, o, direction, **kwargs)

    def score_po(self, p, o, s=None, ground_truth=None, **kwargs):
        return self._base_model.get_scorer().score_po(p, o, s, ground_truth, **kwargs)

    def score_so(self, s, o, p=None, **kwargs):
        return self._base_model.get_scorer().score_so(s, o, p, **kwargs)

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._base_model.get_scorer().score_sp_po(self, s, p, o, entity_subset, **kwargs)