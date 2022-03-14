import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, RelationalScorer
from kge.util.io import load_checkpoint


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

        model1_checkpoint = self.config.get(self.configuration_key + ".checkpoint1")
        model2_checkpoint = self.config.get(self.configuration_key + ".checkpoint2")
        # Initialize base model 1
        if model1_checkpoint:
            model1 = KgeModel.create_from(load_checkpoint(model1_checkpoint), dataset)
        else:
            model1 = KgeModel.create(
                config=config,
                dataset=dataset,
                configuration_key=self.configuration_key + ".model1",
                init_for_load_only=init_for_load_only,
            )

        # Initialize base model 2
        if model2_checkpoint:
            model2 = KgeModel.create_from(load_checkpoint(model2_checkpoint), dataset)
        else:
            model2 = KgeModel.create(
                config=config,
                dataset=dataset,
                configuration_key=self.configuration_key + ".model2",
                init_for_load_only=init_for_load_only,
            )

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=model1.get_scorer(),
            create_embedders=False,
            init_for_load_only=init_for_load_only,
            configuration_key=self.configuration_key
        )
        self._model1 = model1
        self._model2 = model2
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        #self._entity_embedder = self._base_model.get_s_embedder()
        #self._relation_embedder = self._base_model.get_p_embedder()

        self.enable_weights_grad = self.config.get(self.configuration_key + ".enable_weights_grad")
        self.enable_models_grad = self.config.get(self.configuration_key + ".enable_models_grad")

        self.model1_weight = self.config.get(self.configuration_key + ".model1.weight")
        self.model2_weight = self.config.get(self.configuration_key + ".model2.weight")

        self.model1_offset = self.config.get(self.configuration_key + ".model1.offset")
        self.model2_offset = self.config.get(self.configuration_key + ".model2.offset")

        self.alpha = self.config.get(self.configuration_key + ".alpha")

        if self.enable_weights_grad:
            self.model1_weight = torch.nn.Parameter(torch.tensor(self.model1_weight))
            self.model2_weight = torch.nn.Parameter(torch.tensor(self.model2_weight))

            self.model1_offset = torch.nn.Parameter(torch.tensor(self.model1_offset))
            self.model2_offset = torch.nn.Parameter(torch.tensor(self.model2_offset))

            self.alpha = torch.nn.Parameter(torch.tensor(self.alpha))


        # self.r1_sum = 0
        # self.r1_squared_sum = 0
        # self.r1_num = 0
        # self.r2_sum = 0
        # self.r2_squared_sum = 0
        # self.r2_num = 0

    def prepare_job(self, job, **kwargs):
        self._model1.prepare_job(job, **kwargs)
        self._model2.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return self._model1.penalty(**kwargs) + self._model2.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, **kwargs) -> Tensor:
        with torch.set_grad_enabled(self.enable_models_grad):
            r1 = self._model1.score_spo(s, p, o, direction, **kwargs)
            r2 = self._model2.score_spo(s, p, o, direction, **kwargs)
        return self._return_with_self_loss(r1, r2)

    def score_po(self, p, o, s=None, ground_truth=None, **kwargs):
        with torch.set_grad_enabled(self.enable_models_grad):
            r1 = self._model1.score_po(p, o, s, ground_truth, **kwargs)
            r2 = self._model2.score_po(p, o, s, ground_truth, **kwargs)
        return self._return_with_self_loss(r1, r2)

    def score_sp(self, s, p, o=None, ground_truth=None, **kwargs):
        with torch.set_grad_enabled(self.enable_models_grad):
            r1 = self._model1.score_sp(s, p, o, ground_truth, **kwargs)
            r2 = self._model2.score_sp(s, p, o, ground_truth, **kwargs)
        return self._return_with_self_loss(r1, r2)


    def score_so(self, s, o, p=None, **kwargs):
        with torch.set_grad_enabled(self.enable_models_grad):
            r1 = self._model1.score_so(s, o, p, **kwargs)
            r2 = self._model2.score_so(s, o, p, **kwargs)
        return self._return_with_self_loss(r1, r2)

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(self.enable_models_grad):
            r1 = self._model1.score_sp_po(s, p, o, entity_subset, **kwargs)
            r2 = self._model2.score_sp_po(s, p, o, entity_subset, **kwargs)

        # self.r1_sum += r1.sum()
        # self.r1_squared_sum += r1.pow(2).sum()
        # self.r1_num += r1.numel()
        #
        # self.r2_sum += r2.sum()
        # self.r2_squared_sum += r2.pow(2).sum()
        # self.r2_num += r2.numel()

        return self._return_with_self_loss(r1, r2)

    def _return_with_self_loss(self, r1, r2):
        self_loss = 0
        if isinstance(r1, tuple):
            r1, self_loss_tmp = r1
            self_loss += self_loss_tmp * self.model1_weight
        if isinstance(r2, tuple):
            r2, self_loss_tmp = r2
            self_loss += self_loss_tmp * self.model2_weight

        result = (self.alpha * self.model1_weight * (r1 + self.model1_offset)
                  + (1 - self.alpha) * self.model2_weight * (r2 + self.model2_offset))

        if self_loss > 0:
            return result, self_loss
        else:
            return result


class EnsembleScorer(RelationalScorer):
    pass

