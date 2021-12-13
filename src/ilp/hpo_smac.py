"""Hyperparameter optimization on StarE both with SMAC & Random Search."""
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    NumericalHyperparameter, UniformFloatHyperparameter, Constant
import numpy as np
from functools import partial

import logging
from typing import Sequence, Tuple
from .models.layers import rotate

from class_resolver import Hint
from pykeen.losses import BCEWithLogitsLoss, MarginRankingLoss

from .models import QualifierModel, StarE, model_resolver
from .models.layers import rotate
from .pipeline import pipeline

logger = logging.getLogger(__name__)

__all__ = [
    "objective",
    "hpo_pipeline_smac",
]

# TODO: change ng.p.parameter to Configspace in order to pass them to smac
# ng.p.TransitionChoice is an ordered categorical parameter; its transitions
# can be picked in EA only for prev & next.
# ng.p.log: positive parameter represented in log scale
# ng.p.choice unordered categorical parameter

# search spaces
hpo_ranges_cs = [
    CategoricalHyperparameter(name='embedding_dim', choices=list(range(128, 256 + 1, 32)), default_value=128),
    CategoricalHyperparameter(name='batch_size', choices=list(range(128, 1025, 64)), default_value=128),
    UniformFloatHyperparameter(name='learning_rate', lower=0.0001, upper=1.0, default_value=1e-3, log=True),
    CategoricalHyperparameter(name='label_smoothing', choices=[0.1, 0.15]),
    Constant('early_stopping_relative_delta', 3e-3)
]

transformer_hpo_ranges = [
    CategoricalHyperparameter(name='dim_transformer_hidden', choices=[512, 1024]),
    CategoricalHyperparameter(name='num_transformer_heads', choices=[2, 4]),
    CategoricalHyperparameter(name='num_transformer_layers', choices=[2, 4]),
    CategoricalHyperparameter(name='affine_transformation', choices=[True, False])
]

dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
gnn_hpo_ranges = [
    CategoricalHyperparameter(name='num_layers', choices=[2, 3]),
    CategoricalHyperparameter(name='hid_drop', choices=dropout),
    CategoricalHyperparameter(name='gcn_drop', choices=dropout),
    CategoricalHyperparameter(name='attention_slope', choices=[0.1, 0.2, 0.3, 0.4]),
    CategoricalHyperparameter(name='attention_drop', choices=dropout),
    CategoricalHyperparameter(name='num_attention_heads', choices=[2, 4]),
    CategoricalHyperparameter(name='qualifier_aggregation', choices=["sum", "attn"]),
    Constant('triple_qual_weight', 0.8),

    # -->!CAREFULL: these hyperparameters are now fixed in the StarE class signature!
    # Constant('composition_function', rotate),
    # Constant('qualifier_comp_function', rotate),
    # Constant('use_learnable_x', False),  # default
    # Constant('use_bias', False),
    # Constant('use_attention', False)
]


# hpo_ranges = dict(
#     embedding_dim=ng.p.TransitionChoice(list(range(128, 256 + 1, 32))),
#     batch_size=ng.p.TransitionChoice(list(range(128, 1025, 64))),
#     learning_rate=ng.p.Log(lower=0.0001, upper=1.0),
#     label_smoothing=ng.p.TransitionChoice([0.1, 0.15]),
#     early_stopping_relative_delta=0.003,
# )
# transformer_hpo_ranges = dict(
#     dim_transformer_hidden=ng.p.TransitionChoice([512, 1024]),
#     num_transformer_heads=ng.p.TransitionChoice([2, 4]),
#     num_transformer_layers=ng.p.TransitionChoice([2, 4]),
#     affine_transformation=ng.p.Choice([True, False]),
# )
# dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
# gnn_hpo_ranges = dict(
#     use_learnable_x=False,
#     num_layers=ng.p.TransitionChoice([2, 3]),
#     hid_drop=ng.p.Choice(dropout),
#     gcn_drop=ng.p.Choice(dropout),
#     attention_slope=ng.p.TransitionChoice([0.1, 0.2, 0.3, 0.4]),
#     triple_qual_weight=0.8,
#     attention_drop=ng.p.Choice(dropout),
#     num_attention_heads=ng.p.TransitionChoice([2, 4]),
#     composition_function=rotate,
#     qualifier_aggregation=ng.p.TransitionChoice(["sum", "attn"]),
#     qualifier_comp_function=rotate,
#     use_bias=False,
#     use_attention=False,
# )
#

def objective(
        hpkwargs,
        kwargs,
        raise_on_error: bool = False,
        verbose: bool = False,
) -> Tuple[float, int, Sequence[float]]:
    """Optimization objective to minimize."""
    # try:
    return pipeline(
            is_hpo=True,
            verbose=verbose,
            **kwargs, **hpkwargs
        )
    # except Exception as e:
    #     logger.error(f"ERROR: {e}")
    #     if 'CUDA out of memory.' in e.args[0]:
    #         return 1000.0, None, None
    #     if raise_on_error:
    #         raise e
    #     return 1000.0, None, None


def hpo_pipeline_smac(
        *,
        training_approach: str,  # fixme hard coded default loss for stare: BCEWithLogitsLoss()
        num_epochs: int,
        early_stopping_patience: int,
        num_hpo_iterations: int,
        model_cls: Hint[QualifierModel],
        hpo,
        **kwargs,
):
    """Optimize hyperparameters using smac."""
    # normalize model class
    model_cls = model_resolver.lookup(query=model_cls)

    # model specific search spaces
    model_kwargs = dict()
    cs = ConfigurationSpace()
    cs.add_hyperparameters(hpo_ranges_cs)
    if issubclass(model_cls, QualifierModel):
        # model_kwargs.update(transformer_hpo_ranges)
        cs.add_hyperparameters(transformer_hpo_ranges)

    if issubclass(model_cls, StarE):
        # model_kwargs.update(gnn_hpo_ranges)
        cs.add_hyperparameters(gnn_hpo_ranges)

    # test out the configuration
    # pipeparam = dict(cs.get_default_configuration())
    kwargs.update(kwargs)
    kwargs['model_name'] = model_cls
    kwargs['num_epochs'] = num_epochs
    # objective(hpkwargs=pipeparam, kwargs=kwargs)
    # TODO how to pass the configuration with the additional config stuff (non Hp) to smac?
    #  - partial?
    # fixme: kwargs argument was dict that aside the hps contains hpo_pipeline_smac's kwargs e.g. dataset_name ...
    # fixme: move kwargs to tae_runner_kwargs --> and revert the quickfixes "hpkwargs-kwargs" combo
    tae = partial(objective, kwargs=kwargs)

    scenario = Scenario({
        "run_obj": "quality",  # we optimize quality (alternatively runtime)
        "runcount-limit": 50,  # max. number of function evaluations
        "cs": cs,  # configuration space
        "deterministic": "true"})

    # TODO tell w&b if we optimized SMAC or RS: using config!
    if hpo == 'smac':
        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(42),
                        tae_runner=tae)
    elif hpo == 'rs': # Random search
        smac = ROAR(scenario=scenario,
                    rng=np.random.RandomState(42),
                    tae_runner=tae)

    incumbent = smac.optimize()

    # TODO Store the Results in a meaningful and comparable manner.
