from typing import Type, Dict

import torch as th

from cinnamon_core.core.configuration import C, Configuration
from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.configurations.model import NetworkConfig
from components.model import BERTBaseline, DummyBaseline


class BERTBaselineConfig(NetworkConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.epochs = 20
        config.add(name='preloaded_model_name',
                   value='bert-base-uncased',
                   type_hint=str,
                   is_required=True,
                   description='The pre-trained HuggingFace BERT model card to use.')
        config.add(name='optimizer_class',
                   value=th.optim.Adam,
                   is_required=True,
                   description='Optimizer to use for network weights update')
        config.add(name='optimizer_args',
                   value={
                       "lr": 2e-05,
                       "weight_decay": 1e-05
                   },
                   type_hint=Dict,
                   description="Arguments for creating the network optimizer")
        config.add(name='freeze_bert',
                   value=False,
                   type_hint=bool,
                   description="If true, the BERT model weights are freezed.")
        config.add(name='add_premise',
                   value=False,
                   type_hint=bool,
                   description='If enabled, the premise text is added as feature for classification.')
        config.add(name='add_stance',
                   value=False,
                   type_hint=bool,
                   description='If enabled, the premise-to-conclusion stance is added as feature for classification.')

        return config

    @classmethod
    def get_conclusion_config(
            cls
    ):
        config = cls.get_default()
        config.add_premise = False
        config.add_stance = False
        return config

    @classmethod
    def get_premise_config(
            cls
    ):
        config = cls.get_default()
        config.add_premise = True
        return config

    @classmethod
    def get_premise_stance_config(
            cls
    ):
        config = cls.get_default()
        config.add_premise = True
        config.add_stance = True
        return config


class DummyBaselineConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='strategy',
                   variants=['most_frequent', 'uniform'],
                   is_required=True,
                   description='The type of dummy classifier to use.')

        return config


@register
def register_models():
    Registry.add_and_bind_variants(config_class=BERTBaselineConfig,
                                   config_constructor=BERTBaselineConfig.get_conclusion_config,
                                   component_class=BERTBaseline,
                                   name='model',
                                   tags={'bert', 'conclusion'},
                                   namespace='a2')

    Registry.add_and_bind_variants(config_class=BERTBaselineConfig,
                                   config_constructor=BERTBaselineConfig.get_premise_config,
                                   component_class=BERTBaseline,
                                   name='model',
                                   tags={'bert', 'premise', 'conclusion'},
                                   namespace='a2')

    Registry.add_and_bind_variants(config_class=BERTBaselineConfig,
                                   config_constructor=BERTBaselineConfig.get_premise_stance_config,
                                   component_class=BERTBaseline,
                                   name='model',
                                   tags={'bert', 'premise', 'conclusion', 'stance'},
                                   namespace='a2')

    Registry.add_and_bind_variants(config_class=DummyBaselineConfig,
                                   component_class=DummyBaseline,
                                   name='model',
                                   tags={'dummy', 'baseline'},
                                   namespace='a2')
