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
        config.add(name='lstm_weights',
                   value=64,
                   type_hint=int,
                   is_required=True,
                   description='Output features for LSTM controller.'
                               'The LSTM controller is used to provide a contextualized representation'
                               'of each utterance encoding, each of which is fed to the triggers classifier.')

        config.add(name='optimizer_class',
                   value=th.optim.Adam,
                   is_required=True,
                   description='Optimizer to use for network weights update')
        config.add(name='optimizer_args',
                   value={
                       "lr": 1e-03,
                       "weight_decay": 1e-05
                   },
                   type_hint=Dict,
                   description="Arguments for creating the network optimizer")
        config.add(name='freeze_bert',
                   is_required=True,
                   type_hint=bool,
                   variants=[False, True],
                   description="If true, the BERT model weights are freezed.")

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
                                   component_class=BERTBaseline,
                                   name='model',
                                   tags={'bert', 'baseline'},
                                   namespace='sp')

    Registry.add_and_bind_variants(config_class=DummyBaselineConfig,
                                   component_class=DummyBaseline,
                                   name='model',
                                   tags={'dummy', 'baseline'},
                                   namespace='sp')
