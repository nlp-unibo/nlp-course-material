from typing import Dict, Type

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, RegistrationKey, register
from cinnamon_generic.components.processor import ProcessorPipeline
from cinnamon_generic.configurations.pipeline import OrderedPipelineConfig
from cinnamon_generic.nlp.configurations.processor import TokenizerProcessorConfig
from components.processor import BERTTokenizer, EmotionProcessor, THClassifierProcessor, THDataProcessor, \
    RoutineStepsProcessor


class BERTTokenizerConfig(TokenizerProcessorConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='preloaded_model_name',
                   value="bert-base-uncased",
                   type_hint=str,
                   description="Pre-trained tokenizer model name from HuggingFace.")
        config.add(name='tokenization_args',
                   value={
                       'truncation': True
                   },
                   type_hint=Dict,
                   description='Tokenization additional arguments.')
        config.add(name='detokenization_args',
                   value={},
                   type_hint=Dict,
                   description='De-tokenization additional arguments.')

        return config


class THDataProcessorConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='num_workers',
                   value=0,
                   type_hint=int,
                   description='Number of processes to use for data loading.')
        config.add(name='batch_size',
                   value=4,
                   type_hint=int,
                   is_required=True,
                   description='Batch size for aggregating samples.')

        return config


@register
def register_processors():
    # Input

    Registry.add_and_bind(config_class=Configuration,
                          component_class=EmotionProcessor,
                          name='processor',
                          tags={'emotion'},
                          namespace='sp')

    Registry.add_and_bind(config_class=BERTTokenizerConfig,
                          component_class=BERTTokenizer,
                          name='processor',
                          tags={'bert'},
                          namespace='sp')

    Registry.add_and_bind(config_class=THDataProcessorConfig,
                          component_class=THDataProcessor,
                          name='processor',
                          tags={'th', 'data'},
                          namespace='sp')

    # Model

    Registry.add_and_bind(config_class=OrderedPipelineConfig,
                          component_class=ProcessorPipeline,
                          config_constructor=OrderedPipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='processor',
                                                  tags={'bert'},
                                                  namespace='sp'),
                                  RegistrationKey(name='processor',
                                                  tags={'emotion'},
                                                  namespace='sp'),
                                  RegistrationKey(name='processor',
                                                  tags={'th', 'data'},
                                                  namespace='sp')
                              ],
                              'names': [
                                  'bert_tokenizer',
                                  'emotion_processor',
                                  'th_processor'
                              ]
                          },
                          name='processor',
                          tags={'bert', 'emotion', 'th'},
                          namespace='sp')

    Registry.add_and_bind(config_class=Configuration,
                          component_class=THClassifierProcessor,
                          name='processor',
                          tags={'classifier', 'th'},
                          namespace='sp')

    # Routine

    Registry.add_and_bind(config_class=Configuration,
                          component_class=RoutineStepsProcessor,
                          name='processor',
                          tags={'routine', 'step'},
                          namespace='sp')
