from sklearn.metrics import f1_score

from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.components.metrics import LambdaMetric, MetricPipeline
from cinnamon_generic.configurations.metrics import LambdaMetricConfig
from cinnamon_generic.configurations.pipeline import PipelineConfig
from components.metrics import SequenceF1Metric


class SequenceF1MetricConfig(LambdaMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.method = f1_score
        config.method_args = {'average': 'macro'}

        config.add(name='output_key',
                   is_required=True,
                   description='Predictions and ground-truth key name.',
                   allowed_range=lambda value: value in ['emotions', 'triggers'])

        return config

    @classmethod
    def get_emotion_config(
            cls
    ):
        config = cls.get_default()
        config.name = 'emotion_F1'
        config.output_key = 'emotions'
        return config

    @classmethod
    def get_triggers_config(
            cls
    ):
        config = cls.get_default()
        config.name = 'triggers_F1'
        config.output_key = 'triggers'
        return config


@register
def register_metrics_configurations():
    Registry.add_and_bind(config_class=SequenceF1MetricConfig,
                          config_constructor=SequenceF1MetricConfig.get_emotion_config,
                          component_class=SequenceF1Metric,
                          name='metrics',
                          tags={'emotions_f1'},
                          namespace='sp')

    Registry.add_and_bind(config_class=SequenceF1MetricConfig,
                          config_constructor=SequenceF1MetricConfig.get_triggers_config,
                          component_class=SequenceF1Metric,
                          name='metrics',
                          tags={'triggers_f1'},
                          namespace='sp')

    Registry.add_and_bind(config_class=PipelineConfig,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='metrics', tags={'emotions_f1'}, namespace='sp'),
                                  RegistrationKey(name='metrics', tags={'triggers_f1'}, namespace='sp'),
                              ],
                              'names': [
                                  'emotions_f1',
                                  'triggers_f1',
                              ]
                          },
                          component_class=MetricPipeline,
                          name='metrics',
                          tags={'emotions_f1', 'triggers_f1'},
                          namespace='sp')
