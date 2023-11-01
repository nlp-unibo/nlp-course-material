from sklearn.metrics import f1_score

from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.configurations.metrics import LambdaMetricConfig
from components.metrics import MultiLabelMetric


class MultiLabelMetricConfig(LambdaMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.method = f1_score
        config.method_args = {'average': 'binary', 'pos_label': 1}
        config.name = 'avg_f1'

        return config


@register
def register_metrics_configurations():
    Registry.add_and_bind(config_class=MultiLabelMetricConfig,
                          component_class=MultiLabelMetric,
                          name='metrics',
                          namespace='a2')
