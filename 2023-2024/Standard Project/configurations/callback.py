from cinnamon_core.core.registry import Registry, register
from cinnamon_th.components.callback import THEarlyStopping
from cinnamon_th.configurations.callback import THEarlyStoppingConfig


@register
def register_callback_configurations():
    Registry.add_and_bind(config_class=THEarlyStoppingConfig,
                          component_class=THEarlyStopping,
                          config_constructor=THEarlyStoppingConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'patience': 3
                              }
                          },
                          name='callback',
                          tags={'early_stopping', 'th'},
                          namespace='sp')
