from cinnamon_core.core.registry import RegistrationKey, Registry, register
from cinnamon_generic.components.routine import TrainAndTestRoutine
from cinnamon_generic.configurations.routine import RoutineConfig


class Task3RoutineConfig(RoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.data_loader = RegistrationKey(name='data_loader',
                                             namespace='a2')

        config.routine_processor = RegistrationKey(name='routine_processor',
                                                   tags={'average', 'fold'},
                                                   namespace='generic')

        # config.seeds = [2023, 15451, 1337, 2001, 2080]
        config.seeds = [2023]

        return config

    @classmethod
    def get_bert_config(
            cls
    ):
        config = cls.get_default()

        config.callbacks = RegistrationKey(name='callback',
                                           tags={'early_stopping', 'th'},
                                           namespace='a2')

        config.model_processor = RegistrationKey(name='processor',
                                                 tags={'th', 'classifier'},
                                                 namespace='a2')

        config.metrics = RegistrationKey(name='metrics',
                                         namespace='a2')

        config.helper = RegistrationKey(name='helper',
                                        tags={'default'},
                                        namespace='th')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'bert', 'stance', 'th'},
                                               namespace='a2')

        config.get('model').variants = [
            RegistrationKey(name='model',
                            tags={'bert', 'conclusion'},
                            namespace='a2'),
            RegistrationKey(name='model',
                            tags={'bert', 'premise', 'conclusion'},
                            namespace='a2'),
            RegistrationKey(name='model',
                            tags={'bert', 'premise', 'conclusion', 'stance'},
                            namespace='a2')
        ]

        config.post_processor = RegistrationKey(name='processor',
                                                tags={'routine', 'step'},
                                                namespace='a2')

        return config

    @classmethod
    def get_dummy_config(
            cls
    ):
        config = cls.get_default()

        config.helper = RegistrationKey(name='helper',
                                        tags={'default'},
                                        namespace='generic')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'dummy'},
                                               namespace='a2')

        config.model = RegistrationKey(name='model',
                                       tags={'dummy', 'baseline'},
                                       namespace='a2')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'baseline'},
                                         namespace='a2')

        return config


@register
def register_routines():
    Registry.add_and_bind_variants(config_class=Task3RoutineConfig,
                                   config_constructor=Task3RoutineConfig.get_bert_config,
                                   component_class=TrainAndTestRoutine,
                                   name='routine',
                                   tags={'bert'},
                                   namespace='a2')

    Registry.add_and_bind_variants(config_class=Task3RoutineConfig,
                                   config_constructor=Task3RoutineConfig.get_dummy_config,
                                   component_class=TrainAndTestRoutine,
                                   name='routine',
                                   tags={'dummy'},
                                   namespace='a2')
