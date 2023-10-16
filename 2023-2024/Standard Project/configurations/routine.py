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
                                             tags={'task3'},
                                             namespace='sp')

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
                                           namespace='sp')

        config.model_processor = RegistrationKey(name='processor',
                                                 tags={'th', 'classifier'},
                                                 namespace='sp')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'emotions_f1', 'triggers_f1'},
                                         namespace='sp')

        config.helper = RegistrationKey(name='helper',
                                        tags={'default'},
                                        namespace='th')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'bert', 'emotion', 'th'},
                                               namespace='sp')

        config.model = RegistrationKey(name='model',
                                       tags={'bert', 'baseline'},
                                       namespace='sp')

        config.post_processor = RegistrationKey(name='processor',
                                                tags={'routine', 'step'},
                                                namespace='sp')

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
                                               tags={'dummy', 'emotion'},
                                               namespace='sp')

        config.model = RegistrationKey(name='model',
                                       tags={'dummy', 'baseline'},
                                       namespace='sp')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'emotions_f1', 'triggers_f1', 'baseline'},
                                         namespace='sp')

        return config


@register
def register_routines():
    Registry.add_and_bind_variants(config_class=Task3RoutineConfig,
                                   config_constructor=Task3RoutineConfig.get_bert_config,
                                   component_class=TrainAndTestRoutine,
                                   name='routine',
                                   tags={'task3', 'bert'},
                                   namespace='sp')

    Registry.add_and_bind_variants(config_class=Task3RoutineConfig,
                                   config_constructor=Task3RoutineConfig.get_dummy_config,
                                   component_class=TrainAndTestRoutine,
                                   name='routine',
                                   tags={'task3', 'dummy'},
                                   namespace='sp')
