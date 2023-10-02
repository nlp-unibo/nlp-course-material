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

        config.callbacks = RegistrationKey(name='callback',
                                           tags={'early_stopping', 'th'},
                                           namespace='sp')

        config.metrics = RegistrationKey(name='metrics',
                                         tags={'emotions_f1', 'triggers_f1'},
                                         namespace='sp')

        config.helper = RegistrationKey(name='helper',
                                        tags={'default'},
                                        namespace='th')

        config.model_processor = RegistrationKey(name='processor',
                                                 tags={'th', 'classifier'},
                                                 namespace='sp')

        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'bert', 'emotion', 'th'},
                                               namespace='sp')

        config.post_processor = RegistrationKey(name='processor',
                                                tags={'routine', 'step'},
                                                namespace='sp')

        config.routine_processor = RegistrationKey(name='routine_processor',
                                                   tags={'average', 'fold'},
                                                   namespace='generic')

        config.model = RegistrationKey(name='model',
                                       tags={'bert', 'baseline'},
                                       namespace='sp')

        # config.seeds = [2023, 15451, 1337, 2001, 2080]
        config.seeds = [2023]

        return config


@register
def register_routines():
    Registry.add_and_bind_variants(config_class=Task3RoutineConfig,
                                   component_class=TrainAndTestRoutine,
                                   name='routine',
                                   tags={'task3'},
                                   namespace='sp')
