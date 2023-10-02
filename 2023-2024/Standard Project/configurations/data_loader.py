from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey, register, Registry
from cinnamon_generic.configurations.data_loader import DataLoaderConfig
from components.data_loader import Task3Loader


class Task3LoaderConfig(DataLoaderConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.has_val_split = True
        config.has_test_split = True
        config.name = 'Task3'

        config.add(name='file_manager_key',
                   value=RegistrationKey(
                       name='file_manager',
                       tags={'default'},
                       namespace='generic'
                   ),
                   type_hint=RegistrationKey,
                   description="registration info of built FileManager component."
                               " Used for filesystem interfacing")
        config.add(name='json_filename',
                   value='MELD_train_efr.json',
                   type_hint=str,
                   description='Name of the .json file storing data',
                   is_required=True)
        config.add(name='train_percentage',
                   value=0.80,
                   type_hint=float,
                   description='Percentage of the original data to reserve for training.'
                               'The remaining is used for validation and test.')
        config.add(name='val_percentage',
                   value=0.10,
                   type_hint=float,
                   description='Percentage of the original data to reserve for validation.')

        return config


@register
def register_data_loaders():
    Registry.add_and_bind(config_class=Task3LoaderConfig,
                          component_class=Task3Loader,
                          name='data_loader',
                          tags={'task3'},
                          namespace='sp')
