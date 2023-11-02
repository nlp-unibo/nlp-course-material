from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey, register, Registry
from cinnamon_generic.configurations.data_loader import DataLoaderConfig
from components.data_loader import HumanValueLoader


class HumanValueLoaderConfig(DataLoaderConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.has_val_split = True
        config.has_test_split = True
        config.name = 'HumanValue'

        config.add(name='file_manager_key',
                   value=RegistrationKey(
                       name='file_manager',
                       tags={'default'},
                       namespace='generic'
                   ),
                   type_hint=RegistrationKey,
                   description="registration info of built FileManager component."
                               " Used for filesystem interfacing")
        config.add(name='arguments_filename',
                   value='arguments-{}.tsv',
                   type_hint=str,
                   description='Name of the .tsv file storing arguments data',
                   is_required=True)
        config.add(name='labels_filename',
                   value='labels-{}.tsv',
                   type_hint=str,
                   description='Name of the .tsv file storing label_names data',
                   is_required=True)
        config.add(name='merge_labels',
                   value=True,
                   type_hint=bool,
                   description='Whether to use macro categories or not.')
        config.add(name='label_map',
                   value={
                       'Openess to change': [
                           'Self-direction: thought',
                           'Self-direction: action',
                           'Stimulation',
                           'Hedonism'
                       ],
                       'Self-enhancement': [
                           'Hedonism',
                           'Achievement',
                           'Power: dominance',
                           'Power: resources',
                           'Face'
                       ],
                       'Conversation': [
                           'Face',
                           'Security: personal',
                           'Security: societal',
                           'Tradition',
                           'Conformity: rules',
                           'Conformity: interpersonal',
                           'Humility'
                       ],
                       'Self-transcendence': [
                           'Humility',
                           'Benevolence: caring',
                           'Benevolence: dependability',
                           'Universalism: concern',
                           'Universalism: nature',
                           'Universalism: tolerance',
                           'Universalism: objectivity'
                       ]
                   })

        return config


@register
def register_data_loaders():
    Registry.add_and_bind(config_class=HumanValueLoaderConfig,
                          component_class=HumanValueLoader,
                          name='data_loader',
                          namespace='a2')
