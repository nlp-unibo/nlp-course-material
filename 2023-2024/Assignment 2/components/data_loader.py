from pathlib import Path
from typing import Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.file_manager import FileManager
from functools import reduce


class HumanValueLoader(DataLoader):

    def build_split(
            self,
            arguments_path: Path,
            labels_path: Path,
            split: str
    ):
        split_arguments = arguments_path.with_name(arguments_path.name.format(split))
        split_labels = labels_path.with_name(labels_path.name.format(split))

        arguments_df = pd.read_csv(split_arguments, sep='\t')
        labels_df = pd.read_csv(split_labels, sep='\t')

        if self.merge_labels:
            for macro_label, mapped_labels in self.label_map.items():
                labels_df[macro_label] = reduce(lambda a, b: np.logical_or(a, b), [labels_df[m_label].values for m_label in mapped_labels]).astype(int)

            labels_df = labels_df[['Argument ID'] + list(self.label_map.keys())]

        return arguments_df.merge(labels_df, on='Argument ID')

    def load_data(
            self
    ) -> Any:
        file_manager = FileManager.retrieve_component_instance(name='file_manager',
                                                               tags={'default'},
                                                               namespace='generic')
        arguments_path: Path = file_manager.dataset_directory.joinpath(self.arguments_filename)
        labels_path: Path = file_manager.dataset_directory.joinpath(self.labels_filename)

        train_df = self.build_split(arguments_path=arguments_path,
                                    labels_path=labels_path,
                                    split='training')
        val_df = self.build_split(arguments_path=arguments_path,
                                  labels_path=labels_path,
                                  split='validation')
        test_df = self.build_split(arguments_path=arguments_path,
                                   labels_path=labels_path,
                                   split='test')

        return train_df, val_df, test_df

    def get_splits(
            self,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        train_df, val_df, test_df = self.load_data()
        return train_df, val_df, test_df

    def parse(
            self,
            data: Optional[pd.DataFrame] = None,
    ) -> Optional[FieldDict]:
        if data is None:
            return data

        return_field = FieldDict()
        return_field.add(name='argument_id',
                         value=data['Argument ID'].values.astype(str),
                         type_hint=Iterable[str],
                         tags={'metadata'},
                         description='Argument unique identifier.')
        return_field.add(name='premise',
                         value=data.Premise.values.astype(str),
                         type_hint=Iterable[str],
                         tags={'text'},
                         description='Argument premise text.')
        return_field.add(name='conclusion',
                         value=data.Conclusion.values.astype(str),
                         type_hint=Iterable[str],
                         tags={'text'},
                         description='Argument conclusion text.')
        return_field.add(name='stance',
                         value=data.Stance.values.astype(str),
                         type_hint=Iterable[str],
                         tags={'metadata'},
                         description='Argument premise to conclusion stance.')
        for label in data.columns:
            if label not in ['Argument ID', 'Conclusion', 'Premise', 'Stance']:
                return_field.add(name=label,
                                 value=data[label].values.astype(int),
                                 type_hint=Iterable[int],
                                 tags={'label'},
                                 description=f'{label} human value associated with argument.')
        return return_field
