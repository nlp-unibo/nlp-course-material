from pathlib import Path
from typing import Tuple, Optional, Any, Iterable, List

import pandas as pd

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.file_manager import FileManager
from cinnamon_core.utility.json_utility import load_json
import numpy as np


class Task3Loader(DataLoader):

    def load_data(
            self
    ) -> Any:
        file_manager = FileManager.retrieve_component_instance(name='file_manager',
                                                               tags={'default'},
                                                               namespace='generic')
        data_path: Path = file_manager.dataset_directory.joinpath(self.json_filename)
        if not data_path.exists():
            data_path.mkdir(parents=True)

        data = load_json(filepath=data_path)

        return pd.DataFrame.from_records(data)

    def get_splits(
            self,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        data = self.load_data()

        # shuffle
        data = data.sample(frac=1)

        train_size = int(np.ceil(len(data) * self.train_percentage))
        val_size = int(np.ceil(len(data) * self.val_percentage))
        train_df = data[:train_size]
        val_df = data[train_size:train_size + val_size]
        test_df = data[train_size + val_size:]

        return train_df, val_df, test_df

    def parse(
            self,
            data: Optional[pd.DataFrame] = None,
    ) -> Optional[FieldDict]:
        if data is None:
            return data

        return_field = FieldDict()
        return_field.add(name='utterances',
                         value=data.utterances.values,
                         type_hint=Iterable[List[str]],
                         tags={'text'},
                         description='Dialogue utterances.')
        return_field.add(name='emotions',
                         value=data.emotions.values,
                         type_hint=Iterable[List[str]],
                         tags={'label'},
                         description='Per utterance emotion labels.')
        return_field.add(name='triggers',
                         value=[np.nan_to_num(seq) for seq in data.triggers.values],
                         type_hint=Iterable[List[float]],
                         tags={'label'},
                         description='Per utterance trigger labels.')
        return_field.add(name='dialogue_id',
                         value=data.episode.values,
                         type_hint=Iterable[str],
                         tags={'metadata'},
                         description='Unique dialogue identifier')
        return return_field
