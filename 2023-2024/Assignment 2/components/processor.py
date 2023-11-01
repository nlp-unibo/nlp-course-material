from functools import partial
from typing import Dict, Union, List, Iterable, Any

import numpy as np
import torch as th
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper, Zipper
from transformers import BertTokenizer

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.nlp.components.processor import TokenizerProcessor


class StanceProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = LabelEncoder()

    def prepare_save_data(
            self
    ) -> Dict:
        data = super().prepare_save_data()

        data['encoder'] = self.encoder
        return data

    def clear(
            self
    ):
        super().clear()
        self.encoder = LabelEncoder()

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ):
        if is_training_data:
            self.encoder.fit(data.stance)

        data.stance = self.encoder.transform(data.stance)

        return data


class BERTTokenizer(TokenizerProcessor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.tokenizer = BertTokenizer.from_pretrained(self.preloaded_model_name)
        self.vocabulary = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocabulary)

    def tokenize(
            self,
            text: Iterable[str],
            remove_special_tokens: bool = False
    ) -> Union[List[int], np.ndarray]:
        return self.tokenizer(text, **self.tokenization_args)

    def detokenize(
            self,
            ids: Iterable[Union[List[int], np.ndarray]],
            remove_special_tokens: bool = False
    ) -> str:
        return self.tokenizer.decode(token_ids=ids, **self.detokenization_args)

    def process(
            self,
            data: FieldDict,
            tokenize: bool = True,
            remove_special_tokens: bool = False):
        data.add(name='pad_token_id',
                 value=self.tokenizer.pad_token_id)
        data.add(name='premise_ids',
                 value=[])
        data.add(name='premise_mask',
                 value=[])
        data.add(name='conclusion_ids',
                 value=[])
        data.add(name='conclusion_mask',
                 value=[])

        for premise, conclusion in zip(data.premise, data.conclusion):
            premise_info = self.tokenize(text=premise,
                                         remove_special_tokens=remove_special_tokens)
            data.premise_ids.append(premise_info['input_ids'])
            data.premise_mask.append(premise_info['attention_mask'])

            conclusion_info = self.tokenize(text=conclusion,
                                            remove_special_tokens=remove_special_tokens)
            data.conclusion_ids.append(conclusion_info['input_ids'])
            data.conclusion_mask.append(conclusion_info['attention_mask'])

        return data


class DummyDataProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.label_names = None

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        return_dict = FieldDict()

        return_dict.add(name='X',
                        value=[argument_id for argument_id in data.argument_id])

        labels = data.search_by_tag('label')
        return_dict.add(name='y',
                        value={key: value for key, value in labels.items()})
        self.label_names = list(labels.keys())

        return return_dict


class THDataProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.label_names = None

    def batch_data(
            self,
            input_batch,
            device,
            pad_token_id,
            label_names
    ):
        premise_ids, premise_mask = [], []
        conclusion_ids, conclusion_mask = [], []
        stance = []
        y = {}
        for input_x, input_y in input_batch:
            premise_ids.append(th.tensor(input_x[0], dtype=th.int32))
            premise_mask.append(th.tensor(input_x[1], dtype=th.int32))
            conclusion_ids.append(th.tensor(input_x[2], dtype=th.int32))
            conclusion_mask.append(th.tensor(input_x[3], dtype=th.int32))
            stance.append(input_x[4])

            for label_name, label_value in zip(label_names, input_y):
                y.setdefault(label_name, []).append(label_value)

        # Texts
        premise_ids = pad_sequence(premise_ids, batch_first=True, padding_value=pad_token_id)
        premise_mask = pad_sequence(premise_mask, batch_first=True, padding_value=0)
        conclusion_ids = pad_sequence(conclusion_ids, batch_first=True, padding_value=pad_token_id)
        conclusion_mask = pad_sequence(conclusion_mask, batch_first=True, padding_value=0)

        # Stance
        stance = th.tensor(stance, dtype=th.int32)

        # input
        x = {
            'premise_ids': premise_ids.to(device),
            'premise_mask': premise_mask.to(device),
            'conclusion_ids': conclusion_ids.to(device),
            'conclusion_mask': conclusion_mask.to(device),
            'stance': stance.to(device)
        }

        # output
        y = {key: th.tensor(value).to(device) for key, value in y.items()}

        return x, y

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        x_th_premise_ids = SequenceWrapper(data.premise_ids)
        x_th_premise_mask = SequenceWrapper(data.premise_mask)
        x_th_conclusion_ids = SequenceWrapper(data.conclusion_ids)
        x_th_conclusion_mask = SequenceWrapper(data.conclusion_mask)
        x_th_stance = SequenceWrapper(data.stance)
        x_th_data = x_th_premise_ids.zip(x_th_premise_mask,
                                         x_th_conclusion_ids,
                                         x_th_conclusion_mask,
                                         x_th_stance)

        y_data = {key: SequenceWrapper(value) for key, value in data.search_by_tag('label').items()}
        y_th_data = Zipper(*list(y_data.values()))
        self.label_names = list(y_data.keys())

        th_data = x_th_data.zip(y_th_data)

        if is_training_data:
            th_data = th_data.shuffle()

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        th_data = DataLoader(th_data,
                             shuffle=is_training_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             collate_fn=partial(self.batch_data,
                                                device=device,
                                                pad_token_id=data.pad_token_id,
                                                label_names=self.label_names))

        steps = int(np.ceil(len(data.premise_ids) / self.batch_size))

        return FieldDict({'iterator': lambda: iter(th_data),
                          'input_iterator': lambda: iter(th_data.map(lambda x, y: x)),
                          'output_iterator': lambda: iter(th_data.map(lambda x, y: y.detach().cpu().numpy())),
                          'steps': steps})


class THClassifierProcessor(Processor):

    def process(
            self,
            data: Any,
            is_training_data: bool = False
    ) -> Any:
        return {key: th.argmax(value, dim=-1) for key, value in data.items()}


class RoutineStepsProcessor(Processor):

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:

        def convert_numpy_item(item):
            item = np.array(item) if type(item) != np.ndarray else item
            item = item.tolist()

            if type(item) == float:
                return item

            if len(item) == 1:
                item = item[0]
            return item

        def parse_metrics(metrics):
            return {key: float(value) for key, value in metrics.items()}

        if 'fit_info' in data:
            data.fit_info = FieldDict({key: convert_numpy_item(value)
                                       for key, value in data.fit_info.to_value_dict().items()})
        if 'train_info' in data:
            data.train_info = FieldDict({key: convert_numpy_item(value) if key != 'metrics' else parse_metrics(value)
                                         for key, value in data.train_info.to_value_dict().items()})
        if 'val_info' in data:
            data.val_info = FieldDict({key: convert_numpy_item(value) if key != 'metrics' else parse_metrics(value)
                                       for key, value in data.val_info.to_value_dict().items()})
        if 'test_info' in data:
            data.test_info = FieldDict({key: convert_numpy_item(value) if key != 'metrics' else parse_metrics(value)
                                        for key, value in data.test_info.to_value_dict().items()})

        return data
