from functools import partial
from typing import Dict, Union, List, Iterable, Any

import numpy as np
import torch as th
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper
from transformers import BertTokenizer

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.nlp.components.processor import TokenizerProcessor

from itertools import chain


class EmotionProcessor(Processor):

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
        flat_emotions = list(chain(*data.emotions))
        if is_training_data:
            self.encoder.fit(flat_emotions)

        data.emotions = [self.encoder.transform(seq) for seq in data.emotions]

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
        data.add(name='utterance_ids',
                 value=[])
        data.add(name='utterance_mask',
                 value=[])

        for utterances in data.utterances:
            tokenization_info = self.tokenize(text=utterances,
                                              remove_special_tokens=remove_special_tokens)
            data.utterance_ids.append(tokenization_info['input_ids'])
            data.utterance_mask.append(tokenization_info['attention_mask'])

        return data


class THDataProcessor(Processor):

    def batch_data(
            self,
            input_batch,
            device
    ):
        utterance_ids, utterance_mask, dialogue_id, emotions, triggers = [], [], [], [], []
        dialogue_utterance_indexes = []
        for input_x, input_y in input_batch:
            utterance_ids.extend([th.tensor(seq, dtype=th.int32) for seq in input_x[0]])
            utterance_mask.extend([th.tensor(seq, dtype=th.int32) for seq in input_x[1]])
            dialogue_id.append(int(input_x[2].split('utterance_')[1]))
            dialogue_utterance_indexes.append(len(input_x[0]))

            emotions.extend([th.tensor(input_y[0], dtype=th.long)])
            triggers.extend([th.tensor(input_y[1], dtype=th.long)])

        # Utterances, Emotions, Triggers
        # TODO: make sure the padding_value equals tokenizer.pad_token_id
        utterance_ids = pad_sequence(utterance_ids, batch_first=True, padding_value=0)
        utterance_mask = pad_sequence(utterance_mask, batch_first=True, padding_value=0)

        # make sure to replace the padding_value to a valid one in model.batch_loss()
        emotions = pad_sequence(emotions, batch_first=True, padding_value=-1)
        triggers = pad_sequence(triggers, batch_first=True, padding_value=-1.0)

        padded_utterance_ids = []
        padded_utterance_mask = []

        utterance_idx = 0
        maximum_utterances = max(dialogue_utterance_indexes)

        for dialogue_idx, dialogue_utterance_idx in enumerate(dialogue_utterance_indexes):
            to_pad = maximum_utterances - dialogue_utterance_idx
            if to_pad == 0:
                # Utterance
                padded_utterance_ids.append(utterance_ids[utterance_idx:utterance_idx + maximum_utterances])
                padded_utterance_mask.append(utterance_mask[utterance_idx:utterance_idx + maximum_utterances])
            else:
                # Utterances
                pad_utterances = th.zeros((to_pad, utterance_ids[0].shape[0]), dtype=th.int32)

                dialogue_utterances_ids = utterance_ids[utterance_idx:utterance_idx + dialogue_utterance_idx]
                padded_utterance_ids.append(th.concat((dialogue_utterances_ids, pad_utterances), dim=0))

                dialogue_utterances_mask = utterance_mask[utterance_idx:utterance_idx + dialogue_utterance_idx]
                padded_utterance_mask.append(th.concat((dialogue_utterances_mask, pad_utterances), dim=0))

            utterance_idx += dialogue_utterance_idx

        # [bs, # utterances, # token length]
        padded_utterance_ids = th.stack(padded_utterance_ids, dim=0)
        padded_utterance_mask = th.stack(padded_utterance_mask, dim=0)

        assert padded_utterance_ids.shape == padded_utterance_mask.shape
        assert padded_utterance_ids.shape[0] == emotions.shape[0] == triggers.shape[0]
        assert padded_utterance_ids.shape[1] == emotions.shape[1] == triggers.shape[1]

        # Dialogue id
        dialogue_id = th.tensor(dialogue_id, dtype=th.int32)

        # input
        x = {
            'utterance_ids': padded_utterance_ids.to(device),
            'utterance_mask': padded_utterance_mask.to(device),
            'dialogue_id': dialogue_id.to(device)
        }

        # output
        y = {
            'emotions': emotions.to(device),
            'triggers': triggers.to(device)
        }

        return x, y

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        x_th_utterance_ids = SequenceWrapper(data.utterance_ids)
        x_th_utterance_mask = SequenceWrapper(data.utterance_mask)
        x_th_id = SequenceWrapper(data.dialogue_id)
        x_th_data = x_th_utterance_ids.zip(x_th_utterance_mask, x_th_id)

        y_th_emotions = SequenceWrapper(data.emotions)
        y_th_triggers = SequenceWrapper(data.triggers)
        y_th_data = y_th_emotions.zip(y_th_triggers)

        th_data = x_th_data.zip(y_th_data)

        if is_training_data:
            th_data = th_data.shuffle()

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        th_data = DataLoader(th_data,
                             shuffle=is_training_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             collate_fn=partial(self.batch_data, device=device))

        steps = int(np.ceil(len(data.utterance_ids) / self.batch_size))

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
        return {
            'emotions': th.argmax(data['emotions'], dim=-1),
            'triggers': th.argmax(data['triggers'], dim=-1)
        }


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
