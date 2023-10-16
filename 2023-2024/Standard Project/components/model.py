from typing import Optional, Any, Dict, Tuple

import torch as th
from sklearn.dummy import DummyClassifier
from transformers import BertConfig

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Model
from cinnamon_generic.components.processor import Processor
from cinnamon_th.components.model import THNetwork
from modeling.models import M_BERTBaseline
from cinnamon_core.utility import logging_utility
from cinnamon_generic.utility.printing_utility import prettify_statistics


class BERTBaseline(THNetwork):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.optimizer = None
        self.ce = None

        self.bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=self.preloaded_model_name)

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        emotion_classes = len(processor.find('encoder').classes_)
        self.model = M_BERTBaseline(emotion_classes=emotion_classes,
                                    bert_config=self.bert_config,
                                    preloaded_model_name=self.preloaded_model_name,
                                    lstm_weights=self.lstm_weights,
                                    freeze_bert=self.freeze_bert).to('cuda')

        self.optimizer = self.optimizer_class(**self.optimizer_args, params=self.model.parameters())
        self.ce = th.nn.CrossEntropyLoss(reduction='none').to('cuda')

    def batch_loss(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {},
    ) -> Tuple[Any, Any, Dict, Any, Dict]:
        (logits,
         model_additional_info) = self.model(batch_x,
                                             input_additional_info=input_additional_info)

        total_loss = 0
        true_loss = 0

        batch_size = logits['emotions'].shape[0]
        num_utterances = logits['emotions'].shape[1]

        # Emotions CE
        # logits ->     [bs, # utterances, # emotions]
        # labels ->     [bs, # utterances]
        emotion_logits = logits['emotions']
        emotion_logits = emotion_logits.reshape(-1, emotion_logits.shape[-1])

        emotion_y = batch_y['emotions']
        emotion_y_mask = th.where(emotion_y == -1, 0, 1)
        emotion_y = th.where(emotion_y == -1, 0, emotion_y)
        emotion_y = emotion_y.reshape(-1, )

        emotion_ce = self.ce(emotion_logits, emotion_y)
        emotion_ce = emotion_ce.reshape(batch_size, num_utterances)
        emotion_ce = th.sum(emotion_ce * emotion_y_mask, dim=1) / th.sum(emotion_y_mask, dim=1)[:, None]
        emotion_ce = th.mean(emotion_ce)

        total_loss += emotion_ce
        true_loss += emotion_ce
        loss_info = {
            'EM_CE': emotion_ce
        }

        # Triggers CE
        # logits ->     [bs, # utterances, 2]
        # labels ->     [bs, # utterances]
        triggers_logits = logits['triggers']
        triggers_logits = triggers_logits.reshape(-1, triggers_logits.shape[-1])

        triggers_y = batch_y['triggers']
        triggers_y_mask = th.where(triggers_y == -1, 0, 1)
        triggers_y = th.where(triggers_y == -1, 0, triggers_y)
        triggers_y = triggers_y.reshape(-1, )

        triggers_ce = self.ce(triggers_logits, triggers_y)
        triggers_ce = triggers_ce.reshape(batch_size, num_utterances)
        triggers_ce = th.sum(triggers_ce * triggers_y_mask, dim=1) / th.sum(triggers_y_mask, dim=1)[:, None]
        triggers_ce = th.mean(triggers_ce)

        total_loss += triggers_ce
        true_loss += triggers_ce
        loss_info['TR_CE'] = triggers_ce

        return total_loss, true_loss, loss_info, logits, model_additional_info


class DummyBaseline(Model):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.emotion_model = DummyClassifier(strategy=self.strategy)
        self.trigger_model = DummyClassifier(strategy=self.strategy)

    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        self.emotion_model.fit(X=train_data.X, y=train_data.y['emotions'])
        self.trigger_model.fit(X=train_data.X, y=train_data.y['triggers'])

        return_field = FieldDict()

        if val_data is not None:
            val_info = self.evaluate(data=val_data,
                                     callbacks=callbacks,
                                     metrics=metrics)
            val_info = val_info.to_value_dict()

            del val_info['predictions']
            if 'metrics' in val_info:
                val_info = {**val_info, **val_info['metrics']}
                del val_info['metrics']

            logging_utility.logger.info(f'\n{prettify_statistics(val_info)}')

            return_field.add(name='val_info',
                             value=val_info)
        return return_field

    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None,
            suffixes: Optional[Dict] = None
    ) -> FieldDict:
        emotion_predictions = self.emotion_model.predict(X=data.X)
        trigger_predictions = self.trigger_model.predict(X=data.X)
        predictions = {
            'emotions': emotion_predictions,
            'triggers': trigger_predictions,
            'dialogue_id': data.X
        }

        return_field = FieldDict()
        return_field.add(name='predictions', value=predictions)
        if suffixes is not None:
            return_field.add(name='suffixes',
                             value=suffixes)

        if 'y' in data and metrics is not None:
            metrics_result = metrics.run(y_pred=predictions,
                                         y_true=data.y,
                                         as_dict=True)
            return_field.add(name='metrics',
                             value=metrics_result)

        return return_field

    def evaluate(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None,
            suffixes: Optional[Dict] = None
    ) -> FieldDict:
        return self.predict(data=data,
                            callbacks=callbacks,
                            metrics=metrics,
                            model_processor=model_processor,
                            suffixes=suffixes)
