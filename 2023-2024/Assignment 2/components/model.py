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
        label_names = processor.find('label_names')
        self.model = M_BERTBaseline(label_names=label_names,
                                    bert_config=self.bert_config,
                                    preloaded_model_name=self.preloaded_model_name,
                                    freeze_bert=self.freeze_bert).to('cuda')

        self.optimizer = self.optimizer_class(**self.optimizer_args, params=self.model.parameters())
        self.ce = th.nn.CrossEntropyLoss().to('cuda')

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

        avg_ce = 0
        for label_name, label_logits in logits.items():
            label_truth = batch_y[label_name]

            label_ce = self.ce(label_logits, label_truth)
            avg_ce += label_ce

        avg_ce = avg_ce / len(logits)
        total_loss += avg_ce
        true_loss += avg_ce
        loss_info = {
            'CE': avg_ce
        }

        return total_loss, true_loss, loss_info, logits, model_additional_info


class DummyBaseline(Model):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.label_names = None
        self.dummy_models = None

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.label_names = processor.find('label_names')
        self.dummy_models = {label_name: DummyClassifier(strategy=self.strategy) for label_name in self.label_names}

    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        for label_name, label_value in train_data.y.items():
            self.dummy_models[label_name].fit(X=train_data.X, y=label_value)

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
        predictions = {label_name: self.dummy_models[label_name].predict(X=data.X) for label_name in self.label_names}

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
