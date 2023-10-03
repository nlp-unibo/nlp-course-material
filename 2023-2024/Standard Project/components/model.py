from cinnamon_th.components.model import THNetwork
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.processor import Processor
from modeling.models import M_BERTBaseline

from transformers import BertConfig

from typing import Optional, Any, Dict, Tuple

import torch as th


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
                                    lstm_weights=self.lstm_weights).to('cuda')

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
        emotion_y = emotion_y.reshape(-1,)

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
        triggers_y = triggers_y.reshape(-1,)

        triggers_ce = self.ce(triggers_logits, triggers_y)
        triggers_ce = triggers_ce.reshape(batch_size, num_utterances)
        triggers_ce = th.sum(triggers_ce * triggers_y_mask, dim=1) / th.sum(triggers_y_mask, dim=1)[:, None]
        triggers_ce = th.mean(triggers_ce)

        total_loss += triggers_ce
        true_loss += triggers_ce
        loss_info['TR_CE'] = triggers_ce

        return total_loss, true_loss, loss_info, logits, model_additional_info
