import torch as th
from transformers import BertModel


class M_BERTBaseline(th.nn.Module):

    def __init__(
            self,
            preloaded_model_name,
            bert_config,
            lstm_weights,
            emotion_classes,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=preloaded_model_name,
                                              config=bert_config)

        self.emotion_clf = th.nn.Linear(out_features=emotion_classes,
                                        in_features=bert_config.hidden_size)

        self.lstm_ctrl = th.nn.LSTM(input_size=bert_config.hidden_size,
                                    hidden_size=lstm_weights,
                                    num_layers=1,
                                    batch_first=True)

        self.triggers_clf = th.nn.Linear(out_features=2,
                                         in_features=lstm_weights)

    def forward(
            self,
            inputs,
            input_additional_info={}
    ):
        # [bs, # utterances, # token length]
        utterance_ids = inputs['utterance_ids']
        utterance_mask = inputs['utterance_mask']

        # [bs * # utterances, d]
        bert_encoding = self.bert(input_ids=utterance_ids.view(-1, utterance_ids.shape[-1]),
                                  attention_mask=utterance_mask.view(-1, utterance_mask.shape[-1]))

        # [bs, # utterances, d]
        bert_encoding = bert_encoding.view(utterance_ids.shape[0], utterance_ids.shape[1], bert_encoding.shape[-1])

        # Emotion

        # [bs, # utterances, # emotion classes]
        emotion_logits = self.emotion_clf(bert_encoding.view(-1, bert_encoding.shape[-1]))
        emotion_logits = emotion_logits.view(utterance_ids.shape[0], utterance_ids.shape[1], emotion_logits.shape[-1])

        # Triggers

        # [bs, # utterances, d']
        dialogue_utterance_encoding, _ = self.lstm_ctrl(bert_encoding)

        # [bs,
        triggers_logits = self.triggers_clf(dialogue_utterance_encoding.view(-1, dialogue_utterance_encoding.shape[-1]))
        triggers_logits = triggers_logits.view(utterance_ids.shape[0], utterance_ids.shape[1], triggers_logits.shape[-1])

        return {
            'emotions': emotion_logits,
            'triggers': triggers_logits
        }, None

