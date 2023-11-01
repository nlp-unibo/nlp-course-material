import torch as th
from transformers import BertModel


class M_BERTBaseline(th.nn.Module):

    def __init__(
            self,
            preloaded_model_name,
            bert_config,
            label_names,
            freeze_bert=True
    ):
        super().__init__()

        self.label_names = label_names
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=preloaded_model_name,
                                              config=bert_config)

        if freeze_bert:
            for module in self.bert.modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.clfs = th.nn.ModuleDict({
            label_name: th.nn.Linear(out_features=2,
                                     in_features=bert_config.hidden_size * 2 + 1)
            for label_name in label_names
        })

    def forward(
            self,
            inputs,
            input_additional_info={}
    ):
        # Premise
        # [bs, # premise token length]
        premise_ids = inputs['premise_ids']
        premise_mask = inputs['premise_mask']

        # [bs, d]
        premise_encoding = self.bert(input_ids=premise_ids,
                                     attention_mask=premise_mask).pooler_output

        # Conclusion
        # [bs, # conclusion token length]
        conclusion_ids = inputs['conclusion_ids']
        conclusion_mask = inputs['conclusion_mask']

        # [bs, d]
        conclusion_encoding = self.bert(input_ids=conclusion_ids,
                                        attention_mask=conclusion_mask).pooler_output

        # Argument
        # [bs, 2*d + #stances]
        argument_encoding = th.concat((premise_encoding,
                                       conclusion_encoding,
                                       inputs['stance'][:, None]),
                                      dim=-1)

        logits = {}
        for label_name in self.label_names:
            label_clf = self.clfs[label_name]
            logits[label_name] = label_clf(argument_encoding)

        return logits, None
