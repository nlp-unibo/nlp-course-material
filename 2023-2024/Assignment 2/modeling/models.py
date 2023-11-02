import torch as th
from transformers import BertModel


class M_BERTBaseline(th.nn.Module):

    def __init__(
            self,
            preloaded_model_name,
            bert_config,
            label_names,
            freeze_bert=True,
            add_premise=True,
            add_stance=True
    ):
        super().__init__()

        self.label_names = label_names
        self.add_premise = add_premise
        self.add_stance= add_stance

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=preloaded_model_name,
                                              config=bert_config)

        if freeze_bert:
            for module in self.bert.modules():
                for param in module.parameters():
                    param.requires_grad = False

        clf_in_features = bert_config.hidden_size
        if self.add_premise:
            clf_in_features += bert_config.hidden_size
        if self.add_stance:
            clf_in_features += 1

        self.clfs = th.nn.ModuleDict({
            label_name: th.nn.Linear(out_features=2,
                                     in_features=clf_in_features)
            for label_name in label_names
        })

    def forward(
            self,
            inputs,
            input_additional_info={}
    ):
        # Premise
        premise_encoding = None
        if self.add_premise:
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
        # [bs, 2*d + #stances] if self.add_premise and self.add_stance
        # [bs, 2*d]            if self.add_premise
        # [bs, d]              otherwise
        argument_encoding = conclusion_encoding
        if self.add_premise:
            argument_encoding = th.concat((premise_encoding, argument_encoding), dim=-1)
        if self.add_stance:
            argument_encoding = th.concat((argument_encoding, inputs['stance'][:, None]), dim=-1)

        logits = {}
        for label_name in self.label_names:
            label_clf = self.clfs[label_name]
            logits[label_name] = label_clf(argument_encoding)

        return logits, None
