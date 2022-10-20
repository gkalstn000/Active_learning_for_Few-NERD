from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

class BertClassificationWrapper(BertForTokenClassification):
    def __init__(self, *args, **kwargs):
        super(BertClassificationWrapper, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output: TokenClassifierOutput = super().forward(*args)

        logits = output.logits

        logits = logits.transpose(1, 2)

        return logits