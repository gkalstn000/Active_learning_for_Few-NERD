from util.data_util import get_labels, make_dataloader
from torch.nn import CrossEntropyLoss
from wrapper.bert_classification_wrapper import BertClassificationWrapper
from util.data_util import get_labels, load_and_cache_examples

from options.active_option_select import MODEL_CLASSES, HEURISTIC_METHODS

class BertInitializer :
    def __init__(self, args):
        self.args = args

    def initialize(self):
        # Base model & data setting
        labels = get_labels(self.args.labels)
        num_labels = len(labels)
        pad_token_label_id = CrossEntropyLoss().ignore_index

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
        config = config_class.from_pretrained(self.args.model_name_or_path, num_labels=num_labels)
        tokenizer = tokenizer_class.from_pretrained(self.args.model_name_or_path, do_lower_case=True)

        model_class = BertClassificationWrapper
        model = model_class.from_pretrained(self.args.model_name_or_path, config=config)

        train_datasets = load_and_cache_examples(self.args, tokenizer, labels, pad_token_label_id, mode='train')
        test_datasets = load_and_cache_examples(self.args, tokenizer, labels, pad_token_label_id, mode='test')
        # train_loader = make_dataloader(args.batch_size, train_datasets, mode='trian')

        return model, train_datasets, test_datasets