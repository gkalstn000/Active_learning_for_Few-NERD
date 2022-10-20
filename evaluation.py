import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from util.data_util import get_labels, make_dataloader
from options.baal_options import Baal_options
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.functional import f1_score

from copy import deepcopy
import numpy as np
import random

from baal.active import heuristics
from baal.active import get_heuristic
from baal.active.dataset import ActiveLearningDataset
from baal.bayesian.dropout import MCDropoutModule
from baal.active.active_loop import ActiveLearningLoop
from baal.modelwrapper import ModelWrapper

from transformers import BertPreTrainedModel, TrainingArguments
from baal.transformers_trainer_wrapper import BaalTransformersTrainer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from util.data_util import get_labels, load_and_cache_examples
from wrapper.bert_classification_wrapper import BertClassificationWrapper
from base_model.bert_initializer import BertInitializer
from options.active_option_select import MODEL_CLASSES, HEURISTIC_METHODS
import util.custom_io as io
from util.metric import Metrics
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from baal.bayesian.dropout import patch_module
import argparse

def set_random(seed = 1004) :
    random_seed = 1004
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
    parser = Baal_options()
    args = parser.parse()
    args.expr_dir = parser.save() # heuristic method 마다 저장될 폴더 생성


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    random_seed = 1004
    set_random(random_seed)
    model_initializer = BertInitializer(args)

    model, train_datasets, test_datasets = model_initializer.initialize()
    labels = test_datasets.tensors[1]
    file_name = 'uncertainty_pool=130450_labelled=1317_prediction.npy'
    preds = io.pred_load(args, file_name)

    metrics = Metrics()
    precision, recall, f1 = metrics.metrics_by_entity(preds, labels.numpy())
    print(f'precision: {precision}, recall: {recall}, f1: {f1}')