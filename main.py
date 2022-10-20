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

from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from baal.bayesian.dropout import patch_module
import argparse
import gc


def set_random(seed = 1004) :
    random_seed = 1004
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

def main(args):
    # Random Setting
    random_seed = 1004
    set_random(random_seed)
    model_initializer = BertInitializer(args)

    model, train_datasets, test_datasets = model_initializer.initialize()

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()
    print("-------------------------- Data set Load Success-----------------------------")

    # Active Learning Setting
    active_data_train= ActiveLearningDataset(train_datasets)
    active_data_test = ActiveLearningDataset(test_datasets)
    # train active data
    initial_num_train = int(0.01 * float(active_data_train.n_unlabelled.item())) # 1% 떼오기
    active_data_train.label_randomly(initial_num_train)
    NDATA_TO_LABEL_train = int(0.05 * float(active_data_train.n_unlabelled.item()))

    # test active data
    initial_num_test = int(float(active_data_test.n_unlabelled.item()))  # 100% 사용
    active_data_test.label_randomly(initial_num_test)
    
    model = MCDropoutModule(model).to(args.device)
    model = ModelWrapper(model, criterion, replicate_in_memory = True)
    # model.add_metric('F1', )

    assert args.heuristic_method in HEURISTIC_METHODS
    heuristic_method = HEURISTIC_METHODS[args.heuristic_method]
    heuristic_init_kwargs = {
        "shuffle_prop": 0.1,
        "reduction": "mean",
    }
    if args.heuristic_method == "random":
        heuristic_init_kwargs["seed"] = random_seed

    active_loop = ActiveLearningLoop(dataset=active_data_train,
                                     get_probabilities=model.predict_on_dataset,
                                     heuristic=heuristic_method(**heuristic_init_kwargs),
                                     query_size=NDATA_TO_LABEL_train,
                                     uncertainty_folder=args.expr_dir,
                                     batch_size=args.batch_size,
                                     iterations=args.MC_sampling,
                                     use_cuda=True
                                     )


    for al_step in range(args.total_step):
        file_name = f"uncertainty_pool={len(active_data_train.pool)}" f"_labelled={len(active_data_train)}"
        hist = model.train_on_dataset(dataset=active_data_train,
                                                  optimizer=optimizer,
                                                  batch_size=args.batch_size,
                                                  epoch=args.epochs,
                                                  use_cuda=True)

        io.model_save(model.state_dict(), args, file_name + '_model_params.pt')
        io.hist_save(hist, args, file_name+'_hist(train_loss).pkl') # loss

        pred_ = model.predict_on_dataset(dataset=active_data_test,
                                         batch_size=args.batch_size,
                                         iterations = args.MC_sampling,
                                         use_cuda=True,
                                         verbose=True)
        pred_ = np.squeeze(pred_.mean(axis=3)) # (Batch, Labels, Max_len, 1)
        pred_ = pred_.transpose((0, 2, 1)) # (Batch, Max_len, Labels)
        io.pred_save(pred_.argmax(axis = 2), args, file_name+'_prediction.npy') # (Batch, Max_len)
        pred_ = 0
        gc.collect()
        torch.cuda.empty_cache()
        if not active_loop.step():
            # loop 내부에서 uncertainty 높은 문장 저장되도록
            # We're done!
            break 


if __name__ == '__main__':
    parser = Baal_options()
    args = parser.parse()
    args.expr_dir = parser.save() # heuristic method 마다 저장될 폴더 생성


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    main(args)