from __future__ import division, print_function

import torch
import argparse
import os
import util.custom_io as io


def opt_to_str(opt):
    return '\n'.join(['%s: %s' % (str(k), str(v)) for k, v in sorted(vars(opt).items())])


class Baal_options(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser('Active learning model training & experiments script', add_help=False)
        self.initialized = False
        self.opt = None

    def initialize(self):
        parser = self.parser
        # basic params (Path, Methods)
        parser.add_argument('--data_dir', default="./Data", type=str, help="The input data dir.")
        parser.add_argument("--output_dir", default="./checkpoints", type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--heuristic_method", default="bald", type=str,
                            help="Heuristic method selected in the list: [bald, random, entropy, variance, margin, certainty]")
        parser.add_argument("--labels", default="./Data/labels.txt", type=str, help="Path to a file containing Fine-grained-level labels.")
        # model params
        parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: [bert]")
        parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="Path to pre-trained model or shortcut name")
        parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--max_seq_length", default=128, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
        # tarin & eval params
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation on the dev set.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument('--batch_size', '-bs', default=128, type=int, help="Train batch size.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform.")
        parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
        parser.add_argument("--epochs", default=10, type=int, help="# training epochs.")
        parser.add_argument("--total_step", default=10, type=int, help="# total steps.")
        parser.add_argument("--MC_sampling", default=5, type=int, help="# of MC sampling.")
        # Optimizer
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
        parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
        parser.add_argument("--suffix", default="", help="Add suffix in load Dataset and Model save path.")

        self.initialized = True

    def parse(self, display=True):
        '''
        Parse option from terminal command string. If ord_str is given, parse option from it instead.
        '''

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        # display options
        if display:
            print('------------ Options -------------')
            for k, v in sorted(vars(self.opt).items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
        return self.opt

    def save(self, fn=None):
        if self.opt is None:
            raise Exception("parse options before saving!")
        if fn is None:
            expr_dir = os.path.join('checkpoints', self.opt.heuristic_method)
            io.mkdir_if_missing(expr_dir)
            fn = os.path.join(expr_dir, 'train_opt.json')
        io.save_json(vars(self.opt), fn)
        return expr_dir

    def load(self, fn):
        args = io.load_json(fn)
        return argparse.Namespace(**args)

if __name__ == "__main__":
    print('here')
    parser = Baal_options()
    opt = parser.parse()
    parser.save()
    parser.load('train_opt.json')