import os

import numpy as np
import random
import pandas as pd
from collections import defaultdict

import torch
import util.custom_io as io
from options.baal_options import Baal_options
from base_model.bert_initializer import BertInitializer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



def set_random(seed = 1004) :
    random_seed = 1004
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

def get_word_uncertainty_dict(heuristic_method) :
    root = 'checkpoints'
    PATH = os.path.join(root, heuristic_method)
    rule = ''
    uncertainty_filenames = [filename for filename in os.listdir(PATH) if
                             filename[-3:] == 'pkl' and filename[-5:] != ').pkl']
    uncertainty_filenames.sort(key=lambda file_name: int(file_name.split('=')[-1][:-4]))

    query = 6522
    uncertainty_dfs = {}

    for filename in uncertainty_filenames:
        dict_ = io.hist_load(args, filename)

        uncertainty = dict_['uncertainty']
        total_labels = train_datasets.tensors[1]
        index = dict_['dataset']['labelled'] == 0
        uncertainty_sentence = total_labels[index].numpy()

        df = pd.DataFrame(uncertainty_sentence, columns=[f'word{i}' for i in range(uncertainty_sentence.shape[1])])
        df['uncertainty'] = uncertainty
        df.sort_values(by='uncertainty', ascending=False, inplace=True)
        df.drop('uncertainty', axis=1, inplace=True)
        uncertainty_dfs[filename] = df.iloc[:query]

    uncertainty_word = defaultdict(int)
    for file_name, df in uncertainty_dfs.items():
        words, counts = np.unique(df, return_counts=True)
        for word, count in zip(words, counts):
            if word == -100: continue
            uncertainty_word[word] += count
    return uncertainty_word

def get_label_map() :
    label_map = {}
    PATH = 'Data/labels.txt'
    with open(PATH, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines) :
        label_map[i] = line[:-1]
    return label_map

def count_original_label() :
    label_count = defaultdict(int)
    PATH = 'Data/train.txt'
    with open(PATH, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line == '\n' : continue
        _, label = line.split('\t')
        if label[:-1] == 'O' : continue
        label_count[label[:-1]] +=1
    return label_count

def get_class_score() :
    PATH = 'Data/class_report.csv'
    return pd.read_csv(PATH)

if __name__ == '__main__':
    parser = Baal_options()
    args = parser.parse()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random_seed = 1004
    set_random(random_seed)
    model_initializer = BertInitializer(args)

    model, train_datasets, test_datasets = model_initializer.initialize()
    heuristic_methods = ['bald', 'random', 'entropy', 'margin', 'variance', 'certainty']
    # heuristic_methods = ['certainty']

    label_map = get_label_map()
    count_label = count_original_label()
    score_df = get_class_score()

    for heuristic_method in heuristic_methods :
        count_storage = {label : 0 for label in label_map.values() if label != 'O'}

        args.heuristic_method = heuristic_method
        args.expr_dir = f'checkpoints/{heuristic_method}'

        uncertainty_count = get_word_uncertainty_dict(heuristic_method)

        labels = [label_map[key] for key in uncertainty_count.keys()]
        counts = [c for c in uncertainty_count.values()]
        if 'O' in  labels:
            O_index = labels.index('O')
            labels.pop(O_index)
            counts.pop(O_index)

        for label, count in zip(labels, counts) :
            count_storage[label] = count / count_label[label] if count_label[label]  != 0 else 0

        # 상위 20개 클래스
        count_storage = dict(sorted(count_storage.items(), key=lambda item: item[1], reverse=True))

        method = [heuristic_method] * len(labels)

        tmp = pd.DataFrame([method, list(count_storage.keys()), list(count_storage.values())]).T

        tmp.columns = ['method', 'label', 'count']
        tmp = pd.merge(tmp, score_df, how='inner')

        max_count = tmp['count'].max()
        min_count = tmp['count'].min()
        tmp['scaling_count'] = (tmp['count'] - min_count) / (max_count - min_count)

        fig, ax = plt.subplots(figsize=(35, 15))
        g = sns.lineplot(data=tmp[['f1-score']], ax=ax,marker='o', markersize=10)
        ax2 = ax.twinx()
        g = sns.barplot(x='label', y='scaling_count', data=tmp, ax=ax2, alpha=.5)
        ax.tick_params(axis='x', rotation=90)
        plt.title(heuristic_method, fontdict={'fontsize' : 30})
        plt.show()
        # plt.savefig(f'{heuristic_method}_Count.png')
