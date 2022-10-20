import json
import pickle
import os
import shutil
import torch
import numpy as np
#io functions of SCRC
def load_str_list(filename, end = '\n'):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    str_list = [s[:-len(end)] for s in str_list]
    return str_list

def save_str_list(str_list, filename, end = '\n'):
    str_list = [s+end for s in str_list]
    with open(filename, 'w') as f:
        f.writelines(str_list)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        # json.dump(json_obj, f, separators=(',\n', ':\n'))
        json.dump(json_obj, f, indent = 0, separators = (',', ': '))

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    return data

def load_data_json(filename):
    with open(filename, 'rb') as f:
        data = json.load(f)
    return data

def copy(fn_src, fn_tar):
    shutil.copyfile(fn_src, fn_tar)


def model_save(model_weight, args, file_name) :
    PATH = os.path.join(args.expr_dir, file_name)
    torch.save(model_weight, PATH)
    print(f'model save done at [{PATH}]')

def model_load(model, args, file_name) :
    PATH = os.path.join(args.expr_dir, file_name)
    model.load_state_dict(torch.load(PATH))
    print(f'model load done at [{PATH}]')
    return model

def hist_save(hist, args, file_name) :
    hist_dict = {'train_loss' : hist}
    PATH = os.path.join(args.expr_dir, file_name)
    with open(PATH, 'wb') as f:
        pickle.dump(hist_dict, f)
    print(f'hist save done at [{PATH}]')

def hist_load(args, file_name) :
    PATH = os.path.join(args.expr_dir, file_name)
    with open(PATH, 'rb') as f:
        hist = pickle.load(f)
    print(f'hist load done at [{PATH}]')
    return hist

def pred_save(pred, args, file_name) :
    # pred : np.array, (Batch, Label, Tokens, MC_sampling)
    PATH = os.path.join(args.expr_dir, file_name)
    np.save(PATH, pred)
    print(f'pred save done at [{PATH}]')

def pred_load(args, file_name) :
    # pred : np.array, (Batch, Label, Tokens, MC_sampling)
    PATH = os.path.join(args.expr_dir, file_name)
    pred_ = np.load(PATH)
    print(f'pred load done at [{PATH}]')
    return pred_