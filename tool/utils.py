import torch
import os
from torch.optim import AdamW
import random
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
import pandas as pd
# from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from collections import defaultdict
import re
import string

def get_parsere(parser):
    parser.add_argument('-train_path', type = str, default = '/home/pc3/zy/data/src/TOPK_SubIPC/train_t.tsv')
    parser.add_argument('-valid_path', type = str, default = '/home/pc3/zy/data/src/TOPK_SubIPC/dev_t.tsv')
    parser.add_argument('-test_path', type = str, default = '/home/pc3/zy/data/src/TOPK_SubIPC/test_t.tsv')
    parser.add_argument('-ipc_path', type=str, default='/home/pc3/zy/data/src/TOPK_SubIPC/ipc.tsv')
    parser.add_argument('-bert_path', type = str, default = '/home/pc3/zy/premodel/bert_base_uncase')
    parser.add_argument('-bge_path', type=str, default='/home/pc3/zy/premodel/bge-base-en-v1.5')
    parser.add_argument('-patent_path', type=str, default='/home/pc3/zy/data/src/TOPK_SubIPC/patent.tsv')
    parser.add_argument('-ref_path', type=str, default='/home/pc3/zy/data/src/TOPK_SubIPC/references.jsonl')
    parser.add_argument('-ref_text_path', type=str, default='/home/pc3/zy/data/src/TOPK_SubIPC/ref_text.pkl')
    parser.add_argument('-ref_payload_path', type=str, default='/home/pc3/zy/data/src/TOPK_SubIPC/ref_emb_top10.pt')
    parser.add_argument('-gnn_layers', type=int, default=2)
    # parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-batch_size', type = int, default = 32, help = 'input batch size for train and vaild')
    parser.add_argument('-learning_rate', type = float, default = 2e-5)
    parser.add_argument('-seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-epoch', type = int, default = 10, help = 'training rounds')
    parser.add_argument('-dropout', type = float, default = 0.1, help = 'the value of dropout')
    parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    args.device = torch.device(args.device)

    return args

def setting(args, train_data, valid_data):
    print('训练集的大小是{}'.format(len(train_data)))
    print('验证集的大小是{}'.format(len(valid_data)))

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def print_time(s):
    now = datetime.now()
    print(s, now)
    return now

def batch_data(args, data):
    data['label'] = data['label'].to(args.device)
    data['xiaolei'] = data['t'].to(args.device)
    df = pd.read_csv(args.ipc_path, sep='\t', header=None)
    ipc_dict = dict(zip(df[0], df[2]))
    data['ipc_dict'] = ipc_dict
    return data

class load_data_withopen(Dataset):
    def __init__(self, file_path, args):
        self.data = []
        self.args = args

        with open(file_path, 'r') as file:
            headers = file.readline().strip().split('\t')
            for line in file:
                parts = line.strip().split('\t')
                feature_dict = {}
                for i, header in enumerate(headers):
                    feature_dict[header] = parts[i]
                self.data.append(feature_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_dict = self.data[idx]
        title_a = feature_dict.get('#1 title')
        title_b = feature_dict.get('#2 title')
        abstract_a = feature_dict.get('#1 abstract')
        abstract_b = feature_dict.get('#2 abstract')
        text_a = title_a + ' ' + abstract_a
        text_b = title_b + ' ' + abstract_b
        patentA = feature_dict.get('#1 ID')
        patentB = feature_dict.get('#2 ID')
        t = torch.tensor(float(feature_dict.get('ipc_3')))
        label = torch.tensor(float(feature_dict.get('Quality')))
        index = torch.tensor(int(feature_dict.get('Index')))
        # ipc_a = feature_dict.get('#1 IPC')[:10]
        # ipc_b = feature_dict.get('#2 IPC')[:10]
        ipc_a = set(feature_dict.get('#1 IPC').split())
        ipc_b = set(feature_dict.get('#2 IPC').split())
        ipc_list = ';'.join([ipc.lower() for ipc in (ipc_a & ipc_b) if len(ipc) == 4])

        return {
            'title_a': title_a,
            'title_b': title_b,
            'abstract_a': abstract_a,
            'abstract_b': abstract_b,
            'text_a': text_a,
            'text_b': text_b,
            'patentA': patentA,
            'patentB': patentB,
            'label': label,
            'index': index,
            'ipc_list': ipc_list,
            't' : t,
        }

def set_lr(args,net):
    bert_lr = getattr(args, "bert_lr", 2e-5)
    gnn_lr = getattr(args, "gnn_lr", 2e-4)
    oth_lr = getattr(args, "oth_lr", 2e-5)

    bert_params, gnn_params, other_params = [], [], []

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("bert_shared"):
            bert_params.append(p)
        elif name.startswith("gnn"):
            gnn_params.append(p)
        else:
            other_params.append(p)

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": bert_lr, "weight_decay": 5e-4},
            {"params": gnn_params, "lr": gnn_lr, "weight_decay": 5e-4},
            {"params": other_params, "lr": oth_lr, "weight_decay": 5e-4},
        ]
    )
    return optimizer

