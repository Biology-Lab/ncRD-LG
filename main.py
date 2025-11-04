import torch
import numpy as np
import sys, copy, math, time, pdb, warnings, traceback
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import pylab
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from train_eval import *
from models import *
import evaluation_fun
import pandas as pd
import traceback
import warnings
import sys
import matplotlib.pyplot as plt
import networkx as nx
import json



# used to traceback which code cause warnings, can delete
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def logger(info, model, optimizer):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    with open(os.path.join('./model_save/', 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse))
    if type(epoch) == int and epoch % args.save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join('./model_save/', 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(
            './model_save/', 'optimizer_checkpoint{}.pth'.format(epoch)
        )
        if model is not None:
            torch.save(model.state_dict(), model_name)



# Arguments
parser = argparse.ArgumentParser(description='ncRD-LG')
# general settings
parser.add_argument('--no-train', action='store_true', default=False,
                    help='if set, skip the training')
parser.add_argument('--debug', action='store_true', default=False,
                    help='turn on debugging mode which uses a small number of data')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                    help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                    valid only for ml_1m and ml_10m')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--save-interval', type=int, default=20,
                    help='save model states every # epochs ')
# subgraph extraction settings
parser.add_argument('--hop', default=3, metavar='S',
                    help='hop number of extracting local graph')
parser.add_argument('--sample-ratio', type=float, default=1.0,
                    help='if < 1, subsample nodes per hop according to the ratio')
parser.add_argument('--max-nodes-per-hop', default=100,
                    help='if > 0, upper bound the # nodes per hop by another subsampling')
parser.add_argument('--use-features', action='store_true', default=True,
                    help='whether to use node features (side information)')
# edge dropout settings
parser.add_argument('--adj-dropout', type=float, default=0,
                    help='if not 0, random drops edges from adjacency matrix with this prob')
parser.add_argument('--force-undirected', action='store_true', default=False,
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
# optimization settings
parser.add_argument('--continue-from', type=int, default=None,
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--lr-decay-step-size', type=int, default=50,
                    help='decay lr by factor A every B steps')
parser.add_argument('--lr-decay-factor', type=float, default=0.2,
                    help='decay lr by factor A every B steps')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=300, metavar='N',
                    help='batch size during training')
parser.add_argument('--test-freq',
                    type=int, default=40, metavar='N',
                    help='test every n epochs')
parser.add_argument('--ARR', type=float, default=0.00,
                    help='The adjacenct rating regularizer. If not 0, regularize the \
                    differences between graph convolution parameters W associated with\
                    adjacent ratings')

parser.add_argument('--data-name', default='ml_100k', help='dataset name')
parser.add_argument('--data-appendix', default='',
                    help='what to append to save-names when saving datasets')

parser.add_argument('--testing', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')

parser.add_argument('--save-appendix', default='',
                    help='what to append to save-names when saving results')

parser.add_argument('--max-train-num', type=int, default=None,
                    help='set maximum number of train data to use')
parser.add_argument('--max-val-num', type=int, default=None,
                    help='set maximum number of val data to use')

args = parser.parse_args()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

rating_map = None





args.file_dir = os.path.dirname(os.path.realpath('__file__'))

if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
args.res_dir = os.path.join(
    args.file_dir, 'results/{}{}_{}'.format(
        args.data_name, args.save_appendix, val_test_appendix
    )
)

def run_experiment(RNA_features, Drug_features, Pair, Pair_new, args, device):
    train_index = np.where((Pair_new == 1) | (Pair_new == -20))
    val_index = np.where((Pair_new == -1) | (Pair_new == -10))
    test_index = np.where((Pair_new == 2) | (Pair_new == -30))
    adj_train = np.copy(Pair)
    adj_train[test_index] = 0

    train_A_csr = ssp.csr_matrix(adj_train)
    train_label = np.array(adj_train[train_index])

    adj_val = np.zeros(Pair.shape)
    adj_val[val_index] = Pair[val_index]
    val_label = np.array(adj_val[val_index])

    adj_test = np.zeros(Pair.shape)
    adj_test[test_index] = Pair[test_index]
    test_A_csr = ssp.csr_matrix(adj_test)
    test_label = np.array(adj_test[test_index])
    test_label_fold=np.array(Pair[test_index])

    #side features combination
    RNA_features = RNA_features.cpu().numpy()
    Drug_features = Drug_features.cpu().numpy()

    class_values = [0,1]
    data_combo = (args.data_name, args.data_appendix, val_test_appendix)
    train_graphs = eval('MyDynamicDataset')(
        'data/{}{}/{}/train'.format(*data_combo),
        train_A_csr,
        train_index,
        train_label,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        RNA_features,
        Drug_features,
        class_values,
        max_num=args.max_train_num
    )

    val_graphs = eval('MyDynamicDataset')(
    'data/{}{}/{}/val'.format(*data_combo),
        train_A_csr,
        val_index,
        val_label,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        RNA_features,
        Drug_features,
        class_values,
        max_num=args.max_train_num
    )


    test_graphs = eval('MyDynamicDataset')(
    'data/{}{}/{}/test'.format(*data_combo),
        train_A_csr,
        test_index,
        test_label,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        RNA_features,
        Drug_features,
        class_values,
        max_num=args.max_train_num
    )


    num_relations = len(class_values)
    multiply_by = 1
    n_features = RNA_features.shape[1] + Drug_features.shape[1]
    model = LocalGL(
        train_graphs,
        latent_dim=[128, 64, 32, 1],
        num_relations=num_relations,
        num_bases=0,
        regression=True,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        side_features=args.use_features,
        n_side_features=n_features,
        multiply_by=multiply_by
    )

    # train
    if not args.no_train:
        train_multiple_epochs(
            train_graphs,
            test_graphs,
            # val_graphs,
            model,
            args.epochs,
            args.batch_size,
            args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            ARR=args.ARR,
            test_freq=args.test_freq,
        )

def import_data():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, 'Dataset'))
    RNA_features = torch.load(os.path.join(dataset_path, 'LncRNA.pt'))
    Drug_features = torch.load(os.path.join(dataset_path, 'LncDrug.pt'))
    Pair = np.load(os.path.join(dataset_path, 'LncPair.npy'))
    Pair_new = np.load(os.path.join(dataset_path, 'LncPair_new.npy'))
    # RNA_features = torch.load(os.path.join(dataset_path, 'MiRNA.pt'))
    # Drug_features = torch.load(os.path.join(dataset_path, 'MiDrug.pt'))
    # Pair = np.load(os.path.join(dataset_path, 'MiPair.npy'))
    # Pair_new = np.load(os.path.join(dataset_path, 'MiPair_new.npy'))
    return RNA_features, Drug_features, Pair, Pair_new



if __name__ == "__main__":
    RNA_features, Drug_features, Pair, Pair_new = import_data()
    run_experiment(RNA_features, Drug_features, Pair, Pair_new, args, device)