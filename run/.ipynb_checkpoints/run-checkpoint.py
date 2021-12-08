"""
Current working directory: Project root dir

=== usage
python run/run.py -m DM --data cn15k --lr 0.01 --batch_size 300
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import tensorflow as tf
# tf.enable_eager_execution()
import sys

if './src' not in sys.path:
    sys.path.append('./src')

if './' not in sys.path:
    sys.path.append('./')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os.path import join
from src.data import Data

from src.trainer import Trainer
from src.list import ModelList
import datetime

import argparse
from src import param


def get_model_identifier(whichmodel, ver):
    prefix = whichmodel.value
    if ver!=0:
        prefix= prefix +'_v'+ str(ver) + '_' + str(param.n_hidden)
    now = datetime.datetime.now()
    date = '%02d%02d' % (now.month, now.day)  # two digits month/day
    identifier = prefix + '_' + date
    return identifier
import os

os.environ['CUDA_VISIBLE_DEVICES']="0"


parser = argparse.ArgumentParser()
# required
parser.add_argument('--data', type=str, default='ppi5k',
                    help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
# optional
parser.add_argument("--verbose", help="print detailed info for debugging",
                    action="store_true")
parser.add_argument('-m', '--model', type=str, default='rect', help="choose model ('logi' or 'rect'). default: rect")
parser.add_argument('-ver', '--version', type=int, default=0, help="choose model ver 1 to 3")
parser.add_argument('-d', '--dim', type=int, default=128, help="set dimension. default: 128")
parser.add_argument('--epoch', type=int, default=100, help="set number of epochs. default: 100")
parser.add_argument('--lr', type=float, default=0.001, help="set learning rate. default: 0.001")
parser.add_argument('--batch_size', type=int, default=1024, help="set batch size. default: 1024")
parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
parser.add_argument('--save_freq', type=int, default=20,
                    help="how often (how many epochs) to run validation and save tf models. default: 10")
parser.add_argument('--n_hidden', type=int, default=128,help="dim of VAE hidden layer")
parser.add_argument('--pre_trained', type=bool, default=False,help="read pre_trained model")
parser.add_argument('--models_dir', type=str, default='./trained_models',
                    help="the dir path where you store trained models. A new directory will be created inside it.")

parser.add_argument('--resume_model_path', type=str, default=None,
                    help="the dir path where you stored trained models.")

parser.add_argument('--no_psl', action='store_true')
parser.add_argument('--semisupervised_neg', action='store_true')
parser.add_argument('--semisupervised_neg_v2', action='store_true')
parser.add_argument('--semisupervised_v1', action='store_true')
parser.add_argument('--semisupervised_v1_1', action='store_true')
parser.add_argument('--semisupervised_v2', action='store_true', help='enable the pool-based semi-supervised learning.')
parser.add_argument('--semisupervised_v2_2', action='store_true')
parser.add_argument('--semisupervised_v2_3', action='store_true')
parser.add_argument('--sample_balance_v0', action='store_true')
parser.add_argument('--sample_balance_v0_1', action='store_true')
parser.add_argument('--sample_balance_for_semisuper_v0', action='store_true')

parser.add_argument('--no_trail', action='store_true')

# regularizer coefficient (lambda)
parser.add_argument('--reg_scale', type=float, default=0.005,
                    help="The scale for regularizer (lambda). Default 0.0005")

args = parser.parse_args()

# parameters
param.n_hidden = args.n_hidden
param.verbose = args.verbose
param.whichdata = args.data
param.whichmodel = ModelList(args.model)
param.pre_trained = args.pre_trained
param.ver = args.version
param.n_epoch = args.epoch
param.learning_rate = args.lr
param.batch_size = args.batch_size
param.val_save_freq = args.save_freq  # The frequency to validate and save model
param.dim = args.dim  # default 128
param.neg_per_pos = args.n_neg  # Number of negative samples per (h,r,t). default 10.
param.reg_scale = args.reg_scale
param.n_psl = 0 if args.no_psl else param.n_psl
param.semisupervised_negative_sample = args.semisupervised_neg
param.semisupervised_negative_sample_v2 = args.semisupervised_neg_v2
param.semisupervised_v1 = args.semisupervised_v1
param.semisupervised_v1_1 = args.semisupervised_v1_1
param.semisupervised_v2 = args.semisupervised_v2
param.semisupervised_v2_2 = args.semisupervised_v2_2
param.semisupervised_v2_3 = args.semisupervised_v2_3
param.sample_balance_v0 = args.sample_balance_v0
param.sample_balance_v0_1 = args.sample_balance_v0_1
param.sample_balance_for_semisuper_v0 = args.sample_balance_for_semisuper_v0
param.resume_model_path = args.resume_model_path
param.no_train = args.no_trail

# path to save
identifier = get_model_identifier(param.whichmodel, param.ver)
save_dir = join(args.models_dir, param.whichdata, identifier)  # the directory where we store this model

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Trained models will be stored in: ', save_dir)

# input files
data_dir = join('./data', args.data)
file_train = join(data_dir, 'train.tsv')  # training data
file_val = join(data_dir, 'val.tsv')  # validation datan
file_test = join(data_dir, 'test.tsv')  # validation datan
file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
print('file_psl: %s' % file_psl)

more_filt = [file_val, join(data_dir, 'test.tsv')]
print('Read train.tsv from', data_dir)

if not param.no_train:
    # load data
    this_data = Data()
    this_data.load_data(file_train=file_train, file_val=file_val, file_test=file_test, file_psl= file_psl )
    for f in more_filt:
        this_data.record_more_data(f)
    this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

    m_train = Trainer()
    m_train.build(this_data, save_dir, psl=(param.n_psl > 0), semisupervised_negative_sample=param.semisupervised_negative_sample)
    # Model will be trained, validated, and saved in './trained_models'
    ht_embedding, r_embedding = m_train.train(epochs=param.n_epoch, save_every_epoch=param.val_save_freq,
                                          lr=param.learning_rate,
                                          data_dir=param.data_dir(), resume_model_path=param.resume_model_path)
else:
    m_train = Trainer()
    m_train.build(None, save_dir, psl=(param.n_psl > 0))
    m.train.test(file_test = file_test, resume_model_path=param.resume_model_path)
