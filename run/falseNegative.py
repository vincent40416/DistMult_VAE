"""
Current working directory: Project root dir

=== usage
python run/run.py -m DM --data cn15k --lr 0.01 --batch_size 300
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if './src' not in sys.path:
    sys.path.append('./src')

if './' not in sys.path:
    sys.path.append('./')

import os
from os.path import join
from src.data import Data

from src.trainer import Trainer
from src.list import ModelList
from src.testers import *
import datetime
import time

import argparse
from src import param
import csv

def get_model_identifier(whichmodel):
    prefix = whichmodel.value
    now = datetime.datetime.now()
    date = '%02d%02d' % (now.month, now.day)  # two digits month/day
    identifier = prefix + '_' + date
    return identifier


parser = argparse.ArgumentParser()
# required
parser.add_argument('--data', type=str, default='ppi5k',
                    help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
# optional
parser.add_argument("--verbose", help="print detailed info for debugging",
                    action="store_true")
parser.add_argument('-m', '--model', type=str, default='rect', help="choose model ('logi' or 'rect'). default: rect")
parser.add_argument('-d', '--dim', type=int, default=128, help="set dimension. default: 128")
parser.add_argument('--epoch', type=int, default=100, help="set number of epochs. default: 100")
parser.add_argument('--lr', type=float, default=0.001, help="set learning rate. default: 0.001")
parser.add_argument('--batch_size', type=int, default=1024, help="set batch size. default: 1024")
parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
parser.add_argument('--save_freq', type=int, default=10,
                    help="how often (how many epochs) to run validation and save tf models. default: 10")
parser.add_argument('--models_dir', type=str, default='./trained_models',
                    help="the dir path where you store trained models. A new directory will be created inside it.")

parser.add_argument('--resume_model_path', type=str, default=None,
                    help="the dir path where you stored trained models.")

parser.add_argument('--no_psl', action='store_true')
parser.add_argument('--semisupervised_neg', action='store_true')
parser.add_argument('--semisupervised_neg_v2', action='store_true')
parser.add_argument('--semisupervised_v1', action='store_true')
parser.add_argument('--semisupervised_v1_1', action='store_true')


parser.add_argument('--start', type=int, default=10, help="Starting epoch")
parser.add_argument('--to', type=int, default=800, help="Ending epoch")
parser.add_argument('--step', type=int, default=10, help="Epoch step size")


parser.add_argument('--no_trail', action='store_true')

# regularizer coefficient (lambda)
parser.add_argument('--reg_scale', type=float, default=0.005,
                    help="The scale for regularizer (lambda). Default 0.0005")

args = parser.parse_args()

# parameters
param.verbose = args.verbose
param.whichdata = args.data
param.whichmodel = ModelList(args.model)
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
param.resume_model_path = args.resume_model_path
param.no_train = args.no_trail




param.data_surfix = ''

if '_threshold_' in args.data:
    # '_threshold_0.x'
    t = args.data.index('_threshold_')
    param.data_surfix = args.data[t: t + 14]


files_opened = []
def save_loss(losses, filename, columns):
    if not filename in files_opened:
        files_opened.append(filename)
        if os.path.exists(filename):
            os.rename(filename, filename + '.bak')

    df = pd.DataFrame(losses, columns=columns)
    print(df.tail(5))
    df.tail(1).to_csv(filename, header=False, index=False, mode='a')


# path to save
identifier = get_model_identifier(param.whichmodel)
save_dir = join(args.models_dir, param.whichdata, identifier)  # the directory where we store this model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Trained models will be stored in: ', save_dir)

# input files
data_dir = join('./data', args.data)
# file_train = join(data_dir, 'train.tsv')  # training data
file_val = join(data_dir, 'val.tsv')  # validation datan
file_test = join(data_dir, 'test.tsv')  # validation datan
file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
print('file_psl: %s' % file_psl)

# more_filt = [file_val, join(data_dir, 'test.tsv')]
print('Read train.tsv from', data_dir)

losses_classify_triple_testing = []
losses_mean_ndcg = []
losses_mse = []
mean_ranks = []

for n in range(args.start, args.to, args.step):

    model_path = os.path.join(param.resume_model_path, 'model.bin-%d.meta'%n)
    while not os.path.exists(model_path):
        # print('\033[91m[error]', model_path, 'not found. sleeping...\033[91m')
        print('[error]', model_path, 'not found. sleeping...')
        time.sleep(60)
    # get corredponding tester
    tmp_data_obj = Data()
    tmp_trainger = Trainer()
    tmp_trainger.build(tmp_data_obj, '', psl=(param.n_psl > 0))
    tester = tmp_trainger.tester
    del tmp_trainger
    del tmp_data_obj

    tester.build_by_file(file_test, param.resume_model_path, model_filename='model.bin-%d'%n)

    tester.load_hr_map(param.data_dir(), 'data.tsv', []) 

    KG_THRES = 0
    # KG_THRES = 0.85

    hr_map = {}
    for h in tester.hr_map:
        for r in tester.hr_map[h]:
            if len(tester.hr_map[h][r]) > 1 \
                    or list(tester.hr_map[h][r].values())[0]>KG_THRES:
                if h not in hr_map:
                    hr_map[h] = {}
                hr_map[h][r] = tester.hr_map[h][r]


    results = tester.find_false_negatives(hr_map)

    with open('%s_%s_falseNegatives.csv'%(args.model, args.data), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    