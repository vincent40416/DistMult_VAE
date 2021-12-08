from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from src.testers import *
import datetime
import time

import argparse
from src import param
import csv

from analysisUtil import *

def get_model_identifier(whichmodel, ver):
    prefix = whichmodel.value
    if ver!=0:
        prefix= prefix +'_v'+ str(ver) + '_' + str(param.n_hidden)
    now = datetime.datetime.now()
    date = '%02d%02d' % (now.month, now.day)  # two digits month/day
    identifier = prefix + '_' + date
    return identifier


parser = argparse.ArgumentParser()
# WANING: Some of the parameters are useless here!
parser.add_argument('--data', type=str, default='ppi5k',
                    help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")

parser.add_argument("--verbose", help="print detailed info for debugging",
                    action="store_true")
parser.add_argument('-m', '--model', type=str, default='rect', help="choose model ('logi' or 'rect'). default: rect")
parser.add_argument('-ver', '--version', type=int, default=0, help="choose model ver 1 to 3")
parser.add_argument('-d', '--dim', type=int, default=128, help="set dimension. default: 128")
parser.add_argument('--epoch', type=int, default=100, help="useless here")
parser.add_argument('--lr', type=float, default=0.001, help="useless here")
parser.add_argument('--batch_size', type=int, default=1024, help="set batch size. default: 1024")
parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
parser.add_argument('--save_freq', type=int, default=10,
                    help="how often (how many epochs) to run validation and save tf models. default: 10")
parser.add_argument('--models_dir', type=str, default='./trained_models',
                    help="the dir path where you store trained models. A new directory will be created inside it.")
parser.add_argument('--pre_trained', type=bool, default=False,help="read pre_trained model")
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



# regularizer coefficient (lambda)
parser.add_argument('--reg_scale', type=float, default=0.005,
                    help="The scale for regularizer (lambda). Default 0.0005")

args = parser.parse_args()

# parameters
param.verbose = args.verbose
param.whichdata = args.data
param.ver = args.version
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




param.data_surfix = ''

if '_threshold_' in args.data:
    # '_threshold_0.x'
    t = args.data.index('_threshold_')
    param.data_surfix = args.data[t: t + 14]


# files_opened = []
def save_loss(losses, filename, columns):
    # if not filename in files_opened:
    #     files_opened.append(filename)
    #     if os.path.exists(filename):
    #         os.rename(filename, filename + '.bak')

    df = pd.DataFrame(losses, columns=columns)
    print(df.tail(5))
    df.tail(1).to_csv(filename, header=False, index=False, mode='a')



# input files
data_dir = join('./data', args.data)
file_train = join(data_dir, 'train.tsv')  # training data
file_val = join(data_dir, 'val.tsv')  # validation datan
file_test = join(data_dir, 'test.tsv')  # validation datan
file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
# print('file_psl: %s' % file_psl)

# more_filt = [file_val, join(data_dir, 'test.tsv')]
print('Read train.tsv from', data_dir)

losses_classify_triple_testing = []
losses_mean_ndcg = []
losses_mse = []
mean_ranks = []


val_mean_rank_save_path = os.path.join(param.resume_model_path, 'val%s_mean_rank_accurate.csv'%(param.data_surfix,))
val_ndcg_save_path = os.path.join(param.resume_model_path, 'val%s_loss_accurate.csv'%(param.data_surfix,))
f = open(val_mean_rank_save_path, "w")
f.truncate()
f.close()
f = open(val_ndcg_save_path, "w")
f.truncate()
f.close()

test_mean_rank_save_path = os.path.join(param.resume_model_path, 'test%s_mean_rank_accurate.csv'%(param.data_surfix,))
test_ndcg_save_path = os.path.join(param.resume_model_path, 'test%s_loss_accurate.csv'%(param.data_surfix,))
f = open(test_mean_rank_save_path, "w")
f.truncate()
f.close()
f = open(test_ndcg_save_path, "w")
f.truncate()
f.close()

test_training_included_mean_rank_save_path = os.path.join(param.resume_model_path, 'test%s_mean_rank_training_included.csv'%(param.data_surfix,))
test_training_included_ndcg_save_path = os.path.join(param.resume_model_path, 'test%s_loss_training_included.csv'%(param.data_surfix,))
f = open(test_training_included_mean_rank_save_path, "w")
f.truncate()
f.close()
f = open(test_training_included_ndcg_save_path, "w")
f.truncate()
f.close()



def test(n, testtype):
    """
    @param n: the epoch/step to validate
    """

#     if getRecord(val_mean_rank_save_path, 'mean_rank', n) != None and \
#         getRecord(val_ndcg_save_path, 'ndcg', n) != None:
#         return "existed"
    print(testtype, n)
    model_path = os.path.join(param.resume_model_path, 'model.bin-%d.meta'%n)
    while not os.path.exists(model_path):
        # print('\033[91m[error]', model_path, 'not found. sleeping...\033[91m')
        print('[error]', model_path, 'not found. sleeping...')
        time.sleep(60)
    # get corredponding tester
    tmp_data_obj = Data()
    tmp_data_obj.load_data(file_train=file_train, file_val=file_val, file_test=file_test, file_psl= file_psl )
    tmp_trainger = Trainer()
    tmp_trainger.build(tmp_data_obj, '', psl=(param.n_psl > 0))
    tester = tmp_trainger.tester
    del tmp_trainger
    del tmp_data_obj
    
    
    
    if testtype == "val":
    
        tester.build_by_file(file_val, param.resume_model_path, model_filename='model.bin-%d'%n)
        tester.load_hr_map(param.data_dir(), 'val.tsv', [])
        mean_rank_path = val_mean_rank_save_path
        ndcg_path = val_ndcg_save_path
        detail = 'val%s_detail_%d.csv'%(param.data_surfix, n)
    elif testtype == "test":
        tester.build_by_file(file_test, param.resume_model_path, model_filename='model.bin-%d'%n)

        # if not hasattr(tester, 'hr_map'):
        tester.load_hr_map(param.data_dir(), 'test.tsv', [])
        mean_rank_path = test_mean_rank_save_path
        ndcg_path = test_ndcg_save_path
        detail = "test%s_test_only_detail_%d.csv"%(param.data_surfix, n)
    elif testtype == "testwithtraining":
        tester.build_by_file(file_test, param.resume_model_path, model_filename='model.bin-%d'%n)

        # if not hasattr(tester, 'hr_map'):
        tester.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv', 'test.tsv'])
        mean_rank_path = test_training_included_mean_rank_save_path
        ndcg_path = test_training_included_ndcg_save_path
        detail = 'test%s_detail_%d_training_included.csv'%(param.data_surfix, n)
        
        
    KG_THRES = 0
    hr_map = {}
    for h in tester.hr_map:
        for r in tester.hr_map[h]:
            # print('%s %s %d'%(h, r, len(list(tester.hr_map[h][r].values()))))
            # print('%d %d %f'%(h, r, max(list(tester.hr_map[h][r].values()))))
            if len(tester.hr_map[h][r]) > 1 \
                    or list(tester.hr_map[h][r].values())[0]>KG_THRES:
            # if len(tester.hr_map[h][r]) >= 8:
                    # and max(list(tester.hr_map[h][r].values()))>KG_THRES:
                if h not in hr_map:
                    hr_map[h] = {}
                hr_map[h][r] = tester.hr_map[h][r]


    r_N = tester.vec_r.shape[0]
    h_N = tester.vec_c.shape[0]

    #metrics mse
    mse = tester.get_mse(epoch=n, toprint=True)
    mse_neg = tester.get_mse_neg(10)
    mse_neg2 = tester.get_mse_neg(10)

    mean_ndcg, mean_exp_ndcg, mean_ndcg_r, count_r, all_ndcg, _ = tester.mean_ndcg(hr_map, accurate_mode=True)#, verbose=True)
    losses_mean_ndcg.append(np.insert(mean_ndcg_r, 0, n))


    losses_mse.append([n, mse, mse_neg, mse_neg2, mean_ndcg, mean_exp_ndcg])
    
    with open(os.path.join(param.resume_model_path, detail), 'w') as f:
        csv.writer(f).writerows(all_ndcg)

    mean_hitAtK, _ = tester.mean_hitAtK(hr_map, [10,20,40,10,20,40], weighted=[False,False,False,True,True,True], accurate_mode=True,verbose = False)

    mean_rank, _ = tester.mean_rank(hr_map, accurate_mode=True)
    mean_rank_weighted, _ = tester.mean_rank(hr_map, weighted = True, accurate_mode=True)

    mean_hitAt10 = mean_hitAtK[0]
    mean_hitAt20 = mean_hitAtK[1]
    mean_hitAt40 = mean_hitAtK[2]
    mean_hitAt10_weighted = mean_hitAtK[3]
    mean_hitAt20_weighted = mean_hitAtK[4]
    mean_hitAt40_weighted = mean_hitAtK[5]
    mean_ranks.append([n, mean_rank, mean_hitAt10, mean_hitAt20, mean_hitAt40, mean_rank_weighted, mean_hitAt10_weighted, mean_hitAt20_weighted, mean_hitAt40_weighted])
    save_loss(mean_ranks, mean_rank_path,
                               columns=['val_epoch', 'mean_rank', 'mean_hit@10', 'mean_hit@20', 'mean_hit@40', 'mean_rank_weighted', 'mean_hit@10_weighted', 'mean_hit@20_weighted', 'mean_hit@40_weighted'])

    save_loss(losses_mse, ndcg_path,
                               columns=['val_epoch', 'mse', 'mse_neg', 'mse_neg(second)', 'ndcg(linear)', 'ndcg(exp)'])

    return 'done'



def test__1(n):
    """
    @param n: the epoch/step to validate
    Run testing. NO training data will be included as candidate entities.
    """

    if getRecord(test_mean_rank_save_path, 'mean_rank', n) != None and \
        getRecord(test_ndcg_save_path, 'ndcg', n) != None:
        return "existed"

    print('test', n)

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

    # if not hasattr(tester, 'hr_map'):
    tester.load_hr_map(param.data_dir(), 'test.tsv', []) #['train.tsv', 'val.tsv',
                                                            #            'test.tsv'])
    
    # if not hasattr(tester, 'hr_map_sub'):
    #     hr_map200 = hr_map= tester.get_fixed_hr(n=2000)  # use smaller size for faster validation
    # else:
    #     hr_map200 = tester.hr_map_sub
    # hr_map200 = hr_map = tester.hr_map

    KG_THRES = 0
    # KG_THRES = 0.85

    hr_map = {}
    for h in tester.hr_map:
        for r in tester.hr_map[h]:
            # print('%s %s %d'%(h, r, len(list(tester.hr_map[h][r].values()))))
            # print('%d %d %f'%(h, r, max(list(tester.hr_map[h][r].values()))))
            if len(tester.hr_map[h][r]) > 1 \
                    or list(tester.hr_map[h][r].values())[0]>KG_THRES:
            # if len(tester.hr_map[h][r]) >= 8:
                    # and max(list(tester.hr_map[h][r].values()))>KG_THRES:
                if h not in hr_map:
                    hr_map[h] = {}
                hr_map[h][r] = tester.hr_map[h][r]


    r_N = tester.vec_r.shape[0]
    h_N = tester.vec_c.shape[0]

    # mean_r_t_score = np.zeros(r_N)
    # count_r_t = np.zeros(r_N)
    # for r in range(r_N):
    #     for h in hr_map.keys():
    #         if r in hr_map[h]:
    #             mean_r_t_score[r] += sum(hr_map[h][r].values())
    #             count_r_t[r] += len(hr_map[h][r].keys())
    # print(mean_r_t_score / count_r_t)
    # print(count_r_t)

    # print(sum([sum( [len(tv) for tv in v.values()] ) for v in hr_map.values()]))
    # print([len(v.keys()) for v in hr_map.values()])


    # metrics: mse
    mse = tester.get_mse(epoch=n, toprint=True)
    mse_neg = tester.get_mse_neg(10)
    mse_neg2 = tester.get_mse_neg(10)
    

    # metrics: triple classification
    # scores, P, R, F1, Acc = tester.classify_triples(0.7, [0.5, 0.6, 0.7])
    # losses_classify_triple_testing.append(np.insert(np.ndarray.flatten(np.array([P, R, F1, Acc])), 0, n))
    # save_loss(losses_classify_triple_testing, os.path.join(param.resume_model_path, 'test_filtered3_triple_classification.csv'),
    #                            columns=['test_epoch', 'P_0.5', 'P_0.6', 'P_0.7', 'R_0.5', 'R_0.6', 'R_0.7', 
    #                                     'F1_0.5', 'F1_0.6', 'F1_0.7', 'Acc_0.5', 'Acc_0.6', 'Acc_0.7'])

    mean_ndcg, mean_exp_ndcg, mean_ndcg_r, count_r, all_ndcg = tester.mean_ndcg(hr_map, accurate_mode=True)#, verbose=True)
    losses_mean_ndcg.append(np.insert(mean_ndcg_r, 0, n))


    # save_loss(losses_mean_ndcg, os.path.join(param.resume_model_path, 'mean_ndcg_r_filtered3.csv'),
    # save_loss(losses_mean_ndcg, os.path.join(param.resume_model_path, 'mean_ndcg_r_all.csv'),
    #                            columns=['epoch'] + [str(x) for x in range(r_N)])

    losses_mse.append([n, mse, mse_neg, mse_neg2, mean_ndcg, mean_exp_ndcg])
    # print('losses_mse')
    # print(losses_mse)

    with open(os.path.join(param.resume_model_path, 'test%s_test_only_detail_%d.csv'%(param.data_surfix, n)), 'w') as f:
        csv.writer(f).writerows(all_ndcg)

    mean_hitAtK, _ = tester.mean_hitAtK(hr_map, [10,20,40,10,20,40], weighted=[False,False,False,True,True,True], accurate_mode=True)
    # mean_hitAt20, _ = tester.mean_hitAtK(hr_map, 20, accurate_mode=True)
    # mean_hitAt40, _ = tester.mean_hitAtK(hr_map, 40, accurate_mode=True)

    mean_rank, _ = tester.mean_rank(hr_map, accurate_mode=True)
    mean_rank_weighted, _ = tester.mean_rank(hr_map, weighted = True, accurate_mode=True)

    mean_hitAt10 = mean_hitAtK[0]
    mean_hitAt20 = mean_hitAtK[1]
    mean_hitAt40 = mean_hitAtK[2]
    mean_hitAt10_weighted = mean_hitAtK[3]
    mean_hitAt20_weighted = mean_hitAtK[4]
    mean_hitAt40_weighted = mean_hitAtK[5]
    # mean_hitAt10_weighted, _ = tester.mean_hitAtK(hr_map, 10, weighted = True, accurate_mode=True)
    # mean_hitAt20_weighted, _ = tester.mean_hitAtK(hr_map, 20, weighted = True, accurate_mode=True)
    # mean_hitAt40_weighted, _ = tester.mean_hitAtK(hr_map, 40, weighted = True, accurate_mode=True)
    # mean_precision, _ = tester.mean_precision(hr_map, accurate_mode=True)
    mean_ranks.append([n, mean_rank, mean_hitAt10, mean_hitAt20, mean_hitAt40, mean_rank_weighted, mean_hitAt10_weighted, mean_hitAt20_weighted, mean_hitAt40_weighted])
    save_loss(mean_ranks, test_mean_rank_save_path,
                               columns=['test_epoch', 'mean_rank', 'mean_hit@10', 'mean_hit@20', 'mean_hit@40', 'mean_rank_weighted', 'mean_hit@10_weighted', 'mean_hit@20_weighted', 'mean_hit@40_weighted'])

    save_loss(losses_mse, test_ndcg_save_path,
                               columns=['test_epoch', 'mse', 'mse_neg', 'mse_neg(second)', 'ndcg(linear)', 'ndcg(exp)'])



    # save_loss(losses_mse, os.path.join(param.resume_model_path, 'test_all_loss.csv'),
    #                            columns=['test_epoch', 'mse', 'mse_neg', 'ndcg(linear)', 'ndcg(exp)'])
    

def test_training_included(n):
    """
    @param n: the epoch/step to validate
    Run testing. Training data WILL be included as candidate entities.
    """
    print('test_training_include', n)
# # test, include training data
# for n in range(args.start, args.to, args.step):
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


    tester.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv','test.tsv'])

    KG_THRES = 0

    hr_map = {}
    for h in tester.hr_map:
        for r in tester.hr_map[h]:
            # print('%s %s %d'%(h, r, len(list(tester.hr_map[h][r].values()))))
            # print('%d %d %f'%(h, r, max(list(tester.hr_map[h][r].values()))))
            if len(tester.hr_map[h][r]) > 1 \
                    or list(tester.hr_map[h][r].values())[0]>KG_THRES:
            # if len(tester.hr_map[h][r]) >= 8:
                    # and max(list(tester.hr_map[h][r].values()))>KG_THRES:
                if h not in hr_map:
                    hr_map[h] = {}
                hr_map[h][r] = tester.hr_map[h][r]


    r_N = tester.vec_r.shape[0]
    h_N = tester.vec_c.shape[0]


    # metrics: mse
    mse = tester.get_mse(epoch=n, toprint=True)
    mse_neg = tester.get_mse_neg(10)
    mse_neg2 = tester.get_mse_neg(10)
    


    mean_ndcg, mean_exp_ndcg, mean_ndcg_r, count_r, all_ndcg = tester.mean_ndcg(hr_map, accurate_mode=True)#, verbose=True)
    losses_mean_ndcg.append(np.insert(mean_ndcg_r, 0, n))


    losses_mse.append([n, mse, mse_neg, mse_neg2, mean_ndcg, mean_exp_ndcg])
    # print('losses_mse')
    # print(losses_mse)

    with open(os.path.join(param.resume_model_path, 'test%s_detail_%d_training_included.csv'%(param.data_surfix, n)), 'w') as f:
        csv.writer(f).writerows(all_ndcg)

    mean_hitAtK, _ = tester.mean_hitAtK(hr_map, [10,20,40,10,20,40], weighted=[False,False,False,True,True,True], accurate_mode=True)

    mean_rank, _ = tester.mean_rank(hr_map, accurate_mode=True)
    mean_rank_weighted, _ = tester.mean_rank(hr_map, weighted = True, accurate_mode=True)

    mean_hitAt10 = mean_hitAtK[0]
    mean_hitAt20 = mean_hitAtK[1]
    mean_hitAt40 = mean_hitAtK[2]
    mean_hitAt10_weighted = mean_hitAtK[3]
    mean_hitAt20_weighted = mean_hitAtK[4]
    mean_hitAt40_weighted = mean_hitAtK[5]
    mean_ranks.append([n, mean_rank, mean_hitAt10, mean_hitAt20, mean_hitAt40, mean_rank_weighted, mean_hitAt10_weighted, mean_hitAt20_weighted, mean_hitAt40_weighted])
    save_loss(mean_ranks, test_training_included_mean_rank_save_path,
                               columns=['test_epoch', 'mean_rank', 'mean_hit@10', 'mean_hit@20', 'mean_hit@40', 'mean_rank_weighted', 'mean_hit@10_weighted', 'mean_hit@20_weighted', 'mean_hit@40_weighted'])

    save_loss(losses_mse, test_training_included_ndcg_save_path,
                               columns=['test_epoch', 'mse', 'mse_neg', 'mse_neg(second)', 'ndcg(linear)', 'ndcg(exp)'])


    
# val test testwithtraining

for n in range(args.start, args.to, args.step):
    status = test(n, "val")
    if status == "existed": 
        print("val existed")

best_epoch, _ = getBestRecord(val_ndcg_save_path, 'ndcg', 'mse', False)
test(best_epoch, "test")
test(best_epoch, "testwithtraining")
best_epoch, _ = getBestRecord(val_ndcg_save_path, 'ndcg', 'ndcg(linear)', True)
test(best_epoch, "test")
test(best_epoch, "testwithtraining")
best_epoch, _ = getBestRecord(val_mean_rank_save_path, 'mean_rank', 'mean_hit@20', True)
test(best_epoch, "test")
test(best_epoch, "testwithtraining")









