import sys
from subprocess import call

import os

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = 0
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    job_id = 0
    gpu_id = 0
    print('Missing argument: job_id and gpu_id. Use default job_id: {}, gpu_id: {}'.format(job_id, gpu_id))

import random, numpy, torch
seed_val = 0 # the same as the random state in relgan_revised_instrutor
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
##########################################################
# training parameter: if_test, gen_pretrain, batch_size
# ===Program===
if_test = int(False) # int(False)  int(True) # train or test
run_model = 'relganRevised' # call 'relganRevised' class
# #### no use ####

# max_seq_len = 30 # the length of the sentence

gen_lr = 1e-4 #
gen_adv_lr = 1e-4
dis_lr = 1e-4
# the log will not change so much, thus, we use the 10 steps rather than the 5 steps!
pre_log_step = 5
adv_log_step = 3

#####################################
CUDA = int(True)

tips = 'PETGEN experiments'
oracle_pretrain = int(True)
# in the later time, I almost do not use this option: gen_pretrain!
gen_pretrain = int(True) # True; False: means we do not have it, and we will train it!
# #### end ####
dis_pretrain = int(False)
# ===Oracle or Real===
if_real_data = [int(True)]# [int(False), int(True)), int(True], previous, only False, then wrong,
# dataset = ['yelp'] # 'yelp', 'wiki'
loss_type = 'rsgan'
vocab_size =[0] # [5000, 0, 0] # in the wiki data, we still use 5k vocabulary
temp_adpt = 'exp'
temperature = [1, 100, 100]

# ===Basic Param===model_type
data_shuffle = int(False)
# for our testing, we can only use the vanilla model!
model_type = "vanilla" #  'vanilla' only, "lstm" just for quick testing
gen_init = 'truncated_normal'
dis_init = 'uniform'
samples_num = 1000



# ===Generator===
ADV_g_step = 1
gen_embed_dim = 32
gen_hidden_dim = 32
mem_slots = 1
num_heads = 2
head_size = 256

# ===Discriminator===
ADV_d_step = 1 # previously by default, more d step with 5
dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(True)
use_nll_div = int(True)
use_bleu = int(True)
use_self_bleu = int(True)
use_ppl = int(False)

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--cuda', CUDA,
    # '--device', gpu_id,   # comment for auto GPU
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,
    # '--mle_epoch', MLE_train_epoch,
    # '--adv_epoch', ADV_train_epoch,
    '--tips', tips,

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    # '--dataset', dataset[job_id], # : change it and set it in the config.py file
    '--loss_type', loss_type,
    '--vocab_size', vocab_size[job_id],
    '--temp_adpt', temp_adpt,
    '--temperature', temperature[job_id],

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--samples_num', samples_num,
    # '--batch_size', batch_size,
    # '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--gen_adv_lr', gen_adv_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size,

    # Discriminator
    '--adv_d_step', ADV_d_step,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,
    '--num_rep', num_rep,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_ppl', use_ppl,
]

# Executables
executable = 'python'  # specify your own python interpreter path here
rootdir = '../'
scriptname = 'main.py'

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
