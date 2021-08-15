import time
from time import strftime, localtime
import os
import re
import torch

occupied_gpu_ram_memory = 32510 - 1*1024
if_gpu_while_loop_monitor = True # get the requried ram to run the code:
if_use_cpu = False

#
dataset = "wiki" # wiki, yelp
if_unbalance = False # False # if data is imbalanceed, we can set the class weight
if if_unbalance:
    weights = [1, 5]

if dataset == "wiki":
    max_seq_len = 30
    pretrained_gen_paths = {
    }
else:
    max_seq_len = 30
    pretrained_gen_paths = {
    }

# ==== setting ====
if_context = True # include the contxt for the LM training
if_linear_embedding = True # True: linear embedding for GAN training like Text-GAN
# ==== end ====

# ==== clf part ==== #
if_have_clf = True #
use_saved_clf = False  # False True

# ==== different classifiers ====
if use_saved_clf is False:
    clf_by_rnn = True # True
    clf_by_ties = False # False
    clf_by_cnn = False

    if clf_by_cnn:
        if_naccl_clf = True
        use_text_cnn_method = True
        print(f"if_naccl_clf:{if_naccl_clf}, use_text_cnn_method:{use_text_cnn_method}")

    clf_epoches = 200 #
    clf_lr = 0.01 # lr for classifier only, data-sensitive
else:
    clf_by_rnn = True
    clf_by_ties = False # True, False
    clf_by_cnn = False

    if clf_by_rnn:
        pretrained_clf_path = None #
    if clf_by_ties:
        pretrained_clf_path = None #
if if_unbalance:
    clf_model_name = f"{int(clf_by_rnn)}{int(clf_by_ties)}{int(clf_by_cnn)}weight{weights[1]}"
else:
    clf_model_name = f"{int(clf_by_rnn)}{int(clf_by_ties)}{int(clf_by_cnn)}"
if_sav_clf = False # True, False

batch_size = 64 #
# ==== pretrain MLE part ====:
if_use_context_attention_aware = True
if_pretrain_mle = True # True: whether we pretrain the LM, False:; by default, we need to pretrain the LM
MLE_train_epoch = 200 #
if if_pretrain_mle is True:
    if_previous_pretrain_model = False # True:the relgan solution: random initial state  False: text summarization technique
    if_sav_pretrain = False # whether save the LM or not
if_use_saved_gen = False # if we pretrained the LM in the past, we can use saved gen.

# ==== if you use the pretrained model, you can use the following setup. Otherwise, skip it ====
# ======== for the existing pretrained model ========
if if_use_context_attention_aware:
    pretrained_gen_path_used = pretrained_gen_paths.get("stance_aware", None) # stance_aware  stance_aware_ours_all_ties
else:
    pretrained_gen_path_used = pretrained_gen_paths.get("pure_mle", None) # without tasks:
# ======== end ========
# ==== end ====


# ==== adv part ====
# adv_training global setting
if_pretrain_mle_in_adv = False # True, False
ADV_train_epoch = 200 #
if_adv_training = True # False True

# ==== setting for the mode ====
if if_adv_training is False:
    if_adv_gan, if_adv_attack, if_adv_recency, if_adv_relevancy = False, False, False, False
else:
    attack_lr = 1e-5 # value for the sanity check now
    recency_lr = 1e-5 #
    num_recent_posts = 3 # 3, add it to 5
    relevancy_lr = 1e-5
    if_sav_adv = False # whether save the result or not
    if_adv_gan, if_adv_attack, if_adv_recency, if_adv_relevancy = True, True, True, True
# model name definition
model_name = f"{int(if_adv_gan)}{int(if_adv_attack)}{int(if_adv_recency)}{int(if_adv_relevancy)}"
# ==== end ====

# ==== end ====


# ==== RevisedRelGan ====
max_len_seq_lstm = 30 # the number of edits(posts)

# ===Program===
if_save = False # if_save for pretrain genator
if_test = False # no use: overriden by the run_relgan.py
CUDA = True
multi_gpu = False
data_shuffle = False  # False
oracle_pretrain = True  # True
gen_pretrain = False
dis_pretrain = False
clas_pretrain = False

run_model = 'relgan'  # relgan, catgan
k_label = 2  # num of labels, >=2
gen_init = 'truncated_normal'  # normal, uniform, truncated_normal
dis_init = 'uniform'  # normal, uniform, truncated_normal

# ===CatGAN===
n_parent = 1
eval_b_num = 8  # >= n_parent*ADV_d_step
max_bn = 1 if eval_b_num > 1 else eval_b_num
lambda_fq = 1.0
lambda_fd = 0.0
d_out_mean = True
freeze_dis = False
freeze_clas = False
use_all_real_fake = False
use_population = False

# ===Oracle or Real, type===
if_real_data = True  # if use real data
# dataset = 'oracle'  # oracle, image_coco, emnlp_news, amazon_app_book, amazon_app_movie, mr15
model_type = 'vanilla'  # vanilla, RMC (custom)
loss_type = 'rsgan'  # rsgan lsgan ragan vanilla wgan hinge, for Discriminator (CatGAN)
mu_type = 'ragan'  # rsgan lsgan ragan vanilla wgan hinge
eval_type = 'Ra'  # standard, rsgan, nll, nll-f1, Ra, bleu3, bleu-f1
d_type = 'Ra'  # S (Standard), Ra (Relativistic_average)
vocab_size = 5000  # oracle: 5000, coco: 4683, emnlp: 5256, amazon_app_book: 6418, mr15: 6289
# max_seq_len = 20  # oracle: 20, coco: 37, emnlp: 51, amazon_app_book: 40
# ADV_train_epoch = 2000  # SeqGAN, LeakGAN-200, RelGAN-3000
extend_vocab_size = 0  # plus test data, only used for Classifier

temp_adpt = 'exp'  # no, lin, exp, log, sigmoid, quad, sqrt
mu_temp = 'exp'  # lin exp log sigmoid quad sqrt
evo_temp_step = 1
temperature = 1

# ===Basic Train===
samples_num = 10000  # 10000, mr15: 2000,
# MLE_train_epoch = 150  # SeqGAN-80, LeakGAN-8, RelGAN-150
PRE_clas_epoch = 10
inter_epoch = 15  # LeakGAN-10
# batch_size = 64  # 64
start_letter = 1
padding_idx = 0
start_token = 'BOS'
padding_token = 'EOS'
UNK = "UNK"
UNK_IDX = 2
gen_lr = 0.01  # 0.01
gen_adv_lr = 1e-4  # RelGAN-1e-4
dis_lr = 1e-4  # SeqGAN,LeakGAN-1e-2, RelGAN-1e-4
clas_lr = 1e-3
clip_norm = 5.0

pre_log_step = 10
adv_log_step = 10

train_data = 'dataset/' + dataset + '.txt'
test_data = 'dataset/testdata/' + dataset + '_test.txt'
cat_train_data = 'dataset/' + dataset + '_cat{}.txt'
cat_test_data = 'dataset/testdata/' + dataset + '_cat{}_test.txt'

# ===Metrics===
use_nll_oracle = True
use_nll_gen = True
use_nll_div = True
use_bleu = True
use_self_bleu = False
use_clas_acc = True
use_ppl = False

# ===Generator===
ADV_g_step = 1  # 1
rollout_num = 16  # 4
gen_embed_dim = 32  # 32
gen_hidden_dim = 32  # 32
goal_size = 16  # LeakGAN-16
step_size = 4  # LeakGAN-4

mem_slots = 1  # RelGAN-1
num_heads = 2  # RelGAN-2
head_size = 256  # RelGAN-256

# ===Discriminator===
d_step = 5  # SeqGAN-50, LeakGAN-5
d_epoch = 3  # SeqGAN,LeakGAN-3
ADV_d_step = 5  # SeqGAN,LeakGAN,RelGAN-5
ADV_d_epoch = 3  # SeqGAN,LeakGAN-3

dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64  # RelGAN

# ===log===
log_time_str = strftime("%m%d_%H%M_%S", localtime())
log_filename = strftime("log/log_%s" % log_time_str)
if os.path.exists(log_filename + '.txt'):
    i = 2
    while True:
        if not os.path.exists(log_filename + '_%d' % i + '.txt'):
            log_filename = log_filename + '_%d' % i
            break
        i += 1
log_filename = log_filename + '.txt'


# by : another approach
def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    # another approach
    import subprocess, re

    # Nvidia-smi GPU memory parsing.
    # Tested on nvidia-smi 370.23

    def run_command(cmd):
        """Run command, return output as string."""
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return output.decode("ascii")

    def list_available_gpus():
        """Returns list of available GPU ids."""
        output = run_command("nvidia-smi -L")
        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        result = []
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            assert m, "Couldnt parse " + line
            result.append(int(m.group("gpu_id")))
        return result

    def gpu_memory_map():
        """Returns map of GPU id to memory allocated on that GPU."""

        output = run_command("nvidia-smi")
        gpu_output = output[output.find("GPU Memory"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
        memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
        rows = gpu_output.split("\n")
        result = {gpu_id: 0 for gpu_id in list_available_gpus()}
        for row in gpu_output.split("\n"):
            m = memory_regex.search(row)
            if not m:
                continue
            gpu_id = int(m.group("gpu_id"))
            gpu_memory = int(m.group("gpu_memory"))
            result[gpu_id] += gpu_memory
        return result

    best_memory = float("inf")
    while best_memory > occupied_gpu_ram_memory: #
        memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
        # print(memory_gpu_map) # [(31582, 0), (29729, 1), (31143, 2), (24625, 3), (29163, 4), (27903, 5), (27805, 6), (27799, 7)]
        best_memory, best_gpu = sorted(memory_gpu_map)[0]
        # print(best_memory, best_gpu)
        if if_gpu_while_loop_monitor is False:
            break
    return best_gpu

# Automatically choose GPU or CPU
#
#  https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = pick_gpu_lowest_memory()
else:
    device = -1

# comment by
# if torch.cuda.is_available() and torch.cuda.device_count() > 0:
#     os.system('nvidia-smi -q -d Utilization > gpu')
#     with open('gpu', 'r') as _tmpfile:
#         # revised by , previously, we choose by utility, key word to Memory, not correct
#         util_gpu = list(map(int, re.findall(r'Gpu\s+:\s*(\d+)\s*%', _tmpfile.read())))
#     os.remove('gpu')
#     if len(util_gpu):
#         device = util_gpu.index(min(util_gpu))
#     else:
#         device = 0
# else:
#     device = -1
# # device=0
# # print('device: ', device)

if multi_gpu:
    devices = '0,1'
    devices = list(map(int, devices.split(',')))
    device = devices[0]
    torch.cuda.set_device(device)
    os.environ['CUDA_VISIBLE_DIVICES'] = ','.join(map(str, devices))
else:
    devices = str(device)
    # revised by  for some checking
    if if_use_cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(device)


# ===Save Model and samples===
save_root = 'save/{}/{}/{}_{}_dt-{}_lt-{}_mt-{}_et-{}_sl{}_temp{}_lfd{}_T{}_module{}/'.format(time.strftime("%Y%m%d"),
                                                                                     dataset, run_model, model_type,
                                                                                     d_type,
                                                                                     loss_type,
                                                                                     '+'.join(
                                                                                         [m[:2] for m in
                                                                                          mu_type.split()]),
                                                                                     eval_type, max_seq_len,
                                                                                     temperature, lambda_fd,
                                                                                     log_time_str,
                                                                                    model_name)
save_samples_root = save_root + 'samples/'
save_model_root = save_root + 'models/'

oracle_state_dict_path = 'pretrain/oracle_data/oracle_lstm.pt'
oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}.pt'
multi_oracle_state_dict_path = 'pretrain/oracle_data/oracle{}_lstm.pt'
multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}.pt'

pretrain_root = 'pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                   samples_num)
pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                               samples_num)
pretrained_clas_path = pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                 samples_num)
#  # change t
# pretrained_clf_path = pretrain_root + 'clf_pretrain_{}_sl{}.pt'.format(run_model, max_seq_len)
# clf_pretrain_relganRevised_lstm_sl31_sn1000.pt -> clf_pretrain_relganRevised_sl31.pt

signal_file = 'run_signal.txt'

tips = ''

if samples_num == 5000 or samples_num == 2000:
    assert 'c' in run_model, 'warning: samples_num={}, run_model={}'.format(samples_num, run_model)


def param_update_extend_vocab(var):
    global extend_vocab_size
    extend_vocab_size = var
    print("in cfg after", extend_vocab_size)

# Init settings according to parser
def init_param(opt):
    global run_model, model_type, loss_type, CUDA, device, data_shuffle, samples_num, vocab_size, \
        MLE_train_epoch, ADV_train_epoch, inter_epoch, batch_size, max_seq_len, start_letter, padding_idx, \
        gen_lr, gen_adv_lr, dis_lr, clip_norm, pre_log_step, adv_log_step, train_data, test_data, temp_adpt, \
        temperature, oracle_pretrain, gen_pretrain, dis_pretrain, ADV_g_step, rollout_num, gen_embed_dim, \
        gen_hidden_dim, goal_size, step_size, mem_slots, num_heads, head_size, d_step, d_epoch, \
        ADV_d_step, ADV_d_epoch, dis_embed_dim, dis_hidden_dim, num_rep, log_filename, save_root, \
        signal_file, tips, save_samples_root, save_model_root, if_real_data, pretrained_gen_path, \
        pretrained_dis_path, pretrain_root, if_test, dataset, PRE_clas_epoch, oracle_samples_path, \
        pretrained_clas_path, n_parent, mu_type, eval_type, d_type, eval_b_num, lambda_fd, d_out_mean, \
        lambda_fq, freeze_dis, freeze_clas, use_all_real_fake, use_population, gen_init, dis_init, \
        multi_oracle_samples_path, k_label, cat_train_data, cat_test_data, evo_temp_step, devices, \
        use_nll_oracle, use_nll_gen, use_nll_div, use_bleu, use_self_bleu, use_clas_acc, use_ppl, \
        pretrained_clf_path

    if_test = True if opt.if_test == 1 else False
    run_model = opt.run_model
    k_label = opt.k_label
    dataset = opt.dataset
    model_type = opt.model_type
    loss_type = opt.loss_type
    mu_type = opt.mu_type
    eval_type = opt.eval_type
    d_type = opt.d_type
    if_real_data = True if opt.if_real_data == 1 else False
    CUDA = True if opt.cuda == 1 else False
    device = opt.device
    devices = opt.devices
    data_shuffle = opt.shuffle
    gen_init = opt.gen_init
    dis_init = opt.dis_init

    n_parent = opt.n_parent
    eval_b_num = opt.eval_b_num
    lambda_fq = opt.lambda_fq
    lambda_fd = opt.lambda_fd
    d_out_mean = opt.d_out_mean
    freeze_dis = opt.freeze_dis
    freeze_clas = opt.freeze_clas
    use_all_real_fake = opt.use_all_real_fake
    use_population = opt.use_population

    samples_num = opt.samples_num
    vocab_size = opt.vocab_size
    MLE_train_epoch = opt.mle_epoch
    PRE_clas_epoch = opt.clas_pre_epoch
    ADV_train_epoch = opt.adv_epoch
    inter_epoch = opt.inter_epoch
    batch_size = opt.batch_size
    max_seq_len = opt.max_seq_len
    start_letter = opt.start_letter
    padding_idx = opt.padding_idx
    gen_lr = opt.gen_lr
    gen_adv_lr = opt.gen_adv_lr
    dis_lr = opt.dis_lr
    clip_norm = opt.clip_norm
    pre_log_step = opt.pre_log_step
    adv_log_step = opt.adv_log_step
    temp_adpt = opt.temp_adpt
    evo_temp_step = opt.evo_temp_step
    temperature = opt.temperature
    oracle_pretrain = True if opt.ora_pretrain == 1 else False
    gen_pretrain = True if opt.gen_pretrain == 1 else False
    dis_pretrain = True if opt.dis_pretrain == 1 else False

    ADV_g_step = opt.adv_g_step
    rollout_num = opt.rollout_num
    gen_embed_dim = opt.gen_embed_dim
    gen_hidden_dim = opt.gen_hidden_dim
    goal_size = opt.goal_size
    step_size = opt.step_size
    mem_slots = opt.mem_slots
    num_heads = opt.num_heads
    head_size = opt.head_size

    d_step = opt.d_step
    d_epoch = opt.d_epoch
    ADV_d_step = opt.adv_d_step
    ADV_d_epoch = opt.adv_d_epoch
    dis_embed_dim = opt.dis_embed_dim
    dis_hidden_dim = opt.dis_hidden_dim
    num_rep = opt.num_rep

    use_nll_oracle = True if opt.use_nll_oracle == 1 else False
    use_nll_gen = True if opt.use_nll_gen == 1 else False
    use_nll_div = True if opt.use_nll_div == 1 else False
    use_bleu = True if opt.use_bleu == 1 else False
    use_self_bleu = True if opt.use_self_bleu == 1 else False
    use_clas_acc = True if opt.use_clas_acc == 1 else False
    use_ppl = True if opt.use_ppl == 1 else False

    log_filename = opt.log_file
    signal_file = opt.signal_file
    tips = opt.tips

    # comment by : Jan. 9th, since it is repeated
    # # CUDA device
    # if multi_gpu:
    #     if type(devices) == str:
    #         devices = list(map(int, devices.split(',')))
    #     device = devices[0]
    #     torch.cuda.set_device(device)
    #     os.environ['CUDA_VISIBLE_DIVICES'] = ','.join(map(str, devices))
    # else:
    #     devices = str(device)
    #     torch.cuda.set_device(device)

    # Save path
    save_root = 'save/{}/{}/{}_{}_dt-{}_lt-{}_mt-{}_et-{}_sl{}_temp{}_lfd{}_T{}_module{}/'.format(time.strftime("%Y%m%d"),
                                                                                         dataset, run_model, model_type,
                                                                                         d_type,
                                                                                         loss_type,
                                                                                         '+'.join(
                                                                                             [m[:2] for m in
                                                                                              mu_type.split()]),
                                                                                         eval_type, max_seq_len,
                                                                                         temperature, lambda_fd,
                                                                                         log_time_str,
                                                                                                  model_name)

    save_samples_root = save_root + 'samples/'
    save_model_root = save_root + 'models/'

    train_data = 'dataset/' + dataset + '.txt'
    test_data = 'dataset/testdata/' + dataset + '_test.txt'
    cat_train_data = 'dataset/' + dataset + '_cat{}.txt'
    cat_test_data = 'dataset/testdata/' + dataset + '_cat{}_test.txt'

    if max_seq_len == 40:
        oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}_sl40.pt'
        multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}_sl40.pt'

    pretrain_root = 'pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
    pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type,
                                                                                       max_seq_len, samples_num)
    pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                   samples_num)
    pretrained_clas_path = pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                     samples_num)
    # pretrained_clf_path = pretrain_root + 'clf_pretrain_{}_sl{}.pt'.format(run_model, max_seq_len)
    # Assertion
    assert k_label >= 2, 'Error: k_label = {}, which should be >=2!'.format(k_label)
    assert eval_b_num >= n_parent * ADV_d_step, 'Error: eval_b_num = {}, which should be >= n_parent * ADV_d_step ({})!'.format(
        eval_b_num, n_parent * ADV_d_step)

    # Create Directory
    dir_list = ['save', 'savefig', 'log', 'pretrain', 'dataset',
                'pretrain/{}'.format(dataset if if_real_data else 'oracle_data')]
    if not if_test:
        dir_list.extend([save_root, save_samples_root, save_model_root])
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
