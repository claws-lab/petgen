import time
from time import strftime, localtime
import os
import re
import torch
# ==== bing's deployment ====
occupied_gpu_ram_memory = 32510 - 1*1024
if_gpu_while_loop_monitor = True # get the requried ram to run the code:
if_use_cpu = False

# malcom-direction: GOSSIP
dataset = "wiki" # wiki, yelp, yelpEqu, yelpEquLarge :  UN: twitter, unbalanced, TwitterUN, wikiUN, yelpUN, GOSSIP
if_unbalance = False # False
if if_unbalance:
    weights = [1, 5] # 10, 1000, here, note that 10:1 is wrong!

#
server = "dgx1"
if dataset == "wiki":
    pretrained_rnn_clf = r"pretrain/wiki/clf_pretrain_relganRevised_sl31.pt"
    pretrained_rnn_clf_another = r"save/20210124/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0124_2236_57_module01000/models/CLF_model100_epoch1610.pt"
    pretrained_ties_clf = "save/20210114/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0114_2231_38_module00000/models/CLF_model010_epoch680.pt"
    max_seq_len = 30

    pretrained_gen_paths = {
        "pure_mle": "save/20210105/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0105_2341_24_module0000/models/MLE_model0000_epoch220.pt",
        "stance_aware": "save/20210119/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0119_1129_43_module00000/models/MLE_model00000_epoch460.pt",
        "pure_ours_all_rnn": r"save/20210107/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0107_2246_17_module1111/models/ADV_model1111_epoch0.pt",
        "stance_aware_ours_all_ties": r"save/20210121/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0121_2202_10_module11110/models/ADV_model11110_epoch276.pt",
    }
elif dataset == "wikiUN":
    max_seq_len = 30
    pretrained_gen_paths = {
        "pure_mle": "",
        "stance_aware": "",
        "pure_ours_all_rnn": r"",
        "stance_aware_ours_all_ties": r"",
    }
elif dataset == "yelpUN":
    max_seq_len = 30
    pretrained_gen_paths = {
        "pure_mle": "",
        "stance_aware": "",
        "pure_ours_all_rnn": "",
        "stance_aware_ours_all_ties": "",
        "malcom": "",
        "ours": "",
    }
elif dataset == "yelpEqu":
    max_seq_len = 30
    pretrained_rnn_clf = ""
    pretrained_gen_paths = {
        "pure_mle": "",
        "stance_aware": "",
        "pure_ours_all_rnn": r"",
        "stance_aware_ours_all_ties": r"",
        "malcom": r"",
        "ours": r"",
    }
elif dataset == "yelp":
    max_seq_len = 30

    pretrained_rnn_clf_another = "save/20210129/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl101_temp1_lfd0.0_T0129_1634_20_module0000/models/CLF_model100_epoch50.pt"
    pretrained_ties_clf = "save/20210128/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl101_temp1_lfd0.0_T0128_2334_39_module00000/models/CLF_model010_epoch160.pt"

    if server == "dgx1":
        pretrained_rnn_clf = "save/20210128/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl101_temp1_lfd0.0_T0128_2320_00_module00000/models/CLF_model100_epoch480.pt"
        pretrained_gen_paths = {
            "pure_mle": "save/20210129/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl101_temp1_lfd0.0_T0129_1228_03_module0000/models/MLE_model0000_epoch75.pt",
            "stance_aware": "save/20210129/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl101_temp1_lfd0.0_T0129_0013_36_module00000/models/MLE_model00000_epoch310.pt",
            "pure_ours_all_rnn": "",
            "stance_aware_ours_all_ties": "",
        }
    else:
        assert server == "data5"
        pretrained_rnn_clf = "save/20210129/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl105_temp1_lfd0.0_T0129_1814_25_module0000/models/CLF_model100_epoch890.pt"
        pretrained_gen_paths = {
            "pure_mle": "save/20210130/yelp/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl105_temp1_lfd0.0_T0130_2107_09_module0000/models/MLE_model0000_epoch255.pt",
            "stance_aware": "save/20210130/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl32_temp1_lfd0.0_T0130_0504_57_module0000/models/MLE_model0000_epoch575.pt",
            "pure_ours_all_rnn": "",
            "stance_aware_ours_all_ties": "",
        }
elif dataset == "twitter":
    max_seq_len = 30
    pretrained_gen_paths = {
                "pure_mle": "",
                "stance_aware": "",
                "pure_ours_all_rnn": "",
                "stance_aware_ours_all_ties": "",
    }
elif dataset == "yelpEquLarge":
    max_seq_len = 30
    pretrained_rnn_clf = r"save/20210218/yelpEquLarge/RelGANInstructorRevisedAttack_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0218_1702_53_module0000/models/CLF_model100_epoch40.pt" # for item
        # r"/home/bhe46/program/aml/TextGAN-PyTorch/save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1201_01_module0000/models/CLF_model100_epoch430.pt" -- finally used in petgen
        # rf"{header}save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1201_01_module0000/model/CLF_model100_epoch430.pt"
        # r"/home/bhe46/program/aml/TextGAN-PyTorch/save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1201_01_module0000/models/CLF_model100_epoch430.pt"
        # r"save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1201_01_module0000/model/CLF_model100_epoch430.pt"
    pretrained_rnn_clf_another = r"save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1437_36_module0000/models/CLF_model100_epoch310.pt"
    pretrained_ties_clf = r"save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1435_37_module0000/models/CLF_model010_epoch100.pt"

    pretrained_gen_paths = {
                "pure_mle": "",
                "stance_aware": "",
                "pure_ours_all_rnn": "",
                "stance_aware_ours_all_ties": "",
                "malcom": "save/20210205/yelpEquLarge/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0205_1210_24_module0000/models/MLE_model0000_epoch335.pt",
                "ours": "",
    }
elif dataset == "GOSSIP":
    max_seq_len = 30
    # pretrained_rnn_clf = "/home/bhe46/program/aml/TextGAN-PyTorch/save/20210319/GOSSIP/MALCOM_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl103_temp1_lfd0.0_T0319_1040_47_module0000/models/CLF_model100_epoch399.pt"
    pretrained_rnn_clf = "save/20210320/GOSSIP/MALCOM_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl103_temp1_lfd0.0_T0320_0939_58_module0000/models/CLF_model100_epoch110.pt"
    pretrained_rnn_clf_another = ""
    pretrained_ties_clf = ""
    pretrained_gen_paths = {
                "pure_mle": "",
                "stance_aware": "",
                "pure_ours_all_rnn": "",
                "stance_aware_ours_all_ties": "",
                # "malcom": "save/20210319/GOSSIP/MALCOM_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl103_temp1_lfd0.0_T0319_1100_39_module0000/models/MLE_model0000_epoch90.pt",
                # "malcom": "save/20210319/GOSSIP/MALCOM_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl103_temp1_lfd0.0_T0319_2040_11_module0000/models/MLE_model0000_epoch190.pt",
                "malcom": "save/20210319/GOSSIP/MALCOM_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl103_temp1_lfd0.0_T0319_2040_11_module0000/models/MLE_model0000_epoch499.pt",
                "ours": "",
    }
# ==== never change ====
if_context = True # whether we include the contexts: for the vocab and model building! even for the classifier, we need it!
if_linear_embedding = True # True all the time and we do not change it
# ==== never change ====

# ==== clf part ==== #
if_have_clf = True # it should be True all the time, False  True
use_saved_clf = False  # False True
# "save/20210114/wiki/relganRevised_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl31_temp1_lfd0.0_T0114_1256_53_module00000/models/CLF_model010_epoch1390.pt"
if use_saved_clf is False:
    clf_by_rnn = True # True
    clf_by_ties = False # False
    clf_by_cnn = False

    if clf_by_cnn:
        if_naccl_clf = True
        use_text_cnn_method = True # False means that we use the generic f-cnn for the classification
        print(f"if_naccl_clf:{if_naccl_clf}, use_text_cnn_method:{use_text_cnn_method}")

    # 2 epoches for sanity check
    clf_epoches = 100 # after 2000: 400 is fine # 50: good performance # it seems that after 100 in the current dimension setting, the performance drops, or 50 can be ok
    clf_lr = 0.1 # for twitter 0.1 is better! previously, we normally use 0.01, for gossip, 0.1 is better for fast convergence.
else:
    clf_by_rnn = True
    clf_by_ties = False # True, False
    clf_by_cnn = False

    if clf_by_rnn:
        pretrained_clf_path = pretrained_rnn_clf #  + 'clf_pretrain_{}_sl{}.pt'.format(run_model, max_seq_len)
    if clf_by_ties:
        pretrained_clf_path = pretrained_ties_clf
if if_unbalance:
    clf_model_name = f"{int(clf_by_rnn)}{int(clf_by_ties)}{int(clf_by_cnn)}weight{weights[1]}"
else:
    clf_model_name = f"{int(clf_by_rnn)}{int(clf_by_ties)}{int(clf_by_cnn)}"
if_sav_clf = False # True, False

batch_size = 32 # in data5, real data, 64 is oom error, then 32, 16
# ==== pretrain MLE part ====:
if_use_context_attention_aware = True # False True, only True, when stance-aware setting, it will change the hidden dimension, ablation study, it is false when malcom
# check the run_relgan.py file for the details
if_pretrain_mle = True # True: whether we pretrain the LM False:
MLE_train_epoch = 100 # 200 # 150 # 5000, 2 for sanity check
if if_pretrain_mle is True:
    if_previous_pretrain_model = False # True:the relgan solution: random initial state  False: text summarization technique
    if_sav_pretrain = False
if_use_saved_gen = False # sometimes, we just train clf as the base one and we do not need the gen and we choose two Falses
# used when if cfg.gen_pretrain and cfg.if_use_gen:
# ==== possible question ====:
# ======== for the existing pretrained model ========
# stance_aware_ours_all_rnn: should be with rnn?? but, no stance
if if_use_context_attention_aware:
    # 1) when we build the best model, we use stance_aware model
    pretrained_gen_path_used = pretrained_gen_paths["stance_aware"] # stance_aware  stance_aware_ours_all_ties
else:
    # 0) when we do the ablation study, we should use pure_mle
    pretrained_gen_path_used = pretrained_gen_paths["pure_mle"] # without four tasks: pure_mle with 4 tasks: pure_ours_all_rnn
# ==== more generic approach ====:
model_selection = "malcom" # malcom, ours
pretrained_gen_path_used = pretrained_gen_paths.get(model_selection, None)
# ======== end ========
# ==== adv part ====
# adv_training global setting
if_pretrain_mle_in_adv = True # True, False
ADV_train_epoch = 100 # 3000 # 3000 # 2 for sanity check
if_adv_training = True # False True

if if_adv_training is False:
    if_adv_gan, if_adv_attack, if_adv_recency, if_adv_relevancy, if_adv_stance = False, False, False, False, False
else:
    attack_lr = 1e-5 # value for the sanity check now
    recency_lr = 1e-6 # -5 no changes at all
    num_recent_posts = 3 # 3, add it to 5
    relevancy_lr = 1e-6
    if_sav_adv = False
    # adv_xx the submodule for the testing
    # mode 0: True, False, False, False
    # mode 1: False, True, False, False
    # mode 2: False, False, True, False
    # mode 3: False, False, False, True
    # mode 4: True, True, True, True
    # mode 5 (malcom): True, True, False, True
    if_adv_gan, if_adv_attack, if_adv_recency, if_adv_relevancy, if_adv_stance = True, True, True, True, False
# model_name = f"{int(cfg.if_adv_gan)}{int(cfg.if_adv_attack)}{int(cfg.if_adv_recency)}{int(cfg.if_adv_relevancy)}"
model_name = f"{int(if_adv_gan)}{int(if_adv_attack)}{int(if_adv_recency)}{int(if_adv_relevancy)}"

#######################################
# #### when we in the testing mode ####

if_just_transfer_no_write = True # True: write text in txt for human evaluation; False:

if_test_baseline = False # True False # text generator testing
if_test_our_generator = True # text generator testing

if_black_box_test_rnn = False
if_black_box_test_ties = False
if if_black_box_test_rnn or if_black_box_test_ties:
    assert clf_by_rnn == True

#######################################################
# ==== standard solutions ====:
compared_methods = {
    0:"copycat",
    1:"hotflip",
    2:"uniAdBugger",
    3:"textBugger",
}

# ==== end bing's setting ====
# ==== RevisedRelGan ====
max_len_seq_lstm = 20 # the number of edits(posts)

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


# by bing: another approach
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
# bing bing
#  https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = pick_gpu_lowest_memory()
else:
    device = -1

# comment by bing
# if torch.cuda.is_available() and torch.cuda.device_count() > 0:
#     os.system('nvidia-smi -q -d Utilization > gpu')
#     with open('gpu', 'r') as _tmpfile:
#         # revised by bing, previously, we choose by utility, key word to Memory, not correct
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
    # revised by bing for some checking
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
# bing # change t
# pretrained_clf_path = pretrain_root + 'clf_pretrain_{}_sl{}.pt'.format(run_model, max_seq_len)
# clf_pretrain_relganRevised_lstm_sl31_sn1000.pt -> clf_pretrain_relganRevised_sl31.pt

signal_file = 'run_signal.txt'

tips = ''

if samples_num == 5000 or samples_num == 2000:
    assert 'c' in run_model, 'warning: samples_num={}, run_model={}'.format(samples_num, run_model)

# TODO: we should be able to have advanced methods, like dict setting
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

    # comment by bing: Jan. 9th, since it is repeated
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
    # bing
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
