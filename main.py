from __future__ import print_function

import argparse

import config as cfg
from utils.text_process import load_test_dict, text_process


def program_config(parser):
    # Program
    parser.add_argument('--if_test', default=cfg.if_test, type=int)
    parser.add_argument('--run_model', default=cfg.run_model, type=str)
    parser.add_argument('--k_label', default=cfg.k_label, type=int)
    parser.add_argument('--dataset', default=cfg.dataset, type=str)
    parser.add_argument('--model_type', default=cfg.model_type, type=str)
    parser.add_argument('--loss_type', default=cfg.loss_type, type=str)
    parser.add_argument('--mu_type', default=cfg.mu_type, type=str)
    parser.add_argument('--eval_type', default=cfg.eval_type, type=str)
    parser.add_argument('--d_type', default=cfg.d_type, type=str)
    parser.add_argument('--if_real_data', default=cfg.if_real_data, type=int)
    parser.add_argument('--cuda', default=cfg.CUDA, type=int)
    parser.add_argument('--device', default=cfg.device, type=int)
    parser.add_argument('--devices', default=cfg.devices, type=str)
    parser.add_argument('--shuffle', default=cfg.data_shuffle, type=int)
    parser.add_argument('--gen_init', default=cfg.gen_init, type=str)
    parser.add_argument('--dis_init', default=cfg.dis_init, type=str)

    # CatGAN
    parser.add_argument('--n_parent', default=cfg.n_parent, type=int)
    parser.add_argument('--eval_b_num', default=cfg.eval_b_num, type=int)
    parser.add_argument('--lambda_fq', default=cfg.lambda_fq, type=float)
    parser.add_argument('--lambda_fd', default=cfg.lambda_fd, type=float)
    parser.add_argument('--d_out_mean', default=cfg.d_out_mean, type=int)
    parser.add_argument('--freeze_dis', default=cfg.freeze_dis, type=int)
    parser.add_argument('--freeze_clas', default=cfg.freeze_clas, type=int)
    parser.add_argument('--use_all_real_fake', default=cfg.use_all_real_fake, type=int)
    parser.add_argument('--use_population', default=cfg.use_population, type=int)

    # Basic Train
    parser.add_argument('--samples_num', default=cfg.samples_num, type=int)
    parser.add_argument('--vocab_size', default=cfg.vocab_size, type=int)
    parser.add_argument('--mle_epoch', default=cfg.MLE_train_epoch, type=int)
    parser.add_argument('--clas_pre_epoch', default=cfg.PRE_clas_epoch, type=int)
    parser.add_argument('--adv_epoch', default=cfg.ADV_train_epoch, type=int)
    parser.add_argument('--inter_epoch', default=cfg.inter_epoch, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--max_seq_len', default=cfg.max_seq_len, type=int)
    parser.add_argument('--start_letter', default=cfg.start_letter, type=int)
    parser.add_argument('--padding_idx', default=cfg.padding_idx, type=int)
    parser.add_argument('--gen_lr', default=cfg.gen_lr, type=float)
    parser.add_argument('--gen_adv_lr', default=cfg.gen_adv_lr, type=float)
    parser.add_argument('--dis_lr', default=cfg.dis_lr, type=float)
    parser.add_argument('--clip_norm', default=cfg.clip_norm, type=float)
    parser.add_argument('--pre_log_step', default=cfg.pre_log_step, type=int)
    parser.add_argument('--adv_log_step', default=cfg.adv_log_step, type=int)
    parser.add_argument('--train_data', default=cfg.train_data, type=str)
    parser.add_argument('--test_data', default=cfg.test_data, type=str)
    parser.add_argument('--temp_adpt', default=cfg.temp_adpt, type=str)
    parser.add_argument('--evo_temp_step', default=cfg.evo_temp_step, type=int)
    parser.add_argument('--temperature', default=cfg.temperature, type=int)
    parser.add_argument('--ora_pretrain', default=cfg.oracle_pretrain, type=int)
    parser.add_argument('--gen_pretrain', default=cfg.gen_pretrain, type=int)
    parser.add_argument('--dis_pretrain', default=cfg.dis_pretrain, type=int)

    # Generator
    parser.add_argument('--adv_g_step', default=cfg.ADV_g_step, type=int)
    parser.add_argument('--rollout_num', default=cfg.rollout_num, type=int)
    parser.add_argument('--gen_embed_dim', default=cfg.gen_embed_dim, type=int)
    parser.add_argument('--gen_hidden_dim', default=cfg.gen_hidden_dim, type=int)
    parser.add_argument('--goal_size', default=cfg.goal_size, type=int)
    parser.add_argument('--step_size', default=cfg.step_size, type=int)
    parser.add_argument('--mem_slots', default=cfg.mem_slots, type=int)
    parser.add_argument('--num_heads', default=cfg.num_heads, type=int)
    parser.add_argument('--head_size', default=cfg.head_size, type=int)

    # Discriminator
    parser.add_argument('--d_step', default=cfg.d_step, type=int)
    parser.add_argument('--d_epoch', default=cfg.d_epoch, type=int)
    parser.add_argument('--adv_d_step', default=cfg.ADV_d_step, type=int)
    parser.add_argument('--adv_d_epoch', default=cfg.ADV_d_epoch, type=int)
    parser.add_argument('--dis_embed_dim', default=cfg.dis_embed_dim, type=int)
    parser.add_argument('--dis_hidden_dim', default=cfg.dis_hidden_dim, type=int)
    parser.add_argument('--num_rep', default=cfg.num_rep, type=int)

    # Metrics
    parser.add_argument('--use_nll_oracle', default=cfg.use_nll_oracle, type=int)
    parser.add_argument('--use_nll_gen', default=cfg.use_nll_gen, type=int)
    parser.add_argument('--use_nll_div', default=cfg.use_nll_div, type=int)
    parser.add_argument('--use_bleu', default=cfg.use_bleu, type=int)
    parser.add_argument('--use_self_bleu', default=cfg.use_self_bleu, type=int)
    parser.add_argument('--use_clas_acc', default=cfg.use_clas_acc, type=int)
    parser.add_argument('--use_ppl', default=cfg.use_ppl, type=int)

    # Log
    parser.add_argument('--log_file', default=cfg.log_filename, type=str)
    parser.add_argument('--save_root', default=cfg.save_root, type=str)
    parser.add_argument('--signal_file', default=cfg.signal_file, type=str)
    parser.add_argument('--tips', default=cfg.tips, type=str)

    return parser


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opt = parser.parse_args()

    if opt.if_real_data:
        # check this issue for the details:
        # https://github.com/williamSYSU/TextGAN-PyTorch/issues/25

        opt.max_seq_len, opt.vocab_size = text_process('dataset/' + opt.dataset + '.txt')
        print("opt.max_seq_len (the length of one sentence/review/post): ", opt.max_seq_len)
        cfg.extend_vocab_size = len(load_test_dict(opt.dataset)[0])  # init classifier vocab_size

        word2idx_dict, idx2word_dict = load_test_dict(opt.dataset)

        if cfg.if_context:
            txt_path = 'dataset/testdata/{}_context.txt'.format(opt.dataset)
            from utils.text_process import update_dictionaries_by_txt_file
            word2idx_dict, _ = update_dictionaries_by_txt_file(txt_path,
                                           word2idx_dict, idx2word_dict, category=opt.dataset)
            cfg.extend_vocab_size = len(word2idx_dict)
    cfg.init_param(opt)
    opt.save_root = cfg.save_root
    opt.train_data = cfg.train_data
    opt.test_data = cfg.test_data

    # ### here, we will manually set the device
    from config import device
    import torch
    if device == torch.device("cpu"):
        cfg.CUDA = int(False)  # int(True) # False
    else:
        cfg.CUDA = int(True)
    # #### by bing

    # ===Dict===
    # ======== PETGEN class ========
    if cfg.if_real_data:
        from instructor.real_data.relgan_revised_instructor import RelGANInstructorRevised
    else:
        pass

    instruction_dict = {
        'relganRevised': RelGANInstructorRevised,
    }

    inst = instruction_dict[cfg.run_model](opt)
    if not cfg.if_test:
        inst._run()
    else:
        inst._test()
