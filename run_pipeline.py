import argparse
import math
import subprocess
import sys
from datetime import datetime


def run_cmd(cmd, arguments_dict):
    for k, v in arguments_dict.items():
        cmd += f" --{k}={v}"

    print("\nrunning cmd: ", cmd)
    start = datetime.now()
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

    end = datetime.now()
    print("running used time:{}\n".format(end - start))


val_num_beams = 1
test_num_beams = 50

train_batch_size = 1024
eval_batch_size = math.floor(train_batch_size // val_num_beams)
test_batch_size = math.floor(train_batch_size // test_num_beams)

arguments_dict = {
    'dataset': 'Disco',
    'run_time': None,
    'train_batch_size': 256,
    'eval_batch_size': 32,
    'test_batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 0.0,
    'warmup_ratio': 0.1,
    'lr_schedule': 'cosine',
    'epochs': 5,
    'max_users': 1000000,
    'eval_interval': 1,
    'patience': None,
    'val_topk': '[5,10]',
    'val_metric': 'ndcg@5',
    'topk': '[5,10]',
    'metrics': "['ndcg','recall','cr']",
    'metadata': 'embedding',
    'split': 'last_out',
    'seq_size': 4,
    'disco_raw_rows': True,
    'train_without_validation': True,
    'skip_validation': True,
    'test_eval_interval': 10,
}


def run_tokenizer(category):
    # RQVAE2 tokenizer config (rqvae_* keys; see genrec/tokenizers/RQVAE2/config.yaml).
    token_paras = {
        'category': category,
        'tokenizer_name': 'RQVAE2',
        'sent_emb_pca': -1,
        'vq_n_codebooks': 4,
        'vq_codebook_size': 256,
        'rqvae_hidden_sizes': '[512,256]',
        'rqvae_e_dim': 64,
        'rqvae_dropout': 0.0,
        'rqvae_lr': 5e-5,
        'rqvae_l2': 1e-4,
        'rqvae_batch_size': 2048,
        'rqvae_epoch': 10000,
        'rqvae_verbose': 50,
        'rqvae_beta': 0.25,
        'rqvae_sk_epsilon': 0.003,
        'rqvae_sk_iters': 10,
        'rqvae_quant_weight': 1,
        'kmeans_init': True,
        'kmeans_iters': 100,
        'rqvae_save_method': 'collision',
    }
    arguments_dict.update(token_paras)

    arguments_dict['run_time'] = datetime.now().strftime(r"%Y%m%d-%H%M%S")

    cmd = f'"{sys.executable}" -u main_token.py'

    run_cmd(cmd, arguments_dict)

    return arguments_dict['run_time']


def run_model(category, sem_id_time):
    if category == 'PolitiFact':
        model_paras = {
            'category': category,
            'model': 'CreGR',
            'tokenizer_name': 'RQVAE2',
            'sem_ids_path': f"{sem_id_time}.sem_ids",
            'epochs': 500,
            'train_batch_size': 4096,
            'eval_batch_size': 32,
            'test_batch_size': 32,
            'lr': 0.0005,
            'weight_decay': 0.05,
            'max_item_seq_len': 4,
            'dropout_rate': 0.1,
            'd_model': 256,
            'mlp_ratio': 4,
            'n_layers': 2,
            'n_heads': 8,
            'n_kv_heads': 8,
            'his_mask_w': 6,
            'gen_steps': 4,
            'val_num_beams': val_num_beams,
            'num_beams': test_num_beams,
            'temperature': 1.0,
        }
    elif category == 'MHMisinfo':
        model_paras = {
            'category': category,
            'model': 'CreGR',
            'tokenizer_name': 'RQVAE2',
            'sem_ids_path': f"{sem_id_time}.sem_ids",
            'epochs': 500,
            'train_batch_size': 512,
            'eval_batch_size': 32,
            'test_batch_size': 32,
            'lr': 0.0005,
            'weight_decay': 0.05,
            'max_item_seq_len': 4,
            'dropout_rate': 0.1,
            'd_model': 256,
            'mlp_ratio': 4,
            'n_layers': 2,
            'n_heads': 8,
            'n_kv_heads': 8,
            'his_mask_w': 5,
            'gen_steps': 4,
            'val_num_beams': val_num_beams,
            'num_beams': test_num_beams,
            'temperature': 1.0,
        }
    elif category == 'GossipCop':
        model_paras = {
            'category': category,
            'model': 'CreGR',
            'tokenizer_name': 'RQVAE2',
            'sem_ids_path': f"{sem_id_time}.sem_ids",
            'epochs': 500,
            'train_batch_size': 256,
            'eval_batch_size': 256,
            'test_batch_size': 256,
            'lr': 0.003,
            'weight_decay': 0.001,
            'max_item_seq_len': 4,
            'dropout_rate': 0.1,
            'd_model': 256,
            'mlp_ratio': 4,
            'n_layers': 6,
            'n_heads': 8,
            'n_kv_heads': 8,
            'his_mask_w': 5,
            'gen_steps': 4,
            'val_num_beams': val_num_beams,
            'num_beams': test_num_beams,
            'temperature': 1.0,
        }
    else:
        raise NotImplementedError(f'Unknown category: {category}')
    arguments_dict.update(model_paras)

    arguments_dict['run_time'] = datetime.now().strftime(r"%Y%m%d-%H%M%S")

    cmd = f'"{sys.executable}" -u main.py'

    run_cmd(cmd, arguments_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tokenize (RQVAE2) and/or train CreGR for one category.')
    parser.add_argument(
        '--category',
        default='PolitiFact',
        help='PolitiFact | MHMisinfo | GossipCop | ')
    parser.add_argument(
        '--sem_ids_path',
        default=None,
        help='Existing "<run_time>.sem_ids" to reuse; if omitted, a fresh '
        'RQVAE2 tokenizer is trained first.')
    args = parser.parse_args()

    if args.sem_ids_path:
        # Reuse existing semantic IDs, skip tokenizer training.
        sem_id_time = args.sem_ids_path.replace('.sem_ids', '')
    else:
        sem_id_time = run_tokenizer(args.category)

    run_model(args.category, sem_id_time)
