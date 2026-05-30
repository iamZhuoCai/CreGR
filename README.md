# CreGR

Official implementation of **Generate What You Can Trust: Content Credibility in Generative Recommenders**.

## Requirements

- Python **3.11** recommended
- **CUDA 12.4** GPU (recommended for RQ-VAE tokenizer training)
- Disk space for Disco cache (~1–3 GB per category depending on subset)

Install dependencies:

```bash
cd Paper6/CreGR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the core stack:

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.6.0+cu124 | CUDA 12.4 wheels via `--extra-index-url` |
| transformers | 4.55.0 | |
| accelerate | 1.10.0 | |
| datasets | ≥2.14, <4 | |
| sentence_transformers | 5.1.0 | |
| scikit_learn | 1.7.2 | k-means init for RQ-VAE codebooks |
| numpy, PyYAML, tqdm, Requests | pinned | |
| huggingface_hub | ≥0.20 | Disco auto-download (raw data + collaborative embeddings) |

---


---
### Tokenization outputs (`main_token.py`, model `TokenModel`)

`{Category}` ∈ `PolitiFact` | `GossipCop` | `MHMisinfo`. With `--max_item_seq_len=4` and `--tokenizer_name=RQVAE2`:

```
output/{Category}/TokenModel_Seq-4_RQVAE2/
├── logs/{run_time}.log                          # program log (primary)
├── sem_ids/{run_time}.sem_ids                   # semantic IDs for CreGR
├── sem_ids/{run_time}.sem_ids.credreg_meta.json # credibility split metadata
└── ckpts/
    ├── {run_time}.pth                           # weights for sem_id generation
    └── {run_time}.ep{epoch}.ckpt.pth            # training checkpoints
```

CreGR loads sem_ids from the **same** `TokenModel_Seq-{max_item_seq_len}_RQVAE2/sem_ids/` tree via `--sem_ids_path={run_time}.sem_ids`.

### CreGR outputs (`main.py`, model `CreGR`)

```
output/Disco_{Category}/CreGR_Seq-4_RQVAE2/
├── logs/{run_time}.log
├── ckpts/                                       # best / saved model weights
└── results/{run_time}.json                      # test metrics
```

```bash
RUN_TIME=pf_rqvae2_$(date +%Y%m%d-%H%M%S)
mkdir -p logs
nohup python -u main_token.py --category=PolitiFact --run_time=${RUN_TIME} ... \
  > logs/nohup_tok_${RUN_TIME}.txt 2>&1 &
```

---

## Training Overview

1. **Tokenization** (`main_token.py`) — train RQVAE2 to assign semantic IDs (`.sem_ids`) to items
2. **Generation** (`main.py`, model `CreGR`) — train the discrete diffusion recommender on those semantic IDs

See [Output Layout](#output-layout) for full directory structure. Key artifacts:

```
output/Disco_{Category}/TokenModel_Seq-4_RQVAE2/sem_ids/{run_time}.sem_ids
output/Disco_{Category}/CreGR_Seq-4_RQVAE2/ckpts/
output/Disco_{Category}/CreGR_Seq-4_RQVAE2/results/
```

**One-shot pipeline**

```bash
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --category=PolitiFact
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --category=MHMisinfo
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --category=GossipCop
```


### Shared tokenizer arguments (RQVAE2)

Defaults from `genrec/tokenizers/RQVAE2/config.yaml`. Key settings used in our Disco experiments:

- `rqvae_e_dim=64`, `rqvae_lr=5e-5`, `sent_emb_pca=128`
- `cred_reg_inter_weight=100` (credibility-aware regularization)
- `rqvae_epoch=10000`, `vq_n_codebooks=4`, `vq_codebook_size=256`
- `--max_item_seq_len=4` (must match CreGR downstream)

Set a unique `--run_time` (e.g. timestamp); semantic IDs are saved as `{run_time}.sem_ids`.

---

## PolitiFact

### Stage 1 — Tokenization

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_token.py \
  --dataset=Disco \
  --category=PolitiFact \
  --tokenizer_name=RQVAE2 \
  --run_time=pf_rqvae2_$(date +%Y%m%d-%H%M%S) \
  --metadata=embedding \
  --max_item_seq_len=4 \
  --disco_raw_rows=True \
  --train_without_validation=True \
  --skip_validation=True \
  --sent_emb_pca=128 \
  --rqvae_e_dim=64 \
  --rqvae_lr=5e-5 \
  --rqvae_epoch=10000 \
  --rqvae_verbose=50 \
  --cred_reg_inter_weight=100
```

### Stage 2 — CreGR

Replace `{run_time}` with the timestamp from stage 1:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --model=CreGR \
  --dataset=Disco \
  --category=PolitiFact \
  --tokenizer_name=RQVAE2 \
  --sem_ids_path={run_time}.sem_ids \
  --run_time=pf_cregr_$(date +%Y%m%d-%H%M%S) \
  --max_item_seq_len=4 \
  --epochs=500 \
  --train_batch_size=4096 \
  --eval_batch_size=32 \
  --test_batch_size=32 \
  --lr=5e-4 \
  --weight_decay=0.05 \
  --d_model=256 \
  --n_layers=2 \
  --n_heads=8 \
  --n_kv_heads=8 \
  --his_mask_w=6 \
  --gen_steps=4 \
  --num_beams=50 \
  --last_token_cred_mask=known_uncredible_only \
  --known_uncredible_mask_mode=soft
```

---

## MHMisinfo

### Stage 1 — Tokenization

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_token.py \
  --dataset=Disco \
  --category=MHMisinfo \
  --tokenizer_name=RQVAE2 \
  --run_time=mh_rqvae2_$(date +%Y%m%d-%H%M%S) \
  --metadata=embedding \
  --max_item_seq_len=4 \
  --disco_raw_rows=True \
  --train_without_validation=True \
  --skip_validation=True \
  --sent_emb_pca=128 \
  --rqvae_e_dim=64 \
  --rqvae_lr=5e-5 \
  --rqvae_epoch=10000 \
  --rqvae_verbose=50 \
  --cred_reg_inter_weight=100
```

### Stage 2 — CreGR

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --model=CreGR \
  --dataset=Disco \
  --category=MHMisinfo \
  --tokenizer_name=RQVAE2 \
  --sem_ids_path={run_time}.sem_ids \
  --run_time=mh_cregr_$(date +%Y%m%d-%H%M%S) \
  --max_item_seq_len=4 \
  --epochs=500 \
  --train_batch_size=512 \
  --eval_batch_size=32 \
  --test_batch_size=32 \
  --lr=5e-4 \
  --weight_decay=0.05 \
  --d_model=256 \
  --n_layers=2 \
  --n_heads=8 \
  --n_kv_heads=8 \
  --his_mask_w=5 \
  --gen_steps=4 \
  --num_beams=50 \
  --last_token_cred_mask=known_uncredible_only \
  --known_uncredible_mask_mode=soft
```

---

## GossipCop

### Stage 1 — Tokenization

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_token.py \
  --dataset=Disco \
  --category=GossipCop \
  --tokenizer_name=RQVAE2 \
  --run_time=gc_rqvae2_$(date +%Y%m%d-%H%M%S) \
  --metadata=embedding \
  --max_item_seq_len=4 \
  --disco_raw_rows=True \
  --train_without_validation=True \
  --skip_validation=True \
  --sent_emb_pca=128 \
  --rqvae_e_dim=64 \
  --rqvae_lr=5e-5 \
  --rqvae_epoch=10000 \
  --rqvae_verbose=50 \
  --cred_reg_inter_weight=100
```

### Stage 2 — CreGR

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --model=CreGR \
  --dataset=Disco \
  --category=GossipCop \
  --tokenizer_name=RQVAE2 \
  --sem_ids_path={run_time}.sem_ids \
  --run_time=gc_cregr_$(date +%Y%m%d-%H%M%S) \
  --max_item_seq_len=4 \
  --epochs=500 \
  --train_batch_size=256 \
  --eval_batch_size=256 \
  --test_batch_size=256 \
  --lr=0.003 \
  --weight_decay=0.001 \
  --d_model=256 \
  --n_layers=6 \
  --n_heads=8 \
  --n_kv_heads=8 \
  --his_mask_w=5 \
  --gen_steps=4 \
  --num_beams=50 \
  --last_token_cred_mask=known_uncredible_only \
  --known_uncredible_mask_mode=soft
```

---

## Citation

```bibtex
@article{shi2025cregr,
  title={CreGR: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation},
  author={Shi, Teng and Shen, Chenglei and Yu, Weijie and Nie, Shen and Li, Chongxuan and Zhang, Xiao and He, Ming and Han, Yan and Xu, Jun},
  journal={arXiv preprint arXiv:2511.06254},
  year={2025}
}
```
