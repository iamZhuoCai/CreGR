import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset

from genrec.dataset import AbstractDataset


class Disco(AbstractDataset):

    _REQUIRED_PICKLE_FILES = (
        'train_data.df',
        'test_data.df',
        'data_statis.df',
    )

    # HF uses both spellings across categories (see Credible_Content_Rec repo).
    _PRETRAIN_EMBEDDING_FILENAMES = (
        'pretrained_item_embeddings.npy',
        'pretrain_item_embeddings.npy',
        'item_pretrain_embeddings.npy',
    )

    def __init__(self, config: dict):
        super(Disco, self).__init__(config)

        self.category = config['category']
        self.log(f'[DATASET] Disco for category: {self.category}')

        self.disco_root = os.path.join(config['cache_dir'], 'Disco_v2')
        self.cache_dir = os.path.join(self.disco_root, 'data', self.category)
        self._download_and_process_raw()

    def _has_embedding_files(self) -> bool:
        title_path = os.path.join(self.cache_dir,
                                  'news_title_embeddings_llama.npy')
        if os.path.exists(title_path):
            for descr_name in (
                    'news_descri_embeddings_llama.npy',
                    'news_desci_embeddings_llama.npy',
            ):
                if os.path.exists(os.path.join(self.cache_dir, descr_name)):
                    return True
            return False
        return os.path.exists(
            os.path.join(self.cache_dir, 'news_embeddings_llama.npy'))

    def _has_required_raw_data(self) -> bool:
        for name in self._REQUIRED_PICKLE_FILES:
            if not os.path.exists(os.path.join(self.cache_dir, name)):
                return False
        if not os.path.exists(
                os.path.join(self.cache_dir, 'credible_items.npy')):
            return False
        return self._has_embedding_files()

    def _hf_repo(self) -> str:
        return self.config.get(
            'disco_hf_repo', 'anony-user-2025/Credible_Content_Rec')

    def _hf_allow_patterns_for_category(self) -> list[str]:
        allow_patterns = self.config.get('disco_hf_allow_patterns')
        if allow_patterns:
            return list(allow_patterns)
        return [f'data/{self.category}/*']

    def _hf_pretrain_patterns_for_category(self) -> list[str]:
        """Exact HF paths for collaborative / pretrain embeddings per category."""
        patterns = self.config.get('disco_hf_pretrain_patterns')
        if patterns:
            return list(patterns)
        prefix = f'data/{self.category}'
        return [f'{prefix}/{name}' for name in self._PRETRAIN_EMBEDDING_FILENAMES]

    def _snapshot_download(self, allow_patterns: list[str], purpose: str) -> None:
        if self._has_pretrain_embeddings() and purpose == 'pretrain':
            return
        if self._has_required_raw_data() and purpose == 'raw':
            return

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                'Disco cache is missing and huggingface_hub is not installed. '
                'Run: pip install huggingface_hub') from exc

        hf_repo = self._hf_repo()
        self.log(
            f'[DATASET] {purpose}: downloading from Hugging Face repo {hf_repo} '
            f'(patterns={allow_patterns}) -> {self.disco_root}')

        hf_endpoint = self.config.get('disco_hf_endpoint')
        prev_endpoint = os.environ.get('HF_ENDPOINT')
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = str(hf_endpoint)

        try:
            with self.accelerator.main_process_first():
                snapshot_download(
                    repo_id=hf_repo,
                    repo_type='dataset',
                    local_dir=self.disco_root,
                    allow_patterns=allow_patterns,
                )
        finally:
            if hf_endpoint:
                if prev_endpoint is None:
                    os.environ.pop('HF_ENDPOINT', None)
                else:
                    os.environ['HF_ENDPOINT'] = prev_endpoint

    def _download_raw_data(self) -> None:
        if self._has_required_raw_data():
            return

        self._snapshot_download(
            self._hf_allow_patterns_for_category(), purpose='raw')

        if not self._has_required_raw_data():
            raise FileNotFoundError(
                f'Disco raw data still missing under {self.cache_dir} after '
                f'downloading {self._hf_repo()}. Expected files include '
                f'{", ".join(self._REQUIRED_PICKLE_FILES)}, credible_items.npy, '
                'and title+description or single LLaMA embeddings.')

        self.log(f'[DATASET] Disco raw data ready at {self.cache_dir}')

    def _resolve_pretrain_embedding_path(self,
                                         include_package: bool = True) -> str | None:
        """Return first existing collaborative/pretrain embedding file path."""
        search_dirs = [self.cache_dir, os.path.join(self.cache_dir, 'processed')]
        for directory in search_dirs:
            for name in self._PRETRAIN_EMBEDDING_FILENAMES:
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    return path

        if include_package:
            pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   self.category)
            for name in self._PRETRAIN_EMBEDDING_FILENAMES:
                path = os.path.join(pkg_dir, name)
                if os.path.exists(path):
                    return path
        return None

    def _has_pretrain_embeddings(self) -> bool:
        return self._resolve_pretrain_embedding_path(include_package=False) is not None

    def _ensure_pretrain_embeddings(self) -> None:
        """Download collaborative embeddings from HF when cache lacks them."""
        if self._resolve_pretrain_embedding_path(include_package=True) is not None:
            return
        if not self.config.get('disco_auto_download_pretrain', True):
            return

        self.log(
            f'[DATASET] Collaborative/pretrain embeddings not found under '
            f'{self.cache_dir}; fetching from Hugging Face '
            f'({self._hf_repo()})...')
        self._snapshot_download(
            self._hf_pretrain_patterns_for_category(), purpose='pretrain')

        if self._resolve_pretrain_embedding_path(include_package=False) is None:
            self.log(
                '[DATASET] Collaborative embeddings still missing after HF '
                'download; tokenizer will skip pretrain concat unless a packaged '
                f'fallback exists under genrec/datasets/Disco/{self.category}/.')

    def _download_and_process_raw(self):
        self._download_raw_data()
        processed_path = os.path.join(self.cache_dir, 'processed')

        # Load raw data
        self.train_df = pd.read_pickle(
            os.path.join(self.cache_dir, 'train_data.df'))
        self.test_df = pd.read_pickle(
            os.path.join(self.cache_dir, 'test_data.df'))
        data_statis = pd.read_pickle(
            os.path.join(self.cache_dir, 'data_statis.df'))

        self.n_items_raw = int(data_statis['item_num'].iloc[0])
        self.seq_size = int(data_statis['seq_size'].iloc[0])
        self.log(f'[DATASET] Items: {self.n_items_raw}, Seq size: {self.seq_size}')

        # Load pre-computed embeddings and credibility labels
        # Support both old (single embedding) and new (title + description) formats
        title_path = os.path.join(self.cache_dir, 'news_title_embeddings_llama.npy')
        if os.path.exists(title_path):
            # New format: concat title + description
            title_emb = np.load(title_path)
            # Handle MHMisinfo typo in filename
            descr_path = os.path.join(self.cache_dir, 'news_descri_embeddings_llama.npy')
            if not os.path.exists(descr_path):
                descr_path = os.path.join(self.cache_dir, 'news_desci_embeddings_llama.npy')
            descr_emb = np.load(descr_path)
            self.item_embeddings = np.concatenate(
                [title_emb.astype(np.float32), descr_emb.astype(np.float32)], axis=1)
            self.log(f'[DATASET] Using concat title+descr embeddings: {self.item_embeddings.shape}')
        else:
            self.item_embeddings = np.load(
                os.path.join(self.cache_dir, 'news_embeddings_llama.npy'))
            self.log(f'[DATASET] Using single embedding: {self.item_embeddings.shape}')
        self.credible_items = np.load(
            os.path.join(self.cache_dir, 'credible_items.npy'))
        self._ensure_pretrain_embeddings()
        self.pretrain_item_embeddings = self._load_pretrain_item_embeddings()
        self.log(
            f'[DATASET] Embeddings shape: {self.item_embeddings.shape}, '
            f'Credible items: {len(self.credible_items)}')

        # Build ID mapping
        os.makedirs(processed_path, exist_ok=True)
        id_mapping_file = os.path.join(processed_path, 'id_mapping.json')
        if os.path.exists(id_mapping_file):
            self.log(f'[DATASET] Loading id mapping from {id_mapping_file}')
            self.id_mapping = json.load(open(id_mapping_file, 'r'))
        else:
            self._build_id_mapping(self.train_df, self.test_df)
            with open(id_mapping_file, 'w') as f:
                json.dump(self.id_mapping, f)

        # User-level sequences for stats / VQ item mask (no sliding-window merge in raw-row mode)
        all_seqs_file = os.path.join(processed_path, 'all_item_seqs.json')
        if self.config.get('disco_raw_rows', False):
            self.all_item_seqs = {}
            rid = 0
            for idx in range(len(self.train_df)):
                seq = self._raw_row_item_seq(self.train_df.iloc[idx])
                self.all_item_seqs[str(rid)] = seq
                rid += 1
            avg = (np.mean([len(v) for v in self.all_item_seqs.values()])
                   if self.all_item_seqs else 0.0)
            self.log(f'[DATASET] disco_raw_rows: {len(self.all_item_seqs)} train rows '
                     f'(one sequence each), avg len: {avg:.1f}')
        elif os.path.exists(all_seqs_file):
            self.log(f'[DATASET] Loading sequences from {all_seqs_file}')
            self.all_item_seqs = json.load(open(all_seqs_file, 'r'))
        else:
            self.all_item_seqs = self._reconstruct_sequences_train_only(
                self.train_df)
            with open(all_seqs_file, 'w') as f:
                json.dump(self.all_item_seqs, f)

        if not self.config.get('disco_raw_rows', False):
            self.log(f'[DATASET] Users: {len(self.all_item_seqs)}, '
                     f'Avg seq len: {np.mean([len(v) for v in self.all_item_seqs.values()]):.1f}')

        # Build metadata
        self.item2meta = self._build_metadata()

    def _build_id_mapping(self, train_df, test_df):
        """Map Disco 0-indexed item IDs to 1-indexed (0 reserved for PAD)."""
        all_items = set()
        for df in [train_df, test_df]:
            for seq in df['seq']:
                all_items.update([i for i in seq if i != 0])
            all_items.update(df['next'].tolist())

        all_items = sorted(all_items)

        for item in all_items:
            str_item = str(item)
            if str_item not in self.id_mapping['item2id']:
                self.id_mapping['item2id'][str_item] = len(
                    self.id_mapping['item2id'])
                self.id_mapping['id2item'].append(str_item)

        self.log(f'[DATASET] Mapped {len(all_items)} items')

    @staticmethod
    def _raw_row_item_seq(row) -> list:
        """One DataFrame row: padded ``seq`` + ``next`` → ``hist + [next]`` as str ids.

        If ``seq`` is all zeros, keep the row: use ``[next]`` only (no history).
        """
        seq_items = list(row['seq'])
        nxt = int(row['next'])
        hist = [str(x) for x in seq_items if x != 0]
        return hist + [str(nxt)]

    def _fill_splits_from_raw_rows(self):
        """Train/val/test from raw rows only (each row = one ``item_seq``)."""
        datasets = {
            'train': {'user': [], 'item_seq': []},
            'val': {'user': [], 'item_seq': []},
            'test': {'user': [], 'item_seq': []},
        }
        uid = 0

        if self.config.get('train_without_validation', False):
            for idx in range(len(self.train_df)):
                seq = self._raw_row_item_seq(self.train_df.iloc[idx])
                datasets['train']['user'].append(str(uid))
                datasets['train']['item_seq'].append(seq)
                uid += 1
            for idx in range(len(self.test_df)):
                seq = self._raw_row_item_seq(self.test_df.iloc[idx])
                datasets['test']['user'].append(str(uid))
                datasets['test']['item_seq'].append(seq)
                uid += 1
        else:
            n = len(self.train_df)
            val_n = max(1, n // 5)
            cut = n - val_n
            for idx in range(cut):
                seq = self._raw_row_item_seq(self.train_df.iloc[idx])
                datasets['train']['user'].append(str(uid))
                datasets['train']['item_seq'].append(seq)
                uid += 1
            for idx in range(cut, n):
                seq = self._raw_row_item_seq(self.train_df.iloc[idx])
                datasets['val']['user'].append(str(uid))
                datasets['val']['item_seq'].append(seq)
                uid += 1
            for idx in range(len(self.test_df)):
                seq = self._raw_row_item_seq(self.test_df.iloc[idx])
                datasets['test']['user'].append(str(uid))
                datasets['test']['item_seq'].append(seq)
                uid += 1
        return datasets

    def _reconstruct_one_df(self, df):
        """Reconstruct full user sequences from a single sliding-window DataFrame."""
        sequences = []
        i = 0
        while i < len(df):
            seq_items = list(df.iloc[i]['seq'])
            next_item = int(df.iloc[i]['next'])
            full_seq = [x for x in seq_items if x != 0] + [next_item]

            j = i + 1
            while j < len(df):
                curr_seq = list(df.iloc[j]['seq'])
                prev_seq = list(df.iloc[j - 1]['seq'])
                prev_next = int(df.iloc[j - 1]['next'])
                expected = prev_seq[1:] + [prev_next]
                if curr_seq == expected:
                    full_seq.append(int(df.iloc[j]['next']))
                    j += 1
                else:
                    break

            str_seq = [str(item) for item in full_seq]
            sequences.append(str_seq)
            i = j

        return sequences

    def _reconstruct_sequences_train_only(self, train_df):
        """Only reconstruct from train_df for all_item_seqs (used by tokenizer)."""
        all_item_seqs = {}
        sequences = self._reconstruct_one_df(train_df)

        for uid, seq in enumerate(sequences):
            user_key = str(uid)
            all_item_seqs[user_key] = seq
            if user_key not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user_key] = len(
                    self.id_mapping['user2id'])
                self.id_mapping['id2user'].append(user_key)

        self.log(
            f'[DATASET] Reconstructed {len(all_item_seqs)} train user sequences')
        return all_item_seqs

    def split(self):
        """Use Disco's original train/test split instead of leave-one-out.

        Disco provides sliding-window data with (seq, next) pairs.

        If ``disco_raw_rows`` is True: **no** sliding-window reconstruction —
        each row of ``train_data.df`` / ``test_data.df`` is one sample
        (non-zero ``seq`` prefix + ``next``).

        If ``train_without_validation`` is True in config: use **all** of
        ``train_data.df`` for training (no hold-out from train), **all** of
        ``test_data.df`` for testing, and an empty validation split (no val
        loop / early stopping — see ``skip_validation`` in trainer).
        """
        if self.split_data is not None:
            return self.split_data

        if self.config.get('disco_raw_rows', False):
            datasets = self._fill_splits_from_raw_rows()
            for split_name in datasets:
                datasets[split_name] = Dataset.from_dict(datasets[split_name])
            self.log(f'[DATASET] Split sizes (raw rows) - '
                     f'train: {len(datasets["train"])}, '
                     f'val: {len(datasets["val"])}, '
                     f'test: {len(datasets["test"])}')
            self.split_data = datasets
            return self.split_data

        train_seqs = self._reconstruct_one_df(self.train_df)
        test_seqs = self._reconstruct_one_df(self.test_df)

        datasets = {
            'train': {'user': [], 'item_seq': []},
            'val': {'user': [], 'item_seq': []},
            'test': {'user': [], 'item_seq': []},
        }

        if self.config.get('train_without_validation', False):
            for uid, seq in enumerate(train_seqs):
                if len(seq) >= 3:
                    datasets['train']['user'].append(str(uid))
                    datasets['train']['item_seq'].append(seq[:-2])
            offset = len(train_seqs)
            for uid, seq in enumerate(test_seqs):
                if len(seq) >= 2:
                    datasets['test']['user'].append(str(offset + uid))
                    datasets['test']['item_seq'].append(seq)
        else:
            val_size = len(train_seqs) // 5
            val_seqs = train_seqs[-val_size:]
            train_seqs_final = train_seqs[:-val_size]

            for uid, seq in enumerate(train_seqs_final):
                if len(seq) >= 3:
                    datasets['train']['user'].append(str(uid))
                    datasets['train']['item_seq'].append(seq[:-2])

            for uid, seq in enumerate(val_seqs):
                if len(seq) >= 2:
                    datasets['val']['user'].append(
                        str(len(train_seqs_final) + uid))
                    datasets['val']['item_seq'].append(seq[:-1])

            for uid, seq in enumerate(test_seqs):
                if len(seq) >= 2:
                    datasets['test']['user'].append(str(len(train_seqs) + uid))
                    datasets['test']['item_seq'].append(seq)

        for split_name in datasets:
            datasets[split_name] = Dataset.from_dict(datasets[split_name])

        self.log(f'[DATASET] Split sizes - '
                 f'train: {len(datasets["train"])}, '
                 f'val: {len(datasets["val"])}, '
                 f'test: {len(datasets["test"])}')

        self.split_data = datasets
        return self.split_data

    def _build_metadata(self):
        """Build placeholder metadata for items."""
        item2meta = {}
        for item_str in self.id_mapping['item2id']:
            if item_str == '[PAD]':
                continue
            item2meta[item_str] = f'news_item_{item_str}'
        return item2meta

    def get_item_embeddings(self):
        """Return pre-computed LLaMA embeddings aligned with id_mapping."""
        n_items = len(self.id_mapping['item2id'])
        embed_dim = self.item_embeddings.shape[1]
        aligned_embs = np.zeros((n_items - 1, embed_dim), dtype=np.float32)

        for item_str, item_id in self.id_mapping['item2id'].items():
            if item_str == '[PAD]' or item_id == 0:
                continue
            raw_id = int(item_str)
            if raw_id < len(self.item_embeddings):
                aligned_embs[item_id - 1] = self.item_embeddings[raw_id].astype(
                    np.float32)

        return aligned_embs

    def _load_pretrain_item_embeddings(self):
        """Load collaborative/pretrain item embeddings (64-d) for tokenization."""
        path = self._resolve_pretrain_embedding_path(include_package=True)
        if path is None:
            self.log('[DATASET] pretrain_item_embeddings not found; skip pretrain '
                     'concat in tokenizer.')
            return None

        embs = np.load(path)
        self.log(
            f'[DATASET] Loaded collaborative/pretrain item embeddings from '
            f'{path}: {embs.shape}')
        return embs.astype(np.float32)

    def get_pretrain_item_embeddings(self):
        """Return pretrain item embeddings aligned with id_mapping, if available."""
        if self.pretrain_item_embeddings is None:
            return None

        n_items = len(self.id_mapping['item2id'])
        embed_dim = self.pretrain_item_embeddings.shape[1]
        aligned_embs = np.zeros((n_items - 1, embed_dim), dtype=np.float32)

        for item_str, item_id in self.id_mapping['item2id'].items():
            if item_str == '[PAD]' or item_id == 0:
                continue
            raw_id = int(item_str)
            if raw_id < len(self.pretrain_item_embeddings):
                aligned_embs[item_id - 1] = self.pretrain_item_embeddings[
                    raw_id].astype(np.float32)

        return aligned_embs
