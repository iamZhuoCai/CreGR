import importlib
import json
import os

import numpy as np
import yaml

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class CreGRTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(CreGRTokenizer, self).__init__(config, dataset)
        self.accelerator = config['accelerator']

        tokenizer_name = self.config['tokenizer_name']
        try:
            tokenizer_class = getattr(
                importlib.import_module(
                    f'genrec.tokenizers.{tokenizer_name}.tokenizer'),
                f'{tokenizer_name}Tokenizer')
        except:
            raise ValueError(f'Tokenizer "{tokenizer_name}" not found.')

        self.log(f'[TOKENIZER] Loading tokenizer config')
        tokenizer_config: dict = yaml.safe_load(
            open(f'genrec/tokenizers/{tokenizer_name}/config.yaml', 'r'))

        for key in tokenizer_config.keys():
            if key in config.keys():
                tokenizer_config[key] = config[key]
            self.log(f"{key}: {tokenizer_config[key]}")
        config.update(tokenizer_config)
        self.config = config

        self.tokenizer = tokenizer_class(config, dataset)

        self.item2id = dataset.item2id
        self.item2tokens = self.tokenizer.item2tokens
        self.mask_token = self.tokenizer.eos_token + 1
        self.ignored_label = self.tokenizer.ignored_label
        self.known_credible_items = self._build_known_credible_items(dataset)
        self.known_uncredible_items = self._build_known_uncredible_items(dataset)

    @property
    def n_digit(self):
        return self.tokenizer.n_digit

    @property
    def codebook_sizes(self):
        return self.tokenizer.codebook_sizes

    @property
    def max_token_seq_len(self) -> int:
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.mask_token + 1

    def _tokenize_items(self, item_seq: list, test=False):
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids = [0] * pad_lens + input_ids
        attention_mask = [0] * pad_lens + attention_mask

        if test:
            labels = list(self.item2tokens[item_seq[-1]])
        else:
            labels = [self.item2id[item_seq[-1]]]

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]

        if split == 'train':
            # Disco raw rows: one (seq, next) row → exactly one training example
            if self.config.get('disco_raw_rows', False):
                chunk = item_seq[-(max_item_seq_len + 1):]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
                    chunk, test=False)
                label_item = chunk[-1]
                uncredible_no_mask = int(label_item in self.known_uncredible_items)
                return {
                    'input_ids': [input_ids],
                    'attention_mask': [attention_mask],
                    'labels': [labels],
                    'seq_lens': [seq_lens],
                    'uncredible_last_token_no_mask': [uncredible_no_mask],
                }
            all_input_ids, all_attention_mask, all_labels, all_seq_lens= [],[],[],[]
            all_uncredible_no_mask = []
            for i in range(2, len(item_seq) + 1):
                cur_item_seq = item_seq[max(0, i - max_item_seq_len - 1):i]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
                    cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
                all_uncredible_no_mask.append(
                    int(cur_item_seq[-1] in self.known_uncredible_items))
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
                'uncredible_last_token_no_mask': all_uncredible_no_mask,
            }

        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
                item_seq[-(max_item_seq_len + 1):], test=True)
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels],
                'seq_lens': [seq_lens],
                'uncredible_last_token_no_mask': [0],
            }

    def _build_known_uncredible_items(self, dataset: AbstractDataset):
        """Sample known uncredible item IDs (as raw string item ids)."""
        if not self.config.get('cred_reg_enable', False):
            return set()
        if not hasattr(dataset, 'credible_items'):
            return set()

        # Prefer exact uncredible sample exported during tokenization stage.
        sem_ids_name = self.config.get('sem_ids_path')
        if sem_ids_name:
            token_model_dir = (
                f"TokenModel_Seq-{self.config['max_item_seq_len']}_"
                f"{self.config['tokenizer_name']}")
            sem_ids_path = os.path.join(
                self.config['base_result_dir'],
                f"{self.config['dataset']}_{self.config['category']}",
                token_model_dir,
                self.config['sem_ids_dir'], sem_ids_name)
            meta_path = f'{sem_ids_path}.credreg_meta.json'
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path, 'r'))
                known_uncredible = set(
                    str(x)
                    for x in meta.get('known_uncredible_raw_ids', []))
                self.log('[TOKENIZER] Loaded known-uncredible items from '
                         f'{meta_path}: {len(known_uncredible)}')
                return known_uncredible

        credible_set = set(int(x) for x in np.asarray(dataset.credible_items).tolist())
        all_train_targets = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            if len(item_seq) > 0:
                all_train_targets.add(item_seq[-1])

        uncredible_candidates = [
            item for item in all_train_targets if int(item) not in credible_set
        ]
        if len(uncredible_candidates) == 0:
            return set()

        known_ratio = float(self.config.get('cred_known_ratio', 0.2))
        seed = int(self.config.get('cred_reg_seed', 2024))
        rng = np.random.default_rng(seed)
        sample_n = max(1, int(len(uncredible_candidates) * known_ratio))
        picked = rng.choice(uncredible_candidates, size=sample_n, replace=False)
        known_uncredible = set(str(x) for x in picked.tolist())
        self.log('[TOKENIZER] CreGR known-uncredible items for no-mask-last-token '
                 '(fallback sampling): '
                 f'{len(known_uncredible)} / {len(uncredible_candidates)}')
        return known_uncredible

    def _build_known_credible_items(self, dataset: AbstractDataset):
        """Load known credible item IDs (as raw string item ids)."""
        if not self.config.get('cred_reg_enable', False):
            return set()
        if not hasattr(dataset, 'credible_items'):
            return set()

        sem_ids_name = self.config.get('sem_ids_path')
        if sem_ids_name:
            token_model_dir = (
                f"TokenModel_Seq-{self.config['max_item_seq_len']}_"
                f"{self.config['tokenizer_name']}")
            sem_ids_path = os.path.join(
                self.config['base_result_dir'],
                f"{self.config['dataset']}_{self.config['category']}",
                token_model_dir,
                self.config['sem_ids_dir'], sem_ids_name)
            meta_path = f'{sem_ids_path}.credreg_meta.json'
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path, 'r'))
                known_credible = set(
                    str(x) for x in meta.get('known_credible_raw_ids', []))
                if len(known_credible) > 0:
                    self.log('[TOKENIZER] Loaded known-credible items from '
                             f'{meta_path}: {len(known_credible)}')
                    return known_credible

        # Backward compatibility for old meta files without known_credible list:
        # reconstruct tokenization-time 20% known-credible with same ratio+seed.
        if sem_ids_name:
            sem_ids_path = os.path.join(
                self.config['base_result_dir'],
                f"{self.config['dataset']}_{self.config['category']}",
                f"TokenModel_Seq-{self.config['max_item_seq_len']}_{self.config['tokenizer_name']}",
                self.config['sem_ids_dir'], sem_ids_name)
            if os.path.exists(sem_ids_path):
                item2sem = json.load(open(sem_ids_path, 'r'))
                raw_item_ids_for_training = [int(k) for k in item2sem.keys()]
                credible_set = set(
                    int(x) for x in np.asarray(dataset.credible_items).tolist())
                cred_candidates = [
                    x for x in raw_item_ids_for_training if int(x) in credible_set
                ]
                if len(cred_candidates) > 0:
                    known_ratio = float(self.config.get('cred_known_ratio', 0.2))
                    seed = int(self.config.get('cred_reg_seed', 2024))
                    rng = np.random.default_rng(seed)
                    sample_n = max(1, int(len(cred_candidates) * known_ratio))
                    sample_n = min(sample_n, len(cred_candidates))
                    picked = rng.choice(cred_candidates,
                                        size=sample_n,
                                        replace=False)
                    reconstructed = set(str(int(x)) for x in picked.tolist())
                    self.log('[TOKENIZER] Reconstructed known-credible items '
                             f'from sem_ids with ratio={known_ratio}: '
                             f'{len(reconstructed)}')
                    return reconstructed

        # Last-resort fallback.
        fallback = set(str(int(x)) for x in np.asarray(dataset.credible_items).tolist())
        self.log('[TOKENIZER] known_credible_raw_ids missing; '
                 f'fallback to full credible set: {len(fallback)}')
        return fallback

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: ')

        for split in datasets:
            self.log(
                f"Tokenized {split} set: {len(tokenized_datasets[split])}")
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
