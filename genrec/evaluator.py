import torch


class Evaluator:

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {'recall': self.recall_at_k, 'ndcg': self.ndcg_at_k}

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

        self.maxk_eval = max(config['val_topk'])

        raw = config.get('credible_item_raw_ids')
        self.credible_raw_ids = set(raw) if raw else set()
        self.credible_token_tuples = set()
        for item, toks in tokenizer.item2tokens.items():
            if int(item) in self.credible_raw_ids:
                self.credible_token_tuples.add(tuple(toks))

    def calculate_pos_index(self, preds, labels, split):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        if split == 'val':
            assert preds.shape[
                1] == self.maxk_eval, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        elif split == 'test':
            assert preds.shape[
                1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        else:
            raise ValueError(f"Invalid split: {split}")

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            # for j in range(self.maxk):

            maxk = self.maxk if (split == 'test') else self.maxk_eval
            for j in range(maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def cr_at_k(self, preds, k):
        """Credible ratio @K computed in token space.

        A predicted token tuple is counted as credible if it exists in the
        credible token-tuple set built from all credible raw items.
        """
        b, maxk, _ = preds.shape
        device = preds.device
        k_eff = min(k, maxk)
        if not self.credible_token_tuples:
            return torch.zeros(b, device=device)
        scores = torch.zeros(b, device=device)
        preds_cpu = preds.detach().cpu()
        for i in range(b):
            cnt = 0
            for j in range(k_eff):
                tup = tuple(int(x) for x in preds_cpu[i, j].tolist())
                if tup in self.credible_token_tuples:
                    cnt += 1
            scores[i] = cnt / float(k)
        return scores

    def calculate_metrics(self, preds, labels, split):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels, split)
        for metric in self.config['metrics']:
            topk_list = self.config['topk'] if (
                split == 'test') else self.config['val_topk']
            if metric == 'cr':
                if not self.credible_raw_ids:
                    continue
                for k in topk_list:
                    results[f'cr@{k}'] = self.cr_at_k(preds, k)
            else:
                for k in topk_list:
                    results[f"{metric}@{k}"] = self.metric2func[metric](
                        pos_index, k)
        return results
