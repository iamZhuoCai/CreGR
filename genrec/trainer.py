import math
import os
import time
from collections import OrderedDict, defaultdict
from logging import getLogger

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import config_for_log, get_file_name, get_total_steps, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel,
                 tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(config['result_dir'],
                                             config['ckpt_dir'],
                                             f"{config['run_time']}.pth")

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader, test_dataloader=None):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
            test_dataloader: Optional. If ``test_eval_interval`` is set in config,
                run test evaluation every N epochs (not prepared by Accelerate here).
        """
        optimizer = AdamW(self.model.parameters(),
                          lr=self.config['lr'],
                          weight_decay=self.config['weight_decay'])

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        warmup_steps = math.floor(total_n_steps * self.config['warmup_ratio'])
        lr_schedule = self.config.get('lr_schedule', 'cosine')
        self.log(
            f"Total steps: {total_n_steps}, warmup steps: {warmup_steps}, "
            f"lr_schedule: {lr_schedule}")

        if lr_schedule == 'cosine':
            scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_n_steps,
            )
        elif lr_schedule == 'constant_with_warmup':
            scheduler = get_scheduler(
                name="constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_n_steps,
            )
        else:
            raise ValueError(
                f"Unsupported lr_schedule: {lr_schedule}. "
                f"Use 'cosine' or 'constant_with_warmup'.")

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler)
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {
                "flush_secs": 60
            }},
        )

        n_epochs = np.ceil(total_n_steps /
                           (len(train_dataloader) *
                            self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1
        start_epoch = 0

        resume_training_ckpt = self.config.get('resume_training_ckpt')
        if resume_training_ckpt:
            start_epoch, best_epoch, best_val_score = self._load_resume_state(
                resume_training_ckpt, optimizer, scheduler)
            if start_epoch >= n_epochs:
                self.log(
                    f'Resume epoch {start_epoch} >= target epochs {n_epochs}. '
                    f'Nothing to train.')
                return best_epoch, best_val_score

        skip_val = bool(self.config.get('skip_validation', False))
        last_epoch_ran = 0
        test_eval_interval = self.config.get('test_eval_interval')
        if test_eval_interval is not None:
            test_eval_interval = int(test_eval_interval)

        for epoch in range(start_epoch, n_epochs):
            start_time = time.time()
            self.model.train()

            total_loss = 0.0
            total_his_mask_loss = 0.0
            total_target_mask_loss = 0.0

            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
                disable=True,
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(),
                                    self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

                total_his_mask_loss += outputs.his_mask_loss.item()
                total_target_mask_loss += outputs.target_mask_loss.item()

            self.accelerator.log(
                {"Loss/train_loss": total_loss / len(train_dataloader)},
                step=epoch + 1)

            self.log(
                "[Epoch {}] Train Loss: {:.4f} His_Mask Loss: {:.4f} Target_Mask Loss: {:.4f} Running Time: {:.2f}s  lr: {:.6f}"
                .format(epoch + 1, total_loss / len(train_dataloader),
                        total_his_mask_loss / len(train_dataloader),
                        total_target_mask_loss / len(train_dataloader),
                        time.time() - start_time,
                        scheduler.get_last_lr()[0]))

            last_epoch_ran = epoch + 1

            # Evaluation (skipped when training without a validation set)
            if not skip_val and (epoch + 1) % self.config['eval_interval'] == 0:

                all_results = self.evaluate(val_dataloader,
                                            split='val',
                                            epoch=epoch)

                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log(
                            {f"Val_Metric/{key}": all_results[key]},
                            step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        self.save_model()
                        self.log(
                            f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}'
                        )

                if self.config[
                        'patience'] is not None and epoch + 1 - best_epoch >= self.config[
                            'patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break

            # Periodic full test-set metrics (e.g. Disco: ndcg/recall/cr @ K)
            if (test_dataloader is not None and len(test_dataloader) > 0
                    and test_eval_interval and test_eval_interval > 0
                    and (epoch + 1) % test_eval_interval == 0):
                t_test = time.perf_counter()
                test_results = self.evaluate(test_dataloader,
                                               split='test',
                                               epoch=epoch)
                test_elapsed = time.perf_counter() - t_test
                if self.accelerator.is_main_process:
                    for key in test_results:
                        self.accelerator.log(
                            {f'Test_Metric/{key}': test_results[key]},
                            step=epoch + 1)
                    self.log(
                        f'[Epoch {epoch + 1}] Test Running Time: {test_elapsed:.2f}s'
                    )
                    self.log(
                        f'[Epoch {epoch + 1}] Test Results: {test_results}')

            if self.accelerator.is_main_process:
                self._save_training_state(epoch, optimizer, scheduler,
                                          best_epoch, best_val_score)

        if skip_val and last_epoch_ran > 0:
            best_epoch = last_epoch_ran
            best_val_score = None
            if self.accelerator.is_main_process:
                self.save_model()
                self.log(
                    f'[Final epoch {last_epoch_ran}] Saved model checkpoint to '
                    f'{self.saved_model_ckpt} (no validation)')
            self.log('Training finished without validation / early stopping.')
        elif not skip_val:
            self.log(
                f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

        return best_epoch, best_val_score

    def save_model(self, ):
        if self.config['use_ddp']:  # unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
        else:
            torch.save(self.model.state_dict(), self.saved_model_ckpt)

    def _training_state_path(self):
        base, _ = os.path.splitext(self.saved_model_ckpt)
        return f'{base}.training_state.pth'

    def _save_training_state(self, epoch, optimizer, scheduler, best_epoch,
                             best_val_score):
        model = self.accelerator.unwrap_model(
            self.model) if self.config['use_ddp'] else self.model
        state = {
            'epoch': int(epoch),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_epoch': int(best_epoch),
            'best_val_score': best_val_score,
        }
        torch.save(state, self._training_state_path())

    def _load_resume_state(self, resume_ckpt_path, optimizer, scheduler):
        self.log(f'Resuming training state from {resume_ckpt_path}')
        ckpt = torch.load(resume_ckpt_path, map_location='cpu')

        # Backward compatible: allow plain model state_dict ckpt.
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model_state = ckpt['model']
            optimizer_state = ckpt.get('optimizer')
            scheduler_state = ckpt.get('scheduler')
            last_epoch = int(ckpt.get('epoch', -1))
            best_epoch = int(ckpt.get('best_epoch', 0))
            best_val_score = ckpt.get('best_val_score', -1)
        else:
            model_state = ckpt
            optimizer_state = None
            scheduler_state = None
            last_epoch = -1
            best_epoch = 0
            best_val_score = -1

        if self.config['use_ddp']:
            self.accelerator.unwrap_model(self.model).load_state_dict(
                model_state)
        else:
            self.model.load_state_dict(model_state)

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        start_epoch = last_epoch + 1
        self.log(
            f'Resumed from epoch {last_epoch}; training continues at epoch {start_epoch + 1}.'
        )
        return start_epoch, best_epoch, best_val_score

    def evaluate(self, dataloader, split='test', epoch=-1):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
            disable=True,
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {
                    k: v.to(self.accelerator.device)
                    for k, v in batch.items()
                }
                batch['split'] = split
                batch['epoch'] = epoch

                if self.config[
                        'use_ddp']:  # ddp, gather data from all devices for evaluation

                    if split == 'val':
                        preds = self.model.module.generate(
                            batch, n_return_sequences=self.evaluator.maxk_eval)
                    else:
                        preds = self.model.module.generate(
                            batch, n_return_sequences=self.evaluator.maxk)

                    all_preds, all_labels = self.accelerator.gather_for_metrics(
                        (preds, batch['labels']))

                    results = self.evaluator.calculate_metrics(
                        all_preds, all_labels, split)
                else:
                    if split == 'val':
                        preds = self.model.generate(
                            batch, n_return_sequences=self.evaluator.maxk_eval)
                    else:
                        preds = self.model.generate(
                            batch, n_return_sequences=self.evaluator.maxk)

                    results = self.evaluator.calculate_metrics(
                        preds, batch['labels'], split)

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            topk_list = self.config['topk'] if (
                split == 'test') else self.config['val_topk']
            for k in topk_list:
                key = f"{metric}@{k}"
                if key not in all_results:
                    continue
                output_results[key] = torch.cat(all_results[key]).mean().item()

        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
