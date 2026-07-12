from allosaurus.am.utils import move_to_tensor, torch_save
from allosaurus.am.criterion import read_criterion
from allosaurus.am.optimizer import read_optimizer
from allosaurus.am.reporter import Reporter
import editdistance
import numpy as np
import torch
from itertools import groupby
from allosaurus.model import get_model_path
import json

class Trainer:

    def __init__(self, model, train_config):

        self.model = model
        self.train_config = train_config

        self.device_id = self.train_config.device_id

        # criterion, only ctc currently
        self.criterion = read_criterion(train_config)

        # optimizer, only sgd currently
        self.optimizer = read_optimizer(self.model, train_config)

        # reporter to write logs
        self.reporter = Reporter(train_config)

        # best per
        self.best_per = 100.0

        # intialize the model
        self.model_path = get_model_path(train_config.new_model)

        # counter for early stopping
        self.num_no_improvement = 0

        # Keep a model snapshot for every completed epoch.  model.pt remains
        # reserved for the best validation model used by inference.
        self.checkpoint_path = self.model_path / 'checkpoints'
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)


    def save_epoch_checkpoint(self, epoch, validate_phone_error_rate):
        """Save an inference-compatible model snapshot and epoch metadata."""
        epoch_number = epoch + 1
        checkpoint_stem = f'epoch_{epoch_number:04d}'
        checkpoint_model_path = self.checkpoint_path / f'{checkpoint_stem}.pt'
        checkpoint_metadata_path = self.checkpoint_path / f'{checkpoint_stem}.json'

        torch_save(self.model, checkpoint_model_path)
        metadata = {
            'epoch': epoch_number,
            'validate_phone_error_rate': validate_phone_error_rate,
        }
        checkpoint_metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        self.reporter.write(f"saved epoch checkpoint: {checkpoint_model_path}")


    def sum_edit_distance(self, output_ndarray, output_lengths_ndarray, token_ndarray, token_lengths_ndarray):
        """
        compute SUM of ter in this batch

        """

        error_cnt_sum = 0.0

        for i in range(len(token_lengths_ndarray)):
            target_list = token_ndarray[i, :token_lengths_ndarray[i]].tolist()
            logit = output_ndarray[i][:output_lengths_ndarray[i]]

            raw_token = [x[0] for x in groupby(np.argmax(logit, axis=1))]
            decoded_token = list(filter(lambda a: a != 0, raw_token))

            error_cnt_sum += editdistance.distance(target_list, decoded_token)

        return error_cnt_sum


    def step(self, feat_batch, token_batch):

        # prepare torch tensors from numpy arrays
        feat_tensor, feat_lengths_tensor = move_to_tensor(feat_batch, self.device_id)
        token_tensor, token_lengths_tensor = move_to_tensor(token_batch, self.device_id)

        #print(feat_tensor)
        #print(feat_lengths_tensor)
        output_tensor = self.model(feat_tensor, feat_lengths_tensor)

        #print(output_tensor)
        #print(token_tensor)
        #print(token_lengths_tensor)

        loss = self.criterion(output_tensor, feat_lengths_tensor, token_tensor, token_lengths_tensor)
        #print(loss.item())

        # extract numpy format for edit distance computing
        output_ndarray = output_tensor.cpu().detach().numpy()
        feat_ndarray, feat_lengths_ndarray = feat_batch
        token_ndarray, token_lengths_ndarray = token_batch

        phone_error_sum = self.sum_edit_distance(output_ndarray, feat_lengths_ndarray, token_ndarray,
                                                 token_lengths_ndarray)

        phone_count = sum(token_lengths_ndarray)

        return loss, phone_error_sum, phone_count


    def train(self, train_loader, validate_loader):

        self.best_per = 100.0

        expected_validate_size = getattr(self.train_config, 'expected_validate_size', 0)
        validate_size = len(validate_loader.dataset)
        if expected_validate_size and validate_size != expected_validate_size:
            raise ValueError(
                f"expected {expected_validate_size} validation utterances, found {validate_size}"
            )
        self.reporter.write(f"validation utterances per epoch: {validate_size}")

        batch_count = len(train_loader)

        global_step = 0

        for epoch in range(self.train_config.epoch):

            # shuffle
            train_loader.shuffle()

            # set to the training mode
            self.model.train()

            # reset all stats
            all_phone_count = 0.0
            all_loss_sum = 0.0
            all_phone_error_sum = 0.0
            epoch_phone_count = 0.0
            epoch_loss_sum = 0.0
            epoch_phone_error_sum = 0.0

            # training loop
            for ii in range(batch_count):

                self.optimizer.zero_grad()

                feat_batch, token_batch = train_loader.read_batch(ii)

                # forward step
                loss_tensor, phone_error_sum, phone_count = self.step(feat_batch, token_batch)

                # backprop and optimize
                loss_tensor.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)

                self.optimizer.step()

                # update stats
                loss_sum = loss_tensor.item()
                all_phone_count += phone_count
                all_loss_sum += loss_sum
                all_phone_error_sum += phone_error_sum
                epoch_phone_count += phone_count
                epoch_loss_sum += loss_sum
                epoch_phone_error_sum += phone_error_sum
                global_step += 1

                if ii % self.train_config.report_per_batch == 0:
                    train_loss = all_loss_sum / all_phone_count
                    train_per = all_phone_error_sum / all_phone_count
                    message = f'epoch[batch]: {epoch:02d}[{ii:04d}] | train loss {train_loss:0.5f} train per {train_per:0.5f}'
                    self.reporter.write(message)
                    self.reporter.log_metrics({
                        'batch/train_loss': train_loss,
                        'batch/train_per': train_per,
                        'batch/epoch': epoch + 1,
                        'batch/global_step': global_step,
                    })

                    # reset all stats
                    all_phone_count = 0.0
                    all_loss_sum = 0.0
                    all_phone_error_sum = 0.0


            # evaluate this model
            epoch_train_loss = epoch_loss_sum / epoch_phone_count
            epoch_train_per = epoch_phone_error_sum / epoch_phone_count
            validate_loss, validate_phone_error_rate = self.validate(validate_loader)

            self.reporter.write(
                f"epoch{epoch + 1} | train loss: {epoch_train_loss:0.5f} "
                f"train per: {epoch_train_per:0.5f} validate loss: {validate_loss:0.5f} "
                f"validate per: {validate_phone_error_rate:0.5f}"
            )

            # Save every completed epoch, regardless of whether validation PER
            # improved. This is separate from model.pt, which tracks the best.
            self.save_epoch_checkpoint(epoch, validate_phone_error_rate)

            if validate_phone_error_rate <= self.best_per:
                self.best_per = validate_phone_error_rate
                self.num_no_improvement = 0
                self.reporter.write("saving model")

                model_name = f"model_{validate_phone_error_rate:0.5f}.pt"

                # save model
                torch_save(self.model, self.model_path / model_name)

                # overwrite the best model
                torch_save(self.model, self.model_path / 'model.pt')

            else:
                self.num_no_improvement += 1

            self.reporter.log_metrics({
                'epoch': epoch + 1,
                'epoch/train_loss': epoch_train_loss,
                'epoch/train_per': epoch_train_per,
                'epoch/validate_loss': validate_loss,
                'epoch/validate_per': validate_phone_error_rate,
                'epoch/best_validate_per': self.best_per,
                'epoch/validate_utterances': validate_size,
                'epoch/learning_rate': self.optimizer.param_groups[0]['lr'],
            })

            if validate_phone_error_rate > self.best_per:
                if self.num_no_improvement >= 3:
                    self.reporter.write("no improvements for several epochs, early stopping now")
                    break

        # close reporter stream
        self.reporter.close()


    def validate(self, validate_loader):

        self.model.eval()

        batch_count = len(validate_loader)

        all_phone_error_sum = 0
        all_phone_count = 0
        all_loss_sum = 0.0

        # validation loop
        with torch.no_grad():
            for ii in range(batch_count):
                feat_batch, token_batch = validate_loader.read_batch(ii)

                # one step
                loss_tensor, phone_error_sum, phone_count = self.step(feat_batch, token_batch)

                # update stats
                all_loss_sum += loss_tensor.item()
                all_phone_error_sum += phone_error_sum
                all_phone_count += phone_count

        return all_loss_sum/all_phone_count, all_phone_error_sum/all_phone_count
