import os
import pickle
import pprint
import shutil
import inspect
from collections import OrderedDict, defaultdict
import time
from tqdm import tqdm
import yaml

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import apex

from utils import count_params, import_class, init_seed

class Trainer():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg['phase'] == 'train':
            # Added control through the command line
            arg['train_feeder_args']['debug'] = arg['train_feeder_args']['debug'] or self.arg['debug']
            logdir = os.path.join(arg['work_dir'], 'trainlogs')

            print(f'logdir is {logdir}')

            if not arg['train_feeder_args']['debug']:
                # logdir = arg['model_saved_name']
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg['assume_yes']:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()

        self.global_step = 0
        self.lr = self.arg['base_lr']
        self.best_acc = 0
        self.best_acc_epoch = 0

        if self.arg['half']:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg["amp_opt_level"]}'
            )
            if self.arg['amp_opt_level'] != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg['device']) is list:
            if len(self.arg['device']) > 1:
                self.print_log(f'{len(self.arg["device"])} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg['device'],
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg['device'][0] if type(
            self.arg['device']) is list else self.arg['device']
        self.output_device = output_device
        Model = import_class(self.arg['model'])

        # Copy model file and main
        shutil.copy2(inspect.getfile(Model), self.arg['work_dir'])
        shutil.copy2(os.path.join('.', __file__), self.arg['work_dir'])

        model = Model(**self.arg['model_args'])
        
        # INI transfer learning
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(384, self.arg['model_args']['num_class'])
        # END transfer learning

        self.model = model.cuda(output_device)

        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg['weights']:
            try:
                self.global_step = int(self.arg['weights'][:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg["weights"]}')
            if '.pkl' in self.arg['weights']:
                with open(self.arg['weights'], 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg['weights'])

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            # print(weights.keys())

            for w in self.arg['ignore_weights']:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        

        # for name, param in self.model.named_parameters():
        #     print((name, param.shape, param.requires_grad))

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.arg['base_lr'],
                momentum=0.9,
                nesterov=self.arg['nesterov'],
                weight_decay=self.arg['weight_decay'])
        elif self.arg['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.arg['base_lr'],
                weight_decay=self.arg['weight_decay'])
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg['optimizer']))

        # Load optimizer states if any
        if self.arg['checkpoint'] is not None:
            self.print_log(f'Loading optimizer states from: {self.arg["checkpoint"]}')
            self.optimizer.load_state_dict(torch.load(self.arg['checkpoint'])['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.arg['step'], gamma=0.1)
        if self.arg['checkpoint'] is not None:
            scheduler_states = torch.load(self.arg['checkpoint'])['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg["checkpoint"]}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def save_arg(self):
        # save arg
        if not os.path.exists(self.arg['work_dir']):
            os.makedirs(self.arg['work_dir'])
        with open(os.path.join(self.arg['work_dir'], 'config.yaml'), 'w') as f:
            yaml.dump(self.arg, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        if self.arg['print_log']:
            with open(os.path.join(self.arg['work_dir'], 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg['work_dir'], out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg["forward_batch_size"]}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def train(self, train_dl, eval_dl):
        self.print_log(f'Parameters:\n{pprint.pformat(self.arg)}\n')
        self.print_log(f'Model total number of params: {count_params(self.model)}')


        self.global_step = self.arg['start_epoch'] * len(train_dl) / self.arg['batch_size']
        for epoch in range(self.arg['start_epoch'], self.arg['num_epoch']):
            save_model = ((epoch + 1) % self.arg['save_interval'] == 0) or (epoch + 1 == self.arg['num_epoch'])
            self.train_epoch(epoch, train_dl, save_model=save_model)
            self.eval(epoch, eval_dl, save_score=self.arg['save_score'])

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f'Best accuracy: {self.best_acc}')
        self.print_log(f'Epoch number: {self.best_acc_epoch}')
        self.print_log(f'Model name: {self.arg["work_dir"]}')
        self.print_log(f'Model total number of params: {num_params}')
        self.print_log(f'Weight decay: {self.arg["weight_decay"]}')
        self.print_log(f'Base LR: {self.arg["base_lr"]}')
        self.print_log(f'Batch Size: {self.arg["batch_size"]}')
        self.print_log(f'Forward Batch Size: {self.arg["forward_batch_size"]}')
        self.print_log(f'Test Batch Size: {self.arg["test_batch_size"]}')

    def train_epoch(self, epoch, loader, save_model=False):
        self.model.train()
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, label, _) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg['forward_batch_size']
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg["batch_size"]!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_label = data[left:right], label[left:right]

                # forward
                output = self.model(batch_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                loss = self.loss(output, batch_label) / splits

                if self.arg['half']:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())

                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            #####################################

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.arg['batch_size'] // self.arg['forward_batch_size']
        self.print_log(f'\tMean training loss: {mean_loss:.4f} (BS {self.arg["batch_size"]}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def eval(self, epoch, loader, save_score=False):
        # Skip evaluation if too early
        if epoch + 1 < self.arg['eval_start']:
            return

        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            loss_values = []
            all_scores = []
            all_idxs = []
            step = 0
            process = tqdm(loader, dynamic_ncols=True)
            for batch_idx, (data, label, idxs) in enumerate(process):
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output = self.model(data)
                loss = self.loss(output, label)
                all_scores.append(output.data.cpu().numpy())
                all_idxs.append(idxs.cpu().numpy())
                loss_values.append(loss.item())

                step += 1

            all_scores = np.concatenate(all_scores)
            all_idxs = np.concatenate(all_idxs)
            loss = np.mean(loss_values)
            accuracy = loader.dataset.accuracy(all_idxs, all_scores)
            print('Accuracy: ', accuracy, ' model: ', self.arg['work_dir'])

            auc=None
            if all_scores.shape[1] == 2:
                auc = loader.dataset.auc(all_idxs, all_scores[:,1])
                print(f'ROC AUC: {auc}')
                self.val_writer.add_scalar('auc', auc,      self.global_step)

            self.val_writer.add_scalar('loss', loss,    self.global_step)
            self.val_writer.add_scalar('acc', accuracy, self.global_step)
                        
            self.print_log(f'\tMean loss of {len(loader)} batches: {np.mean(loss_values)}.')

        metrics = {
            'accuracy': accuracy,
            'loss': loss
        }
        if auc is not None: metrics['auc'] = auc

        # Empty cache after evaluation
        torch.cuda.empty_cache()
        return metrics, all_scores

    def test(self, test_dl):
        if self.arg['weights'] is None:
            raise ValueError('Please appoint --weights.')

        self.print_log(f'Model:   {self.arg["model"]}')
        self.print_log(f'Weights: {self.arg["weights"]}')

        metrics = self.eval(
            epoch=0,
            loader=test_dl,
            save_score=self.arg['save_score']
        )

        self.print_log('Done.\n')
        return metrics
