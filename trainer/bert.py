from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import autograd
import json
import numpy as np
from abc import *
from pathlib import Path


''' 新加的'''
# class Discriminator(nn.Module):
#     def __init__(self, args,hiddens):
#         super(Discriminator, self).__init__()
#         self.args = args
#         self.device = args.device
#         self.hiddens = hiddens
#         self.discriminator = nn.Sequential(
#             nn.Linear(self.hiddens, self.hiddens),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(self.hiddens, self.hiddens // 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(self.hiddens // 2, 1)
#         ).to(self.device)
#     def forward(self, hidden_states):
#         return self.discriminator(hidden_states)
''' 新加的'''
  
# 检查是否可用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 定义一个简单的模型类
# class YourModel(nn.Module):
#     def __init__(self):
#         super(YourModel, self).__init__()
#         self.linear = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.linear(x)

# # 创建模型实例并将其移动到GPU
# model = YourModel()
# model.to(device)

# # 检查模型是否在GPU上
# if next(model.parameters()).is_cuda:
#     print('模型已在GPU上运行')
# else:
#     print('模型在CPU上运行')






class BERTTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        print(self.device)

        # self.discriminator = Discriminator(self.args,  hiddens=64).to(self.device)



        self.is_parallel = args.num_gpu > 1
        
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)#假设gpu运算的
            #print('it is GPU')
        self.KL = nn.KLDivLoss(reduction='batchmean')
        #self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.fake_net = nn.Sequential(
            nn.Linear(args.bert_hidden_units, args.bert_hidden_units),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.bert_hidden_units, args.bert_hidden_units)).to(self.device)
        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mask_token = self.args.num_items + 1
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        
        self.optimizer = self._create_optimizer()
        
        
        # self.discriminator_optimizer = self._create_doptimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)
            
        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.ce = nn.CrossEntropyLoss(ignore_index=0,reduction='mean')

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)
        completed_steps=0
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)

            batch = [x.to(self.device) for x in batch]
             
            seqs, labels = batch
            
            x, kl, mask = self.model.embedding(seqs)#输入是长度为200的序列seqs
            
            #scores, kl, logit = self.model(x)
            real_inputs_embeds = x #x==>64*200*64

            # ------ 攻击开关 --------以后再重构 先测 可以注释和打开注释来决定是否攻击
            # random_num = torch.rand_like(real_inputs_embeds)
            # fake_inputs_embeds = self.fake_net(real_inputs_embeds) * random_num + (1 - random_num) * real_inputs_embeds
            # 修改拼接逻辑：把攻击产生的“假数据”拼接到训练集里
            # inputs_embeds = torch.cat([real_inputs_embeds, fake_inputs_embeds], dim=0)
            # attention_mask = torch.cat([mask, mask], dim=0) # mask 也要对应翻倍

            # 标签也要对应翻倍，否则 Loss 会报错
            # elselabel = torch.cat([labels, labels], dim=0)
            # -------------------------

            #扰动序列Fake = α*MLP(REAL) + (1-α)REAL 
            
            # ------- 原始逻辑 先注释 以后再重构 先测 -------
            attention_mask = torch.cat([mask], dim=0)
            inputs_embeds = torch.cat([real_inputs_embeds], dim=0)
            elselabel = torch.cat([labels], dim=0)
            

            logits2 = self.model.model(inputs_embeds, self.model.embedding.token2.weight, attention_mask)
            
            
            kl = kl[:, -1, :]
            nan_indices = torch.isnan(kl)  # 使用 torch.isnan 函数检测NaN值的位置       
            kl_1 = kl[ ~nan_indices] # 通过索引操作删除包含NaN值的元素
            #labels ===> 64*200
            ##logits2 = logit.view(-1, logit.size(-1))
            logits2 = logits2.view(-1, logits2.size(-1))
            elselabel = elselabel.view(-1)
            ##loss = self.ce(logits2, elselabel)
            #loss = self.ce(logits2, elselabel)
            #loss = loss + 0.1 * kl_1.mean()
            
            self.optimizer.zero_grad()
            loss = self.calculate_loss(kl_1, logits2, elselabel)
            self.loss = loss           
            loss.backward()
            self.clip_gradients(5)
            self.optimizer.step()           
            self.lr_scheduler.step()
            completed_steps += 1
            
            
            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} ,a_loss {:.3f}'.format(epoch+1, average_meter_set['loss'].avg,loss.item()))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)
            

            with open(os.path.join(self.export_root, 'logs', 'val_metrics.json'), 'a', encoding='utf-8') as fw:
                str = json.dumps(average_meter_set.averages(),indent=4, ensure_ascii=False)
                fw.write(str)
                fw.write('\n')
            
        

    def test(self):
        best_model_dict = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        all_scores = []
        average_scores = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)
                
                # seqs, candidates, labels = batch
                # scores = self.model(seqs)
                # scores = scores[:, -1, :]
                # scores_sorted, indices = torch.sort(scores, dim=-1, descending=True)
                # all_scores += scores_sorted[:, :100].cpu().numpy().tolist()
                # average_scores += scores_sorted.cpu().numpy().tolist()
                # scores = scores.gather(1, candidates)
                # metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics

    # def calculate_loss(self, batch):
    #     seqs, labels = batch
    #     logits1  = self.model(seqs)
    #     x,mask = self.model.embedding(seqs)
    #     real_inputs_embeds = x
    #     random_num = torch.rand_like(real_inputs_embeds)
    #     fake_inputs_embeds = self.fake_net(real_inputs_embeds) * random_num + (1 - random_num) * real_inputs_embeds
    #     attention_mask = torch.cat([mask, mask], dim=0)
    #     inputs_embeds = torch.cat([real_inputs_embeds, fake_inputs_embeds], dim=0)
    #     print(inputs_embeds.size())
    #     elselabel = torch.cat([labels, labels], dim=0)
    #     logits2 = self.model.model(inputs_embeds, self.model.embedding.token.weight, attention_mask)
    #     logits1 = logits1.view(-1, logits1.size(-1))
    #     logits2 = logits2.view(-1, logits2.size(-1))
    #     elselabel = elselabel.view(-1)
    #     labels = labels.view(-1)
    #     loss = self.ce(logits1, labels)
    #     loss1 = self.ce(logits2, elselabel)
    #     loss = loss + loss1
    #     return loss
    def calculate_loss(self,  kl, logits, labels) :
        
        loss = self.ce(logits, labels)
        #loss = loss + 0.1 * kl.mean()
        return loss


    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch #candidates 表示输出的序列=answer+negs

        scores = self.model(seqs)
        logit, kl, mask = self.model.embedding(seqs)
        logit = logit[:, -1, :]
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)#candidates 输出序列的分数

        #logit = logit.gather(1,candidates)
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        #metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            nn.utils.clip_grad_norm_(p, 5)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError
    
    def _create_doptimizer(self):
        args = self.args
        discriminator_optimizer = list(self.discriminator.named_parameters())
        no_decay = ['bias', 'layer_norm']
        doptimizer_grouped_parameters = [
            {
                'params': [p for n, p in discriminator_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in discriminator_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(doptimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(doptimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(doptimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        # based on hugging face get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
