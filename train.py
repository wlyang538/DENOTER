from datasets import DATASETS
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *
import time 


def train(args, export_root=None, resume=True):
    args.lr = 0.001
    fix_random_seed_as(args.model_init_seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)

    if args.model_code == 'locker':
        model = Locker(args)
    elif args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)
    elif args.model_code == 'IB':
        model = BERT(args)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    export_root = None
    resume = True
    if export_root == None:
        export_root = 'experiments/' + args.model_code + '/' + args.dataset_code

    if resume:
        try:
            model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')

    if args.model_code == 'locker':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    elif args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    elif args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    elif args.model_code == 'narm':
        trainer = RNNTrainer(args, model, train_loader, val_loader, test_loader, export_root)


    # trainer.train()
    # print("这是测试结果")
    # trainer.test()


    # ================== 记录训练时间 ==================
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    print(f"⏱ Train time: {train_end_time - train_start_time:.2f} seconds")

    # ================== 记录测试时间 ==================
    print("这是测试结果")
    test_start_time = time.time()
    trainer.test()
    test_end_time = time.time()
    print(f"⏱ Test time: {test_end_time - test_start_time:.2f} seconds")


if __name__ == "__main__":
    set_template(args)
    train(args, resume=False)
