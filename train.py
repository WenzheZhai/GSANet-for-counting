import argparse
import importlib
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tools import AverageMeter, np2jpg

from tools import get_logger


class Trainer(object):
    def __init__(self, args):
        self.seed = 3035
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.config()

        self.model = Trainer.model_get(args.net, args.gpu)

        self.optimizer = optim.Adam(
            self.model.parameters(), args.lr, weight_decay=args.weight_decay
        )

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.lr_decay)

        if args.loss == "GLCL":
            from misc.glcl_loss import GLCL

            self.loss = GLCL().cuda()
        else:
            self.loss = nn.MSELoss().cuda()

        if args.dataset == "SHHA":
            from datasets.SHHA.loading_data import loading_data
        else:
            from datasets.SHHB.loading_data import loading_data

        self.train_loader, self.val_loader, self.restore_transform = loading_data()

        self.max_epoch = args.max_epoch
        self.epoch = 0

        self.best_mae = 1e10
        self.best_mse = 1e10

    def forward(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch

            self.train()

            self.scheduler.step()

            self.validation()

    def train(self):
        self.model.train()

        for i, data in enumerate(self.train_loader):
            img, gt = data
            img = img.cuda()
            gt = gt.cuda()

            output = self.model(img)

            if args.loss == "GLCL":
                loss = self.loss(output, gt)
            else:
                loss = self.loss(output.squeeze(), gt.squeeze())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"] * 10000
                self.logger.info(f"[i: {i+1}] [loss: {loss.item()}] [lr: {lr}]")

                self.writer.add_scalar("loss", loss.item(), self.tensorboard_x)
                self.tensorboard_x += 1

    def validation(self):
        self.model.eval()

        maes = AverageMeter()
        mses = AverageMeter()

        for i, data in enumerate(self.val_loader):
            img, gt = data

            with torch.no_grad():
                img = img.cuda()
                output = self.model(img)

                gt = gt.squeeze().detach().cpu().numpy()
                output = output.squeeze().detach().cpu().numpy()

                gt = np.sum(gt) / 100
                est = np.sum(output) / 100

                if (i + 1) % 50 == 0:
                    self.logger.info(f"[i: {i+1}] [est: {est:.2f}] [gt: {gt:.2f}]")

                maes.update(abs(est - gt))
                mses.update((abs(est - gt) ** 2))

        mae = maes.avg
        mse = np.sqrt(mses.avg)

        if mae < self.best_mae or mse < self.best_mse:
            self.best_mae = mae
            self.best_mse = mse

            torch.save(
                self.model.state_dict(),
                f"{self.output_path}/epoch_{self.epoch+1}_mae_{self.best_mae:.1f}_mse_{self.best_mse:.1f}.pth",
            )
        self.logger.info(
            f"[epoch: {self.epoch+1}] [mae: {mae:.1f}] [mse: {mse:.1f}]  \
                [best_mae: {self.best_mae:.1f}] [best_mse: {self.best_mse:.1f}]"
        )

    @staticmethod
    def model_get(net="VGG", gpu="0,1"):
        model = importlib.import_module(f"models.SCC_Model.{net}")
        net = getattr(model, net)

        model = net()

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        if gpu == "0,1":
            model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)

        model = model.cuda()

        return model

    @staticmethod
    def test(args):
        model_root = "./results/"
        model_path = "2023-03-14_08-57_MyNet_MSE_SHHA/"
        model_name = "epoch_2729_mae_71.9_mse_118.7.pth"

        log_path = model_root + model_path
        pth = log_path + model_name
        gt_path = log_path + "gt/"
        pred_path = log_path + "pred/"

        if not os.path.exists(gt_path):
            os.mkdir(gt_path)
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)

        model = Trainer.model_get(args.net, args.gpu)
        model.load_state_dict(torch.load(pth))

        model.eval()

        from datasets.SHHA.loading_data import loading_data

        train_loader, val_loader, restore_transform = loading_data()

        logger = get_logger("Test", log_path + "test.log", "w")

        maes = AverageMeter()
        mses = AverageMeter()

        for i, data in enumerate(val_loader):
            img, gt = data

            with torch.no_grad():
                img = img.cuda()
                gt = gt.cuda()
                output = model(img)

                gt = gt.squeeze().detach().cpu().numpy()
                output = output.squeeze().detach().cpu().numpy()

                np2jpg(gt, gt_path + f"{i+1}.jpg")
                np2jpg(output, pred_path + f"{i+1}.jpg")

                gt = np.sum(gt) / 100
                est = np.sum(output) / 100

                logger.info(f"[i: {i+1}] [est: {est:.2f}] [gt: {gt:.2f}]")

                maes.update(abs(est - gt))
                mses.update((abs(est - gt) ** 2))

        mae = maes.avg
        mse = np.sqrt(mses.avg)

        logger.info(f"[mae: {mae:.1f}] [mse: {mse:.1f}]")

    def config(self):
        self.time = time.strftime("%Y-%m-%d_%H-%M")

        self.output_path = "./results"
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.output_path = (
            f"{self.output_path}/{self.time}_{args.net}_{args.loss}_{args.dataset}"
        )
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.logger = get_logger("Train", f"{self.output_path}/train.log", "w")

        self.logger.info(f"[net: {args.net}]")
        self.logger.info(f"[loss: {args.loss}]")
        self.logger.info(f"[lr: {args.lr}]")
        self.logger.info(f"[lr_decay: {args.lr_decay}]")
        self.logger.info(f"[weight_decay: {args.weight_decay}]")
        self.logger.info(f"[max_epoch: {args.max_epoch}]")
        self.logger.info(f"[dataset: {args.dataset}]")

        self.writer = SummaryWriter(self.output_path)
        self.tensorboard_x = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test")
    parser.add_argument("--mode", default="Train", type=str)
    parser.add_argument("--gpu", default="0,1", choices=["0", "1", "0,1"], type=str)
    parser.add_argument("--net", default="GSANet", type=str)
    parser.add_argument("--loss", default="MSE", choices=["MSE", "GLCL"], type=str)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--lr_decay", default=1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, help="weight decay", type=float)
    parser.add_argument("--max_epoch", default=3000, help="max epoch", type=int)
    parser.add_argument("--dataset", default="SHHA", choices=["SHHA", "SHHB"], type=str)
    args = parser.parse_args()

    if args.mode == "Train":
        train_me = Trainer(args)
        train_me.forward()
    elif args.mode == "Test":
        Trainer.test(args)
