# -*- coding: utf-8 -*-
import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn
from utils import mkdir_p


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # # Training
        self.parser.add_argument('--epochs', default=100, type=int, metavar='N',
                                 help='number of total epochs to run')
        self.parser.add_argument('--batchSize', default=8, type=int, metavar='N',
                                 help='input batch size')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-7, type=float,
                                 metavar='LR', help='initial learning rate')
        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: \
                                e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--manualSeed', type=int, help='manual seed')
        # Dataset
        self.parser.add_argument('--phase', type=str, default='train', help='train, val,\
                                test, etc')
        self.parser.add_argument('--cropWidth', type=int, default=224, help='Crop to\
                                this width')
        self.parser.add_argument('--cropHeight', type=int, default=224, help='Crop to\
                                this height')
        self.parser.add_argument('--length', type=int, default=13, help='the length of\
                                time sequence')
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                 help='number of data loading workers (default: 4)')
        # Checkpoints
        self.parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar= \
            'PATH', help='path to save checkpoint (default: checkpoint)')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                 help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of\
                                the experiment. It decides where to store samples and models')
        # miscs
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                                 help='evaluate model on validation set')
        self.parser.add_argument('--model_arch', type=str, default='cnn_mstcn', help='The model arch you selected')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        # GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        self.opt.use_cuda = torch.cuda.is_available()
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # Random seed
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if self.opt.use_cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)
            cudnn.benchmark = True
            cudnn.enabled = True

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        expr_dir = os.path.join(self.opt.checkpoint, self.opt.name)
        mkdir_p(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
