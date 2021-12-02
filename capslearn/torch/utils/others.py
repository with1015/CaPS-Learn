import argparse
import os
import time


def argument_parsing():
    parser = argparse.ArgumentParser(description='PyTorch Training with CaPS-Learning System.')

    # Common arguments
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')

    # Hyperparameters for training
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='Batch size for local machine learning (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # Distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--rank', default=None, type=int,
                        help='Define DDP rank')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Define world size for DDP')
    parser.add_argument('--master_addr', default='localhost', type=str,
                        help='Define master address of DistributedDataParallel')
    parser.add_argument('--master_port', default='28000', type=str,
                        help='Define master port of DistributedDataParallel')


    # CapsOptimizer option
    parser.add_argument('--log-mode', dest='log_mode', action='store_true',
                        help='Enable log for CaPS optimizer')
    parser.add_argument('--log-dir', default=None, type=str,
                        help='Log directory of CaPS optimizer')

    return parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

