import os
import random
import argparse

import torch
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from networks.efficientnet import EfficientNet

from dataloader import imagenet_loader

from utils import AverageMeter
from utils import ProgressMeter

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', default='/mnt/home/20160022', type=str,
    help=('path to a dataset. ')
)
parser.add_argument(
    '--model_saving_dir', default='/mnt/home/20160022/saved_models',
    type=str, help=('path to a dataset. ')
)
parser.add_argument(
    '--seed', default=None, type=int,
    help=('seed for initializing training. ')
)
parser.add_argument(
    '--workers', default=4, type=int,
    help=('number of data loading workers. ')
)
parser.add_argument(
    '--dist_url', default='tcp://127.0.0.1:3000', type=str,
    help=('url used to set up distributed training. ')
)
parser.add_argument(
    '--dist_backend', default='nccl', type=str,
    help=('distributed backend. ')
)
parser.add_argument(
    '--gpu', default=0, type=int,
    help=('GPU id to use. ')
)
parser.add_argument(
    '--multiprocessing_distributed', default=True, type=bool,
    help=('Use multi-processing distributed training to launch. ')
)
parser.add_argument(
    '--model_type', default='efficientnet-b0', type=str,
    help=('Type of model to use, '
          'efficientnet-b0 to efficientnet-b9 are possible. ')
)
parser.add_argument(
    '--lr', default=0.256, type=float,
    help=('initial learning rate. ')
)
parser.add_argument(
    '--momentum', default=0.9, type=float,
    help=('momentum. ')
)
parser.add_argument(
    '--weight_decay', default=1e-5, type=float,
    help=('weight decay. ')
)
parser.add_argument(
    '--batch_size', default=256, type=int,
    help=('mini-batch size. ')
)
parser.add_argument(
    '--epochs', default=200, type=int,
    help=('Epochs to train. ')
)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print('You have chosen to seed training. ')

    # TODO: Handling distributed training with multiple nodes
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print('You use multi GPUs in a single node. ')
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args))
    else:
        # If args.multiprocessing_distributed is False, we use single GPU
        assert args.gpu is not None
        print('You use single GPU. ')
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.multiprocessing_distributed:
        print('Initializing to use multi GPUs. ')
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=ngpus_per_node,
            rank=gpu)

    if args.model_type.startswith('efficientnet'):
        print('Creating model: {}'.format(args.model_type))
        model = EfficientNet.from_name(args.model_type)

    if args.multiprocessing_distributed:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    train_loader, train_sampler = imagenet_loader(
            is_train=True,
            is_distributed=args.multiprocessing_distributed,
            dataset_dir=os.path.join(args.data_dir, 'train'),
            batch_size=args.batch_size,
            num_workers=args.workers,
            num_replicas=ngpus_per_node,
            rank=gpu)
    print('Complete loading training data, length: {}'
        .format(len(train_loader)))

    val_loader, val_sampler = imagenet_loader(
        is_train=False,
        is_distributed=args.multiprocessing_distributed,
        dataset_dir=os.path.join(args.data_dir, 'val'),
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_replicas=ngpus_per_node,
        rank=gpu)
    print('Complete loading validation data, length: {}'.
        format(len(val_loader)))

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    # Experiment settings from https://arxiv.org/pdf/1905.11946.pdf
    optimizer = torch.optim.RMSprop(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # learning rate decays by 0.97 every 2.4 epochs
    # TODO: Warm up learning rate https://arxiv.org/pdf/1706.02677.pdf
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=len(train_loader) * 2.4,
        gamma=0.97)

    best_acc1 = 0
    for epoch in range(args.epochs):
        print('Starts epoch: {} '.format(epoch + 1))
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        train(
            args=args,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            gpu=gpu,
            ngpus_per_node=ngpus_per_node,
            sampler=train_sampler)
        acc1, acc5 = validate(
            args=args,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            gpu=gpu,
            ngpus_per_node=ngpus_per_node,
            sampler=val_sampler)

        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and gpu == 0):
            if best_acc1 < acc1:
                best_acc1 = acc1
                state = {
                    'epoch': epoch + 1,
                    'model_type': args.model_type,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(
                    args.model_saving_dir, 'checkpoint.pth'))

            print('Epoch: {}, current acc: {}, best acc: {}'
                .format(epoch + 1, acc1, best_acc1))


def train(
        args, train_loader, model, criterion,
        optimizer, scheduler, epoch, gpu,
        ngpus_per_node, sampler):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        print('acc1: ', acc1)
        print('acc5: ', acc5)

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print('Training epoch: {}, batch : {} / {}, target size: {}, acc@1: {top1.avg:.3f}'
                .format(epoch + 1, i, len(train_loader),
                        images.size(0), top1=top1))

    if args.multiprocessing_distributed:
        top1_acc = metric_average(top1.avg / ngpus_per_node)
        top5_acc = metric_average(top5.avg / ngpus_per_node)
        if gpu == 0:
            print('Ends training epoch: {}, top1 acc: {}, top5 acc: {}'
                .format(epoch + 1, top1_acc, top5_acc))

    if not args.multiprocessing_distributed:
        top1_acc = top1.avg
        top5_acc = top5.avg

        print('Ends training epoch: {}, top1 acc: {}, top5 acc: {}'
              .format(epoch + 1, top1_acc, top5_acc))

    return top1_acc, top5_acc


def validate(
        args, val_loader, model, criterion,
        epoch, gpu, ngpus_per_node, sampler):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_acc = None
    top5_acc = None

    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        if args.multiprocessing_distributed:
            top1_acc = metric_average(top1.avg / ngpus_per_node)
            top5_acc = metric_average(top5.avg / ngpus_per_node)
            if gpu == 0:
                print('Ends validating epoch: {}, top1 acc: {}, top5 acc: {}'
                    .format(epoch + 1, top1_acc, top5_acc))

        if not args.multiprocessing_distributed:
            top1_acc = top1.avg
            top5_acc = top5.avg

            print('Ends validating epoch: {}, top1 acc: {}, top5 acc: {}'
                  .format(epoch + 1, top1_acc, top5_acc))

    return top1_acc, top5_acc


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Reference from https://github.com/Jongchan/Pytorch-Horovod-Examples/blob/master/examples/cifar100/main_horovod.py#L298
# Reference form https://github.com/leo-mao/dist-mnist/blob/tensorflow-tutorial-amended/torch-dist/mnist-dist.py
def metric_average(target):
    tensor = torch.tensor(target, requires_grad=False, device='cuda')
    dist.reduce(tensor, 0)
    return float(tensor)


if __name__ == '__main__':
    main()
