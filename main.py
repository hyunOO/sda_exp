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

from tensorboardX import SummaryWriter

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
    '--use_parallel', action='store_true',
    help='Use DataParallel API. '
)
parser.add_argument(
    '--use_distributed', action='store_true',
    help='Use multi-processing distributed training to launch. '
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
    '--weight_decay', default=1e-4, type=float,
    help=('weight decay. ')
)
parser.add_argument(
    '--batch_size', default=256, type=int,
    help=('mini-batch size. ')
)
parser.add_argument(
    '--epochs', default=90, type=int,
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
    if args.use_distributed:
        print('You use DistributedDataParallel API with multi-GPUs. ')
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args))
    elif args.use_parallel:
        print('You use DataParallel API with mulit-GPUs. ')
        main_worker(args.gpu, ngpus_per_node, args)
    else:
        # If args.use_distributed is False, we use single GPU
        assert args.gpu is not None
        print('You use single GPU. ')
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.use_distributed:
        print('Initializing to use multi GPUs. ')
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=ngpus_per_node,
            rank=gpu)

    if args.model_type.startswith('efficientnet'):
        print('Creating model: {}'.format(args.model_type))
        model = EfficientNet.from_name(args.model_type)

    if args.use_distributed:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        # https://discuss.pytorch.org/t/is-average-the-correct-way-for-the-gradient-in-distributeddataparallel-with-multi-nodes/34260/6
        args.lr = args.lr / ngpus_per_node
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu])
    elif args.use_parallel:
        model = nn.DataParallel(model)
        model.cuda()
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    train_loader, train_sampler = imagenet_loader(
            is_train=True,
            is_distributed=args.use_distributed,
            dataset_dir=os.path.join(args.data_dir, 'train'),
            batch_size=args.batch_size,
            num_workers=args.workers,
            num_replicas=ngpus_per_node,
            rank=gpu)
    print('Complete loading training data, length: {}'
        .format(len(train_loader)))

    val_loader, val_sampler = imagenet_loader(
        is_train=False,
        is_distributed=args.use_distributed,
        dataset_dir=os.path.join(args.data_dir, 'val'),
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_replicas=ngpus_per_node,
        rank=gpu)
    print('Complete loading validation data, length: {}'.
        format(len(val_loader)))

    criterion = nn.CrossEntropyLoss()
    # Experiment settings from https://arxiv.org/pdf/1608.06993.pdf Section 4.2
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    # learning rate decays by 0.97 every 2.4 epochs
    # TODO: Warm up learning rate https://arxiv.org/pdf/1706.02677.pdf
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60],
        gamma=0.1)

    best_acc1 = 0
    best_acc5 = 0
    writer = SummaryWriter(
        log_dir=os.path.join(os.getcwd(), 'log', args.model_type))

    for epoch in range(args.epochs):
        print('Starts epoch: {} '.format(epoch + 1))
        if args.use_distributed:
            train_sampler.set_epoch(epoch)

        train_top1, train_top5 = train(
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
        writer.add_scalar('top1_acc/train', train_top1, epoch)
        writer.add_scalar('top5_acc/train', train_top5, epoch)

        acc1, acc5 = validate(
            args=args,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            gpu=gpu,
            ngpus_per_node=ngpus_per_node,
            sampler=val_sampler)
        writer.add_scalar('top1_acc/val', acc1, epoch)
        writer.add_scalar('top5_acc/val', acc5, epoch)

        if not args.use_distributed or \
                (args.use_distributed and gpu == 0):
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

            if best_acc5 < acc5:
                best_acc5 = acc5

            print('Epoch: {}, current acc: {}, best acc1: {}, best_acc5: {}'
                .format(epoch + 1, acc1, best_acc1, best_acc5))


def train(
        args, train_loader, model, criterion,
        optimizer, scheduler, epoch, gpu,
        ngpus_per_node, sampler):
    top1 = 0
    top5 = 0
    total = 0

    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1 += acc1
        top5 += acc5
        total += target.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            top1_acc = float(acc1 / target.size(0))
            print('Training epoch: {}, batch : {} / {}, target size: {}, batch top1 acc: {}'
                .format(epoch + 1, i, len(train_loader),
                        target.size(0), top1_acc))

    if args.use_distributed:
        top1_acc = float(metric_average(top1 / ngpus_per_node) / total)
        top5_acc = float(metric_average(top5 / ngpus_per_node) / total)
        if gpu == 0:
            print('Ends training epoch: {}, top1 acc: {}, top5 acc: {}'
                .format(epoch + 1, top1_acc, top5_acc))

    if not args.use_distributed:
        top1_acc = float(top1 / total)
        top5_acc = float(top5 / total)

        print('Ends training epoch: {}, top1 acc: {}, top5 acc: {}'
              .format(epoch + 1, top1_acc, top5_acc))

    return top1_acc, top5_acc


def validate(
        args, val_loader, model, criterion,
        epoch, gpu, ngpus_per_node, sampler):
    top1 = 0
    top5 = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1 += acc1
            top5 += acc5
            total += target.size(0)

        if args.use_distributed:
            top1_acc = float(metric_average(top1 / ngpus_per_node) / total)
            top5_acc = float(metric_average(top5 / ngpus_per_node) / total)
            if gpu == 0:
                print('Ends validating epoch: {}, top1 acc: {}, top5 acc: {}'
                    .format(epoch + 1, top1_acc, top5_acc))

        if not args.use_distributed:
            top1_acc = float(top1 / total)
            top5_acc = float(top5 / total)

            print('Ends validating epoch: {}, top1 acc: {}, top5 acc: {}'
                  .format(epoch + 1, top1_acc, top5_acc))

    return top1_acc, top5_acc


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


# Reference from https://github.com/Jongchan/Pytorch-Horovod-Examples/blob/master/examples/cifar100/main_horovod.py#L298
# Reference form https://github.com/leo-mao/dist-mnist/blob/tensorflow-tutorial-amended/torch-dist/mnist-dist.py
def metric_average(target):
    tensor = torch.tensor(target, requires_grad=False, device='cuda')
    dist.reduce(tensor, 0)
    return float(tensor)


if __name__ == '__main__':
    main()
