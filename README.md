# SDA_EXP

# Notes

  - The codes only support training on single node. (single GPU with single server, or multi-GPUs with single server)

# Things to implement

  - Current ImageNet training uses warmup scheduler; Learning rate is increased from 0 to
0.256 in the first 5 epochs. (https://arxiv.org/abs/1706.02677)
  - Various augmentation techniques.

# Things to verify

  - Do we need `average_gradients` before `optimizer.step()`? (https://github.com/pytorch/examples/issues/659)
  - How many epochs do we have to train?
  - How large batch size do we have to set?

# References

  - How to use `DistributedSampler`
    <https://discuss.pytorch.org/t/distributedsampler-for-validation-set-in-imagenet-example/35273> \
    <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler>
  - How to use `DistributedDataParallel`
    <https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html>
  - How to get summaries of metrics
    <https://github.com/leo-mao/dist-mnist/blob/tensorflow-tutorial-amended/torch-dist/mnist-dist.py>
  - EfficientNet model code
    <https://github.com/lukemelas/EfficientNet-PyTorch>
  - ImageNet training code
    <https://github.com/pytorch/examples/tree/master/imagenet>

# How to run

```sh
$ python3 main.py
```

