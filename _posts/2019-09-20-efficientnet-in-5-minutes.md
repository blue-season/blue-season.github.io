---
layout: post
title: Efficientnet in 5 minutes
categories: [Machine Learning]
---

[EfficientNet](https://arxiv.org/abs/1905.11946) is evolved from the [MobileNet V2](https://arxiv.org/abs/1905.11946) building blocks, with the key insight
that scaling up the *width*, *depth* or *resolution* can improve a network's performance, and a balanced scaling of all three
is the key to maximizing improvements. It achieves the state of the art performance with much less parameters and FLOPS than other architectures.

All code shown here are powered by [PyWarm](https://github.com/blue-season/pywarm), a high level PyTorch API that makes the network definitions super clean.

## Building blocks

- The main building block, called *MBConv*, is similar to the bottleneck block from MobileNet V2.

```python
def conv_pad_same(x, size, kernel=1, stride=1, **kw):
    """ Same padding so that out_size*stride == in_size. """
    pad = 0
    if kernel != 1 or stride != 1:
        in_size, s, k = [torch.as_tensor(v)
            for v in (x.shape[2:], stride, kernel)]
        pad = torch.max(((in_size+s-1)//s-1)*s+k-in_size, torch.tensor(0))
        left, right = pad//2, pad-pad//2
        if torch.all(left == right):
            pad = tuple(left.tolist())
        else:
            left, right = left.tolist(), right.tolist()
            pad = sum(zip(left[::-1], right[::-1]), ())
            x = F.pad(x, pad)
            pad = 0
    return W.conv(x, size, kernel, stride=stride, padding=pad, **kw)


def conv_bn_act(x, size, kernel=1, stride=1, groups=1, 
        bias=False, eps=1e-3, momentum=1e-2, act=swish):
    x = conv_pad_same(x, size, kernel, stride=stride, groups=groups, bias=bias)
    return W.batch_norm(x, eps=eps, momentum=momentum, activation=act)


def mb_block(x, size_out, expand=1, kernel=1, stride=1,
        se_ratio=0.25, dc_ratio=0.2):
    """ Mobilenet Bottleneck Block. """
    size_in = x.shape[1]
    size_mid = size_in*expand
    y = conv_bn_act(x, size_mid, 1) if expand > 1 else x
    y = conv_bn_act(y, size_mid, kernel, stride=stride, groups=size_mid)
    y = squeeze_excitation(y, int(size_in*se_ratio))
    y = conv_bn_act(y, size_out, 1, act=None)
    if stride == 1 and size_in == size_out:
        y = drop_connect(y, dc_ratio)
        y += x
    return y
```

- EfficientNet uses [Swish](https://arxiv.org/abs/1710.05941) instead of [ReLU6](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf).

```python
def swish(x):
    return x*torch.sigmoid(x)
```

- Adds [squeeze and excitation](https://arxiv.org/abs/1709.01507) at the block output.

    - Use [global average pooling](https://arxiv.org/abs/1312.4400) to squeeze the entire receptive field into 1 channel embedding.

    - Remap the 1 channel embedding into more channels via excitation.

```python
def squeeze_excitation(x, size_se):
    if size_se == 0:
        return x
    size_in = x.shape[1]
    x = F.adaptive_avg_pool2d(x, 1)
    x = W.conv(x, size_se, 1, activation=swish)
    return W.conv(x, size_in, 1, activation=swish)
```

- Uses [drop connect](https://arxiv.org/abs/1603.09382) instead of dropout.

    - Randomly bypass layers by setting them to identity.

```python
def drop_connect(x, rate):
    if rate == 0:
        return x
    rate = 1.00-rate
    drop_mask = rate + torch.rand([x.shape[0], 1, 1, 1],
        device=x.device, requires_grad=False)
    return x/rate*drop_mask.floor()
```

## Scaling

The paper proposes to scale up *width*, *depth*, and *resolution* propotionally in order to maximize performance gains.

- Width: number of channels (hidden dimensions).

- Depth: total number of layers.

- Resolution: the input image size. The authors used bicubic upsampling to increase the resolution of input images.

- Intuitively, if we treat the network as a cylinder, then depth will determine its height and the width and resolution will determine its diameter.

- Therefore, when scaling these three dimensions, the ratio should be roughly:

```
depth : width*width : resolution*resolution ~= 1 : 1 : 1
```

- The paper propsed 8 baselines, B0 to B7, by gradually scaling up theses dimensions under some total FLOPS constraints.

- The B0 model has a specification as follows:

```python
spec_b0 = (
# size, expand, kernel, stride, repeat, squeeze_excitation, drop_connect
    (16, 1, 3, 1, 1, 0.25, 0.2),
    (24, 6, 3, 2, 2, 0.25, 0.2),
    (40, 6, 5, 2, 2, 0.25, 0.2),
    (80, 6, 3, 2, 3, 0.25, 0.2),
    (112, 6, 5, 1, 3, 0.25, 0.2),
    (192, 6, 5, 2, 4, 0.25, 0.2),
    (320, 6, 3, 1, 1, 0.25, 0.2), )


class WarmEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        warm.up(self, [2, 3, 32, 32])
    def forward(self, x):
        x = conv_bn_act(x, 32, kernel=3, stride=2)
        for size, expand, kernel, stride, repeat, se, dc in spec_b0:
            for i in range(repeat):
                stride = stride if i == 0 else 1
                x = mb_block(x, size, expand, kernel, stride, se, dc)
        x = conv_bn_act(x, 1280)
        x = F.adaptive_avg_pool2d(x, 1)
        x = W.dropout(x, 0.2)
        x = x.view(x.shape[0], -1)
        x = W.linear(x, 1000)
        return x
```

## Other tricks

- [AutoAugment](https://arxiv.org/abs/1805.09501) of inputs

    - Automatically search for best input augmentations that lead to highest validation accuracy

- Training weight decay

- Learning rate decay

## Notes

- Some users have suggested that EfficientNet, though theoretically uses much fewer FLOPS and number of parameters, does not translate directly
  to faster GPU inference and less GPU memory usage. In fact, some complained that the training speed is much slower than the well-established ResNets.

- People have also mentioned that EfficientNet is very hard to train. You will need to copy the hyperparameters very precisely and empoly all
  the tricks mentioned in the paper in order to getting close to the reported performance.
