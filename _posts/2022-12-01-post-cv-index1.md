---
title: "CV-首页(博客迁移中)"
date: 2022-12-01
toc: true
categories:
  - 内容-CV
classes: wide
words_per_minute: 10
sidebar:
  - nav: "cv_docs"
---

# 内容理解（分类居多）

## AlexNet
- 年份：2012年
- 效果：ILSVRC图像分类任务 top-5错误率降至 15.3%
- 改进：
  - 用ReLU代替sigmoid做激活函数，缓解梯度消失问题
  - 局部响应归一化（LRN）
  - dropout、数据扩充等技巧

## VGGNet
- 年份：由Karen Simonyan和Andrew Zisserman在2014年提出
- 简介：VGG网络采用了多个小卷积核和深层网络结构，可以进一步提高图像分类和检测的准确度。相比AlexNet，VGG使用了更深的网络结构，证明了增加网络深度能够在一定程度上影响网络性能
- 效果：ILSVRC图像分类任务 top-5错误率降至 8%，模型集成后可以到6.8%
- 改进：
  - 用 $3 \times 3$ 代替$5 \times 5$ 或者 $7 \times 7$的大卷积核。参数量和计算量更少，但获得同样的感受野，同时增加网络的深度
  - 用 $2 \times 2$ 代替$3 \times 3$ 池化核
  - 网络更深特征图更宽。卷积核专注于扩大通道数，池化专注于缩小高和宽，使得模型更深更宽的同时，计算量的增加不断放缓；
  - 多尺度。作者从多尺度训练可以提升性能受到启发，训练和测试时使用整张图片的不同尺度的图像，以提高模型的性能。
  - 去掉LRN
  - 层级：深度达到20层

<figure>
  <img src="{{ '/assets/images/vgg-img1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

## Inception-V1
- 年份：Google团队, 2014年
- 效果：采用了多个并行的卷积核和池化层，以及降维和升维等技术，可以有效地提取图像中的多尺度特征。ImageNet2012 数据集，突破性分类的top-5错误率降至 6.67%
- 改进：
  - 网络中的大通道卷积层替换为多个小通道卷积层的多分支结构；同时利用 1、3、5 三种卷积核进行多路特征提取，让网络稀疏化，增加对多个尺度特征的适应
  - 提出bottleneck结构：在计算较大的卷积层之前，使用 $1 \times 1$ 卷积层进行对通道数进行压缩，减少计算量，然后再通过 $1 \times 1$ 卷积层进行复原
  - ***辅助分类器***：在模型的中间层，拉出辅助分类器（推理时不参与），来计算误差损失进行反向传播，缓解深度网络的梯度消失问题. $total_{loss} = real_{loss} + 0.3 * aux_{loss1} + 0.3 * aux_{loss2}$

<figure>
  <img src="{{ '/assets/images/inception-v1-img1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

<figure>
  <img src="{{ '/assets/images/inception-v1-img2.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

## ResNet
- 背景：随着网络层数的增加，网络的训练误差和测试误差都会上升。称之为网络的退化。（不是过拟合：训练误差降低，测试误差上升）
- 解决：采用跳层连接（shortcut connection)
- 效果：ImageNet 2012 数据集，单模型可以让top-5错误率降低至4.49%，集成可以达到 3.57%
- 改进
  - 缩短反向传播到各层的路径，有效抑制梯度消失现象；在不断加深网络时，性能不会显著下降
  - 如果网络在加深层数时发现性能退化，它可以在控制网络里 近道 和 非近道 的组合比例，来退回到之前浅层的状态
  - 使得现有的网络，可以加到上千层


# 其它问题

$1 \times 1$ 卷积核为何可以减少参数量？

<figure>
  <img src="{{ '/assets/images/cv-others-1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>
