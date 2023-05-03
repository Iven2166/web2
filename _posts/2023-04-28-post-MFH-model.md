---
title: "MFH模型介绍"
date: 2023-03-10
toc: true
categories:
  - 内容-多模态
tags:
  - MFH
  - 多模态
  - VQA视觉问答
classes: wide
sidebar:
  - nav: "modal_docs"
---

# 论文介绍

论文 [Beyond Bilinear: Generalized Multimodal Factorized High-order Pooling for Visual Question Answering][mfh-paper] 
- 中文名称：多模态分解高阶池化方法
- 年份：2017
- 文章应用任务：视觉问答VQA
- 来源：IEEE 

## 摘要

> 视觉问答 (VQA) 具有挑战性，因为它需要同时理解图像的视觉内容和问题的文本内容。
> 
> 为了支持 VQA 任务，我们需要为以下三个问题找到好的解决方案：1）图像和问题的细粒度特征表示； 2）多模态特征融合，能够捕获多模态特征之间的复杂相互作用； 3）自动答案预测，能够考虑同一问题的多个不同答案之间的复杂相关性。
> 
> 1、对于细粒度的图像和问题表示，通过使用深度神经网络架构开发了一种“共同注意”机制来共同学习图像和问题的注意力，这可以让我们有效地减少不相关的特征并获得 图像和问题表示的更具辨别力的特征。
>
> 2、对于多模态特征融合，开发了一种广义的多模态分解高阶池化方法（MFH），通过充分利用它们的相关性来实现多模态特征的更有效融合
> 
> 3、对于答案预测，采用KL(Kullback-Leibler)散度作为损失函数，实现对多个意义相同或相似的不同答案之间复杂相关性的精确刻画，可以使我们获得更快的收敛速度，获得稍好的答案预测准确率 。
> 
> 结果：深度神经网络架构旨在将所有上述模块集成到一个统一模型中，以实现卓越的 VQA 性能。 借助我们的 MFH 模型集合，我们在大规模 VQA 数据集上实现了最先进的性能，并在 2017 年 VQA 挑战赛中获得亚军。

## 模型

### Multi-modal Low-rank Bilinear Pooling (MLB) 

基于两个特征向量的低维映射后的哈达玛乘积

$$z=MLB(x, y)=(U^Tx)\odot(V^Ty)$$

$$x\inR^m,y\inR^n,U\inR^{m\times o},V\inR^{n\times o}$$

$o$ 是输出特征的维度，为了提升模型的capacity（捕捉非线性特征的能力），在z之后一般会加上 tanh 激活函数


### Generalized Multi-modal Factorized High-order Pooling (MFH)


[mfh-paper]: https://arxiv.org/abs/1708.03619