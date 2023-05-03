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

$$x \in R^m, y \in R^n,U \in R^{m \times o},V \in R^{n \times o}$$

$o$ 是输出特征的维度，为了提升模型的capacity（捕捉非线性特征的能力），在$z$之后一般会加上 $tanh$ 激活函数

### Generalized Multi-modal Factorized High-order Pooling (MFH)

- 首先回顾了 MFB（Multi-modal Factorized Bilinear Polling），并且给出跟MLB的关系。
- 将MFB作为基础的模块，扩展 bilinear pooling 到  generalized high-order pooling (MFH)方法，做法是利用多个MFB叠加起来

#### A. Multi-modal Factorized Bilinear Pooling (MFB)

**模型介绍** 

最简单的多模态的线性融合方式: 

$$z_i = x^T W_i y, where x\in R^m, y\in R^n, W_i \in R ^{m \times n}, z_i \in R$$

学习 $o$ 维的输出 $z$，我们需要  $W = [W_i, ..., W_o] \in R^{m\times n \times o}$ 虽然 bilinear poolng 能够有效学习到成对特征的交互，但也有*很高的参数量级*。

因此，需要矩阵分解技巧。$W_i \in R^{m \times n}$ 能够分解为两个低秩（low-rank）的矩阵。模型压缩，对于i，参数量从 m*n 下降到 m+n（比如图片和文本分别$500$维度，原需$500 /times 500=25w$参数量，分解后只需 $1k$ 参数量）

$$z_i = x^T U_i V_i^{T} y = \sum^{k}_{d=1} x^T u_d v_d^{T} y = 1^T(U_i^T x \odot V_i^T y)$$

$$where, U_i=[u_1, ..., u_k] \in R^{m \times k}, V_i=[v_1, ..., v_k] \in R^{n \times k}, 1 \in R^k, z \in R^o$$

对于$o$个输出（学习到的交叉类型），我们学习到参数：

$$U = [U_1, ..., U_o] \in R^{m \times k \times o}, V = [V_1, ..., V_o] \in R^{n \times k \times o}$$

再通过 reshape 得到 $U'\in R^{m \times ko}, V'\in R^{n \times ko}$

最终，通过聚合函数，
$$z = SumPool(U'^Tx \odot V'^Ty, k), where U'^Tx \odot V'^Ty \in R^{ko}$$

**MFB 模型图**

MFB 的详细过程如图 2（a）所示。 该方法可以通过组合一些常用层（例如全连接层、逐元素乘法层和池化层）轻松实现。 此外，为了防止过度拟合，在逐元素乘法层之后添加了 dropout 层。 由于引入了逐元素乘法，输出神经元的大小可能会发生巨大变化，并且模型可能会收敛到一个不令人满意的局部最小值。 因此，power normalization 和 layer normalization 层附加在 MFB 输出之后。 整个 MFB 模块的流程图如图 2（b）所示。

<figure>
  <img src="{{ '/assets/images/mfh-img1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

实际源码处理示例（含矩阵shape）

<figure>
  <img src="{{ '/assets/images/mfh-img2.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

**与MLB的关系**

MLB：回顾最前面的 Multi-modal Low-rank Bilinear Pooling，可以理解为最基础的多模态交叉、融合的方法

- MLB 是 MFB 在 $k=1$ 的特例
- MLB 相比 MFB 可能表征不充分
  - MFB 是两个阶段：（1）不同模态的特征扩充为高维度空间，并且进行元素层面的乘积 （2）依次进行 sum pooling、归一化层 来压缩高维度空间至紧凑的表征空间
  - MLB 是将不通模态的特征映射到低维度空间，再进行元素层面的乘积

#### B. From Bilinear Pooling to Generalized High-order Pooling

我们将 MFB 表述为以下两个公式（阶段），首先将 $x$, $y$ 进行映射到高维度空间，进行点乘，进行交叉，再dropout，获得$z_{exp} /in R^{ko}$。再降维得到$z \in R^{o}$：

$$z_{exp} = MFB_{exp}(x,y) = Dropout(U'^{T}x \odot V'^{T}y) /in R^{ko}$$

$$z = MFB_{sqz}(z_{exp}) = Norm(SumPool(z_{exp})) \in R^{o}$$

如果将 $p$个 MFB 模块进行堆叠，我们将 $z_{exp}$ 表达过程修改为下式子，（$i \in {1,2,...,p}$ 为模块index）。这样改造后，每一层会和上一层的 $z_{exp}$ 再进行一次融合。原文里使用 $p < 4$ 作为实验参数。MFB 实际上是 MFH 的 $p=1$ 特例。

$$z_{exp}^{i} = MFB_{exp}^{i}(x,y) = z_{exp}^{i-1} \odot Dropout(U'^{T}x \odot V'^{T}y)$$



[mfh-paper]: https://arxiv.org/abs/1708.03619