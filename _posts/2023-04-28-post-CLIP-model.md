---
title: "CLIP模型介绍及应用"
date: 2023-04-28
toc: true
categories:
  - 内容-多模态
tags:
  - CLIP
  - 多模态
  - 对比学习
classes: wide
sidebar:
  - nav: "modal_docs"
header:
  - image: /assets/images/clip-img.png
---

本文旨在通过CLIP模型的原论文介绍，来剖析模型结构以及重点细节。并且进行应用，观察效果。

# 先看应用

应用：[github链接][my-github-clip-1]

我用clip模型，跑通了 zero-shot，进行了图片识别、OCR识别。感知效果还是不错的。我们可以用个初始印象，再跳到论文细节里，理解作者的用意。


# clip论文讲解

论文 [Learning Transferable Visual Models From Natural Language Supervision][clip-paper] 
- 2021年
- 发表于 arxiv
- 源码: [https://github.com/OpenAI/CLIP](https://github.com/OpenAI/CLIP)

## 摘要

1、提出CV的现状问题：预训练需要大量的标注数据 

2、数据解决方法：从图像的介绍文本（相对丰富）获取数据，4亿图像和文本对 

3、模型解决方法：构建标题和图像对应的任务，对比学习 

4、效果：zero-shot，用于OCR、图像分类、视频动作识别等下游任务;利用文本信息监督视觉任务自训练，本质就是将分类任务转化成了图文匹配任务，效果可与全监督方法相当 

> 最先进的计算机视觉系统经过训练可以预测一组固定的预定对象类别。 这种受限的监督形式限制了它们的通用性和可用性，因为需要`额外的标记数据`来指定任何其他视觉概念。 直接从有关`图像的原始文本`中学习是一种很有前途的替代方案，它可以利用更广泛的监督资源。 
> 
> 我们证明了预测`哪个标题与哪个图像对应的简单预训练任务`是一种有效且可扩展的方式，可以在从互联网收集的 `4 亿（图像、文本）对`数据集上从头开始学习 SOTA 图像表示。 
> 
> 预训练后，使用自然语言来引用学习到的视觉概念（或描述新概念），从而实现模型到下游任务的零样本迁移。 我们通过对 30 多个不同的现有计算机视觉数据集进行基准测试来研究这种方法的性能，涵盖 OCR、视频中的动作识别、地理定位和许多类型的细粒度对象分类等任务。 该模型可以轻松地迁移到大多数任务，并且通常可以与完全监督的基线相媲美，而无需任何数据集特定的训练。 例如，我们在 ImageNet zero-shot 上匹配原始 ResNet-50 的准确性，而无需使用它所训练的 128 万个训练示例中的任何一个。 

## 1. 介绍学界进展、gap、clip优越性

该部分介绍了学界在NLP领域的预训练进展，以及CV的预训练对应发展。

### NLP预训练任务的启发

文本预训练里，“文本到文本”作为标准化输入输出接口的发展使得与任务无关的架构能够零样本传输到下游数据集，从而消除了对专门输出头或数据集特定定制的需求。典型代表：GPT系列 能够在下游多种任务进行表现。

这些结果表明，在网络规模的文本集合（web-scale collections of text）中的现代预训练方法可获得的总体监督超过了用高质量密集标记（high-quality crowd-labeled）NLP数据集得到的监督效果。

### CV的过往进展
- Mori et al. (1999) 通过训练模型来预测与图像配对的文本文档中的名词和形容词，探索改进基于内容的图像检索。
- Quattoni et al. (2007)：证明可以通过在分类器X的权重空间中进行学习来学习更多数据有效的图像表示，这些分类器X是训练预测与图像相关的字幕中的单词
- Srivastava & Salakhutdinov (2012)：通过在低层级的图像和文本标签特征之上训练多模态深度玻尔兹曼机来探索深度表征学习。
- Joulin et al. (2016)：对这一工作领域进行了现代化改造，并证明经过训练以预测图像说明中的单词的 CNN 学习了有用的图像表示。 
- Thomee et al.，2016：将YFCC100M 数据集中图像的标题、描述和主题标签元数据转换为词袋多标签分类任务
- 采用更新的架构和预训练方法，VirTex (Desai & Johnson, 2020)、ICMLM (Bulent Sariyildiz et al., 2020) 和 ConVIRT (Zhang et al., 2020) 最近展示了基于转换器的语言的潜力 建模、掩码语言建模和对比学习，以从文本中学习图像表示

### CV在该领域为何还没有较强效果？--即本论文填补的gap

相对而言，CV 类似的做法效果较差，文中也列举了很多案例。
>This is likely because demonstrated performance on common benchmarks is much lower than alternative approaches.

相比于有监督学习（1000 个label），利用文本进行预训练：学习 18291 个label。但均使用了`静态`的softmax判别分类，以及缺乏`动态生成`的机制。限制了灵活性以及 zero-shot 的能力。
> Both approaches also use static softmax classifiers to
perform prediction and lack a mechanism for dynamic outputs. This severely curtails their flexibility and limits their
“zero-shot” capabilities

另外，规模差异也是一个因素。本文的CLIP直接上大规模，减少了gap，创造了 400百万 的大规模图片文本数据对。
>While Mahajan et al. (2018) and Kolesnikov et al. (2019) trained their models for accelerator years on millions to billions of images, VirTex, ICMLM, and ConVIRT trained for accelerator days on one to two hundred thousand images. 

## 2. 做法

### 小总结

这一部分介绍clip模型的基本做法。

模型：对比学习，预测N * N对图文数据，将图片分类任务转换成图文匹配任务

- 双流，2个encoder分别处理文本和图片数据，text encoder使用Transformer，image encoder用了2种模型，ResNet和Vision Transformer(ViT)；
- encoder representation直接线性投影到multi-modal embedding space； 
- 计算两个模态之间的cosine similarity，让N个匹配的图文对相似度最大，不匹配的图文对相似度最小； 
- 对称的cross-entropy loss
- 数据增强：对resized图片进行random square crop

<figure>
  <img src="{{ '/assets/images/clip-img.png' | relative_url }}" alt="clip-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

### 2.1 自然语言监督（Natural Language Supervision）

本段主要阐述了NLP相比CV在预训练的"天然优势"：从语言里学习，有诸多好处。
- 不必拘泥于传统图像学习里的静态机器学习的样式（`“machine learning
compatible format”`），比如图像的固定分类标准
- 网络的文本数据量级巨大
- 无监督或者半监督，不仅学习表征，还将表征联系到了语言，所以可以灵活地进行zero-shot（比如，输入图片，有一段话进行表示）
>Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot  transfer

### 2.2 构建数据集（Creating a Sufficiently Large Dataset）

- 通过搜索query，构造了 400百万 的图像-文本对
- 总词数规模达到 WebText （GPT-2）

### 2.3 选择高效预训练方法（Selecting an Efficient Pre-Training Method）

- 首先，CV之前的预训练模型，仅在预测1000分类时计算量已十分巨大；开放式预测表达图像的语言表述似乎更难。因此，要做好语言相关的预训练，关键得寻找高效的训练任务和方法。
- 最开始的做法，其实联合训练了 CNN 和 文本transformer 去做图像的语言分类预测； 发现同等准确率时，效率低于基准的词袋预测模型
> Our initial approach, similar to VirTex, jointly trained an image CNN and text transformer from scratch to predict the caption of an image. In Figure 2 we show that a 63 million parameter transformer language model, which already uses twice the compute of its ResNet-50 image encoder, learns to recognize ImageNet classes three times slower than a much simpler baseline that predicts a bag-ofwords encoding of the same text
- 究其原因，是因为预测具体的词是非常困难的任务（一张图片会有同性质但不完全相同的描述）。因此，转化为对比学习。不仅zero-shot迁移到 IMAGENET 的准确度高，而且效率高。
> Recent work in contrastive representation learning for images has found that contrastive objectives can learn better representations than their equivalent predictive objective (Tian et al., 2019)

<figure>
  <img src="{{ '/assets/images/clip-img2.png' | relative_url }}" alt="clip-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

有几点训练的细节

- 学习多模态表征的方式：对比学习任务，最大化 N 真实图像-文本对的概率，最小化 N^2 - N 的负样本图像-文本对的概率。
- 由于数据量级很大，过拟合不是主要问题
- 从头训练CLIP，不会初始化 文本和图片的encoder
- 从encoder到多模态的同一空间里，不使用 非线性的映射，使用linear
- 数据增强：仅进行了图片缩放后的裁剪
- 温度参数：在计算 logits 里的 tau 系数，不是超参数，而是放到训练里 

<figure>
  <img src="{{ '/assets/images/clip-img3.png' | relative_url }}" alt="clip-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

### 2.4 选择和缩放模型架构（Choosing and Scaling a Model）

- 图像encoder：（1）ResNet-50 （2）Vision Transformer（ViT）
- 文本encoder：Transformer基础版本
  - 63M-parameter 12-layer 512-wide model with 8 attention heads
  - 文本截断：76
  - 采用BPE编码
  - 将 [EOS] token 视为表征提取

### 2.5 训练

细节见原文吧～ 

> We train all models for 32 epochs.
> 
> Initial hyperparameters were set using a combination of grid searches, random search, and manual tuning on the baseline ResNet-50 model when trained for 1 epoch.
> 
> The learnable temperature parameter τ was initialized to the equivalent of 0.07 from (Wu et al.,2018) and clipped to prevent scaling the logits by more than 100 which we found necessary to prevent training instability.
> 
> We usea very large minibatch size of 32,768.
> 
> Mixed-precision (Micikevicius et al., 2017) was used to accelerate training and save memory.
> 
> To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016), half-precision Adam statistics (Dhariwalet al., 2020), and half-precision stochastically rounded text encoder weights were used.
> 
> The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs.

## 3. 实验

### 3.1 zero-shot 迁移

图片分类的zero-shot指的是对未知类别进行推理。本文的zero-shot指的是对未知任务进行推理，通过zero-shot transfer衡量任务学习的能力。`Visual N-Grams (Li et al., 2017)` 是第一个将zero-shot transfer应用到图片分类任务上的模型。模型用于学习长度为1~5grams的共142806个visual n-grams，对输入的图片，最大化对应的n-grams的probability。

同样的，CLIP在进行zero-shot transfer时，将数据集中的类别标签转换为文字描述，主要步骤如下：

- 输入：考虑到大部分的数据集的标签都是以单词的形式存在的，比如“bird”，“cat”等等，然而在预训练阶段的文本描述大多都是某个短句，为了填补这种数据分布上的差别，作者考虑用“指示上下文”（guide context）对标签进行扩展。可以用a photo of a “object".作为文本端的输入，其中的``恰恰是需要预测的zero-shot标签。（100个类别就是100个文本描述）； 
- 转换向量：经过2个encoder，分别输出image和text的feature embedding； 
- 计算cosine similarity； 
- 预测类别：multinomial logistic regression classifier。
> Note that this prediction layer is a multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling. 

#### 3.1.3 对比可视化n-grams算法

虽然大幅领先，但除了模型架构之外，还有模型规模、数据采样和模型推出时间gap、训练数据量规模等原因。

#### 3.1.4. 提示词 PROMPT ENGINEERING AND ENSEMBLING

- 提示词能够提升模型表现，比如相比无提示词，提升 ImageNet 预测准确率 1.3pp
- 针对任务细节，还可以调整提示词
>This often improves performance over the baseline of using only the label text. For instance, just using this prompt improves accuracy on ImageNet by 1.3%.
> 
> For example on Oxford-IIIT Pets, using “A photo of a flabelg, a type of pet.”
> 
> For OCR datasets, we found that putting quotes around the text or number to be recognized improved performance.
> 
> we found that on satellite image classification datasets it helped to specify that the images were of this form and we use variants of “a satellite photo of a flabelg.”

原文后面再提供了大量数据集和对比细节。

[//]: # (### 3.2 学习表征)
[//]: # (### 3.3 鲁棒性 Robustness to Natural Distribution Shift)

## 衍生问题

- 如何把握数据质量？比如想让图片和文本能够有ocr任务能力，一张具备文本的限速照片，文本应该讲述限速多少
- 这种规模的大模型，学习率等超参数如何调节？
- 用小模型作为encoder提取图片和文本特征，效果会减弱多少？
- 公司业务如何应用？做图片qa、图片分类，但似乎做不了文本生成图片？



[clip-paper]: https://arxiv.org/abs/2103.00020
[my-github-clip-1]: https://github.com/Iven2166/models-learning/blob/main/deep-learning/modals-models/clip/clip-hugging.ipynb
