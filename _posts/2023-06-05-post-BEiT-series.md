---
title: "BEiT系列"
date: 2023-06-05
toc: true
categories:
  - content-multi-modal
tags:
  - BEiT
  - 多模态
classes: wide
sidebar:
  - nav: "modal_docs"
---

在2022年，微软推出 [BEiT-3](https://arxiv.org/abs/2208.10442)

## 背景

### 多模态预训练大一统
语言、视觉和多模态等领域的预训练开始呈现大一统（big convergence）趋势。通过对大量数据的大规模预训练，我们可以更轻松地将模型迁移到多种下游任务上。
- 骨干网络统一：微软亚洲研究院提出了一个统一的骨干网络 Multiway Transformer，可以同时编码多种模态。
- 基于掩码数据建模（masked data modeling）的预训练已成功应用于多种模态，如文本和图像。微软亚洲研究院的研究员们将图像看作一种语言，实现了以相同的方式处理文本和图像两种模态任务的目的。
- 扩大模型规模和数据大小可提高基础模型的泛化能力，从而提升模型的下游迁移能力

# 视觉基础模型：BEiT-1

通过对图像的掩码建模，推出BEiT模型；文本是离散信号，而图片是连续信号，所以需要经过特殊处理。

- 编码学习 tokenizer，将图像变成离散的视觉符号（visual token），类似文本
- 图像切成多个小“像素块”(patch)，每个像素块相当于一个字符

在用 BEiT 预训练时，模型可以随机遮盖图像的部分像素块，并将其替换为特殊的掩码符号[M]，然后在骨干网络 ViT 中不断学习、预测实际图片的样子

<figure>
  <img src="{{ '/assets/images/beit1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

一些细节 [https://huggingface.co/docs/transformers/model_doc/beit](https://huggingface.co/docs/transformers/model_doc/beit)
- BEiT模型是 ViT，但是预训练而非监督的。超过了  original model (ViT) ， Data-efficient Image Transformers (DeiT)
- 需要将每个图片进行缩放到统一的size
- BEiT uses relative position embeddings, inspired by the T5 model.


# 多模态基础大模型：BEiT-3

## 骨架

[Multiway Transformer](https://arxiv.org/abs/2111.02358)

输入：图片、文本、多模态（前两者concat），其中图片是设定好 (P,P) 的预定patch大小，然后将RGB图像像素reshape进去，再通过线性全连接层得到patch embedding。还可以加入可以学习的 CLS 的特殊token

<figure>
  <img src="{{ '/assets/images/multiway-transformer-inputshape.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

## BEiT-3

BEiT-3 利用一个共享的 Multiway Transformer 结构，通过在单模态和多模态数据上进行掩码数据建模完成预训练，并可迁移到各种视觉、视觉-语言的下游任务中。

- 骨干网络：Multiway Transformer
  - 每个 Multiway Transformer 由一个共享的自注意力模块（self-attention）和多个模态专家(modality experts)组成，每个模态专家都是一个前馈神经网络（feed-forward network） 
  - **共享自注意力模块可以有效学习不同模态信息的对齐**，并对不同模态信息深度融合编码使其更好地应用在多模态理解任务上
- 预训练任务：掩码数据建模 （masked data modeling）。研究员们在单模态（即图像与文本）和多模态数据（即图像-文本对）上通过统一的掩码-预测任务进行 BEiT-3 预训练。预训练期间，会随机掩盖一定百分比的文本字符或像素块，模型通过被训练恢复掩盖的文本字符或其视觉符号，来学习不同模态的表示及不同模态间的对齐。
  - 相对而言，对比学习则需要较大的batch，引致GPU内存相关的工程挑战
- 扩大模型规模：BEiT-3 由40层 Multiway Transformer 组成，模型共包含19亿个参数。在预训练数据上，BEiT-3 基于多个单模态和多模态数据进行预训练，多模态数据从五个公开数据集中收集了大约1,500万图像和2,100万图像-文本对；单模态数据使用了1,400万图像和160GB文本语料。

附录：
- [通用多模态基础模型BEiT-3：引领文本、图像、多模态预训练迈向“大一统”](https://www.msra.cn/zh-cn/news/features/beit-3)
- [AGI机构](https://thegenerality.com/agi/about.html)