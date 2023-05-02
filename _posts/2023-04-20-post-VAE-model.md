---
title: "VAE模型介绍及应用"
date: 2023-04-20
toc: true
categories:
  - 内容-多模态
tags:
  - VAE
  - 多模态生成
classes: wide
sidebar:
  - nav: "modal_docs"
---

# 1.背景
- 为何会出现VAE？填补了什么gap？

使用普通GAN能够生成逼真的图像，但具备几个缺点：（1）图像是根据任意噪声生成的，我们只能通过搜索整个分布，才可以知道初始噪声值生成某个图片（2）GAN训练的任务是区分真实和生成图片，没有限制图片的类型（比如生成猫图片必需像猫），只是样式像正常图片。

- VAE能做什么？

它作为生成模型，可以接受我们输入的*服从高斯分布*的数据，来生成图像。

- VAE原理简述 
  - 标准的自动编码器（AE）的原理是能够基于训练数据，压缩图片信息，即保存图片的编码向量；再通过解码器，重建这个图片。
  - 变分自动编码器（VAE）：我们希望模型不是仅"记忆并重建"图片的数据结构，还可以做到"生成"。因此，编码网络加入了约束，即通过训练，迫使隐藏层服从高斯分布。在推理阶段，我们可以从高斯分布里进行采样，并且送到解码器进行生成图片。这也是跟AE的本质区别。


# 2.VAE模型理解

- 论文[https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- 时间：2013

## 2.1 直白理解

AE模型学习一个隐向量，每个维度可能表达某些图片信息。并且通过该向量进行解码，生成图像。

<figure>
  <img src="{{ '/assets/images/vae-img2.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

但我们更希望做到的是，每个潜在属性表示为概率分布，我们可以以此进行调节，生成不通风格的图像。


<figure>
  <img src="{{ '/assets/images/vae-img3.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>


<figure>
  <img src="{{ '/assets/images/vae-img4.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>


## 2.2 模型框架

**模型图**

<figure>
  <img src="{{ '/assets/images/vae-img5.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

（1）模型框架

- 编码器：`q(z|x)` 输入图片x，训练获取最可能生成x的隐向量 z
- 解码器：`p(z|x)` 输入隐向量 z，推理生成图片x
- 损失函数：最大化`log(p(x|z)) + KL(q(z|x)||p(z))`
- 重新参数化技巧：如果直接从高斯分布里随机采样z，那么因为这个独立的动作，梯度是无法更新的。所以，假设我们经过encoder获取是高斯分布，可以抽离出均值和标准差，即 `z_i = \mu_i + \epsilon_i \odot \sigma_i, \epsilon_i \in N(0,1)`

（2）理解为什么是这样的计算流程？（不断更新中...）

>论文[https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)的 2.1 Problem scenario 详细讲解了为何需要如此设计

假设已经有某个分布的 z，我们通过解码器生成了 x 。我们能观察到 x ，但也想推理出 隐变量 z 的分布/性质：

`p(z|x)=\frac{p(x|z)p(z)}{p(x)}`

`p(z)` 可知，因为限定了高斯分布。本质在于学习 `p(x|z)` 此时，引入编码器。

编码器 `q(z|x)` 是能够近似 `p(x|z)` 并且具备 `tractable distribution` 

因此，引入 KL 散度去让`q(z|x)` 和 `p(z)` 的差异最小化（这里会有推理）。 此外，让模型学习时最大化复原图片的概率

# 3. 应用

[Jupyter地址][my-github-vae-1]

- 2个epoch: 相对模糊，部分非数字形状

<figure>
  <img src="{{ '/assets/images/vae-img-gen1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>

- 20个epoch: 相对清晰数字有轮廓

<figure>
  <img src="{{ '/assets/images/vae-img-gen2.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:600px">
</figure>


# 附录
- 参考链接：
  - openAI文章：https://openai.com/research/generative-models
  - CLIPdraw：https://www.crosslabs.org/blog/clipdraw-exploring-text-to-drawing-synthesis
  - 博客-VAE：
    - https://www.jeremyjordan.me/variational-autoencoders/
    - https://zhuanlan.zhihu.com/p/79536532
    - https://kvfrans.com/variational-autoencoders-explained/
    - https://zhuanlan.zhihu.com/p/108262170
  - 博客-GAN：
    - https://kvfrans.com/generative-adversial-networks-explained/
  - openAI-DALL-E：https://openai.com/research/dall-e

[clip-paper]: https://arxiv.org/abs/2103.00020
[my-github-vae-1]: https://github.com/Iven2166/models-learning/blob/main/deep-learning/modals-models/vae/VAE-demo1.ipynb