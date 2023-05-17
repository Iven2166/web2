---
title: "diffusion model"
date: 2023-04-27
categories:
  - AIGC
tags:
  - diffusion
  - multi-modal
---

# 模型简述

<figure>
  <img src="{{ '/assets/images/ddpm-algorithm1.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

- 从 0 到 T 时刻，不断的加入高斯噪声，公式内带有温度参数等超参数，能够控制

## 如何做 text-image 生成？

[dataset-LAION] 有巨大的文本-图片数据集，一般文生图都是在上面进行训练的。

在 denoise 的阶段，不仅是输入T时刻带有高斯噪声的图片，还需要输入噪声的严重程度（哪个阶段），以及文本，这三个因素放到 noise-preditor 里面，生成 高斯噪声，并且减去该高斯噪声，生成了 T-1 时刻带有高斯噪声的图片。

<figure>
  <img src="{{ '/assets/images/ddpm-img10.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img11.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


 $x_0$ 的生成概率需要1到T的步骤及计算

<figure>
  <img src="{{ '/assets/images/ddpm-img12.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img13.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

虽然想象中 $q(x_t | x_0)$ 是逐步叠加噪声得到的，但其实可以直接计算

<figure>
  <img src="{{ '/assets/images/ddpm-img14.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

虽然看起来不同 $t$需要有不同的高斯分布（i.i.d），但实际操作可以只抽样一次。

<figure>
  <img src="{{ '/assets/images/ddpm-img15.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

<figure>
  <img src="{{ '/assets/images/ddpm-img16.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img17.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

天。。。这推导

<figure>
  <img src="{{ '/assets/images/ddpm-img18.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

- 第一项：取 给定分布的 0、1阶段的 P 的期望值
- 第二项：KL跟 参数 $\theta$ 无关
- 第三项：$P(x_{t-1} | x_t)$是和参数有关的

<figure>
  <img src="{{ '/assets/images/ddpm-img19.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

求 $q(x_{t-1} | x_{t}, x_{0})$

<figure>
  <img src="{{ '/assets/images/ddpm-img20.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

mean 和 variance 分别是 

<figure>
  <img src="{{ '/assets/images/ddpm-img21.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

$q$ 这部分为高斯分布，mean和var是常量，固定的。而$P$也是固定的，var不变，mean是可动的。实际上，就是mean越接近越好。

<figure>
  <img src="{{ '/assets/images/ddpm-img22.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

1:02

<figure>
  <img src="{{ '/assets/images/ddpm-img23.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img24.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img25.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img26.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


<figure>
  <img src="{{ '/assets/images/ddpm-img27.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


语音合成同样的道理，要生成比较自然的声音，在推理时加上dropout

<figure>
  <img src="{{ '/assets/images/ddpm-img28.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

<figure>
  <img src="{{ '/assets/images/ddpm-img29.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

所以再denoise时，进行抽样效果会好一点。

<figure>
  <img src="{{ '/assets/images/ddpm-img30.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

案例

<figure>
  <img src="{{ '/assets/images/ddpm-img31.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

- training的时候为什么一步到位？
- sampling 的时候为什么加noise

<figure>
  <img src="{{ '/assets/images/ddpm-img32.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

从文字到图像
- Diffusion-LM：从噪声堆积上进行denoise，得到word emb，再推出来原有的text
- 

<figure>
  <img src="{{ '/assets/images/ddpm-img33.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

<figure>
  <img src="{{ '/assets/images/ddpm-img33.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

<figure>
  <img src="{{ '/assets/images/ddpm-img34.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

<figure>
  <img src="{{ '/assets/images/ddpm-img35.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>

Mask-Predict

- 在经过第一次decoder时，第二个字出现 员，但第一个字 "演"并非最高概率。那么第二次时，把低概率的掩盖 mask 掉，再进行预测。

<figure>
  <img src="{{ '/assets/images/ddpm-img36.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>


MVTM：类似于上面的nlp方面，预测部分，在预测低概率的地方
- 训练：把一张图片的部分位置进行mask，进行预测
- 推理：最开始是 mask ，再丢到decoder，再 mask 掉低概率的位置，再进行预测

<figure>
  <img src="{{ '/assets/images/ddpm-img37.png' | relative_url }}" alt="vae-paper"  class="center" style="max-height:600px; max-width:800px">
</figure>




[ppt-link]:https://www.bilibili.com/video/BV16c411J7WW/?spm_id_from=333.880.my_history.page.click&vd_source=4089d4a51ca3637483befeb898ed1a46
[dataset-LAION]:https://laion.ai/projects/