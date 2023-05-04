---
title: "diffusion model"
date: 2023-04-27
categories:
  - AIGC
tags:
  - diffusion
  - multi-modal
---

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

[ppt-link]:https://www.bilibili.com/video/BV16c411J7WW/?spm_id_from=333.880.my_history.page.click&vd_source=4089d4a51ca3637483befeb898ed1a46