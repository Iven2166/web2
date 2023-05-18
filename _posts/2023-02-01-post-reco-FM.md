---
title: "Reco-FM模型"
date: 2023-02-01
toc: true
categories:
  - reco
classes: wide
words_per_minute: 10
---

矩阵分解：实质上是 user 行 对 物品列的打分矩阵，然后根据矩阵分解，分拆为 两个矩阵向量。而用户向量与物品向量的内积计算，就是该行该列的得分。跟NLP里的文本相似计算非常类似，两个文本emb向量的内积用于表示文本相似度。

<figure>
  <img src="{{ '/assets/images/reco-fm1.png' | relative_url }}" alt="xgboost"  class="center" style="max-height:600px; max-width:800px">
</figure>


[mygithub-rec-part]:https://github.com/Iven2166/models-learning/tree/main/deep-learning/REC-models
[old-application-FM]: https://github.com/Iven2166/models-learning/blob/main/ctr-predict-models/models/FM-large-10%25data-v2.ipynb
[application-deepFM]:https://github.com/Iven2166/models-learning/blob/main/deep-learning/REC-models/deepFM/deepFM-criteoSmall.ipynb
[application-MMOE]:https://github.com/Iven2166/models-learning/blob/main/deep-learning/REC-models/MMOE/%E5%A4%9A%E4%BB%BB%E5%8A%A1%E7%9B%AE%E6%A0%87%E5%AD%A6%E4%B9%A0-mmoe.ipynb
[old-mygithub]:https://github.com/Iven2166/models-learning/tree/main/ctr-predict-models