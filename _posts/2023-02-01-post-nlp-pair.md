---
title: "NLP-文本匹配"
date: 2023-02-01
toc: true
categories:
  - content-nlp
classes: wide
words_per_minute: 10
sidebar:
  - nav: "nlp_docs"
---

# 总体介绍

## 任务

文本匹配任务，含有较多任务：
- 文本语义相似度的计算，复述（paraphrase）的判断
- 问答匹配
  - 给定问题和该问题的答案候选池，从候选池中找出可以准确回答问题的最佳答案候选
- 对话匹配
  - 在问答匹配上面，加入了历史session（多轮）
- 自然语言推理/文本蕴含识别（Natural Language Inference/ Recognizing Textual Entailment）
  - 简介：给定一个句子A作为前提（premise），另一个句子B作为假设（hypothesis），若A能推理出B，则A、B为蕴含关系（entailment），若A与B矛盾，则A、B为矛盾关系（contradiction），否则A、B独立（neutral），可以看做一个三分类问题
- 信息检索（Information Retrieval）
  - 信息检索场景下，一般先通过检索方法召回相关项，再对相关项进行rerank，文本匹配的方法同样可以套用在这个场景

## 对比学习概念

本质是 "learn to compare"，找到一个 encoder $f$

$$ sim_{score}(f(x), f(x^{+})) >> sim_{score}(f(x), f(x^{-})) $$

其中，$x$ 与 $x^{+}$ 相似，与 $x^{-}$ 不相似， sim_score 是相似性度量函数。相当于在一个空间向量的球体里，相似的文本应该是在球体里是靠近的，跟不相似的距离尽可能"拉远"。

常用的loss：

NCE，InfoNCE 可以参考这个[文章](https://zhuanlan.zhihu.com/p/506544456)，当encoder使得样本跟相似样本距离小时（向量内积、余弦距离），loss更小。同时，温度 $\tau$ 如果变大，则所有logits的数值都变小，所以logits分布更加平滑，那么对比损失对于所有的负样本都"一视同仁"，导致模型学习没有轻重。如果温度系数设的过小，则模型会越关注特别困难的负样本，但其实那些负样本很可能是潜在的正样本，这样会导致模型很难收敛或者泛化能力差。

$$ Loss = - log(\frac{exp(q \dot k_{+} / \tau)}{\sum{i=0}{k}exp(q \dot k_{i} / \tau)})$$

<figure>
  <img src="{{ '/assets/images/nlp-ner1-nce-img1.png' | relative_url }}" alt="xgboost"  class="center" style="max-height:600px; max-width:800px">
</figure>

## 如何获取相似文本：数据增强

通过数据增强的概念，一般NLP为 加入噪声（dropout等），或者 回译（A语言翻译到B语言，再回到A语言），可以获得更相似的文本。

# 模型

## 基本类型

- Representation based（双塔式）
  - 文本a 和 文本b 分别进入encoder，获得两个向量后，进行计算、计算相似概率
  - 优点：双塔模式较快，实际业务应用时先提前计算好文本的编码，实际来时再进行浅层计算。
  - 缺点：相比交互式缺少了信息
- Interaction based（交互式）
  - 文本a 和 文本b 在早期时进行交互，再进行计算、计算相似概率
  - 优点：模型理论上更有优势，因为交互早、信息多
  - 缺点：计算慢

<figure>
  <img src="{{ '/assets/images/nlp-pairlearn-img1.png' | relative_url }}" alt="xgboost"  class="center" style="max-height:600px; max-width:800px">
</figure>

# 应用

## 应用1：双塔计算

数据集[quora-question-pairs]
- kaggle 的 leaderboard是 Log-loss 达到 top50=0.13）
- paperwithcode 表现：F1、acc top水平分别在 90%

应用项目链接：[project1-glove-bilstm] 
- 词emb：glove作为emb先验知识
- 句子encoder：biLSTM 作为两个句子共享的encoder，获取表征
- 双塔模式：核心部分在于 $torch.cat((sent1,sent2,torch.abs(sent1-sent2),sent1*sent2), dim=1)$ 最终输出分类概率

验证集效果（epoch = 11）：
- accuracy: 0.8461
- recall: 0.8755
- f1: 0.8080
- precision: 0.7502
- auc: 0.9249

## 应用2：前置使用 simCSE 进行表征计算

[simCSE-paper]


[blog1]: https://www.zhihu.com/question/31623490/answer/2900998167
[blog2]: https://zhuanlan.zhihu.com/p/357864974
[quora-question-pairs]:https://www.kaggle.com/competitions/quora-question-pairs/overview/evaluation
[simCSE-paper]: https://arxiv.org/abs/2104.08821
[project1-glove-bilstm]: https://github.com/Iven2166/models-learning/blob/main/deep-learning/NLP-models/sentences-pair-relation/quora-ques-1-lstm.ipynb
[project2-simcse]:
