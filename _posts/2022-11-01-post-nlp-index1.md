---
title: "NLP-首页(博客迁移中)"
date: 2022-11-01
toc: true
categories:
  - content-nlp
classes: wide
words_per_minute: 10
sidebar:
  - nav: "nlp_docs"
---

# 文本匹配

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

$$ sim_score(f(x), f(x^{+})) >> sim_score(f(x), f(x^{-})) $$

其中，$x$ 与 $x^{+}$ 相似，与 $x^{-}$ 不相似， sim_score 是相似性度量函数。

对比学习是自监督的，因此有效地构造相似pair，不相似pair 很重要。




# 附录
- 序列标注：
  - [nlp_pos1]


[nlp_my_project]:https://github.com/Iven2166/models-learning/tree/main/deep-learning/NLP-models
[nlp_pos1]:https://zhuanlan.zhihu.com/p/268579769