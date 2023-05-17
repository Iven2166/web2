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

文本分类

命名实体识别

短文本匹配
文本摘要
- 生成方式：抽取式、生成式 
  - 抽取式：抽取式方法直接从原文中选取关键词、关键句组成摘要。 
    - 无监督抽取：不需要平行语料，不需要人工标记。基本是基于统计层面的算法，最大化抽取的摘要句子对原始文章的表达能力。最著名的无监督抽取算法为 TextRank。 
    - 有监督抽取：可以建模为序列标注任务，为原文的每一个句子打上标签0或1，最终抽取所有标签为1的句子形成摘要内容。有监督的抽取可以采用深度学习模型Bert来建模学习。
  - 生成式：生成式方法将任务建模成一个生成任务。随着近几年神经网络模型的发展，基于 Encoder-Decoder 架构的序列到序列（seq2seq）模型，被广泛的用于生成式摘要任务
    - 优点：生成式摘要允许摘要中包含新的单词或短语，灵活性较高。 
    - 问题：未登录词（OOV）、**生成重复**




# 附录
- 序列标注：
  - [nlp_pos1]


[nlp_my_project]:https://github.com/Iven2166/models-learning/tree/main/deep-learning/NLP-models
[nlp_pos1]:https://zhuanlan.zhihu.com/p/268579769
[nlp-blog]: https://zhuanlan.zhihu.com/p/444574498