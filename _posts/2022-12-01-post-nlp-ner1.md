---
title: "NER-命名实体标注"
date: 2022-12-01
toc: true
categories:
  - content-nlp
classes: wide
words_per_minute: 10
sidebar:
  - nav: "nlp_docs"
---

# 实验记录

项目地址：[ner_conll2003]

**从各个版本的效果提升来看，不同模块对应的指标提升有小许差异**
- biLSTM：能够捕捉到时序特征，但仍需要注意正则来减少过拟合风险，体现于提升测试集的recall和precision指标
- GLOVE、bert等：因为预训练特征提前学习了实体、非实体的词语特征，能够很有效地提升整体的accuracy指标
- CRF：捕捉转移概率，能够有效地提升实体与非实体之间的关系，实体词组里的B和I关系，有效提升precision指标

### V0: biLSTM单独预测
- 名称：Embrand200-bilstm1Layer200Hidden16Batch1e-3Learn
- token-emb：nn.embedding随机初始化， 200维度
- bilstm：200hidden、1layer、16batch-size、1e-3learning-rate
- 效果：
  - Train metrics recall = 91.54%, precision = 87.91%, accuracy = 96.48%
  - Test metrics recall = 61.03%, precision = 54.01%, accuracy = 84.12%

### V1: biLSTM单独预测（结构调整，进行调参缓解过拟合）
- 名称：Embrand200-bilstmLayer=1Hidden=200Dropout0.2Batch=32Learn=1e-1
- 基于V0改动了layer=3，发现效果并不好
- 加入了两层FC，并且在中间进行dropout=0.2进行训练，可能缓解了部分过拟合情况
- 效果：
  - Train :Total accu = 94.76% recall = 98.45%
  - Test :Total accu = 83.41% (- 1pp) recall = 87.19% (+ 36pp)

### V2: 预训练模型 + biLSTM
- 名称：Emb=Glove300-bilstmLayer=1Hidden=200Dropout0.1Batch=32Learn=1e-1
- 引入预训练：GLove作为emb的pretrained，效果提升很多
- 效果：Epoch = 49 
  - Train metrics recall = 98.70%, precision = 94.87%, accuracy = 98.89% 
  - Test metrics recall = 84.66%（- 3pp）, precision = 86.14%, accuracy = 94.94% (+ 11pp) -- 召回有所削减

### V3: 预训练模型 + biLSTM + CRF
- 名称：Glove300-bilstmCRFLayer=1Hidden=200Dropout0.1Batch=32Learn=1e-1
- 加入CRF层：捕捉转移概率
- 首尾加入start和stop，所以需要在token原文、token_id的seq里加入、token_emb里加入。
- 优化细节：token在首尾分别加入：START_TAG = "< START >"，STOP_TAG = "< STOP >"
- 效果
  - Train metrics: recall = 81.90%, precision = 94.26%, accuracy = 96.14%, f1 = 87.65%
  - Test metrics: recall = 82.04% (-2.4pp), precision = 92.75% (+ 6pp), accuracy = 95.74% (+ 0.8pp), f1 = 87.07%


[ner_conll2003]:https://github.com/Iven2166/models-learning/tree/main/deep-learning/NLP-models/ner/ner_conll2003