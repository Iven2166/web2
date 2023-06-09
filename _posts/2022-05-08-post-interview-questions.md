---
title: "面试题回顾"
date: 2022-05-08
toc: true
categories:
  - interview
classes: wide
words_per_minute: 10
---


*以最简要（少字符）进行记录*

# NLP 常见问题

# CV 常见问题

# 树模型常见问题

## GBDT

### GBDT 原理

全程是 Gradient Boosting Decision Tree，是一种基于boosting增强策略的加法模型，训练的时候采用前向分布算法进行贪婪的学习，训练弱分类器，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。

每一轮迭代里，会计算当前模型在所有样本的负梯度，然后以该值为目标，训练一个新的弱分类器，并且拟合计算该弱分类器的权重，从而最小化 boosting 模型的损失。

### GBDT 的优点、局限性？

- 优点
  - 推理时，树可以并行化计算，速度快
  - 在分布稠密的数据集上效果、泛化能力均表现较好
  - 决策树作为弱分类器，让GBDT模型解释性较强，自动发现特征之间的*高阶关系*
- 局限性
  - 训练时需要串行训练
  - 在高维稀疏数据里，表现不如SVM或者DNN

### 梯度提升（GBDT）和梯度下降有什么区别和联系？

两者其实都是在每一步的迭代里，定义了损失函数，然后在损失函数相对于模型的负梯度的方向里，对模型进行更新。
- 梯度下降：一般就是对已定义的模型的参数进行更新
- 梯度提升：比如 boosting 的 GBDT 模型，是不断地叠加模型。模型是定义在函数空间里的，该函数空间影响到损失函数。因此，扩大了可以使用的模型种类（我理解是可以有完全不同的树结构，而不是同一个树结构不同的参数）。

## xgboost 

### xgboost 介绍

xgboost的加法函数定义为：第t次迭代结果为 前t-1次迭代结果，加上第t棵树的结果

损失函数定义为 loss 加上 树整体的复杂度（叶子节点数目）

泰勒展开部分：

<figure>
  <img src="{{ '/assets/images/xgboost-img1.png' | relative_url }}" alt="xgboost"  class="center" style="max-height:600px; max-width:800px">
</figure>

> https://zhuanlan.zhihu.com/p/92837676

我们定义一颗树的复杂度 Ω，它由两部分组成： 叶子结点的数量； 叶子结点权重向量的L2范数；

Q：xgboost进行最佳分裂点？

XGBoost在训练前预先将特征按照特征值进行了排序，并存储为block结构，以后在结点分裂时可以重复使用该结构。
因此，可以采用特征并行的方法利用多个线程分别计算每个特征的最佳分割点，根据每次分裂后产生的增益，最终选择增益最大的那个特征的特征值作为最佳分裂点。

### xgboost的最佳分裂点

XGBoost在训练前预先将特征按照特征值进行了排序，并存储为block结构，以后在结点分裂时可以重复使用该结构。

因此，可以采用特征并行的方法利用多个线程分别计算每个特征的最佳分割点，根据每次分裂后产生的增益，最终选择增益最大的那个特征的特征值作为最佳分裂点。

### xgboost的并行化训练

- XGBoost的并行，并不是说每棵树可以并行训练，XGB本质上仍然采用boosting思想，每棵树训练前需要等前面的树训练完成才能开始训练。 
- XGBoost的并行，指的是特征维度的并行
  - 在训练之前，每个特征按特征值对样本进行预排序，并存储为Block结构，在后面查找特征分割点时可以重复使用，而且特征已经被存储为一个个block结构，那么在寻找每个特征的最佳分割点时，可以利用多线程对每个block并行计算。

### xgboost防止过拟合方法？
XGBoost在设计时，为了防止过拟合做了很多优化，具体如下：
- 目标函数添加正则项：叶子节点个数+叶子节点权重的L2正则化 
- 列抽样：训练的时候只用一部分特征（不考虑剩余的block块即可） 
- 子采样：每轮计算可以不使用全部样本，使算法更加保守 
- shrinkage: 可以叫学习率或步长，为了给后面的训练留出更多的学习空间

### xgboost 的参数

- max_depth：每棵子树的最大深度
- min_child_weight：子节点的权重阈值
- gamma：也称为 min_split_loss 对于一个叶子节点，最小划分损失阈值。大于该阈值就继续划分，反之就不划分。一般取 0.1-0.5
- subsample：训练的采样比例
- colsample_bytree：特征的采样比例
- alpha: 参数正则化 L1
- lambda: 参数正则化 L2

过拟合如何解决：
- 第一类参数：用于直接控制模型的复杂度。包括max_depth,min_child_weight,gamma 等参数
- 第二类参数：用于增加随机性，从而使得模型在训练时对于噪音不敏感。包括subsample,colsample_bytree
- 还有就是直接减小learning rate，但需要同时增加estimator 参数。

### xgboost 对缺失值的处理

- 从模型角度来看：在寻找最佳的split point时，只对非缺失的样本对应特征值进行遍历
- 在缺失值情况下，样本分别分配到左叶子结点和右叶子结点，选择分裂后增益最大的方向，作为预测时特征值缺失样本的默认分支方向
- 如果训练中没有缺失值，但是预测里有缺失，自动将缺失值的划分方向放到右子节点


## 树模型之间的对比

### xgboost 和 GBDT 的差异

- 剪支、正则化
  - GBDT：基于经验损失的负梯度来构造新的决策树，到最后才剪枝
  - xgboost：在定义决策树构建阶段就剪枝，涉及到 叶子节点数限制、叶子节点的预测值。


### xgboost和DNN的优劣对比

- 如果讨论DNN是比较简单的全连接神经网络，那么相当于可以设计网络层数、激活函数等架构。
- 可解释性：xgboost能够通过信息增益，计算出每个特征的贡献度。而DNN偏向于黑盒模型。
- 针对数据：表格型数据，或者数据量不多情况下，xgboost能够比较好的表现。DNN需要有一定数量才可以收敛。
- 特征稀疏性：实际上xboost不适合处理高维稀疏特征。一方面ID类特征不好处理，因为one-hot情况下，树需要找到熵减最大程度的特征，会收敛很慢。另一方面，因为稀疏数据很可能导致树的分裂误判，从而会过拟合，泛化性不强。



### GBDT和LR的比较？
- LR虽然是分类，但是简单的广义线性模型，可解释性强，容易并行化。但学习能力有限，需要大量的人工特征工程
- gbdt，非线性模型，具有天然的特征组合优势，特征表达能力强，但是树与树之间无法并行训练，而且树模型很容易过拟合
当在高维稀疏特征的场景下，LR的效果一般会比GBDT好

这也就是为什么在高维稀疏特征的时候，线性模型会比非线性模型好的原因了：带正则化的线性模型比较不容易对稀疏特征过拟合


### xgboost 如何调参

Q：xgboost如何处理不平衡数据？

对于不平衡的数据集，例如用户的购买行为，肯定是极其不平衡的，这对XGBoost的训练有很大的影响，XGBoost有两种自带的方法来解决：

第一种，如果你在意AUC，采用AUC来评估模型的性能，那你可以通过设置scale_pos_weight来平衡正样本和负样本的权重。例如，当正负样本比例为1:10时，scale_pos_weight可以取10（正样本应该有的权重）；

第二种，如果你在意概率(预测得分的合理性)，你不能重新平衡数据集(会破坏数据的真实分布)，应该设置max_delta_step为一个有限数字来帮助收敛（基模型为LR时有效）。




## RNN

## LSTM

<figure>
  <img src="{{ '/assets/images/lstm-img1.png' | relative_url }}" alt="-"  class="center" style="max-height:600px; max-width:800px">
</figure>

如果输入 x 的维度是 batch-n * d, 那么隐向量维度为 d，则总体参数量= $4*(d \times h + h \times h + h)$


## TextCNN

## bert

### bert与lstm比较

- 底层的语义表征、模型架构是差别很大的。
- 语义表征里，bert对于token的表征，是通过position-emb以及token-emb来表示。
- lstm是时序模型，串行的运算，后面的需要等前面的计算完成，所以不是并行的。
- bert 是通过self-attention来进行运算，得到最后的序列的编码。除了层与层之间需要依赖，但self-attention的矩阵运算，让计算并行加快。

按照原论文作者说，实质上模型变复杂、参数变多，但训练时间跟lstm实际上相差不多。


## albert

- Factorized Embedding Parameterization，降低embedding维度，再通过全链接层扩充到高维度，匹配后续层
- Cross-layer Parameter Sharing，不同层之间共享参数。几种方案，最终albert选择所有层共享，因为压缩力度最大并且损失可控
  - 所有层共享
  - attention 层共享
  - feed forward 层共享
  - 每M层一组，组内共享
- Sentence Order Prediction（SOP），跟NSP都是 next sentence 任务。区别在于负样本选取不同。SOP 将正样本的两个句子调换位置，也作为负样本。

## roberta

A Robustly Optimized BERT Pretraining Approach

相比于bert，两个差异：
- 使用更多训练数据：160GB，而 bert 使用 16GB数据
- 动态掩码机制：bert 在准备数据时，就进行掩码操作（静态）；而 roberta 是在输入模型时进行操作，即同一条训练数据在不同的训练 epoch 里是不同的掩码，能够学习到更多的样式

## BART

Bidirectional and Auto-Regressive Transformers

兼顾上下文信息（bert）以及自回归特点（GPT）的模型

- 噪声预训练：尽可能减少模型对"结构化信息"-即位置等的依赖
  - token masking：替换为 mask 后进行单词预测
  - token deletion：随机删除个别词，训练预测单词以及位置的能力
  - text infilling：连续的词替换为 mask，并且进行包含词以及长度的预测
  - sentence permutation：随机打乱文本顺序，加强模型对词的关联性的提取能力
- 下游任务微调


## resnet

## vgg

## 深度学习网络的其他技巧

### 梯度消失的情况

- 做好网络的参数初始化：梯度爆炸或者消失的最初原因在于，网络各层的参数过大或者过小，导致多层情况下，乘以激活函数的梯度，还是会爆炸或者消失。因此通过xavier、正态等方法去初始化。
- 激活函数：选择更好的激活函数，比如ReLU或者leaky relu函数，在左侧导数为0，右侧导数恒为1. 但恒为1的导数也有梯度爆炸风险，合适的阈值能够解决
- 加入归一化层：将输入拉伸到正太分布，减少输入的向量对于参数的过大或者过小的影响。能够加速收敛，控制过拟合
- 改变传播结构：减少层数、或者从RNN改为LSTM等方法
- 残差网络的思想：拟合残差，但在计算最后输出是增加了原数值，梯度不会显著变小。

### 网络构建和初始化

<figure>
  <img src="{{ '/assets/images/deep-learning-param-init1.png' | relative_url }}" alt="-"  class="center" style="max-height:600px; max-width:800px">
</figure>

### 网络调参技巧

深度学习
网络优化技巧
调整网络参数
除了以上对网络结构的尝试，我们也进行了多组超参的调优。神经网络最常用的超参设置有：隐层层数及节点数、学习率、正则化、Dropout Ratio、优化器、激活函数、Batch Normalization、Batch Size等。不同的参数对神经网络的影响不同，神经网络常见的一些问题也可以通过超参的设置来解决：
过拟合：网络宽度深度适当调小，正则化参数适当调大，Dropout Ratio适当调大等。
欠拟合：网络宽度深度适当调大，正则化参数调小，学习率减小等。
梯度消失/爆炸问题：合适的激活函数，添加Batch Normalization，网络宽度深度变小等。
局部最优解：调大Learning Rate，合适的优化器，减小Batch Size等。
Covariate Shift：增加Batch Normalization，网络宽度深度变小等。
影响神经网络的超参数非常多，神经网络调参也是一件非常重要的事情。工业界比较实用的调参方法包括：
网格搜索/Grid Search：这是在机器学习模型调参时最常用到的方法，对每个超参数都敲定几个要尝试的候选值，形成一个网格，把所有超参数网格中的组合遍历一下尝试效果。简单暴力，如果能全部遍历的话，结果比较可靠。但是时间开销比较大，神经网络的场景下一般尝试不了太多的参数组合。
随机搜索/Random Search：Bengio在“Random Search for Hyper-Parameter Optimization”https://tech.meituan.com/2018/06/07/searchads-dnn.html#fn:10中指出，Random Search比Grid Search更有效。实际操作的时候，可以先用Grid Search的方法，得到所有候选参数，然后每次从中随机选择进行训练。这种方式的优点是因为采样，时间开销变小，但另一方面，也有可能会错过较优的超参数组合。
分阶段调参：先进行初步范围搜索，然后根据好结果出现的地方，再缩小范围进行更精细的搜索。或者根据经验值固定住其他的超参数，有针对地实验其中一个超参数，逐次迭代直至完成所有超参数的选择。这个方式的优点是可以在优先尝试次数中，拿到效果较好的结果。
我们在实际调参过程中，使用的是第3种方式，在根据经验参数初始化超参数之后，按照隐层大小->学习率->Batch Size->Drop out/L1/L2的顺序进行参数调优。


# spark

- 开 SparkSession 
- 初始化数据集
- 各类算子
  - 选择和切片
    - filter：对数据的筛选
    - where
  - 