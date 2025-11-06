# 0. Introduction
- TRM在做一个什么事情?(以机器翻译为例)
<img src="assets/Transformer从零解读/image.png" alt="alt text" style="zoom: 33%;" />
<img src="assets/Transformer从零解读/image-1.png" alt="alt text" style="zoom: 33%;" />
<img src="assets/Transformer从零解读/image-2.png" alt="alt text" style="zoom: 33%;" />
- 结构上相同,参数上不同(独立训练)
- 原论文的图
<img src="assets/Transformer从零解读/image-3.png" alt="alt text" style="zoom:33%;" />

# 1.Encoder
<img src="assets/Transformer从零解读/image-4.png" alt="alt text" style="zoom:33%;" />

## 输入部分
### [1]Embedding
<img src="assets/Transformer从零解读/image-5.png" alt="alt text" style="zoom:33%;" />

- 一共是12个字,按字切分,每个字对应一个512维度的字向量
  - 对于字向量,可以使用word2vec,或者随机初始化
  - 如果数据量非常大的话,使用随机还是word2vec差别不会特别大,可以忽略

### [2]位置编码
> 为什么需要
>
> - 对于RNN来讲,一个共识,RNN的参数$U$,输入参数$W$、隐藏参数$V$、输出参数、这是一套参数,对于所有RNN的所有的`time steps`,它都共享一套参数.
> - 也就是说,一百个`times steps`,一百个词,一百个字,但是你只有一套参数,你在更新的时候是更新这一套的$U$、$W$、$V$.

- RNN的梯度消失有什么不同?
  - RNN的梯度消失和普通网络的梯度消失含义不相同.
  - RNN的梯度是一个总的梯度和,它的梯度消失并不是变为零,而是说总梯度被近距离梯度主导,被远距离梯度忽略不计

- RNN是串行的，而TRM是并行处理的，但是忽略了单词之间的序列关系（先后关系）
  - 引出了位置编码


<img src="assets/Transformer从零解读/image-7.png" alt="alt text" style="zoom:33%;" />

#### 位置编码公式
$$
\begin{aligned}
P E_{(p o s, 2 i)} & =\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} & =\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$
<img src="assets/Transformer从零解读/image-8.png" alt="alt text" style="zoom:33%;" />
<img src="assets/Transformer从零解读/image-9.png" alt="alt text" style="zoom:33%;" />
- 位置编码与词向量相加

#### 为什么位置嵌入会有用
<img src="assets/Transformer从零解读/image-10.png" alt="alt text" style="zoom:33%;" />

- 但是这种相对位置信息会在注意力机制那里消失
## 注意力机制
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
- `softmax`之后得到相似度向量,再与$V$相乘得到最后的加权和

### [1]从公式角度来看
<img src="assets/Transformer从零解读/image-11.png" alt="alt text" style="zoom: 33%;" />
<img src="assets/Transformer从零解读/image-12.png" alt="alt text" style="zoom: 33%;" />
<img src="assets/Transformer从零解读/image-13.png" alt="alt text" style="zoom:33%;" />
- 点乘：反映两个向量之间的相似度，也就说两个向量越相似，它的点乘结果就越大，说明距离越靠近，也就是我越关注。

### [2]在只有单词向量的情况下，如何获取QKV
<img src="assets/Transformer从零解读/image-14.png" alt="alt text" style="zoom:33%;" />

- 与不同的权重矩阵相乘得到q,k,v

<img src="assets/Transformer从零解读/image-15.png" alt="alt text" style="zoom:33%;" />

- 计算QK相似度,得到attention值
- 如果$q \times k$的值很大，`Softmax`在反向传播的时候梯度很小，梯度很小就容易造成梯度消失。为什么用根号$d_k$，是为了保持方差为1

- 实际代码中使用矩阵,方便并行操作

<img src="assets/Transformer从零解读/image-16.png" alt="alt text" style="zoom:33%;" />

### [3]多头注意力机制
<img src="assets/Transformer从零解读/image-17.png" alt="alt text" style="zoom:33%;" />

- 为什么这么做?
  - 多头相当于把原始信息打到了不同空间。相当于放到几个不同的地方，保证Transform可以注意到不同子控件的信息
- 多个头就会有多个输出,需要合在一起输出

<img src="assets/Transformer从零解读/image-18.png" alt="alt text" style="zoom:33%;" />

## 残差和layerNorm

### [1]残差
> 残差的作用

<img src="assets/Transformer从零解读/image-19.png" alt="alt text" style="zoom:33%;" />

### [2]Layer Normalization
#### Batch Normalization
> BN的效果差，所以不用

- 什么是BN,以及使用场景
  - 无论在机器学习还是深度学习，我们有这样一个操作，就是在特征处理的时候都要做feature scaling，就是说feature scaling就是为了消除量纲的影响，让模型收敛的更快。

<img src="assets/Transformer从零解读/image-20.png" alt="alt text" style="zoom:33%;" />

- 重点:它针对这个batch中的同一维度特征(体重...)做处理

<img src="assets/Transformer从零解读/image-21.png" alt="alt text" style="zoom:25%;" />

> BN的优点和缺点
>
> - 优点
>   1. 可以解决内部协变量偏移
>   2. 缓解了梯度饱和问题（如果使用sigmoid激活函数的话），加快收敛。
> - 缺点
>   1. batch_size较小的时候，效果差。因为BN的过程使用的是它的一个假设，是使用整个batch中的样本的
>      均值和方差来模拟全部数据的均值和方差。（例如10个人代替全班）
>   2. BN 在RNN中效果比较差。在RNN中，BN在RNN中效果不好的原因，就是RNN的输入是动态的，所以它不能有效地得到整个batch size中均值和方差，接下来我们聊一下为什么使用LN，我们首先明白一点，

<img src="assets/Transformer从零解读/image-22.png" alt="alt text" style="zoom:25%;" />

#### Layer-Norm

<img src="assets/Transformer从零解读/image-23.png" alt="alt text" style="zoom:25%;" />

- 为什么使用layer-norm?
  - 理解:LayerNorm单独对一个样本的所有单词做缩放可以起到效果


## 前馈神经网络 FNN
<img src="assets/Transformer从零解读/image-24.png" alt="alt text" style="zoom:33%;" />

# 2. Decoder
<img src="assets/Transformer从零解读/image-25.png" alt="alt text" style="zoom:33%;" />

## 多头注意力机制(Masked)
<img src="assets/Transformer从零解读/image-26.png" alt="alt text" style="zoom:33%;" />

### 为什么需要mask?
> 比如说在输入love的时候，我输出you。如果decoder的输入没有mask，所有的词为生成you这个预测结果都会提供信息。但是这样训练出来的模型在预测的时候就会出现一个问题，因为在预测阶段，比如说我同样的是预测这个you，我没有ground truth，就是没有后面这个信息的，你的模型看不见未来时刻的单词，模型在预测的时候和训练的时候就是存在gap的。也就是说你可以看到这两个单词，但是预测的时候看不到，这个模型的效果肯定不好

<img src="assets/Transformer从零解读/image-27.png" alt="alt text" style="zoom: 25%;" />


## 交互层
<img src="assets/Transformer从零解读/image-28.png" alt="alt text" style="zoom:33%;" />
<img src="assets/Transformer从零解读/image-29.png" alt="alt text" style="zoom:33%;" />

- Encoder的输出与每个Decoder都进行交互
- Encoder 生成$K$和$V$矩阵
- Decoder 生成$Q$矩阵，$Q$来自于本身

<img src="assets/Transformer从零解读/image-30.png" alt="alt text" style="zoom:33%;" />