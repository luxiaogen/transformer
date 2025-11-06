import torch 
from torch import nn
from functools import partial  # 引入 functools 模块中的 partial 函数，用于创建函数的偏应用版本
from collections import OrderedDict  # 引入 OrderedDict 类，用于保持字典的插入顺序

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths（随机深度）每个样本（在残差块的主路径中应用时）。
    这个实现类似于 DropConnect，用于 EfficientNet 等网络，但名字不同，DropConnect 是另一种形式的 dropout。
    链接中有详细的讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    我们使用 'drop path' 而不是 'DropConnect' 来避免混淆，并将参数名用 'survival rate' 来代替。
    
    参数：
    - x: 输入张量。
    - drop_prob: 丢弃路径的概率。
    - training: 是否处于训练模式。

    返回：
    - 如果不在训练模式或丢弃概率为 0，返回输入张量 x；
    - 否则，返回经过丢弃操作后的张量。
    """
    if drop_prob == 0. or not training:  # 如果丢弃概率为 0 或不处于训练模式，直接返回原始输入
        return x
    keep_prob = 1 - drop_prob  # 保持路径的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 生成与 x 的维度匹配的形状，只保持 batch 维度
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 生成一个与 x 大小相同的随机张量
    random_tensor.floor_()  # 将随机张量二值化（小于 keep_prob 的值为 0，其他为 1）
    output = x.div(keep_prob) * random_tensor  # 将输入 x 缩放并与随机张量相乘，实现部分路径的丢弃
    return output  # 返回经过 drop path 操作后的张量

class DropPath(nn.Module):
    """
    Drop paths（随机深度）每个样本（在残差块的主路径中应用时）。
    
    这是一个 PyTorch 模块，用于在训练期间随机丢弃某些路径，以增强模型的泛化能力。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, x):
        """
        前向传播函数，调用 drop_path 函数。
        
        参数：
        - x: 输入张量。

        返回：
        - 经过 drop path 操作后的张量。
        """
        return drop_path(x, self.drop_prob, self.training)  # 调用上面定义的 drop_path 函数


class PatchEmbed(nn.Module):
  """
    :img_size: 输入图像的大小
    :patch_size: 每个patch的大小
    :in_c: 输入图像的通道数
    :embed_dim: 每个patch映射到的维度
    :norm_layer: 归一化层
  """
  def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768,norm_layer=None):
    super().__init__()
    img_size = (img_size, img_size) # 将输入图像大小变为二维元组
    patch_size = (patch_size, patch_size) # 将patch大小变为二维元组
    self.img_size = img_size
    self.patch_size = patch_size
    # patch网格大小   224/16=14  (14,14) --> 一共有14*14个patch
    self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) 
    # 14*14=196
    self.num_patches = self.grid_size[0] * self.grid_size[1] # patch的总数
    # (B, 3, 224, 224) -> (B, 768, 14, 14)       (224 + 2*0 - 16) / 16 + 1 =14
    self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size) 
    # 如果有则使用,没有则默认保持不变
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    

    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

  """
    :x: 输入图像 形状为 (B, C, H, W)
  """
  def forward(self, x):
    B, C, H, W = x.shape # 获取输入张量的形状
    assert H == self.img_size[0] and W == self.img_size[1], \
      f"输入图像大小({H}*{W})与模型期望大小({self.img_size[0]}*{self.img_size[1]}不匹配)."  
    # (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 768, 196) -> (B, 196, 768)
    x = self.proj(x).flatten(2).transpose(1, 2)
    x = self.norm(x) # 若有归一化层 则使用
    return x
    
class Attention(nn.Module):
  def __init__(self, 
              dim, # 输入的token维度,768
              num_heads=8, # 注意力头数
              qkv_bias=False, # 生成QKV的时候是否添加偏置
              qk_scale=None, # 用于缩放QK的系数,如果None,则使用1/sqrt(embed_dim_pre_head)
              att_drop_ration=0., # 注意力分数的dropout的比率,防止过拟合
              proj_drop_ration=0.):  # 最终投影层的dropout比例
    super().__init__()
    self.num_heads = num_heads # 注意力头数
    head_dim = dim // num_heads # 每个注意力头的维度
    self.scale = qk_scale or head_dim ** -0.5 # qk的缩放系数 | d_k
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 通过全连接层生成QKV,为了并行计算,提高计算效率,参数更少
    self.att_drop = nn.Dropout(att_drop_ration) # 注意力分数的dropout层
    self.proj_drop = nn.Dropout(proj_drop_ration) #
    # 将每个head得到的输出进行concat拼接,然后通过线性变化映射回原本的嵌入dim
    self.proj = nn.Linear(dim, dim) # 最终的投影层
  
  def forward(self, x):
    B, N, C = x.shape # 获取输入张量的形状 (Batch, num_patch+1(class token), embed_dim)
    qkv = self.qkv(x) # 通过全连接层生成QKV (B, N, 3*embed_dim)
    # (B, N, 3*C) → (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 方便后面做运算
    # 用切片拿到QKV,形状(B, num_heads, N, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    """
      * 计算qk的点积冰进行缩放,得到注意力分数
      * Q.shape: q: (B, num_heads, N, head_dim)
      * K.shape: k.transpose(-2, -1): (B, num_heads, N, head_dim) -> (B, num_heads, head_dim, N)
    """
    attn = (q @ k.transpose(-2, -1)) * self.scale # 计算注意力分数 (B, num_heads, N, N)
    attn = attn.softmax(dim=-1) # 对每行进行处理,使得每行和为1
    """
      * 注意力权重对V进行加权求和
      * attn @ v: (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
      * transpose(1, 2): (B, N, num_heads, head_dim)
      * reshape(B, N, C): (B, N, num_heads*head_dim) = (B, N, C) 将最后两个维度信息拼接,合并多个头的输出,回到总的嵌入维度
    """
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    # 通过线性变化映射回原本的嵌入dim
    x = self.proj(x) 
    x = self.proj_drop(x) # 防止过拟合

    return x


class MLP(nn.Module):
  def __init__(self, 
              in_features, # 输入的维度
              hidden_features=None,  # 隐藏层的维度,通常=in_features*4
              out_features=None, # 输出的维度,通常=in_features
              act_layer=nn.GELU, # 激活函数
              drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features) # 第一个全连接层
    self.act = act_layer() # 激活函数
    self.fc2 = nn.Linear(hidden_features, out_features) # 第二个全连接层
    self.drop = nn.Dropout(drop) # dropout层

  def forward(self, x):
    x = self.fc1(x) # 通过第一个全连接层
    x = self.act(x) # 激活函数
    x = self.drop(x) # dropout,随机丢弃一定比例的神经元
    x = self.fc2(x) # 通过第二个全连接层
    x = self.drop(x) # dropout
    return x
  
# 构建Block
class Block(nn.Module):
  def __init__(self,
              dim, # 每个token的维度
              num_heads, # 多头自注意力的头数
              mlp_ratio=4, # 计算hidden_features大小,通常=dim*4
              qkv_bias=False,
              qk_scale=None,
              drop_ration=0., # 多头自注意力机制最后的linear后使用的dropout
              att_drop_ration=0., # 生成qkv的dropout
              drop_path_ration=0., # droppath的比例,会用在Encoder里面,在Multi-Head Attention和MLP之后
              act_layer=nn.GELU, # 激活函数
              norm_layer=nn.LayerNorm): # 正则化层
    super(Block, self).__init__()
    self.norm1 = norm_layer(dim) # transformer encoder block中得一个 layer norm
    # 实例化多头自注意力机制
    self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          att_drop_ration=att_drop_ration, proj_drop_ration=drop_ration) 
    # 如果drop_path_ration>0,则使用DropPath,否则使用恒等映射(不做任何更改)
    self.drop_path = DropPath(drop_path_ration) if drop_path_ration > 0. else nn.Identity()
    self.norm2 = norm_layer(dim) # 定义第二个layer_norm层
    mlp_hidden_dim = int(dim * mlp_ratio) # 计算mlp第一个全连接层的节点个数
    # 定义MLP层,传入 dim = mlp_hidden_dim
    self.mlp = MLP(in_features=dim,mlp_hidden_dim=mlp_hidden_dim,act_layer=act_layer,drop=drop_ration)

  def forward(self, x):
    # 前向传播部分 输入的x先经过layernorm再经过multiheadatte
    x = x + self.drop_path(self.attn(self.norm1(x))) 
    # 将得到的x依次通过layernorm2,mlp,drop_path
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x