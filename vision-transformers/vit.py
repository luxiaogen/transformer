import torch 
from torch import nn

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
              att_drop_ration=0, # 注意力分数的dropout的比率,防止过拟合
              proj_drop_ration=0):  # 最终投影层的dropout比例
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

