import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMS归一化 (Root Mean Square Normalization)
    相比LayerNorm，RMSNorm去掉了均值中心化，只保留方差缩放
    计算更简单，效果相当，在大模型中广泛使用
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: 归一化的维度大小
            eps: 防止除零的小常数
        """
        super().__init__()  # 调用父类nn.Module的构造函数
        self.dim = dim  # 存储维度大小
        self.eps = eps  # 存储epsilon值
        # nn.Parameter: 将tensor注册为可学习参数，会自动加入optimizer
        # torch.ones(dim): 创建全1的tensor作为缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        RMSNorm的核心计算：x / sqrt(mean(x^2) + eps)
        """
        # x.pow(2): 对x每个元素平方
        # .mean(-1, keepdim=True): 在最后一维求均值，keepdim=True保持维度不变，结果形状为[batch, seq_len, 1]
        # 加上eps防止除零，然后计算平方根的倒数，最后乘以x实现归一化
        # x形状为[batch, seq_len, dim]，归一化后仍然是这个形状
        # torch.rsqrt(): 计算平方根的倒数，即 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，shape为[batch, seq_len, dim]
        Returns:
            归一化后的tensor
        """
        # .float(): 转换为float32进行计算，提高数值稳定性
        # .type_as(x): 将结果转换回x的原始数据类型
        # self.weight *: 可学习的缩放参数
        return self.weight * self._norm(x.float()).type_as(x)
