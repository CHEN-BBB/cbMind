import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class cbMindConfig(PretrainedConfig):
    model_type = "cbmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,  # 高频边界，波长比例大于此值的维度不缩放
                "beta_slow": 1,  # 低频边界，波长比例小于此值的维度全量缩放
                "factor": 16,  # 从原始长度扩展到目标长度的倍数（例如从 2048 扩展到 32768，factor 就是 16）
                "original_max_position_embeddings": 2048,  # 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
                "attention_factor": 1.0,  # 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
                "type": "yarn",  # 可选 "yarn" 或 "linear"，分别对应 YaRN 和线性插值两种 RoPE 缩放方法
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            # inv_dim = lambda b: (
            #     (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # )
            # 使用define定义函数，方便后续调用
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),  # 0 到 low 之间的高频维度不缩放
                min(
                    math.ceil(inv_dim(beta_slow)), dim // 2 - 1
                ),  # high 到 dim//2 之间的低频维度全量缩放
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (
                    torch.arange(dim // 2, device=freqs.device).float() - low
                )  # 计算每个维度相对于 low 的偏移量
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    # freqs_cos 和 freqs_sin 的形状都是 (end, dim)，分别对应每个位置的 Cos 和 Sin 频率矩阵。
    freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    )  # 将 cos 部分复制一份，拼接在一起，得到最终的频率矩阵
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # q, k: (seq_len, dim) 或 (batch_size, seq_len, dim)
    # cos, sin: (max_position_embeddings, dim)
    # 将输入张量 x 沿最后一个维度切分成两半，并交换这两半的位置
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


# GQA 中的重复 KV 方案：将原本 num_key_value_heads 的 KV 复制 n_rep 份，得到 num_key_value_heads * n_rep 的 KV。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(
            bs, slen, num_key_value_heads, n_rep, head_dim
        )  # 在原来的 KV 维度后面增加一个维度，并复制 n_rep 份
        .reshape(
            bs, slen, num_key_value_heads * n_rep, head_dim
        )  # 将复制后的维度和原来的 KV 维度合并，得到新的 KV 维度是 num_key_value_heads * n_rep
    )


class Attention(nn.Module):
    def __init__(self, args: cbMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads  # 注意力头的总数，例如 8
            if args.num_key_value_heads is None
            else args.num_key_value_heads  # KV 头的数量，例如 2
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.n_local_heads = args.num_attention_heads  # 注意力头的总数，例如 8
        self.n_local_kv_heads = self.num_key_value_heads  # KV 头的数量，例如 2
        self.n_rep = (
            self.n_local_heads // self.n_local_kv_heads
        )  # 每个 KV 头需要被复制的次数，例如 4（因为 8 个 Q 头对应 2 个 KV 头，每个 KV 头需要被复制 4 份才能匹配 8 个 Q 头）
        self.head_dim = (
            args.hidden_size // args.num_attention_heads
        )  # 每个头的维度，例如 512 // 8 = 64

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )  # 将输入的隐藏状态映射到查询向量空间，输出维度是 num_attention_heads * head_dim，例如 8 * 64 = 512
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )  # 将输入的隐藏状态映射到键向量空间，输出维度是 num_key_value_heads * head_dim，例如 2 * 64 = 128
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )  # 将输入的隐藏状态映射到值向量空间，输出维度是 num_key_value_heads * head_dim，例如 2 * 64 = 128
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )  # 将多头注意力的输出映射回隐藏状态维度，例如 8 * 64 = 512 -> 512

        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力权重的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接的 dropout
        self.dropout = args.dropout  # 注意力的 dropout 概率
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention
        )  # 是否使用 PyTorch 2.0 的原生 Flash Attention，如果满足条件则为 True，否则为 False

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # x: (batch_size, seq_len, hidden_size)，输入的隐藏状态张量
        # position_embeddings: (cos, sin)，预先计算好的 RoPE 频率矩阵，分别是 cos 和 sin，形状都是 (max_position_embeddings, head_dim)
        # past_key_value: ((batch_size, seq_len_past, num_key_value_heads * head_dim), (batch_size, seq_len_past, num_key_value_heads * head_dim))，前面时间步的 KV 缓存，如果有的话
        # use_cache: 是否返回新的 KV 缓存，供后续时间步使用
        bsz, seq_len, _ = x.shape
        # 1. 线性变换得到 Q、K、V 张量，并调整形状以便后续计算
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 将 Q、K、V 张量从 (batch_size, seq_len, num_heads * head_dim) 形状调整为 (batch_size, seq_len, num_heads, head_dim) 以便进行多头注意力计算
        xq = xq.view(
            bsz, seq_len, self.n_local_heads, self.head_dim
        )  # 将查询张量调整为 (batch_size, seq_len, num_attention_heads, head_dim)，例如 (32, 128, 8, 64)
        xk = xk.view(
            bsz, seq_len, self.n_local_kv_heads, self.head_dim
        )  # 将键张量调整为 (batch_size, seq_len, num_key_value_heads, head_dim)，例如 (32, 128, 2, 64)
        xv = xv.view(
            bsz, seq_len, self.n_local_kv_heads, self.head_dim
        )  # 将值张量调整为 (batch_size, seq_len, num_key_value_heads, head_dim)，例如 (32, 128, 2, 64)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(
            xq, xk, cos, sin
        )  # 将 RoPE 位置编码应用到 Q 和 K 上，增强它们的位置信息，使模型能够更好地捕捉序列中元素之间的相对位置关系

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat(
                [past_key_value[0], xk], dim=1
            )  # 将过去的 K 缓存与当前的 K 张量在序列维度上拼接，例如 (batch_size, seq_len_past + seq_len, num_key_value_heads, head_dim)
            xv = torch.cat(
                [past_key_value[1], xv], dim=1
            )  # 将过去的 V 缓存与当前的 V 张量在序列维度上拼接，例如 (batch_size, seq_len_past + seq_len, num_key_value_heads, head_dim)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(
                1, 2
            ),  # 将 Q 张量的序列维度和头维度进行转置，例如 (batch_size, num_attention_heads, seq_len, head_dim)
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            # 将 K 张量进行 KV 复制后再转置，例如 (batch_size, num_attention_heads, seq_len, head_dim)，其中 num_attention_heads 是通过复制 KV 头得到的
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 计算注意力分数，得到 (batch_size, num_attention_heads, seq_len, seq_len_kv)，其中 seq_len_kv 是 K 张量的序列长度，可能包含过去时间步的 KV 缓存
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )
            # 上面的代码通过添加一个上三角矩阵（triu）来实现 causal mask，确保每个位置只能关注它之前的位置，从而防止信息泄露。

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 对注意力分数进行 softmax 归一化，得到注意力权重，形状仍然是 (batch_size, num_attention_heads, seq_len, seq_len_kv)
            # dim=-1 表示在最后一个维度上进行 softmax，即对每个查询位置的注意力分数进行归一化，使它们的和为 1。
            scores = self.attn_dropout(
                scores
            )  # 对注意力权重进行 dropout，以增加模型的鲁棒性，防止过拟合
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 将输出张量的形状从 (batch_size, num_attention_heads, seq_len, head_dim) 调整为 (batch_size, seq_len, num_attention_heads * head_dim)
        output = self.resid_dropout(
            self.o_proj(output)
        )  # 通过线性变换将多头注意力的输出映射回隐藏状态维度，并应用残差连接的 dropout
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: cbMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 为了更高的计算效率，通常会将 intermediate_size 调整为 64 的倍数，因为大多数硬件（尤其是 GPU）在处理 64 的倍数的维度时效率更高。
            # 因此，下面的代码将 intermediate_size 调整为大于或等于计算值的最小的 64 的倍数。
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,  # 这个线性层用于计算门控机制中的 gate 张量，输入维度是 hidden_size，输出维度是 intermediate_size，例如 512 -> 1024
        )
        # 门控的作用是通过一个额外的线性层计算出一个 gate 张量，并将其与 FFN 的另一个线性层的输出进行逐元素相乘，
        # 从而实现对 FFN 输出的动态调节，增强模型的表达能力和非线性建模能力。
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,  # 这个线性层用于将 FFN 的输出映射回隐藏状态维度，输入维度是 intermediate_size，输出维度是 hidden_size，例如 1024 -> 512
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,  # 这个线性层用于将输入的隐藏状态映射到 FFN 的中间维度，输入维度是 hidden_size，输出维度是 intermediate_size，例如 512 -> 1024
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[
            config.hidden_act
        ]  # 激活函数，根据配置中的 hidden_act 参数选择，例如 "silu" 对应 SiLU 激活函数，也就是 Swish 激活函数

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)  # 计算门控 FFN 的输出，首先通过 gate_proj 计算 gate 张量，并应用激活函数，然后与通过 up_proj 计算的另一个张量逐元素相乘，得到 gated 张量
        return self.dropout(self.down_proj(gated))  # 将 gated 张量通过 down_proj 映射回隐藏状态维度，并应用 dropout，得到最终的 FFN 输出


class MoEGate(nn.Module):
    def __init__(self, config: cbMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数量，例如 2
        self.n_routed_experts = config.n_routed_experts  # 专家总数，例如 4

        self.scoring_func = config.scoring_func  # 门控的评分函数，例如 "softmax"
        self.alpha = config.aux_loss_alpha  # 辅助损失的权重，例如 0.01
        self.seq_aux = config.seq_aux  # 是否使用序列级的辅助损失，如果为 False 则使用 token 级的辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否对 top-k 专家的概率进行归一化，确保它们的和为 1，这在 top-k 大于 1 时尤其重要
        self.gating_dim = config.hidden_size  # 门控层的输入维度，例如 512
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))  # 门控层的权重矩阵，形状是 (n_routed_experts, gating_dim)，例如 (4, 512)
        )
        self.reset_parameters()  # 初始化门控层的权重参数，通常使用 Kaiming 初始化方法来确保权重的初始分布适合训练深度神经网络

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 使用 Kaiming 均匀分布初始化门控层的权重参数，这种初始化方法有助于保持前向传播和反向传播过程中信号的稳定，促进模型的训练效果,公式如下：
        # weight ~ U(-bound, bound)，其中 bound = sqrt(6 / fan_in)，fan_in 是权重矩阵输入维度的大小，这里是 gating_dim，例如 512，
        # a 是激活函数的负半轴斜率，对于 ReLU 和 SiLU 激活函数，a 的值是 sqrt(5)。

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape  # hidden_states 的形状是 (batch_size, seq_len, gating_dim)，例如 (32, 128, 512)
        hidden_states = hidden_states.view(-1, h)  # 将 hidden_states 调整为 (batch_size * seq_len, gating_dim)，例如 (4096, 512)，以便进行线性变换
        logits = F.linear(hidden_states, self.weight, None)  # 计算门控层的 logits，形状是 (batch_size * seq_len, n_routed_experts)，例如 (4096, 4)，每个 token 对每个专家的评分

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)  # 对 logits 进行 softmax 归一化，得到每个 token 对每个专家的选择概率，形状仍然是 (batch_size * seq_len, n_routed_experts)，例如 (4096, 4)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"  # 如果配置中指定了不支持的评分函数，则抛出异常，提示用户选择正确的评分函数，例如 "softmax"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # 选择每个 token 的 top-k 专家及其权重，得到 topk_weight 和 topk_idx，
        # 形状都是 (batch_size * seq_len, num_experts_per_tok)，例如 (4096, 2)，分别对应每个 token 选择的专家的权重和索引

        # ============第一步：对 top-k 权重进行归一化（如果需要）===========
        # 对 top-k 权重进行归一化，使它们的和为 1，形状仍然是 (batch_size * seq_len, num_experts_per_tok)，例如 (4096, 2)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 计算 top-k 权重的和，形状是 (batch_size * seq_len, 1)，例如 (4096, 1)，加上一个小常数防止除零
            topk_weight = topk_weight / denominator

        # ============第二步：计算辅助损失（如果需要）===========
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 形状是 (batch_size * seq_len, n_routed_experts)，例如 (4096, 4)，用于计算辅助损失的专家选择概率
            aux_topk = self.top_k  # 每个 token 选择的专家数量，例如 2
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # 将 topk_idx 调整为 (batch_size, seq_len * num_experts_per_tok)，例如 (32, 256)，用于计算辅助损失的专家索引
            # ===========方式1：序列级辅助损失===========
            if self.seq_aux:  # 如果使用序列级的辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # 将 scores_for_aux 调整为 (batch_size, seq_len, n_routed_experts)，例如 (32, 128, 4)，用于计算辅助损失的专家选择概率
                # 创建一个零张量，用于统计每个专家被选择的次数，形状是 (batch_size, n_routed_experts)，例如 (32, 4)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                # 使用 scatter_add_ 将 topk_idx_for_aux_loss 中每个 token 选择的专家索引对应的位置加上 1，统计每个专家被选择的总次数，形状仍然是 (batch_size, n_routed_experts)，例如 (32, 4)
                ce.scatter_add_(
                    1,  # 1 表示在列维度上进行 scatter_add_
                    topk_idx_for_aux_loss,  # 形状是 (batch_size, seq_len * num_experts_per_tok)，例如 (32, 256)，每个元素是一个专家索引，表示对应 token 选择的专家
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),  # 形状是 (batch_size, seq_len * num_experts_per_tok)，例如 (32, 256)，每个元素都是 1，表示对应 token 选择了一个专家
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                # 将统计结果除以 (seq_len * num_experts_per_tok / n_routed_experts)，得到每个专家被选择的平均概率，形状仍然是 (batch_size, n_routed_experts)，例如 (32, 4)
                # 将每个专家的平均选择概率与其在整个序列中的平均选择概率相乘，并对所有专家求和，得到每个序列的辅助损失，形状是 (batch_size,)，例如 (32,)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha  # 对所有序列的辅助损失求平均，并乘以 alpha 权重，得到最终的辅助损失标量
            # ===========方式2：Token级辅助损失===========
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
            # 将 topk_idx_for_aux_loss 调整为 (batch_size * seq_len * num_experts_per_tok,)，例如 (4096,)，每个元素是一个专家索引，然后使用 one_hot 将其转换为独热编码，得到 mask_ce，
            # 形状是 (batch_size * seq_len * num_experts_per_tok, n_routed_experts)，例如 (4096, 4)，每行表示对应 token 选择的专家
                ce = mask_ce.float().mean(0)  # 计算每个专家被选择的平均概率，形状是 (n_routed_experts,)，例如 (4,)
                Pi = scores_for_aux.mean(0)  # 计算每个专家在整个输入中的平均选择概率，形状是 (n_routed_experts,)，例如 (4,)
                fi = ce * self.n_routed_experts  # 计算每个专家的负载平衡因子，形状是 (n_routed_experts,)，例如 (4,)，通过将每个专家的平均选择概率乘以专家总数来得到负载平衡因子，这样可以鼓励模型更均匀地使用所有专家，避免某些专家过载而其他专家闲置
                aux_loss = (Pi * fi).sum() * self.alpha  # 计算辅助损失，通过将每个专家的平均选择概率与其负载平衡因子相乘，并对所有专家求和，得到一个标量，然后乘以 alpha 权重，得到最终的辅助损失标量
        else:
            aux_loss = scores.new_zeros(1).squeeze()  # 如果不使用辅助损失，则创建一个值为 0 的标量张量，形状是 ()，并将其作为辅助损失返回
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):  # ！修正：原MoEFeedForaward拼写错误
    def __init__(self, config: cbMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        # 复制输入张量以便后续处理，保持原始输入不变
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理，将其调整为 (batch_size * seq_len, hidden_size)，例如 (4096, 512)，以便后续根据 topk_idx 选择专家进行处理
        x = x.view(-1, x.shape[-1])
        # 将 topk_idx 展开为一维张量，以便后续处理(4096,2) -> (8192,)，每个元素是一个专家索引，表示对应 token 选择的专家
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


class cbMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: cbMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)  # 每个 Transformer 层包含一个自注意力模块，负责捕捉序列中元素之间的关系和依赖

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )  # 输入的 LayerNorm，应用于自注意力模块的输入，使用 RMSNorm 替代传统的 LayerNorm，具有更高的数值稳定性和效率
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)  # ！修正：原MoEFeedForaward拼写错误
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states  # 残差连接，将自注意力模块的输出与输入的隐藏状态相加，形成残差连接，有助于缓解深层网络中的梯度消失问题，并促进信息在层与层之间的流动

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)  # post-norm，先对残差连接后的隐藏状态进行 LayerNorm，然后通过 FFN 模块计算 FFN 的输出，并与残差连接后的隐藏状态相加，形成第二个残差连接
        )
        return hidden_states, present_key_value


class cbMindModel(nn.Module):
    def __init__(self, config: cbMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [cbMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置中的最大位置编码长度和 RoPE 参数预计算频率矩阵，供后续的注意力模块使用
        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # 将预计算的频率矩阵注册为模型的 buffer，这样它们就会随着模型一起保存和加载，但不会被优化器更新，因为 persistent=False 表示这些 buffer 不需要持久化到模型的 state_dict 中。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids: [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):  # 兼容 transformers 4.30 之前的版本，past_key_values 可能是一个对象，包含一个 layers 属性
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)  # 如果 past_key_values 是 None，则创建一个长度等于层数的列表，每个元素都是 None，表示每层都没有 KV 缓存

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 通过检查第一个层的 past_key_values 是否存在来确定 start_pos，如果存在，则取其序列长度作为 start_pos，否则为 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],  # 从预计算的频率矩阵中切片出当前输入序列长度对应的频率，作为位置编码，传递给后续的注意力模块使用
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)  # 将每一层的 past_key_value 与对应的层进行配对，方便在前向传播中使用
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # ！修正：原MoEFeedForaward拼写错误
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss  # aux_loss


class cbMindForCausalLM(PreTrainedModel, GenerationMixin):  # 继承 PreTrainedModel 和 GenerationMixin，使得 cbMindForCausalLM 既具有预训练模型的功能，又支持文本生成的功能
    config_class = cbMindConfig

    def __init__(self, config: cbMindConfig):
        self.config = config
        super().__init__(config)  # 调用父类 PreTrainedModel 的构造函数，传入配置对象 config，完成模型的初始化
        self.model = cbMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 输出层，将模型的隐藏状态映射到词汇表大小的维度，输出每个位置上每个词的 logits 分数，供后续计算损失或生成文本使用
        self.model.embed_tokens.weight = self.lm_head.weight
        # 将输入嵌入层的权重与输出层的权重共享，这样可以减少模型参数的数量，并且在某些情况下可以提高模型的性能和泛化能力，因为输入和输出使用相同的词向量表示。

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # labels 参数用于计算语言模型的训练损失，如果提供了 labels，则会计算交叉熵损失；如果 labels 是 None，则不计算损失，通常用于推理阶段。
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,  # 这个参数用于指定在计算输出 logits 时保留输入序列的多少部分，支持整数和张量两种类型，如果是整数，则表示保留最后 logits_to_keep 个位置的 logits；如果是张量，则直接使用该张量作为切片索引来选择要保留的 logits。
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        # 根据 logits_to_keep 参数计算切片索引 slice_indices，用于从 hidden_states 中选择要保留的部分来计算输出 logits。
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # 通过输出层 lm_head 将选择的 hidden_states 切片映射到词汇表大小的维度，
        # 得到输出 logits，形状是 (batch_size, seq_len_to_keep, vocab_size)，其中 seq_len_to_keep 是根据 logits_to_keep 参数计算得到的要保留的序列长度。
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()  # 将 logits 的最后一个时间步切掉，因为它没有对应的标签
            shift_labels = labels[..., 1:].contiguous()      # 将 labels 的第一个时间步切掉，因为它没有对应的输入
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )  # 计算交叉熵损失，将 shift_logits 和 shift_labels 展平为二维张量，忽略标签值为 -100 的位置，这些位置通常用于填充或不参与损失计算

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output

