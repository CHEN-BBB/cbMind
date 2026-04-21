# cbMind Trainer 训练体系文档

> 覆盖 6 种训练范式：Pretrain → Full SFT → LoRA → DPO → PPO → GRPO

---

## 目录

1. [整体训练流水线](#整体训练流水线)
2. [公共基础设施（trainer_utils.py）](#公共基础设施)
3. [Pretrain — 预训练](#1-pretrain--预训练)
4. [Full SFT — 全参数监督微调](#2-full-sft--全参数监督微调)
5. [LoRA — 低秩适配微调](#3-lora--低秩适配微调)
6. [DPO — 直接偏好优化](#4-dpo--直接偏好优化)
7. [PPO — 近端策略优化](#5-ppo--近端策略优化)
8. [GRPO — 组相对策略优化](#6-grpo--组相对策略优化)
9. [各方法横向对比](#各方法横向对比)
10. [关键技术细节汇总](#关键技术细节汇总)

---

## 整体训练流水线

```
原始语料
    │
    ▼
[1] Pretrain（train_pretrain.py）
    └─ 从零学习语言建模能力
    │
    ▼
[2] Full SFT（train_full_sft.py）
    └─ 全参数微调，注入指令跟随能力
    │
    ▼ ─────────────────────────────────
    │                                  │
[3] LoRA（train_lora.py）         [4] DPO（train_dpo.py）
    └─ 轻量化垂直领域适配              └─ 偏好学习，离线对齐
    │
    ▼
[5] PPO（train_ppo.py）
    └─ 在线强化学习，多模型协作
    │
    ▼
[6] GRPO（train_grpo.py）
    └─ 组内相对奖励，高效在线对齐
```

---

## 公共基础设施

**文件：** `trainer_utils.py`

所有训练脚本共享以下工具函数：

| 函数 / 类 | 功能说明 |
|---|---|
| `is_main_process()` | 判断是否为分布式主进程（rank==0） |
| `Logger(content)` | 仅主进程打印日志 |
| `get_lr(current_step, total_steps, lr)` | 余弦退火学习率：`step=0 → lr`，`step=T → 0.1*lr` |
| `init_distributed_mode()` | 初始化 NCCL 分布式进程组，返回 local_rank |
| `setup_seed(seed)` | 设置全局随机种子（Python/NumPy/PyTorch） |
| `init_model(lm_config, from_weight, ...)` | 加载 tokenizer + cbMindForCausalLM，支持 strict=False |
| `lm_checkpoint(lm_config, weight, ...)` | 保存/加载完整训练状态（支持断点续训） |
| `SkipBatchSampler` | 跳过前 N 个 batch 的自定义采样器（断点续训用） |

### `get_lr` 公式

```python
lr = lr_base * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
# step=0:     lr_base * 1.0
# step=T/2:   lr_base * 0.55
# step=T:     lr_base * 0.1
```

### `lm_checkpoint` 保存格式

```python
{
    "model":      state_dict,        # 模型权重（半精度）
    "optimizer":  optimizer.state_dict(),
    "epoch":      int,
    "step":       int,
    "world_size": int,               # 用于多卡→单卡续训的 step 自动换算
    "wandb_id":   str | None,
    # kwargs 支持额外字段：scaler、scheduler、critic_model、...
}
```

---

## 1. Pretrain — 预训练

**文件：** `train_pretrain.py`  
**目标：** 从头训练，让模型学习语言建模（next-token prediction）  
**数据集：** `PretrainDataset`，输出 `(input_ids, labels, attention_mask)`

### 训练步骤

```
Step 1  初始化分布式环境 + 随机种子
Step 2  构建 cbMindConfig + 创建保存目录
Step 3  设置混合精度上下文（bfloat16/float16）
Step 4  初始化 WandB/SwanLab 实验跟踪（可选）
Step 5  init_model() → 加载模型（from_weight='none' 则从零初始化）
Step 6  构建 PretrainDataset + DistributedSampler + DataLoader
Step 7  初始化 AdamW 优化器 + GradScaler
Step 8  如有 checkpoint，恢复 model/optimizer/scaler/epoch/step
Step 9  DDP 包装（忽略 freqs_cos/freqs_sin 同步）
Step 10 训练循环：
          ├─ 动态学习率（余弦退火）
          ├─ AMP 前向：res = model(input_ids, labels=labels, ...)
          ├─ loss = res.loss + res.aux_loss（MoE 辅助损失）
          ├─ loss /= accumulation_steps
          ├─ scaler.scale(loss).backward()
          ├─ 每 accumulation_steps 步：unscale → clip_grad_norm → step → update → zero_grad
          ├─ 每 log_interval 步：打印 loss/lr/ETA
          └─ 每 save_interval 步：保存 .pth（半精度）+ lm_checkpoint
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `learning_rate` | 5e-4 | 预训练学习率（相对较大） |
| `batch_size` | 32 | — |
| `accumulation_steps` | 8 | 等效 batch=256 |
| `epochs` | 1 | 预训练通常 1 轮 |
| `max_seq_len` | 512 | — |
| `from_weight` | `none` | 从零开始 |

### 代码结构

```
train_pretrain.py
├── train_epoch(epoch, loader, iters, start_step, wandb)  # 训练单 epoch
└── __main__
    ├── argparse 解析
    ├── 初始化（分布式/混合精度/wandb）
    ├── 模型 + 数据 + 优化器初始化
    └── for epoch in range(start_epoch, epochs): train_epoch(...)
```

---

## 2. Full SFT — 全参数监督微调

**文件：** `train_full_sft.py`  
**目标：** 在预训练模型上全参数微调，注入指令跟随能力  
**数据集：** `SFTDataset`，包含 instruction/response 对，`(input_ids, labels, attention_mask)`

### 训练步骤

```
Step 1  初始化分布式环境 + 随机种子
Step 2  构建 cbMindConfig + 创建保存目录
Step 3  设置混合精度上下文
Step 4  初始化实验跟踪
Step 5  init_model(from_weight='pretrain') → 加载预训练权重
Step 6  （可选）torch.compile 加速（20%~40% 提升）
Step 7  构建 SFTDataset + 采样器 + DataLoader
Step 8  初始化 AdamW 优化器 + GradScaler
Step 9  恢复 checkpoint（如开启）
Step 10 DDP 包装
Step 11 训练循环：
          ├─ setup_seed(42 + epoch)  →  torch.randperm 随机打乱数据
          ├─ SkipBatchSampler 处理断点续训
          ├─ 同 Pretrain 的 AMP + 梯度累积 + 梯度裁剪流程
          └─ 保存时额外 del state_dict 释放内存
Step 12 dist.destroy_process_group() 清理分布式
```

### 与 Pretrain 的主要差异

| 对比项 | Pretrain | Full SFT |
|---|---|---|
| 数据集类 | `PretrainDataset` | `SFTDataset` |
| 加载权重 | `none`（从零） | `pretrain` |
| 学习率 | 5e-4 | 1e-6（更小，防止遗忘） |
| batch_size | 32 | 16 |
| torch.compile | 不支持 | 支持 |
| epoch 数据打乱 | DataLoader shuffle | `torch.randperm` 手动打乱 |
| 日志字段 | loss/lr | loss/logits_loss/aux_loss/lr |

---

## 3. LoRA — 低秩适配微调

**文件：** `train_lora.py`  
**目标：** 参数高效微调，冻结原模型权重，仅训练低秩分解矩阵  
**数据集：** `SFTDataset`

### LoRA 核心原理

```
原始权重 W ∈ R^{d×d}   →  冻结，requires_grad=False
新增矩阵 A ∈ R^{d×r}   →  可训练（r << d）
新增矩阵 B ∈ R^{r×d}   →  可训练

前向计算：output = W·x + B·A·x
参数占比：通常 1%~5%
```

### 训练步骤

```
Step 1-4  同 Full SFT（环境初始化/混合精度/实验跟踪）
Step 5    init_model(from_weight='full_sft') → 加载 SFT 权重
Step 6    apply_lora(model) → 注入 LoRA 适配器（A/B 矩阵）
Step 7    统计并打印 LoRA 参数占比
Step 8    参数冻结：
            for name, param in model.named_parameters():
                if "lora" in name:  param.requires_grad = True
                else:               param.requires_grad = False
Step 9    AdamW 优化器仅传入 lora_params
Step 10   训练循环（与 Full SFT 相同流程）
Step 11   保存时调用 save_lora()，仅保存 A/B 矩阵（体积极小）
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `learning_rate` | 1e-4 | 比 Full SFT 大，因参数少 |
| `epochs` | 50 | 数据少时需更多轮 |
| `from_weight` | `full_sft` | 基于 SFT 模型 |
| `lora_name` | `lora_identity` | 标识任务类型 |

### 保存格式对比

```
Full SFT: 完整模型 state_dict（~100%参数）
LoRA:     仅 A/B 矩阵（~1-5%参数）→ save_lora(model, path)
```

---

## 4. DPO — 直接偏好优化

**文件：** `train_dpo.py`  
**目标：** 利用人类偏好数据对齐模型，无需显式奖励模型  
**数据集：** `DPODataset`，包含 `(x_chosen, x_rejected, y_chosen, y_rejected, mask_*)`

### DPO 核心公式

```
L_DPO = -log σ( β · [ (log π(y_w|x) - log π_ref(y_w|x))
                      - (log π(y_l|x) - log π_ref(y_l|x)) ] )

其中：
  π       = 策略模型（需优化）
  π_ref   = 参考模型（冻结，与策略模型初始权重相同）
  y_w     = chosen（人类偏好回答）
  y_l     = rejected（人类不偏好回答）
  β       = 优化强度（默认 0.1）
```

### 训练步骤

```
Step 1-4  同前（环境/混合精度/实验跟踪）
Step 5    初始化双模型：
            policy_model  = init_model(from_weight='full_sft')  →  可训练
            ref_model     = init_model(from_weight='full_sft')  →  ref_model.requires_grad_(False)
Step 6    构建 DPODataset + 优化器（仅 policy_model）
Step 7    训练循环：
            ├─ 合并 chosen/rejected → x = cat([x_chosen, x_rejected])
            ├─ ref_model（no_grad）→ ref_logits → ref_log_probs
            ├─ policy_model → logits → policy_log_probs
            ├─ dpo_loss(ref_log_probs, policy_log_probs, mask, β)
            ├─ loss += aux_loss（MoE 辅助）
            └─ 标准 AMP + 梯度累积 + 保存
```

### 核心函数详解

```python
def logits_to_log_probs(logits, labels):
    # [B, S, V] → log_softmax → gather 对应 token 位置 → [B, S]
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 序列级均值 log 概率
    ref_seq   = (ref_log_probs * mask).sum(1) / mask.sum(1)
    policy_seq = (policy_log_probs * mask).sum(1) / mask.sum(1)
    # 拆分 chosen/rejected（batch 前一半/后一半）
    pi_logratios  = policy_seq[:B//2] - policy_seq[B//2:]
    ref_logratios = ref_seq[:B//2]    - ref_seq[B//2:]
    # DPO 损失
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `learning_rate` | 4e-8 | 极小，防止遗忘（建议 ≤ 5e-8） |
| `beta` | 0.1 | 控制与参考模型的偏差强度 |
| `batch_size` | 4 | DPO 需同时处理 chosen+rejected，显存占用较大 |

---

## 5. PPO — 近端策略优化

**文件：** `train_ppo.py`  
**目标：** 在线强化学习，使用奖励模型引导生成对齐  
**数据集：** `RLAIFDataset`，仅包含 prompt

### PPO 四模型架构

```
Actor Model     ──→ 需优化的策略模型（生成回答）
Old Actor       ──→ 上一版策略（used for重要性采样比率 ratio）
Critic Model    ──→ 价值函数（估计状态价值 V(s)），基于 cbMindForCausalLM + value_head
Reference Model ──→ 冻结的 SFT 基线（KL 惩罚参考）
Reward Model    ──→ internlm2-1.8b-reward（外部打分 + 格式奖励）
```

### CriticModel 结构

```python
class CriticModel(cbMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)  # 额外的价值头

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.model.norm(self.model(input_ids, attention_mask)[0])  # [B, L, H]
        return self.value_head(hidden).squeeze(-1)  # [B, L]
```

### 奖励计算（两部分叠加）

```
Reward = 格式奖励（规则 based）+ 奖励模型评分

格式奖励（reasoning 模式）：
  - <think>...</think><answer>...</answer> 完整格式 → +0.5
  - 每个正确 tag（<think>/<think>/...） → +0.25（共4个）
  
奖励模型评分：
  - internlm2-1.8b-reward.get_score(messages)
  - 分数 clamp 至 [-3, 3]
  - 若有 <answer> 标签：score = score*0.4 + answer_score*0.6
```

### 训练步骤

```
Step 1-4  环境/混合精度/实验跟踪初始化
Step 5    初始化5个模型：actor / old_actor / ref / critic / reward
Step 6    构建 RLAIFDataset + 双优化器（actor_optimizer / critic_optimizer）
Step 7    CosineAnnealingLR 学习率调度（区别于其他方案的手动 get_lr）
Step 8    训练循环（ppo_train_epoch）：
            ├─ tokenize prompts（左侧 padding）
            ├─ actor.generate() 采样回答（no_grad）
            ├─ calculate_rewards() 计算奖励
            ├─ critic_model() 估计价值 V(s)
            ├─ advantage = reward - V(s).detach()
            ├─ actor 前向 → actor_logp
            ├─ old_actor/ref 前向（no_grad）→ old_logp / ref_logp
            ├─ ratio = exp(actor_logp - old_logp)
            ├─ kl_ref = actor_logp - ref_logp（KL 惩罚）
            ├─ PPO Clip 损失：
            │     surr1 = ratio * advantage
            │     surr2 = clamp(ratio, 1±ε) * advantage
            │     policy_loss = -min(surr1, surr2).mean()
            ├─ value_loss = MSE(V(s), reward)
            ├─ total_loss = policy_loss + vf_coef * value_loss + kl_coef * kl_ref
            ├─ backward + clip_grad_norm（actor + critic 各自裁剪）
            ├─ actor_optimizer.step() + critic_optimizer.step()
            ├─ actor/critic scheduler.step()
            └─ 每 update_old_actor_freq 步同步 old_actor 权重
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `learning_rate` | 2e-5 | Actor 学习率 |
| `critic_learning_rate` | 2e-5 | Critic 学习率 |
| `clip_epsilon` | 0.1 | PPO 裁剪范围 |
| `vf_coef` | 0.1 | 价值损失权重 |
| `kl_coef` | 0.01 | KL 惩罚权重 |
| `update_old_actor_freq` | 4 | old actor 同步频率 |
| `max_gen_len` | 768 | 生成回答最大长度 |

---

## 6. GRPO — 组相对策略优化

**文件：** `train_grpo.py`  
**目标：** 无 Critic 模型的高效在线对齐，通过组内奖励标准化计算 advantage  
**数据集：** `RLAIFDataset`，仅包含 prompt

### GRPO vs PPO 核心区别

```
PPO：  advantage = reward - V(s)      需要 Critic 模型估计价值
GRPO： advantage = (r - mean_r) / std_r  同一 prompt 多次采样后组内标准化
```

### GRPO 优势计算

```python
# 每个 prompt 生成 num_generations 个回答
grouped_rewards = rewards.view(-1, num_generations)       # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G) # [B*G]
std_r  = grouped_rewards.std(dim=1).repeat_interleave(G)  # [B*G]
advantages = (rewards - mean_r) / (std_r + 1e-4)
advantages = clamp(advantages, -10, 10)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 二次标准化
```

### 损失函数（per-token 级别）

```python
# KL 散度（per token）
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = exp(kl_div) - kl_div - 1  # 近似无偏 KL

# GRPO per-token 损失
per_token_loss = -(
    exp(logps - logps.detach()) * advantages  # 策略梯度项
    - beta * per_token_kl                     # KL 惩罚项
)

# 序列级均值（仅计算 EOS 之前的 token）
loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
```

### 训练步骤

```
Step 1-2  环境/实验跟踪初始化
Step 3    初始化 policy_model + ref_model + reward_model（无 Critic）
Step 4    构建 RLAIFDataset + AdamW + CosineAnnealingLR
Step 5    训练循环（grpo_train_epoch）：
            ├─ tokenize prompts
            ├─ model.generate(num_return_sequences=G) → completion_ids
            ├─ get_per_token_logps(model, outputs, n)  → per_token_logps
            ├─ get_per_token_logps(ref_model, outputs, n) → ref_per_token_logps（no_grad）
            ├─ decode → calculate_rewards()
            ├─ 组内标准化 → advantages
            ├─ 构建 completion_mask（EOS 之前有效）
            ├─ per_token_kl + per_token_loss 计算
            ├─ loss.backward()
            ├─ clip_grad_norm + optimizer.step() + scheduler.step()
            └─ 每步 del + cuda empty_cache（显存管理）
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `learning_rate` | 8e-8 | 极小，GRPO 更新幅度小 |
| `num_generations` | 8 | 每 prompt 采样数，越大优势估计越准 |
| `beta` | 0.02 | KL 惩罚系数（比 DPO 更小） |
| `max_gen_len` | 1536 | 支持长思维链 |
| `batch_size` | 2 | 因每 prompt 生成 G 个样本，显存极大 |

---

## 各方法横向对比

| 维度 | Pretrain | Full SFT | LoRA | DPO | PPO | GRPO |
|---|---|---|---|---|---|---|
| **训练目标** | 语言建模 | 指令跟随 | 垂直适配 | 偏好对齐 | 在线RL对齐 | 在线RL对齐 |
| **需要哪个权重** | 无（从零） | pretrain | full_sft | full_sft | full_sft | full_sft |
| **可训练参数** | 全部 | 全部 | LoRA A/B（~1-5%） | 全部（policy） | Actor全部 | 全部 |
| **模型数量** | 1 | 1 | 1 | 2（policy+ref） | 5（actor/old/critic/ref/reward） | 3（policy+ref+reward） |
| **数据格式** | 纯文本 | instruction+response | instruction+response | chosen+rejected | prompt only | prompt only |
| **奖励信号** | 无 | 无 | 无 | 隐式（偏好对） | 外部 reward model | 外部 reward model |
| **默认学习率** | 5e-4 | 1e-6 | 1e-4 | 4e-8 | 2e-5 | 8e-8 |
| **显存需求** | 中 | 中 | 低 | 中（2倍模型） | 极高（5个模型） | 高（3个模型+多采样） |
| **训练复杂度** | ★★☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **学习率调度** | 余弦退火（get_lr） | 余弦退火 | 余弦退火 | 余弦退火 | CosineAnnealingLR | CosineAnnealingLR |

---

## 关键技术细节汇总

### 1. 混合精度训练

```python
# bfloat16（推荐，数值稳定）
autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled=False)  # bfloat16 不需要 scaler

# float16（更高效，需要 scaler）
autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=True)
```

### 2. 梯度累积

```python
loss = loss / accumulation_steps
scaler.scale(loss).backward()
if step % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

### 3. DDP 特殊处理

```python
# RoPE 位置编码缓存不需要同步
model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
model = DistributedDataParallel(model, device_ids=[local_rank])

# 保存时需要 .module 访问真实模型
state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
```

### 4. 权重保存策略

```python
# 推理用：半精度，节省 ~50% 磁盘空间
torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

# 续训用：包含完整训练状态
lm_checkpoint(lm_config, weight=..., model=..., optimizer=..., scaler=..., epoch=..., step=...)

# LoRA：仅保存 A/B 矩阵
save_lora(model, lora_save_path)
```

### 5. 断点续训机制

```python
# 检测 checkpoint
ckp_data = lm_checkpoint(lm_config, weight=save_weight, save_dir="../checkpoints")

# 恢复状态
model.load_state_dict(ckp_data["model"])
optimizer.load_state_dict(ckp_data["optimizer"])
start_epoch = ckp_data["epoch"]
start_step  = ckp_data.get("step", 0)

# 跳过已训练的 batch
batch_sampler = SkipBatchSampler(sampler, batch_size, skip_batches=start_step)

# 多卡→单卡续训时，step 自动换算
step = step * saved_world_size // current_world_size
```

### 6. MoE 辅助损失

```python
# MoE（Mixture of Experts）架构额外引入路由均衡损失
loss = res.loss + res.aux_loss
# res.aux_loss 为 None 时模型自动处理，日志中单独记录
```

### 7. PPO vs GRPO 优势计算对比

```
PPO：
  advantage = reward - V(s)
  需要额外的 Critic 网络学习 V(s)
  value_loss = MSE(V(s), reward)

GRPO：
  对每个 prompt 生成 G 个回答 {r_1, ..., r_G}
  advantage_i = (r_i - mean(r)) / (std(r) + ε)
  无需 Critic，通过组内对比消除基线偏差
```

---

*文档生成时间：2026-04-21 | 基于 cbMind trainer 代码库*
