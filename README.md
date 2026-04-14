<div align="center">

# 🧠 cbMind

**一个从零构建的中文小型语言模型**

基于 [minimind](https://github.com/jingyaogong/minimind) 复刻 · Transformer 架构 · 支持预训练 / SFT / LoRA / DPO / PPO / GRPO

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 项目简介

cbMind 是一个面向学习与研究目的的中文小型语言模型，从零实现了完整的 LLM 训练流程，包括：

- **预训练（Pretrain）**：基于海量文本的下一个 token 预测
- **监督微调（SFT）**：指令跟随与多轮对话能力
- **LoRA 微调**：参数高效的垂直领域适配
- **强化学习**：DPO / PPO / GRPO 偏好对齐
- **MoE 架构**：可选的混合专家模块

模型支持多种规格配置，最小仅 **26M** 参数，适合单卡 / 低资源环境快速复现。

---

## 🏗️ 项目结构

```
cbMind/
├── model/
│   ├── model.py              # 模型主体（cbMindConfig + cbMindForCausalLM）
│   ├── model_lora.py         # LoRA 模块实现
│   ├── tokenizer.json        # 自定义分词器（词表大小 6400）
│   └── tokenizer_config.json
├── trainer/
│   ├── train_pretrain.py     # 预训练脚本
│   ├── train_full_sft.py     # 全量 SFT 脚本
│   ├── train_lora.py         # LoRA 微调脚本
│   ├── train_dpo.py          # DPO 强化学习脚本
│   ├── train_ppo.py          # PPO 强化学习脚本
│   ├── train_grpo.py         # GRPO 强化学习脚本
│   └── trainer_utils.py      # 公共工具函数
├── dataset/
│   ├── lm_dataset.py         # 数据集加载（Pretrain / SFT / DPO）
│   └── dataset.md            # 数据集说明文档
├── eval.py                   # 推理与交互对话脚本
├── main.py                   # 项目入口
└── pyproject.toml            # 项目依赖配置
```

---

## 🚀 安装步骤

### 1. 环境要求

| 依赖 | 版本要求 |
|------|---------|
| Python | ≥ 3.11 |
| PyTorch | ≥ 2.9.0 |
| transformers | ≥ 4.57.1 |
| CUDA（推荐） | ≥ 12.0 |

### 2. 克隆仓库

```bash
git clone https://github.com/your-username/cbMind.git
cd cbMind
```

### 3. 安装依赖

推荐使用 `uv`（快速）或 `pip` 安装：

```bash
# 方式一：使用 uv（推荐）
pip install uv
uv sync

# 方式二：使用 pip
pip install -e .

# 方式三：手动安装
pip install torch>=2.9.0 transformers>=4.57.1 numpy>=2.3.4 pandas>=2.3.3
```

> 如果需要 Flash Attention 加速（推荐）：
> ```bash
> pip install flash-attn --no-build-isolation
> ```

### 4. 下载数据集

训练数据托管在 ModelScope / HuggingFace，按需下载：

```bash
# 下载轻量预训练数据（推荐快速复现）
modelscope download --dataset gongjy/cbMind_dataset pretrain_t2t_mini.jsonl --local_dir ./dataset

# 下载轻量 SFT 数据
modelscope download --dataset gongjy/cbMind_dataset sft_t2t_mini.jsonl --local_dir ./dataset

# 下载 RLAIF 数据（强化学习阶段）
modelscope download --dataset gongjy/cbMind_dataset rlaif.jsonl --local_dir ./dataset
```

完整数据集地址：[ModelScope](https://www.modelscope.cn/datasets/gongjy/cbMind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/cbMind_dataset/tree/main)

---

## 💡 使用示例

### 模型规格对照表

| 规格 | `hidden_size` | `num_hidden_layers` | 参数量 | 场景 |
|------|:---:|:---:|:---:|------|
| Small | 512 | 8 | ~26M | 快速实验 / 低资源 |
| MoE | 640 | 8 | ~145M | 稀疏专家架构 |
| Base | 768 | 16 | ~104M | 完整复现 |

---

### 阶段一：预训练

```bash
cd trainer
python train_pretrain.py \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --epochs 1 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --max_seq_len 512 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_dir ../out \
  --device cuda:0
```

多卡分布式训练（DDP）：

```bash
torchrun --nproc_per_node=4 trainer/train_pretrain.py \
  --hidden_size 512 \
  --batch_size 16 \
  --data_path ./dataset/pretrain_t2t_mini.jsonl
```

---

### 阶段二：监督微调（SFT）

```bash
cd trainer
python train_full_sft.py \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --max_seq_len 512 \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --save_dir ../out
```

---

### 阶段三：LoRA 微调（可选）

适合垂直领域（医疗、法律等）的低资源适配：

```bash
cd trainer
python train_lora.py \
  --hidden_size 512 \
  --data_path ../dataset/lora_medical.jsonl \
  --from_weight full_sft \
  --lora_name lora_medical
```

---

### 阶段四：强化学习对齐（可选）

```bash
# DPO 偏好优化
python trainer/train_dpo.py --hidden_size 512 --data_path ./dataset/dpo.jsonl

# GRPO
python trainer/train_grpo.py --hidden_size 512 --data_path ./dataset/rlaif.jsonl

# PPO
python trainer/train_ppo.py --hidden_size 512 --data_path ./dataset/rlaif.jsonl
```

---

### 推理 & 交互对话

```bash
# 交互对话（支持自动测试 / 手动输入两种模式）
python eval.py \
  --load_from model \
  --weight full_sft \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --temperature 0.85 \
  --top_p 0.85 \
  --max_new_tokens 1024
```

携带历史对话轮次（多轮对话）：

```bash
python eval.py --weight full_sft --historys 4
```

启用 RoPE 长文本外推（4×）：

```bash
python eval.py --weight full_sft --inference_rope_scaling
```

使用 LoRA 权重：

```bash
python eval.py --weight full_sft --lora_weight lora_medical --hidden_size 512
```

加载 Transformers 格式模型（HuggingFace Hub 等）：

```bash
python eval.py --load_from /path/to/hf_model_dir
```

---

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|:---:|
| `--hidden_size` | 隐藏层维度（512/640/768） | 512 |
| `--num_hidden_layers` | 隐藏层数（Small/MoE=8, Base=16） | 8 |
| `--use_moe` | 是否使用 MoE 架构（0/1） | 0 |
| `--weight` | 权重前缀（pretrain/full_sft/rlhf/reason/grpo/ppo_actor） | pretrain |
| `--temperature` | 生成温度（0~1，越大越随机） | 0.85 |
| `--top_p` | Nucleus 采样阈值 | 0.85 |
| `--max_new_tokens` | 最大生成 token 数 | 8192 |
| `--historys` | 携带历史对话轮数（需为偶数） | 0 |
| `--device` | 运行设备（cuda/cpu） | 自动检测 |

---

## 📊 训练流程总览

```
原始语料
   │
   ▼
Pretrain（下一 token 预测）
   │
   ▼
Full SFT（指令微调 + 多轮对话 + Tool Call）
   │
   ├──► LoRA（可选：垂直领域适配）
   │
   ▼
RLHF（可选：DPO / PPO / GRPO 偏好对齐）
```

---



### Issue 规范

- 🐛 **Bug 报告**：请附上复现步骤、报错信息、Python / PyTorch 版本
- 💡 **功能建议**：描述使用场景和期望行为
- 📖 **文档改进**：直接指出错误或缺失内容

### 开发重点方向

- [ ] 模型权重开源（ModelScope / HuggingFace）
- [ ] 推理性能优化（量化、KV Cache）
- [ ] Web Demo / API 服务封装
- [ ] 评测基准接入（C-Eval、MMLU 等）
- [ ] Tool Calling 端到端评测

---

## 📄 致谢 & License

本项目基于 [minimind](https://github.com/jingyaogong/minimind) 复刻，感谢原作者的开源贡献。

数据集来源包括：[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)、[Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align)、[R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)、[DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k) 等，遵循相关开源协议。

MIT License © 2025 cbMind Contributors
