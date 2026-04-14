# 📌 数据介绍

## Ⅰ Tokenizer

分词器可以粗略理解成 LLM 使用的一本“词典”，负责把自然语言映射成 token id，再把 token id 解码回文本；项目中也提供了`train_tokenizer.py`作为词表训练示例。不建议重新训练 tokenizer，因为词表和切分规则一旦变化，模型权重、数据格式、推理接口与社区生态的兼容性都会下降，也会削弱模型的传播性。同时，tokenizer 还会影响 PPL 这类按 token 统计的指标，因此跨 tokenizer 比较时，BPB（Bits Per Byte）往往更有参考价值，可参考[这篇](https://skeptric.com/perplexity/)。
对 cbMind 这类小模型来说，词表大小还会直接影响 embedding 层和输出层的参数占比，因此保持词表精简通常是更合适的取舍。

<details>
<summary>Tokenizer介绍</summary>


第三方强大的开源模型例如 Yi、Qwen2、ChatGLM、Mistral、Llama 3 的 tokenizer 词表长度如下：

<table>
  <tr><th>Tokenizer模型</th><th>词表大小</th><th>来源</th></tr>
  <tr><td>Yi</td><td>64,000</td><td>01万物（中国）</td></tr>
  <tr><td>Qwen2</td><td>151,643</td><td>阿里云（中国）</td></tr>
  <tr><td>ChatGLM</td><td>151,329</td><td>智谱AI（中国）</td></tr>
  <tr><td>Mistral</td><td>32,000</td><td>Mistral AI（法国）</td></tr>
  <tr><td>Llama 3</td><td>128,000</td><td>Meta（美国）</td></tr>
  <tr><td>cbMind</td><td>6,400</td><td>自定义</td></tr>
</table>


> 当前主线为避免历史版本歧义并控制整体体积，统一使用 `cbMind_tokenizer`，不再维护 `mistral_tokenizer` 版本。

尽管 `cbMind_tokenizer` 的词表只有 `6400`，编解码效率弱于 `qwen2`、`glm` 等更偏中文友好的 tokenizer，但它能显著压缩 embedding 层和输出层的参数占比，更适合 cbMind 这类小模型的体积约束。
从实际使用效果看，这套 tokenizer 并没有明显带来生僻词解码失败的问题，整体仍然足够稳定可用；因此当前主线训练也统一沿用这套词表，而不再额外分叉维护其他 tokenizer 版本。

</details>

## Ⅱ Pretrain数据

`cbMind-3` 当前主线预训练数据为 `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`。  
这两份数据已经整理成统一的 `text -> next token prediction` 训练格式，目标是在较小算力下兼顾：

- 文本质量；
- 长度分布；
- 中英混合能力；
- 与后续 SFT / Tool Calling / RLAIF 阶段的模板衔接。

数据来源包括但不限于通用文本语料、对话整理语料、蒸馏补充语料，以及各类**宽松开源协议**可用的数据集；主线数据会在清洗、去重、长度控制与格式统一后再进入训练。数据来源于：[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)、[Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align) 等公开数据源。

其中：

- `pretrain_t2t_mini.jsonl` 更适合快速复现；
- `pretrain_t2t.jsonl` 更适合完整训练 `cbMind-3` 主线模型。

文件数据格式为

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

## Ⅲ SFT数据

`cbMind-3` 当前主线 SFT 数据为 `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`。相比更早期的 `sft_512 / sft_1024 / sft_2048` 方案，当前版本更强调：

- 统一模板；
- 更适合对话 + 思考标签 + Tool Calling 的混合训练；
- 尽量减少数据预处理分叉，降低复现成本。

其数据来源包括但不限于高质量指令跟随数据、公开对话数据、模型蒸馏合成数据，以及协议友好的开源数据集；在进入 `t2t` 主线前，会统一为当前仓库使用的多轮对话格式。当前主线中也包含大量合成数据，例如本人基于 `qwen3-4b` 合成的约 `10w` 条 `tool call` 数据，以及 `qwen3` 系列的 `reasoning` 数据等。其中社区主要来源有：[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)、[Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align)、[R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)、[COIG](https://huggingface.co/datasets/BAAI/COIG)、[Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT) 等。公布版本会确保数据来源与处理链路符合对应开源协议的可传递性约束，并遵守 Apache-2.0、CC-BY-NC-2.0 等相关协议要求。

其中：

- `sft_t2t_mini.jsonl`：适合快速训练对话模型；
- `sft_t2t.jsonl`：适合完整复现主线版本；
- `toolcall` 能力已经并入主线 SFT 数据。

所有 SFT 文件数据格式均为（包含对话数据、Tool Use数据）

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

## Ⅳ RL 数据

`cbMind` 当前主线 RL 数据为 `dpo.jsonl`。数据抽样自 [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)。

主线中会将这部分样本统一重组为当前仓库使用的偏好学习格式，用于奖励模型或偏好优化阶段训练；其中 `chosen` 表示更符合偏好的回复，`rejected` 表示相对较差的回复。

其中 `dpo.jsonl` 数据格式为

```json
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

除此之外，其他 RL 数据与 SFT 数据格式保持一致，通常是从 SFT 数据中按总长度和对话轮次筛选得到，并将最后一个 `assistant` 位置留空，供 rollout 阶段续写使用。

## Ⅴ cbMind 训练数据集

> [!NOTE]
> 当前主线训练所需的核心数据集已开源，因此无需再自行预处理大规模数据集，避免重复性的数据处理工作。

cbMind训练数据集下载地址： [ModelScope](https://www.modelscope.cn/datasets/gongjy/cbMind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/cbMind_dataset/tree/main)

> 无需全部clone，可单独下载所需的文件
> 下载单个文件到指定本地文件夹（以下载pretrain_t2t_mini.jsonl到当前路径下“dataset”目录为例）
> modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl --local_dir ./dataset

将下载的数据集文件放到`./dataset/`目录下（✨为推荐的必须项）

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, ✨)
├── lora_exam.jsonl (24.65MB)
├── lora_identity.jsonl (22.70KB)
├── lora_medical.jsonl (34.00MB)
├── pretrain_t2t.jsonl (10GB)
├── pretrain_t2t_mini.jsonl (1.24GB)
├── rlaif.jsonl (24MB, ✨)
├── sft_t2t_mini.jsonl (1.6GB, ✨)
└── sft_t2t.jsonl (14GB)

```

<details>
<summary>注：各数据集简介</summary>


* `agent_rl.jsonl` --Agentic RL 主线训练数据，用于 `train_agent.py` 的多轮 Tool-Use / CISPO / GRPO 训练
* `agent_rl_math.jsonl` --Agentic RL 纯数学补充数据，适合带最终校验目标的多轮推理/工具使用场景（用于RLVR）
* `dpo.jsonl` --RLHF阶段偏好训练数据（DPO）
* `pretrain_t2t_mini`✨ --`cbMind-3` 轻量预训练数据，适合快速复现（推荐设置`max_seq_len≈768`）
* `pretrain_t2t` --`cbMind-3` 主线预训练数据（推荐设置`max_seq_len≈380`）
* `rlaif.jsonl`✨ --RLAIF训练数据集，用于PPO/GRPO/CISPO等强化学习算法训练
* `sft_t2t_mini.jsonl`✨ --`cbMind-3` 轻量SFT数据（用于快速训练Zero模型），推荐设置`max_seq_len≈768`，其中已混入一部分 Tool Call 样本
* `sft_t2t.jsonl` --`cbMind-3` 主线SFT数据，适合完整复现，其中同样已混入 Tool Call 样本


训练参数 `max_seq_len` 目前指的是 tokens 长度，而非绝对字符数。
本项目tokenizer在中文文本上大约`1.5~1.7 字符/token`，纯英文的压缩比在`4~5 字符/token`，不同数据分布会有波动。
数据集命名标注的“最大长度”均为字符数，100长度的字符串可粗略换算成`100/1.5≈67`的tokens长度。

例如：

* 中文：`白日依山尽`5个字符可能被拆分为[`白日`,`依`,`山`,`尽`] 4个tokens；
* 英文：`The sun sets in the west`24个字符可能被拆分为[`The `,`sun `,`sets `,`in `,`the`,`west`] 6个tokens

“推荐设置”给出了各个数据集上最大tokens长度的粗略估计。
须知 `max_seq_len` 可以激进 / 保守 / 均衡地调整，因为更大或更小均无法避免副作用：一些样本短于 `max_seq_len` 后被 padding 浪费算力，一些样本长于 `max_seq_len` 后被截断语义。

在算力效率与语义完整性之间找到平衡点即可

</details>


![dataset](./dataset.jpg)

> cbMind 主线训练数据组成与推荐组合示意图

<details>
<summary>说明 & 推荐训练方案</summary>


* `cbMind-3` 主线推荐采用 `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl` 的阶段式训练组合。

* 想要最快速度从0实现Zero模型，推荐使用`pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl` 的数据组合

* 推荐具备一定算力资源或更在意效果的朋友完整复现 `cbMind-3`；仅有单卡GPU或更在意快速复现的朋友强烈推荐 mini 组合。

* 当前 `sft_t2t / sft_t2t_mini` 已经混入 Tool Call 数据，因此通常不需要再额外做一轮独立的 Tool Calling 监督微调。

</details>