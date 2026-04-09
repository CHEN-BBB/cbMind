import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import cbMindConfig, cbMindForCausalLM
#from model.model_lora import apply_lora, load_lora  # ！修正：原缺少LoRA加载支持
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if "model" in args.load_from:
        model = cbMindForCausalLM(
            cbMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(
                    args.use_moe
                ),  # ！修正：原缺少use_moe参数，MoE模型无法正确加载
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        moe_suffix = "_moe" if hasattr(args, "use_moe") and args.use_moe else ""
        # pth权重加载逻辑
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        # 加载模型权重，strict=True确保权重完全匹配，避免潜在的训练/推理不一致问题
        model.load_state_dict(
            torch.load(ckp, map_location=args.device), strict=True
        )  # ！修正：原strict=False会静默忽略丢失/多余的权重键

        # ！修正：原缺少LoRA加载逻辑
        if args.lora_weight != "None":
            apply_lora(model)
            load_lora(
                model,
                f"./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )
    print(
        f"cbMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)"  # ！修正：原残留MiniMind命名
    )
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="cbMind模型推理与对话"
    )
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="模型权重目录")
    parser.add_argument(
        "--weight",
        default="pretrain",
        type=str,
        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）",
    )
    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=8,
        type=int,
        help="隐藏层数量（Small/MoE=8, Base=16）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=False,
        action="store_true",
        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=8192,
        type=int,
        help="最大生成长度（注意：并非模型实际长文本能力）",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）",
    )
    parser.add_argument(
        "--top_p", default=0.85, type=float, help="nucleus采样阈值（0-1）"
    )
    parser.add_argument(
        "--historys",
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="运行设备",
    )
    args = parser.parse_args()

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    conversation = []
    # 获取模型和分词器
    model, tokenizer = init_model(args)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    # 设置文本流式输出，skip_prompt=True跳过输入提示，skip_special_tokens=True跳过特殊token（如BOS/EOS），让输出更干净自然
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)  # or setup_seed(random.randint(0, 2048))
        if input_mode == 0:
            print(f"👶: {prompt}")
        conversation = conversation[-args.historys :] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 对话模板构建：根据模型类型选择是否使用对话模板，pretrain模型直接拼接文本，其他模型使用apply_chat_template构建对话格式
        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Reasoning模型启用思考链提示，激励模型生成更具推理过程的回答，其他模型不使用该提示以保持简洁
        if args.weight == "reason":
            templates["enable_thinking"] = True  # 仅Reason模型使用
        # 构建输入文本：根据模型类型选择是否使用对话模板，pretrain模型直接拼接文本，其他模型使用apply_chat_template构建对话格式
        inputs = (
            tokenizer.apply_chat_template(**templates)
            if args.weight != "pretrain"
            else (tokenizer.bos_token + prompt)
        )
        # 分词并转换为模型输入格式，truncation=True确保输入不会超过模型最大长度，to(args.device)将输入移动到指定设备
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print("🤖️: ", end="")
        # 生成文本，使用模型的generate方法，传入输入ID和注意力掩码，设置生成参数如max_new_tokens、temperature、top_p等，
        # streamer参数实现流式输出，pad_token_id和eos_token_id确保生成过程正确处理文本边界
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        # 从生成的ID中提取新生成的部分，跳过输入文本对应的ID，使用tokenizer.decode将ID转换回文本，skip_special_tokens=True确保输出不包含特殊token
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()
