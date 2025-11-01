from functools import lru_cache
import threading
import logging
from llama_cpp import Llama
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    ReasoningEffort,
)

# -------- Settings --------
LLM_FILE = "model/gpt-oss-20b-F16.gguf"
N_CTX = 4096  # 0は無効。モデルに合わせて適切に
N_GPU_LAYERS = -1  # Metal/GPU を使うなら -1（全層）。CPUのみなら 0
SEED = 0

# 推論のスレッドセーフ化
_infer_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_llm(model_path: str = LLM_FILE) -> Llama:
    """
    モデルを一度だけロードして返す。
    lru_cacheによりプロセス内でシングルトンとして再利用される。
    """
    return Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,  # Metal ビルド時に有効
        seed=SEED,
        embedding=False,
        logits_all=False,
        use_mmap=True,   # メモリ効率を少し改善
        use_mlock=False, # 必要に応じてTrue（メモリ固定: 要権限）
        vocab_only=False,
    )


@lru_cache(maxsize=1)
def get_harmony_encoding():
    # エンコーディングもキャッシュ
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def create_harmony_prompt(user_prompt: str) -> str:
    encoding = get_harmony_encoding()

    system_message = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date("2025-06-28")
    )

    developer_message = (
        DeveloperContent.new()
        .with_instructions("Please answer the last question with a single character.")
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system_message),
            Message.from_role_and_content(Role.DEVELOPER, developer_message),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    text = encoding.decode_utf8(tokens)
    return text


def run_llm(prompt: str, *, max_tokens: int = 1000000) -> str:
    """
    共有の Llama インスタンスを使って推論。
    """
    logger = logging.getLogger("mmlu_logger")
    llm = get_llm()

    temperature = 1.0
    top_p = 1.0

    # llama-cpp-python の Llama.__call__ はスレッドセーフでない前提でロック
    with _infer_lock:
        logger.debug(f"Running LLM inference with max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # echo=True,  # もしプロンプト含めて返したい場合
        )

    return output["choices"][0]["text"]


def parse_llm_output(output: str):
    """
    LLMの出力から analysis / final を抽出
    """
    import re

    logger = logging.getLogger("mmlu_logger")
    result = {"analysis": "", "final": ""}

    analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
    final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'

    m = re.search(analysis_pattern, output, re.DOTALL)
    if m:
        result["analysis"] = m.group(1).strip()

    m = re.search(final_pattern, output, re.DOTALL)
    if m:
        result["final"] = m.group(1).strip()

    if result["analysis"] == "" or result["final"] == "":
        logger.error("LLM output parsing failed: analysis or final is empty")
        logger.error(f"Raw output: {output[:200]}...")  # 最初の200文字だけログ
        return ""

    if not result["final"] in ["A", "B", "C", "D"]:
        original_answer = result["final"]
        if len(original_answer) > 0:
            result["final"] = original_answer[0].upper()
            logger.warning(f"Invalid answer detected: '{original_answer}'. Using first character: '{result['final']}'")
        else:
            result["final"] = "A"  # デフォルト値
            logger.warning(f"Empty answer detected. Using default: 'A'")

    logger.debug(f"Successfully parsed answer: {result['final']}")
    return result


def generate(prompt: str):
    harmony_prompt = create_harmony_prompt(prompt)
    raw = run_llm(harmony_prompt)
    return parse_llm_output(raw)


def generate_from_prompt(harmony_prompt: str):
    """
    既に構築されたHarmony形式のプロンプトから答えを生成

    Args:
        harmony_prompt: Harmony形式のプロンプト（generate_early_answer_promptなどで作成済み）

    Returns:
        dict: parse_llm_outputの結果
    """
    raw = run_llm(harmony_prompt)
    return parse_llm_output(raw)


if __name__ == "__main__":
    print(generate("What is the weather in Tokyo?"))
