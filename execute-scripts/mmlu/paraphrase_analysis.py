"""
言い換えテスト用スクリプト
results/mmlu/original から数個のサンプルを抽出して言い換えをテストする

使用例:
    python paraphrase_analysis.py

環境変数:
    OPENAI_API_KEY: OpenAI APIキー
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# -------- 設定 --------
INPUT_DIR = "results/mmlu/original"
OUTPUT_DIR = "results/mmlu/paraphrase_test"
MODEL = "gpt-4.1"
NUM_SAMPLES = 50  # テストするサンプル数

# 言い換えプロンプト
PARAPHRASE_SYSTEM_PROMPT = """You are a text paraphrasing assistant. Rewrite text using different words while preserving content.

RULES:
1. Rephrase each sentence - do not summarize or skip content
2. Keep ALL mathematical calculations and step-by-step reasoning
3. Preserve mathematical expressions, formulas, option letters (A, B, C, D) exactly
4. Output ONLY the paraphrased text

LENGTH CONTROL (CRITICAL):
- Output must be SAME length as input (within 10% difference)
- Do NOT add explanations, elaborations, or extra context
- Do NOT make sentences more verbose - just use different words
- If input is concise, output must also be concise
- NEVER output just a single letter - paraphrase the full text

"""

PARAPHRASE_USER_PROMPT = """Paraphrase the following text. Keep the SAME length (within 10%). Do not add extra content or make it longer.

{text}"""


def paraphrase_text(client: OpenAI, text: str) -> tuple[str, dict]:
    """OpenAI APIを使ってテキストを言い換える。テキストとトークン使用量を返す"""
    if not text or len(text.strip()) == 0:
        return text, {"input": 0, "output": 0, "total": 0}

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
            {"role": "user", "content": PARAPHRASE_USER_PROMPT.format(text=text)}
        ],
        temperature=1,
        #reasoning_effort="medium",
    )
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }
    return response.choices[0].message.content.strip(), usage


def load_samples(input_dir: Path, num_samples: int) -> list:
    """複数のファイルからサンプルを抽出"""
    samples = []
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    for jsonl_file in jsonl_files[:num_samples]:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            line = f.readline()
            if line.strip():
                record = json.loads(line)
                record["_source_file"] = jsonl_file.name
                samples.append(record)

    return samples


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}")
    print(f"Testing with {NUM_SAMPLES} samples\n")

    samples = load_samples(input_dir, NUM_SAMPLES)
    results = []

    for i, sample in enumerate(samples, 1):
        print(f"Processing sample {i}/{NUM_SAMPLES}: {sample['_source_file']}")

        original = sample.get("analysis", "")
        paraphrased, usage = paraphrase_text(client, original)
        length_diff = abs(len(paraphrased) - len(original)) / len(original) * 100 if original else 0

        results.append({
            "source_file": sample["_source_file"],
            "question_number": sample.get("question_number"),
            "original": original,
            "paraphrased": paraphrased,
            "original_length": len(original),
            "paraphrased_length": len(paraphrased),
            "length_diff_percent": round(length_diff, 1),
            "tokens": usage,
        })

    # トークン合計を計算
    total_tokens = {"input": 0, "output": 0, "total": 0}
    for r in results:
        total_tokens["input"] += r["tokens"]["input"]
        total_tokens["output"] += r["tokens"]["output"]
        total_tokens["total"] += r["tokens"]["total"]

    # JSONで保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"comparison_{timestamp}.json"
    output_data = {
        "model": MODEL,
        "samples": NUM_SAMPLES,
        "total_tokens": total_tokens,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
