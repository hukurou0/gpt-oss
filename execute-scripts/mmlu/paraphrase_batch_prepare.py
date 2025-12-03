"""
Batch API用の言い換えリクエストJSONLファイルを生成するスクリプト
results/mmlu/original の全ファイル・全問題を処理
科目ごとにファイルを分けて出力

使用例:
    python paraphrase_batch_prepare.py

出力:
    results/mmlu/paraphrase_batch/{subject}.jsonl  (科目ごと)
    results/mmlu/paraphrase_batch/_metadata.json   (全体メタデータ)
"""

import json
from datetime import datetime
from pathlib import Path

# -------- 設定 --------
INPUT_DIR = "results/mmlu/original"
OUTPUT_DIR = "results/mmlu/paraphrase_batch"
MODEL = "gpt-4.1"

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


def create_batch_request(custom_id: str, text: str) -> dict:
    """Batch API用のリクエストオブジェクトを作成"""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                {"role": "user", "content": PARAPHRASE_USER_PROMPT.format(text=text)}
            ],
            "temperature": 1,
        }
    }


def process_subject_file(input_file: Path, output_file: Path) -> dict:
    """1つの科目ファイルを処理してBatch用JSONLを生成"""
    subject = input_file.stem
    request_count = 0
    skipped_count = 0
    metadata_entries = []

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("analysis", "")

            # 空のテキストはスキップ
            if not text or len(text.strip()) == 0:
                skipped_count += 1
                continue

            # custom_idを生成: {subject}_q{question_number}
            question_number = record.get("question_number", line_num)
            custom_id = f"{subject}_q{question_number}"

            request = create_batch_request(custom_id, text)
            fout.write(json.dumps(request, ensure_ascii=False) + "\n")

            metadata_entries.append({
                "custom_id": custom_id,
                "question_number": question_number,
                "original_length": len(text)
            })

            request_count += 1

    return {
        "subject": subject,
        "requests": request_count,
        "skipped": skipped_count,
        "metadata": metadata_entries
    }


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} subject files in {INPUT_DIR}\n")

    total_requests = 0
    total_skipped = 0
    subject_stats = []

    for input_file in jsonl_files:
        output_file = output_dir / input_file.name
        result = process_subject_file(input_file, output_file)

        total_requests += result["requests"]
        total_skipped += result["skipped"]
        subject_stats.append({
            "subject": result["subject"],
            "file": input_file.name,
            "requests": result["requests"],
            "skipped": result["skipped"]
        })

        print(f"  {result['subject']}: {result['requests']} requests")

    # メタデータを保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = output_dir / "_metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL,
            "created_at": timestamp,
            "total_requests": total_requests,
            "total_skipped": total_skipped,
            "subjects": subject_stats
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"Total subjects: {len(jsonl_files)}")
    print(f"Total requests: {total_requests}")
    print(f"Skipped (empty): {total_skipped}")
    print(f"Metadata: {metadata_file}")

    # 合計ファイルサイズ
    total_size = sum((output_dir / f.name).stat().st_size for f in jsonl_files)
    if total_size > 1024 * 1024:
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    else:
        print(f"Total size: {total_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
