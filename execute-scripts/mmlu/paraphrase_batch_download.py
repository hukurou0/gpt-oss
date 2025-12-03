"""
Batch APIの結果をダウンロードするスクリプト
完了したバッチの結果を科目ごとにダウンロード

使用例:
    # 完了した全バッチの結果をダウンロード
    python paraphrase_batch_download.py

    # 特定の科目のみダウンロード
    python paraphrase_batch_download.py --subjects abstract_algebra anatomy

    # 元データとマージして保存
    python paraphrase_batch_download.py --merge

環境変数:
    OPENAI_API_KEY: OpenAI APIキー
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# -------- 設定 --------
BATCH_DIR = "results/mmlu/paraphrase_batch"
STATUS_FILE = "results/mmlu/paraphrase_batch/_batch_status.json"
OUTPUT_DIR = "results/mmlu/paraphrase_batch/results"
ORIGINAL_DIR = "results/mmlu/original"
MERGED_DIR = "results/mmlu/paraphrased"


def load_status() -> dict:
    """保存された状態を読み込み"""
    status_path = Path(STATUS_FILE)
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"batches": []}


def save_status(status: dict):
    """状態を保存"""
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def download_batch_result(client: OpenAI, batch_info: dict, output_dir: Path) -> Path | None:
    """バッチ結果をダウンロード"""
    subject = batch_info["subject"]

    # 状態を確認
    batch = client.batches.retrieve(batch_info["batch_id"])

    if batch.status != "completed":
        print(f"  Skipping {subject} (status: {batch.status})")
        return None

    if not batch.output_file_id:
        print(f"  Skipping {subject} (no output file)")
        return None

    # 結果ファイルをダウンロード
    print(f"  Downloading {subject}...")
    content = client.files.content(batch.output_file_id)

    output_file = output_dir / f"{subject}.jsonl"
    with open(output_file, "wb") as f:
        f.write(content.read())

    # エラーファイルがあればダウンロード
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_file = output_dir / f"{subject}_errors.jsonl"
        with open(error_file, "wb") as f:
            f.write(error_content.read())
        print(f"    Error file saved: {error_file.name}")

    return output_file


def parse_batch_result(result_file: Path) -> dict:
    """バッチ結果をパースしてcustom_id -> response のマップを作成"""
    results = {}

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            custom_id = data["custom_id"]

            if data.get("error"):
                results[custom_id] = {
                    "error": data["error"],
                    "paraphrased": None
                }
            else:
                response = data["response"]
                if response["status_code"] == 200:
                    body = response["body"]
                    content = body["choices"][0]["message"]["content"]
                    usage = body["usage"]
                    results[custom_id] = {
                        "paraphrased": content,
                        "tokens": {
                            "input": usage["prompt_tokens"],
                            "output": usage["completion_tokens"],
                            "total": usage["total_tokens"]
                        }
                    }
                else:
                    results[custom_id] = {
                        "error": f"HTTP {response['status_code']}",
                        "paraphrased": None
                    }

    return results


def merge_with_original(subject: str, results: dict, original_dir: Path, merged_dir: Path):
    """元データとマージして新しいファイルを作成"""
    original_file = original_dir / f"{subject}.jsonl"
    merged_file = merged_dir / f"{subject}.jsonl"

    if not original_file.exists():
        print(f"    Warning: Original file not found: {original_file}")
        return

    merged_records = []
    total_tokens = {"input": 0, "output": 0, "total": 0}

    with open(original_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            question_number = record.get("question_number")
            custom_id = f"{subject}_q{question_number}"

            if custom_id in results:
                result = results[custom_id]
                record["paraphrased_analysis"] = result.get("paraphrased")
                if result.get("tokens"):
                    record["paraphrase_tokens"] = result["tokens"]
                    total_tokens["input"] += result["tokens"]["input"]
                    total_tokens["output"] += result["tokens"]["output"]
                    total_tokens["total"] += result["tokens"]["total"]
                if result.get("error"):
                    record["paraphrase_error"] = result["error"]

            merged_records.append(record)

    with open(merged_file, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "records": len(merged_records),
        "tokens": total_tokens
    }


def cmd_download(client: OpenAI, subjects: list[str] | None, merge: bool):
    """結果をダウンロード"""
    status = load_status()

    if not status["batches"]:
        print("No batches found. Run submit first.")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge:
        merged_dir = Path(MERGED_DIR)
        merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading results...\n")

    downloaded = 0
    total_tokens = {"input": 0, "output": 0, "total": 0}

    for batch_info in status["batches"]:
        subject = batch_info["subject"]

        if subjects and subject not in subjects:
            continue

        # 既にダウンロード済みかチェック
        result_file = output_dir / f"{subject}.jsonl"
        if result_file.exists() and not merge:
            print(f"  Skipping {subject} (already downloaded)")
            continue

        try:
            downloaded_file = download_batch_result(client, batch_info, output_dir)

            if downloaded_file:
                downloaded += 1
                batch_info["downloaded"] = True
                batch_info["downloaded_at"] = datetime.now().isoformat()

                # マージオプションが有効な場合
                if merge:
                    print(f"    Merging with original...")
                    results = parse_batch_result(downloaded_file)
                    merge_stats = merge_with_original(
                        subject, results, Path(ORIGINAL_DIR), Path(MERGED_DIR)
                    )
                    if merge_stats:
                        total_tokens["input"] += merge_stats["tokens"]["input"]
                        total_tokens["output"] += merge_stats["tokens"]["output"]
                        total_tokens["total"] += merge_stats["tokens"]["total"]
                        print(f"    Merged {merge_stats['records']} records")

        except Exception as e:
            print(f"  Error downloading {subject}: {e}")

    save_status(status)

    print(f"\n{'='*50}")
    print(f"Downloaded: {downloaded} file(s)")
    print(f"Output directory: {output_dir}")

    if merge:
        print(f"Merged directory: {MERGED_DIR}")
        print(f"\nTotal tokens used:")
        print(f"  Input:  {total_tokens['input']:,}")
        print(f"  Output: {total_tokens['output']:,}")
        print(f"  Total:  {total_tokens['total']:,}")

        # 費用計算 (gpt-4.1: input $2.00, output $8.00 per 1M)
        input_cost = total_tokens["input"] / 1_000_000 * 2.00
        output_cost = total_tokens["output"] / 1_000_000 * 8.00
        # Batch APIは50%割引
        total_cost = (input_cost + output_cost) * 0.5
        print(f"\nEstimated cost (with 50% batch discount): ${total_cost:.2f}")


def cmd_summary(client: OpenAI):
    """ダウンロード済み結果のサマリーを表示"""
    output_dir = Path(OUTPUT_DIR)

    if not output_dir.exists():
        print("No results downloaded yet.")
        return

    result_files = sorted(output_dir.glob("*.jsonl"))
    result_files = [f for f in result_files if not f.name.endswith("_errors.jsonl")]

    if not result_files:
        print("No result files found.")
        return

    print(f"{'Subject':<30} {'Requests':<10} {'Errors':<10}")
    print("=" * 50)

    total_requests = 0
    total_errors = 0

    for result_file in result_files:
        subject = result_file.stem
        requests = 0
        errors = 0

        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                requests += 1
                if data.get("error") or data.get("response", {}).get("status_code") != 200:
                    errors += 1

        total_requests += requests
        total_errors += errors
        print(f"{subject:<30} {requests:<10} {errors:<10}")

    print("=" * 50)
    print(f"{'Total':<30} {total_requests:<10} {total_errors:<10}")


def main():
    parser = argparse.ArgumentParser(description="Download batch results")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to download")
    parser.add_argument("--merge", action="store_true",
                        help="Merge results with original data")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary of downloaded results")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if args.summary:
        cmd_summary(client)
    else:
        cmd_download(client, args.subjects, args.merge)


if __name__ == "__main__":
    main()
