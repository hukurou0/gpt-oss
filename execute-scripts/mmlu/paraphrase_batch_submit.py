"""
Batch APIにリクエストをアップロード・実行するスクリプト
paraphrase_batch_prepare.py で生成したファイルを処理

使用例:
    # 全科目を一括でバッチ実行
    python paraphrase_batch_submit.py

    # 特定の科目のみ実行
    python paraphrase_batch_submit.py --subjects abstract_algebra anatomy

    # バッチ状態を確認
    python paraphrase_batch_submit.py --status

    # failedバッチを再送信
    python paraphrase_batch_submit.py --retry

環境変数:
    OPENAI_API_KEY: OpenAI APIキー
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# -------- 設定 --------
BATCH_DIR = "results/mmlu/paraphrase_batch"
STATUS_FILE = "results/mmlu/paraphrase_batch/_batch_status.json"


def upload_and_create_batch(client: OpenAI, file_path: Path) -> dict:
    """ファイルをアップロードしてバッチを作成"""
    subject = file_path.stem

    # 1. ファイルをアップロード
    print(f"  Uploading {file_path.name}...")
    with open(file_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")

    print(f"    File ID: {uploaded_file.id}")

    # 2. バッチを作成
    print(f"  Creating batch...")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"subject": subject}
    )

    print(f"    Batch ID: {batch.id}")

    return {
        "subject": subject,
        "file_name": file_path.name,
        "file_id": uploaded_file.id,
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": datetime.now().isoformat()
    }


def check_batch_status(client: OpenAI, batch_id: str) -> dict:
    """バッチの状態を確認"""
    batch = client.batches.retrieve(batch_id)
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id
    }


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


def cmd_submit(client: OpenAI, subjects: list[str] | None):
    """バッチを送信"""
    batch_dir = Path(BATCH_DIR)

    if not batch_dir.exists():
        print(f"Error: Batch directory not found: {BATCH_DIR}")
        print("Run paraphrase_batch_prepare.py first.")
        sys.exit(1)

    # 処理するファイルを取得
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))

    if subjects:
        jsonl_files = [f for f in jsonl_files if f.stem in subjects]

    if not jsonl_files:
        print("No files to process.")
        return

    print(f"Submitting {len(jsonl_files)} batch(es)...\n")

    status = load_status()
    existing_subjects = {b["subject"] for b in status["batches"]}

    for file_path in jsonl_files:
        subject = file_path.stem

        # 既に送信済みかチェック
        if subject in existing_subjects:
            print(f"Skipping {subject} (already submitted)")
            continue

        print(f"Processing: {subject}")
        try:
            result = upload_and_create_batch(client, file_path)
            status["batches"].append(result)
            save_status(status)
            print(f"  Done!\n")
        except Exception as e:
            print(f"  Error: {e}\n")

        # レート制限対策で少し待つ
        time.sleep(1)

    print(f"\nStatus saved to: {STATUS_FILE}")


def cmd_status(client: OpenAI):
    """全バッチの状態を確認"""
    status = load_status()

    if not status["batches"]:
        print("No batches found. Run submit first.")
        return

    print(f"Checking {len(status['batches'])} batch(es)...\n")
    print(f"{'Subject':<30} {'Status':<15} {'Progress':<15}")
    print("=" * 60)

    updated_batches = []
    for batch_info in status["batches"]:
        try:
            current = check_batch_status(client, batch_info["batch_id"])
            batch_info["status"] = current["status"]
            batch_info["request_counts"] = current["request_counts"]
            batch_info["output_file_id"] = current["output_file_id"]
            batch_info["error_file_id"] = current["error_file_id"]

            counts = current["request_counts"]
            progress = f"{counts['completed']}/{counts['total']}"
            print(f"{batch_info['subject']:<30} {current['status']:<15} {progress:<15}")
        except Exception as e:
            print(f"{batch_info['subject']:<30} {'error':<15} {str(e)[:15]}")

        updated_batches.append(batch_info)

    status["batches"] = updated_batches
    status["last_checked"] = datetime.now().isoformat()
    save_status(status)

    # サマリー
    completed = sum(1 for b in status["batches"] if b["status"] == "completed")
    in_progress = sum(1 for b in status["batches"] if b["status"] == "in_progress")
    failed = sum(1 for b in status["batches"] if b["status"] == "failed")

    print(f"\nSummary: {completed} completed, {in_progress} in progress, {failed} failed")


def cmd_cancel(client: OpenAI, subjects: list[str] | None):
    """バッチをキャンセル"""
    status = load_status()

    if not status["batches"]:
        print("No batches found.")
        return

    for batch_info in status["batches"]:
        if subjects and batch_info["subject"] not in subjects:
            continue

        if batch_info["status"] in ["completed", "failed", "cancelled"]:
            print(f"Skipping {batch_info['subject']} (status: {batch_info['status']})")
            continue

        try:
            client.batches.cancel(batch_info["batch_id"])
            print(f"Cancelled: {batch_info['subject']}")
            batch_info["status"] = "cancelled"
        except Exception as e:
            print(f"Error cancelling {batch_info['subject']}: {e}")

    save_status(status)


def cmd_retry(client: OpenAI, subjects: list[str] | None):
    """failedバッチを再送信"""
    batch_dir = Path(BATCH_DIR)
    status = load_status()

    if not status["batches"]:
        print("No batches found.")
        return

    # failedバッチを抽出
    failed_batches = [b for b in status["batches"] if b["status"] == "failed"]

    if subjects:
        failed_batches = [b for b in failed_batches if b["subject"] in subjects]

    if not failed_batches:
        print("No failed batches to retry.")
        return

    print(f"Retrying {len(failed_batches)} failed batch(es)...\n")

    # 既存のバッチリストからfailedを除外
    status["batches"] = [b for b in status["batches"]
                         if b["status"] != "failed" or
                         (subjects and b["subject"] not in subjects) or
                         (not subjects and False)]

    # failedを除外した状態を保存
    remaining_failed = {b["subject"] for b in status["batches"] if b["status"] == "failed"}
    status["batches"] = [b for b in status["batches"]
                         if b["subject"] not in {fb["subject"] for fb in failed_batches}]

    for batch_info in failed_batches:
        subject = batch_info["subject"]
        file_path = batch_dir / f"{subject}.jsonl"

        if not file_path.exists():
            print(f"Skipping {subject} (file not found)")
            continue

        print(f"Retrying: {subject}")
        try:
            result = upload_and_create_batch(client, file_path)
            status["batches"].append(result)
            save_status(status)
            print(f"  Done!\n")
        except Exception as e:
            print(f"  Error: {e}\n")

        # レート制限対策で少し待つ
        time.sleep(2)

    print(f"\nStatus saved to: {STATUS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Batch API submit and status checker")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--cancel", action="store_true", help="Cancel batches")
    parser.add_argument("--retry", action="store_true", help="Retry failed batches")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to process")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if args.status:
        cmd_status(client)
    elif args.cancel:
        cmd_cancel(client, args.subjects)
    elif args.retry:
        cmd_retry(client, args.subjects)
    else:
        cmd_submit(client, args.subjects)


if __name__ == "__main__":
    main()
