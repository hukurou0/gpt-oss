#!/usr/bin/env python3
"""
results/original内の正答率を計算するスクリプト
"""

import json
from pathlib import Path
from collections import defaultdict


def calculate_accuracy(folder_path):
    """
    指定されたフォルダ内のすべてのJSONLファイルを読み込み、正答率を計算する

    Args:
        folder_path: JSONLファイルが格納されているフォルダのパス

    Returns:
        dict: 統計情報を含む辞書
    """
    total_questions = 0
    correct_answers = 0
    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    # フォルダ内のすべてのJSONLファイルを処理
    for jsonl_file in Path(folder_path).glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    subject = data.get("subject", "unknown")
                    is_correct = data.get("is_correct", False)

                    total_questions += 1
                    subject_stats[subject]["total"] += 1

                    if is_correct:
                        correct_answers += 1
                        subject_stats[subject]["correct"] += 1

                except json.JSONDecodeError:
                    continue

    # 正答率を計算
    overall_accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

    # サブジェクトごとの正答率を計算
    for subject in subject_stats:
        stats = subject_stats[subject]
        stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0

    return {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "overall_accuracy": overall_accuracy,
        "subject_stats": dict(subject_stats)
    }


def main():
    original_path = Path("results/mmlu/original")

    if not original_path.exists():
        print(f"エラー: {original_path} が見つかりません")
        return

    print("=" * 80)
    print("Original結果の正答率")
    print("=" * 80)
    print()

    stats = calculate_accuracy(original_path)

    if stats["total_questions"] > 0:
        accuracy = stats["overall_accuracy"]
        total = stats["total_questions"]
        correct = stats["correct_answers"]

        print(f"正答率: {accuracy:.2f}% ({correct}/{total})")
        print()

        # 詳細な結果をJSONファイルに保存
        output_file = "results/mmlu/original_accuracy.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"詳細な結果を {output_file} に保存しました")
    else:
        print("データが見つかりませんでした")


if __name__ == "__main__":
    main()
