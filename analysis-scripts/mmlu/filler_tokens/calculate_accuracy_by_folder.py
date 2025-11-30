#!/usr/bin/env python3
"""
results/filler_tokens内のフォルダごとに正答率を計算するスクリプト
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def calculate_accuracy_for_folder(folder_path):
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
    base_path = Path("results/mmlu/filler_tokens")

    if not base_path.exists():
        print(f"エラー: {base_path} が見つかりません")
        return

    # すべてのサブディレクトリを探索
    all_results = {}

    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        dataset_results = {}

        # データセット内のフォルダ（00, 01, 02...）を処理
        for folder in sorted(dataset_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name
            stats = calculate_accuracy_for_folder(folder)

            if stats["total_questions"] > 0:
                dataset_results[folder_name] = stats

        if dataset_results:
            all_results[dataset_name] = dataset_results

    # 結果を表示
    print("=" * 80)
    print("フォルダごとの正答率サマリー")
    print("=" * 80)
    print()

    for dataset_name, dataset_results in all_results.items():
        print(f"Dataset: {dataset_name}")
        print("-" * 80)

        for folder_name, stats in sorted(dataset_results.items()):
            accuracy = stats["overall_accuracy"]
            total = stats["total_questions"]
            correct = stats["correct_answers"]

            print(f"  フォルダ {folder_name}: {accuracy:.2f}% ({correct}/{total})")

        print()

    # 詳細な結果をJSONファイルに保存
    output_file = "results/mmlu/accuracy_by_folder.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"詳細な結果を {output_file} に保存しました")

    # サマリーCSVも作成
    csv_file = "results/mmlu/accuracy_summary.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("dataset,folder,accuracy,correct,total\n")
        for dataset_name, dataset_results in all_results.items():
            for folder_name, stats in sorted(dataset_results.items()):
                f.write(f"{dataset_name},{folder_name},{stats['overall_accuracy']:.2f},"
                       f"{stats['correct_answers']},{stats['total_questions']}\n")

    print(f"サマリーを {csv_file} に保存しました")


if __name__ == "__main__":
    main()
