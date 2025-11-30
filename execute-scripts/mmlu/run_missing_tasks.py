import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
import time
from model.call_gpt_oss import generate
from logs.utils.logger import setup_logger, get_logger

# ========== 設定 ==========
# 結果を保存するディレクトリ
OUTPUT_DIR = "results/mmlu/original"

# 欠損タスクの定義（第1回実行: 27問中18問成功）
# MISSING_TASKS = {
#     "professional_law": [52, 246, 284, 433, 460, 482, 596, 886, 919, 1024, 1122, 1159, 1210, 1293, 1515],
#     "college_computer_science": [6, 49, 78],
#     "miscellaneous": [316, 341, 388],
#     "college_chemistry": [18, 35],
#     "college_mathematics": [14, 92],
#     "college_medicine": [74],
#     "high_school_macroeconomics": [216],
#     "high_school_mathematics": [82],
#     "high_school_us_history": [41],
#     "professional_accounting": [7],
# }

# 残り9問（第1回実行で失敗）
""" MISSING_TASKS = {
    "professional_law": [52, 482, 919, 1122, 1210, 1293],
    "college_computer_science": [49],
    "college_chemistry": [35],
    "college_mathematics": [92],
} """
# ==========================

MISSING_TASKS = {
    "professional_law": [52, 1122, 1293],
    "college_computer_science": [49],
}

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval_single_question(subject, dev_df, test_df, question_number):
    """
    単一の問題を評価して結果を返す

    Args:
        subject: サブジェクト名
        dev_df: 開発データ
        test_df: テストデータ
        question_number: 問題番号（1ベース）

    Returns:
        結果の辞書、またはエラー時はNone
    """
    logger = get_logger()
    i = question_number - 1  # 0ベースのインデックス

    # プロンプトを生成
    k = 5
    prompt_end = format_example(test_df, i, include_answer=False)
    train_prompt = gen_prompt(dev_df, subject, k)
    prompt = train_prompt + prompt_end

    label = test_df.iloc[i, test_df.shape[1]-1]

    try:
        ai_start_time = time.time()
        output = generate(prompt)
        ai_elapsed_time = time.time() - ai_start_time

        pred = output["final"]
        analysis = output.get("analysis", "")

        result_str = f"pred: {pred}, label: {label} (AI time: {ai_elapsed_time:.2f}s)"
        logger.info(f"{subject} Q{question_number}: {result_str}")
        print(f"{subject} Q{question_number}: {result_str}")

        result = {
            "subject": subject,
            "question_number": question_number,
            "analysis": analysis,
            "predicted_answer": pred,
            "correct_answer": label,
            "is_correct": pred == label,
            "ai_time_seconds": round(ai_elapsed_time, 2)
        }

        return result

    except Exception as e:
        logger.error(f"Error processing {subject} Q{question_number}: {str(e)}")
        print(f"Error processing {subject} Q{question_number}: {str(e)}")
        return None

def load_existing_results(filepath):
    """既存のJSONLファイルを読み込む"""
    results = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results

def save_results(filepath, results):
    """結果をJSONLファイルに保存（question_numberでソート）"""
    # question_numberでソート
    sorted_results = sorted(results, key=lambda x: x["question_number"])

    with open(filepath, 'w', encoding='utf-8') as f:
        for result in sorted_results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

def main():
    logger = setup_logger()
    logger.info("Starting missing tasks execution")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_missing = sum(len(questions) for questions in MISSING_TASKS.values())
    logger.info(f"Total missing tasks: {total_missing}")
    print(f"Total missing tasks: {total_missing}")

    completed = 0
    failed = 0

    for subject, question_numbers in MISSING_TASKS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subject: {subject} ({len(question_numbers)} missing)")
        print(f"\n{'='*60}")
        print(f"Processing subject: {subject} ({len(question_numbers)} missing)")

        # データを読み込み
        dev_df = pd.read_csv(os.path.join("dataset/mmlu/data", "dev", subject + "_dev.csv"), header=None)[:5]
        test_df = pd.read_csv(os.path.join("dataset/mmlu/data", "test", subject + "_test.csv"), header=None)

        # 既存の結果を読み込み
        jsonl_path = os.path.join(OUTPUT_DIR, f"{subject}.jsonl")
        existing_results = load_existing_results(jsonl_path)
        existing_questions = {r["question_number"] for r in existing_results}

        logger.info(f"Existing results: {len(existing_results)} questions")

        new_results = []

        for question_number in question_numbers:
            # 既存の結果に含まれている場合はスキップ（predicted_answerが空でない場合）
            existing = next((r for r in existing_results if r["question_number"] == question_number), None)
            if existing and existing.get("predicted_answer", ""):
                logger.info(f"Skipping Q{question_number} (already has valid answer)")
                print(f"Skipping Q{question_number} (already has valid answer)")
                continue

            # 問題を実行
            result = eval_single_question(subject, dev_df, test_df, question_number)

            if result and result["predicted_answer"]:
                new_results.append(result)
                completed += 1
            else:
                failed += 1
                logger.warning(f"Failed to get valid answer for {subject} Q{question_number}")

        if new_results:
            # 既存の結果から該当question_numberを削除
            filtered_existing = [r for r in existing_results
                               if r["question_number"] not in [nr["question_number"] for nr in new_results]]

            # 新しい結果をマージ
            merged_results = filtered_existing + new_results

            # 保存
            save_results(jsonl_path, merged_results)
            logger.info(f"Updated {jsonl_path} with {len(new_results)} new results")
            print(f"Updated {jsonl_path} with {len(new_results)} new results")

    # サマリー
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total attempted: {completed + failed}")

    logger.info(f"\nCompleted: {completed}, Failed: {failed}")

if __name__ == "__main__":
    main()
