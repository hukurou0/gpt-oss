import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

"""
欠損しているタスクだけを再実行して埋めるスクリプト

使用例:
    python run_mmlu_early_answer_missing.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from model.experiment_call import generate_early_answer
from logs.utils.logger import setup_logger, get_logger
from logs.utils.result_saver import ResultSaver

# -------- パラメータ設定 --------
ORIGINAL_RESULTS_DIR = "results/mmlu/original"
OUTPUT_BASE_DIR = "results/mmlu/early_answer"
ANALYSIS_PERCENTAGES = [0.9]
DATA_DIR = "dataset/mmlu/data"

# 欠損リスト: タスク名 -> 欠損しているquestion_numberのリスト
MISSING_RECORDS = {
    "professional_law": [52],
    "college_computer_science": [49],
    #"miscellaneous": [316, 341, 388],
    #"college_chemistry": [18, 35],
    #"college_mathematics": [14, 92],
    #"college_medicine": [74],
   # "high_school_macroeconomics": [216],
    #"high_school_mathematics": [82],
    #"high_school_us_history": [41],
    #"professional_accounting": [7],
}

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    return " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def load_analysis_map(results_file):
    """結果ファイルからanalysisマップを作成"""
    analysis_map = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                key = (result["subject"], result["question_number"])
                analysis_map[key] = result["analysis"]
    return analysis_map


def process_missing_questions(
    subject: str,
    missing_questions: list,
    analysis_map: dict,
    generate_func,
    dev_df,
    test_df,
    output_file: str,
    logger,
):
    """
    欠損しているquestion_numberだけを処理する

    Args:
        subject: サブジェクト名
        missing_questions: 欠損しているquestion_numberのリスト
        analysis_map: (subject, question_number) -> analysis のマップ
        generate_func: generate関数
        dev_df: 開発データ
        test_df: テストデータ
        output_file: 出力ファイルパス
        logger: ロガー

    Returns:
        tuple: (cors, ai_times, processed_count)
    """
    cors = []
    ai_times = []
    processed_count = 0

    train_prompt = gen_prompt(dev_df, subject, k=5)

    for question_number in missing_questions:
        # question_numberは1-indexed、DataFrameは0-indexed
        i = question_number - 1

        if i >= test_df.shape[0]:
            logger.warning(f"Question {question_number} out of range for {subject}")
            continue

        # analysisを取得
        key = (subject, question_number)
        if key not in analysis_map:
            logger.warning(f"No analysis found for {subject} Q{question_number}, skipping")
            continue

        original_analysis = analysis_map[key]

        # プロンプトを作成
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = train_prompt + prompt_end

        # 質問文と選択肢を取得
        question_text = test_df.iloc[i, 0]
        num_choices = test_df.shape[1] - 2
        choice_texts = [test_df.iloc[i, j + 1] for j in range(num_choices)]
        label = test_df.iloc[i, test_df.shape[1] - 1]

        try:
            ai_start_time = time.time()
            pred = generate_func(prompt, original_analysis)
            ai_elapsed_time = time.time() - ai_start_time
            ai_times.append(ai_elapsed_time)

            cor = pred == label
            cors.append(cor)

            result_str = f"pred: {pred}, label: {label} (AI time: {ai_elapsed_time:.2f}s)"
            logger.info(f"{subject} Q{question_number}: {result_str}")
            print(result_str)

            # 結果をファイルに追記
            result_record = {
                "subject": subject,
                "question_number": question_number,
                "question": question_text,
                "choices": choice_texts,
                "analysis": "tmp",  # early_answerでは使用しない
                "predicted_answer": pred,
                "correct_answer": label,
                "is_correct": cor,
                "ai_time": ai_elapsed_time,
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {subject} Q{question_number}: {str(e)}")
            continue

    return cors, ai_times, processed_count


if __name__ == "__main__":
    logger = setup_logger()

    logger.info("=" * 80)
    logger.info("RUNNING MISSING RECORDS RECOVERY")
    logger.info("=" * 80)
    logger.info(f"Original results directory: {ORIGINAL_RESULTS_DIR}")
    logger.info(f"Output base directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Missing tasks: {list(MISSING_RECORDS.keys())}")
    logger.info(f"Total missing records per directory: {sum(len(v) for v in MISSING_RECORDS.values())}")

    grand_total_start_time = time.time()
    total_processed = 0

    # analysis_percentageごとに処理
    for analysis_percentage in ANALYSIS_PERCENTAGES:
        percentage_int = int(analysis_percentage * 10)
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{percentage_int:02d}")

        logger.info(f"\n{'#'*80}")
        logger.info(f"Processing analysis_percentage={analysis_percentage:.1f} ({percentage_int:02d})")
        logger.info(f"{'#'*80}")

        # generate関数をラップ
        def generate_wrapper(prompt: str, analysis: str, pct=analysis_percentage):
            return generate_early_answer(prompt, analysis, pct)

        percentage_processed = 0
        percentage_cors = []
        percentage_ai_times = []

        # 各欠損タスクを処理
        for subject, missing_questions in MISSING_RECORDS.items():
            logger.info(f"\n--- {subject}: {len(missing_questions)} missing records ---")

            # originalからanalysisを読み込む
            original_file = os.path.join(ORIGINAL_RESULTS_DIR, f"{subject}.jsonl")
            if not os.path.exists(original_file):
                logger.error(f"Original file not found: {original_file}")
                continue

            analysis_map = load_analysis_map(original_file)

            # データをロード
            dev_df = pd.read_csv(
                os.path.join(DATA_DIR, "dev", f"{subject}_dev.csv"), header=None
            )[:5]
            test_df = pd.read_csv(
                os.path.join(DATA_DIR, "test", f"{subject}_test.csv"), header=None
            )

            # 出力ファイル
            output_file = os.path.join(output_dir, f"{subject}.jsonl")

            cors, ai_times, processed = process_missing_questions(
                subject,
                missing_questions,
                analysis_map,
                generate_wrapper,
                dev_df,
                test_df,
                output_file,
                logger,
            )

            percentage_cors.extend(cors)
            percentage_ai_times.extend(ai_times)
            percentage_processed += processed

        logger.info(f"\nPercentage {percentage_int:02d} completed: {percentage_processed} records processed")
        if percentage_cors:
            logger.info(f"Accuracy for recovered records: {np.mean(percentage_cors):.3f}")
        total_processed += percentage_processed

    grand_total_elapsed_time = time.time() - grand_total_start_time

    # サマリー
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info("MISSING RECORDS RECOVERY COMPLETE")
    logger.info(separator)
    logger.info(f"Total records processed: {total_processed}")
    logger.info(f"Total execution time: {grand_total_elapsed_time:.2f}s ({grand_total_elapsed_time/60:.2f}min)")
    logger.info(separator)

    print(f"\n{separator}")
    print("MISSING RECORDS RECOVERY COMPLETE")
    print(separator)
    print(f"Total records processed: {total_processed}")
    print(f"Total execution time: {grand_total_elapsed_time:.2f}s ({grand_total_elapsed_time/60:.2f}min)")
    print(separator)
