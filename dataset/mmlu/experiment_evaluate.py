import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
from logs.utils.logger import setup_logger, get_logger
from logs.utils.result_saver import ResultSaver

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

def eval(generate, subject, dev_df, test_df, result_saver=None, analysis_map=None):
    logger = get_logger()
    cors = []
    ai_times = []

    subject_start_time = time.time()
    logger.info(f"Starting evaluation for subject: {subject}")

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # 質問文と選択肢を取得
        question_text = test_df.iloc[i, 0]
        num_choices = test_df.shape[1] - 2
        choice_texts = [test_df.iloc[i, j+1] for j in range(num_choices)]
        label = test_df.iloc[i, test_df.shape[1]-1]

        # 元のanalysisを取得
        key = (subject, i+1)
        original_analysis = None
        if analysis_map and key in analysis_map:
            original_analysis = analysis_map[key]
            logger.info(f"Using original analysis for {subject} Q{i+1}")

        try:
            ai_start_time = time.time()
            # generate関数にpromptとanalysisを渡す
            if original_analysis:
                output = generate(prompt, original_analysis)
            else:
                logger.warning(f"No original analysis found for {subject} Q{i+1}, skipping")
                continue

            ai_elapsed_time = time.time() - ai_start_time
            ai_times.append(ai_elapsed_time)

            pred = output
            analysis = "tmp"

            result_str = "pred: {}, label: {} (AI time: {:.2f}s)".format(pred, label, ai_elapsed_time)
            logger.info(f"Question {i+1}/{test_df.shape[0]}: {result_str}")
            print(result_str)

            # CSV保存用に結果を記録
            if result_saver is not None:
                cor = pred == label
                result_saver.add_result(
                    subject=subject,
                    question_number=i+1,
                    question=question_text,
                    choices=choice_texts,
                    analysis=analysis,
                    predicted_answer=pred,
                    correct_answer=label,
                    is_correct=cor,
                    ai_time=ai_elapsed_time
                )
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {str(e)}")
            logger.error(f"Prompt: {prompt}")
            raise Exception("Error: " + prompt)

        cor = pred == label
        cors.append(cor)

    subject_elapsed_time = time.time() - subject_start_time
    acc = np.mean(cors)
    cors = np.array(cors)
    avg_ai_time = np.mean(ai_times) if ai_times else 0

    summary = "Average accuracy {:.3f} - {}".format(acc, subject)
    time_summary = "Subject total time: {:.2f}s, Average AI time: {:.2f}s, Total questions: {}".format(
        subject_elapsed_time, avg_ai_time, len(ai_times))

    logger.info(summary)
    logger.info(time_summary)
    print(summary)
    print(time_summary)

    return cors, acc, subject_elapsed_time, ai_times

def load_analysis_map(results_file):
    """
    結果ファイルからanalysisマップを作成

    Args:
        results_file: JSON Linesファイルのパス

    Returns:
        dict: (subject, question_number) -> analysis のマップ
    """
    analysis_map = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                key = (result['subject'], result['question_number'])
                analysis_map[key] = result['analysis']
    return analysis_map


def main(generate, results_file=None):
    """
    early_answer評価のメイン関数

    Args:
        generate: generate関数（prompt, analysisを受け取る）
        results_file: 元の結果ファイルのパス（JSON Lines形式）
    """
    # ロガーの初期化
    logger = setup_logger()

    # 結果ファイルが指定されている場合、analysisマップをロード
    analysis_map = None
    if results_file:
        logger.info(f"Loading analysis from: {results_file}")
        analysis_map = load_analysis_map(results_file)
        logger.info(f"Loaded {len(analysis_map)} analyses")
    else:
        logger.error("results_file is required for early_answer evaluation")
        raise ValueError("results_file must be provided")

    # 結果保存用のインスタンスを作成
    result_saver = ResultSaver(filename=f"mmlu_early_answer_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    logger.info(f"Results will be saved to: {result_saver.get_filepath()}")

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join("dataset/mmlu/data", "test")) if "_test.csv" in f])

    logger.info(f"Found {len(subjects)} subjects to evaluate")

    all_cors = []
    all_subject_times = []
    all_ai_times = []

    total_start_time = time.time()

    n = 0

    for subject in subjects:
        logger.info(f"Loading data for subject: {subject}")
        dev_df = pd.read_csv(os.path.join("dataset/mmlu/data", "dev", subject + "_dev.csv"), header=None)[:5]
        test_df = pd.read_csv(os.path.join("dataset/mmlu/data", "test", subject + "_test.csv"), header=None)

        cors, _, subject_time, ai_times = eval(generate, subject, dev_df, test_df, result_saver, analysis_map)
        all_cors.append(cors)
        all_subject_times.append(subject_time)
        all_ai_times.extend(ai_times)
        n += 1
        if n >= 1:
            break

    total_elapsed_time = time.time() - total_start_time

    weighted_acc = np.mean(np.concatenate(all_cors))

    # サマリーの出力
    separator = "="*60
    logger.info("\n" + separator)
    logger.info("EARLY ANSWER SUMMARY")
    logger.info(separator)
    logger.info("Average accuracy: {:.3f}".format(weighted_acc))
    logger.info("Total execution time: {:.2f}s ({:.2f}min)".format(total_elapsed_time, total_elapsed_time/60))
    logger.info("Average time per subject: {:.2f}s".format(np.mean(all_subject_times)))
    logger.info("Average time per AI call: {:.2f}s".format(np.mean(all_ai_times)))
    logger.info("Total AI calls: {}".format(len(all_ai_times)))
    logger.info(separator)

    # コンソールにも出力
    print("\n" + separator)
    print("EARLY ANSWER SUMMARY")
    print(separator)
    print("Average accuracy: {:.3f}".format(weighted_acc))
    print("Total execution time: {:.2f}s ({:.2f}min)".format(total_elapsed_time, total_elapsed_time/60))
    print("Average time per subject: {:.2f}s".format(np.mean(all_subject_times)))
    print("Average time per AI call: {:.2f}s".format(np.mean(all_ai_times)))
    print("Total AI calls: {}".format(len(all_ai_times)))
    print(separator)

    logger.info(f"Results saved to: {result_saver.get_filepath()}")
    print(f"\nResults saved to: {result_saver.get_filepath()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="dataset/mmlu/data")
    parser.add_argument("--save_dir", "-s", type=str, default="dataset/mmlu/results")
    args = parser.parse_args()
    main(args)

