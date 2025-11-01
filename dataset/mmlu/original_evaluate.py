import argparse
import os
import time
import numpy as np
import pandas as pd
from utils.logger import setup_logger, get_logger
from utils.result_saver import ResultSaver

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

def eval(generate, subject, dev_df, test_df, result_saver=None):
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

        try:
            ai_start_time = time.time()
            output = generate(prompt)
            ai_elapsed_time = time.time() - ai_start_time
            ai_times.append(ai_elapsed_time)

            pred = output["final"]
            analysis = output.get("analysis", "")

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

def main(generate, start_from=None, start_index=None, output_dir="results/original/mmlu"):
    # ロガーの初期化
    logger = setup_logger()

    # 結果保存用のインスタンスを作成
    result_saver = ResultSaver()
    logger.info(f"Results will be saved to: {result_saver.get_filepath()}")

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join("dataset/mmlu/data", "test")) if "_test.csv" in f])

    logger.info(f"Found {len(subjects)} subjects to evaluate")

    # 開始位置の決定
    start_idx = 0
    if start_index is not None:
        start_idx = start_index
        logger.info(f"Starting from index {start_idx}")
    elif start_from is not None:
        if start_from in subjects:
            start_idx = subjects.index(start_from)
            logger.info(f"Starting from subject '{start_from}' (index {start_idx})")
        else:
            logger.warning(f"Subject '{start_from}' not found. Starting from beginning.")

    all_cors = []
    all_subject_times = []
    all_ai_times = []

    total_start_time = time.time()

    for i, subject in enumerate(subjects):
        # 開始インデックスより前はスキップ
        if i < start_idx:
            logger.info(f"Skipping {subject} (before start index)")
            continue

        logger.info(f"Loading data for subject: {subject} ({i+1}/{len(subjects)})")
        dev_df = pd.read_csv(os.path.join("dataset/mmlu/data", "dev", subject + "_dev.csv"), header=None)[:5]
        test_df = pd.read_csv(os.path.join("dataset/mmlu/data", "test", subject + "_test.csv"), header=None)

        cors, _, subject_time, ai_times = eval(generate, subject, dev_df, test_df, result_saver)
        all_cors.append(cors)
        all_subject_times.append(subject_time)
        all_ai_times.extend(ai_times)

        # subjectごとの結果をJSONLとして保存
        jsonl_path = result_saver.save_subject_jsonl(subject, output_dir=output_dir)
        logger.info(f"Saved results for {subject} to: {jsonl_path}")
        print(f"Saved results for {subject} to: {jsonl_path}")

    total_elapsed_time = time.time() - total_start_time

    # サマリーの出力
    separator = "="*60
    logger.info("\n" + separator)
    logger.info("SUMMARY")
    logger.info(separator)

    if all_cors:
        weighted_acc = np.mean(np.concatenate(all_cors))
        logger.info("Average accuracy: {:.3f}".format(weighted_acc))
        logger.info("Average time per subject: {:.2f}s".format(np.mean(all_subject_times)))
        logger.info("Average time per AI call: {:.2f}s".format(np.mean(all_ai_times)))
        logger.info("Total AI calls: {}".format(len(all_ai_times)))
    else:
        logger.info("No subjects were processed")

    logger.info("Total execution time: {:.2f}s ({:.2f}min)".format(total_elapsed_time, total_elapsed_time/60))
    logger.info(separator)

    # コンソールにも出力
    print("\n" + separator)
    print("SUMMARY")
    print(separator)

    if all_cors:
        print("Average accuracy: {:.3f}".format(weighted_acc))
        print("Average time per subject: {:.2f}s".format(np.mean(all_subject_times)))
        print("Average time per AI call: {:.2f}s".format(np.mean(all_ai_times)))
        print("Total AI calls: {}".format(len(all_ai_times)))
    else:
        print("No subjects were processed")

    print("Total execution time: {:.2f}s ({:.2f}min)".format(total_elapsed_time, total_elapsed_time/60))
    print(separator)

    # 結果をCSVに保存
    saved_path = result_saver.save()
    logger.info(f"Results saved to: {saved_path}")
    print(f"\nResults saved to: {saved_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="dataset/mmlu/data")
    parser.add_argument("--save_dir", "-s", type=str, default="dataset/mmlu/results")
    args = parser.parse_args()
    main(args)

