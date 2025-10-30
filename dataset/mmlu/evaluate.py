import argparse
import os
import time
import numpy as np
import pandas as pd

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

def eval(generate, subject, dev_df, test_df):
    cors = []
    ai_times = []

    subject_start_time = time.time()

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        try:
            ai_start_time = time.time()
            output = generate(prompt)
            ai_elapsed_time = time.time() - ai_start_time
            ai_times.append(ai_elapsed_time)

            pred = output["final"]
            print("pred: {}, label: {} (AI time: {:.2f}s)".format(pred, label, ai_elapsed_time))
        except:
            raise Exception("Error: " + prompt)

        cor = pred == label
        cors.append(cor)

    subject_elapsed_time = time.time() - subject_start_time
    acc = np.mean(cors)
    cors = np.array(cors)
    avg_ai_time = np.mean(ai_times) if ai_times else 0

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("Subject total time: {:.2f}s, Average AI time: {:.2f}s, Total questions: {}".format(
        subject_elapsed_time, avg_ai_time, len(ai_times)))

    return cors, acc, subject_elapsed_time, ai_times

def main(generate):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join("dataset/mmlu/data", "test")) if "_test.csv" in f])

    all_cors = []
    all_subject_times = []
    all_ai_times = []

    total_start_time = time.time()

    n = 0

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join("dataset/mmlu/data", "dev", subject + "_dev.csv"), header=None)[:5]
        test_df = pd.read_csv(os.path.join("dataset/mmlu/data", "test", subject + "_test.csv"), header=None)

        cors, _, subject_time, ai_times = eval(generate, subject, dev_df, test_df)
        all_cors.append(cors)
        all_subject_times.append(subject_time)
        all_ai_times.extend(ai_times)
        n += 1
        if n >= 1:
            break

    total_elapsed_time = time.time() - total_start_time

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Average accuracy: {:.3f}".format(weighted_acc))
    print("Total execution time: {:.2f}s ({:.2f}min)".format(total_elapsed_time, total_elapsed_time/60))
    print("Average time per subject: {:.2f}s".format(np.mean(all_subject_times)))
    print("Average time per AI call: {:.2f}s".format(np.mean(all_ai_times)))
    print("Total AI calls: {}".format(len(all_ai_times)))
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="dataset/mmlu/data")
    parser.add_argument("--save_dir", "-s", type=str, default="dataset/mmlu/results")
    args = parser.parse_args()
    main(args)

