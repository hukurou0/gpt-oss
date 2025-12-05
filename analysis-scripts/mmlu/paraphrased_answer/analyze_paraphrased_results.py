#!/usr/bin/env python3
"""
Paraphrased Answer Analysis Script

このスクリプトは、MMLUベンチマークにおける選択肢言い換え実験の結果を分析する。

分析内容:
1. 言い換え失敗の判定（too_short基準: 言い換え後のanalysisが元の10%未満）
2. 正答率の比較（全データ、正常な言い換えのみ、言い換え失敗のみ）
3. 回答変化パターンの分析（both_correct, both_wrong, wrong_to_correct, correct_to_wrong）
4. Analysis内の結論変化の検出

使用方法:
    python analyze_paraphrased_results.py
"""

import json
import os
import re
from collections import defaultdict


# Directories (relative to project root)
ORIGINAL_DIR = "results/mmlu/original"
PARAPHRASED_DIR = "results/mmlu/paraphrased_answer"
PARAPHRASE_BATCH_DIR = "results/mmlu/paraphrase_batch/results"

# Failure detection threshold
TOO_SHORT_THRESHOLD = 0.10  # less than 10% of original length


def load_jsonl(filepath):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_all_data(original_dir, paraphrased_dir, paraphrase_batch_dir):
    """Load all data from the three directories."""
    original_data = {}
    paraphrased_data = {}
    paraphrase_batch_data = {}

    # Load original data
    for filename in os.listdir(original_dir):
        if filename.endswith('.jsonl'):
            subject = filename.replace('.jsonl', '')
            for item in load_jsonl(os.path.join(original_dir, filename)):
                key = (subject, item['question_number'])
                original_data[key] = item

    # Load paraphrased answer data
    for filename in os.listdir(paraphrased_dir):
        if filename.endswith('.jsonl'):
            subject = filename.replace('.jsonl', '')
            for item in load_jsonl(os.path.join(paraphrased_dir, filename)):
                key = (subject, item['question_number'])
                paraphrased_data[key] = item

    # Load paraphrase batch results (different format - contains the actual paraphrased analysis)
    for filename in os.listdir(paraphrase_batch_dir):
        if filename.endswith('.jsonl'):
            subject = filename.replace('.jsonl', '')
            for item in load_jsonl(os.path.join(paraphrase_batch_dir, filename)):
                # Extract question_number from custom_id like "anatomy_q1"
                custom_id = item.get('custom_id', '')
                if '_q' in custom_id:
                    q_num = int(custom_id.split('_q')[1])
                    # Extract paraphrased analysis from nested structure
                    para_analysis = (item.get('response', {})
                                    .get('body', {})
                                    .get('choices', [{}])[0]
                                    .get('message', {})
                                    .get('content', ''))
                    key = (subject, q_num)
                    paraphrase_batch_data[key] = {'paraphrased_analysis': para_analysis}

    return original_data, paraphrased_data, paraphrase_batch_data


def detect_failures(original_data, paraphrased_data, paraphrase_batch_data):
    """Detect paraphrase failures using too_short criterion."""
    failures = {'too_short': []}
    valid_keys = []

    for key in original_data:
        if key not in paraphrased_data:
            continue
        if key not in paraphrase_batch_data:
            continue

        orig = original_data[key]
        orig_analysis = orig.get('analysis', '')
        para_analysis = paraphrase_batch_data[key].get('paraphrased_analysis', '')

        orig_len = len(orig_analysis)
        para_len = len(para_analysis)

        is_failure = False

        # too_short check
        if orig_len > 0 and para_len < orig_len * TOO_SHORT_THRESHOLD:
            failures['too_short'].append({
                'subject': key[0],
                'question_number': key[1],
                'orig_len': orig_len,
                'para_len': para_len,
                'ratio': para_len / orig_len if orig_len > 0 else 0
            })
            is_failure = True

        if not is_failure:
            valid_keys.append(key)

    # Create set of failed keys
    failed_keys = set()
    for failure_list in failures.values():
        for item in failure_list:
            failed_keys.add((item['subject'], item['question_number']))

    return failures, valid_keys, failed_keys


def calculate_accuracy(keys, original_data, paraphrased_data):
    """Calculate accuracy for a set of keys."""
    if not keys:
        return 0, 0, 0
    orig_correct = sum(1 for k in keys if original_data[k].get('is_correct', False))
    para_correct = sum(1 for k in keys if paraphrased_data[k].get('is_correct', False))
    return orig_correct, para_correct, len(keys)


def analyze_changes(keys, original_data, paraphrased_data):
    """Analyze answer change patterns for a set of keys."""
    patterns = {
        'both_correct': [],
        'both_wrong': [],
        'wrong_to_correct': [],
        'correct_to_wrong': []
    }

    for key in keys:
        orig = original_data[key]
        para = paraphrased_data[key]
        orig_correct = orig.get('is_correct', False)
        para_correct = para.get('is_correct', False)

        item = {
            'subject': key[0],
            'question_number': key[1],
            'orig_answer': orig.get('predicted_answer', ''),
            'para_answer': para.get('predicted_answer', ''),
            'correct_answer': orig.get('correct_answer', '')
        }

        if orig_correct and para_correct:
            patterns['both_correct'].append(item)
        elif not orig_correct and not para_correct:
            patterns['both_wrong'].append(item)
        elif not orig_correct and para_correct:
            patterns['wrong_to_correct'].append(item)
        else:
            patterns['correct_to_wrong'].append(item)

    return patterns


def extract_conclusion(analysis):
    """Extract conclusion from analysis text."""
    if not analysis:
        return None

    patterns = [
        r'(?:final\s+)?answer[:\s]+["\']?([A-D])["\']?\.?\s*$',
        r'(?:the\s+)?answer\s+is[:\s]+["\']?([A-D])["\']?',
        r'correct\s+answer[:\s]+["\']?([A-D])["\']?',
        r'output[:\s]+["\']?([A-D])["\']?\.?\s*$',
        r'\*\*([A-D])\*\*\.?\s*$',
    ]

    # Check last 500 chars
    tail = analysis[-500:] if len(analysis) > 500 else analysis

    for pattern in patterns:
        match = re.search(pattern, tail, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def detect_conclusion_changes(valid_keys, original_data, paraphrased_data, paraphrase_batch_data):
    """Detect cases where the analysis conclusion changed after paraphrasing."""
    conclusion_changes = []

    for key in valid_keys:
        orig = original_data[key]
        para = paraphrased_data[key]

        orig_analysis = orig.get('analysis', '')
        para_analysis = paraphrase_batch_data[key].get('paraphrased_analysis', '')

        orig_conclusion = extract_conclusion(orig_analysis)
        para_conclusion = extract_conclusion(para_analysis)

        if orig_conclusion and para_conclusion and orig_conclusion != para_conclusion:
            # Validate by checking if conclusions match predicted answers
            orig_pred = orig.get('predicted_answer', '')
            para_pred = para.get('predicted_answer', '')
            correct = orig.get('correct_answer', '')

            if orig_conclusion == orig_pred and para_conclusion == para_pred:
                conclusion_changes.append({
                    'subject': key[0],
                    'question_number': key[1],
                    'orig_conclusion': orig_conclusion,
                    'para_conclusion': para_conclusion,
                    'correct': correct,
                    'orig_correct': orig_conclusion == correct,
                    'para_correct': para_conclusion == correct
                })

    return conclusion_changes


def print_report(original_data, paraphrased_data, paraphrase_batch_data,
                 failures, valid_keys, failed_keys, all_keys,
                 all_patterns, valid_patterns, failed_patterns, conclusion_changes):
    """Print the analysis report."""

    print("=" * 60)
    print("言い換え失敗の判定結果")
    print("=" * 60)
    print(f"\n判定基準: too_short (言い換え後のanalysisが元の10%未満)")
    print(f"\ntoo_short: {len(failures['too_short'])}件")
    print(f"総失敗件数: {len(failed_keys)}件")
    print(f"正常な言い換え: {len(valid_keys)}件")
    print(f"総問題数: {len(original_data)}件")
    print(f"失敗率: {len(failed_keys)/len(original_data)*100:.2f}%")

    # Accuracy comparison
    orig_all, para_all, total_all = calculate_accuracy(all_keys, original_data, paraphrased_data)
    print(f"\n{'='*60}")
    print("正答率の比較")
    print("=" * 60)
    print(f"\n全データ ({total_all}問):")
    print(f"  Original: {orig_all}/{total_all} = {orig_all/total_all*100:.2f}%")
    print(f"  Paraphrased: {para_all}/{total_all} = {para_all/total_all*100:.2f}%")
    print(f"  差: {(para_all-orig_all)/total_all*100:+.2f}pt")

    orig_valid, para_valid, total_valid = calculate_accuracy(valid_keys, original_data, paraphrased_data)
    print(f"\n正常な言い換えのみ ({total_valid}問):")
    print(f"  Original: {orig_valid}/{total_valid} = {orig_valid/total_valid*100:.2f}%")
    print(f"  Paraphrased: {para_valid}/{total_valid} = {para_valid/total_valid*100:.2f}%")
    print(f"  差: {(para_valid-orig_valid)/total_valid*100:+.2f}pt")

    failed_key_list = list(failed_keys)
    orig_failed, para_failed, total_failed = calculate_accuracy(failed_key_list, original_data, paraphrased_data)
    print(f"\n言い換え失敗のみ ({total_failed}問):")
    print(f"  Original: {orig_failed}/{total_failed} = {orig_failed/total_failed*100:.2f}%")
    print(f"  Paraphrased: {para_failed}/{total_failed} = {para_failed/total_failed*100:.2f}%")
    print(f"  差: {(para_failed-orig_failed)/total_failed*100:+.2f}pt")

    # Answer change patterns
    print(f"\n{'='*60}")
    print("回答変化パターン")
    print("=" * 60)
    print(f"\n全データ ({len(all_keys)}問):")
    for pattern, items in all_patterns.items():
        print(f"  {pattern}: {len(items)}件 ({len(items)/len(all_keys)*100:.2f}%)")

    print(f"\n正常な言い換えのみ ({len(valid_keys)}問):")
    for pattern, items in valid_patterns.items():
        print(f"  {pattern}: {len(items)}件 ({len(items)/len(valid_keys)*100:.2f}%)")

    print(f"\n言い換え失敗のみ ({len(failed_key_list)}問):")
    for pattern, items in failed_patterns.items():
        print(f"  {pattern}: {len(items)}件 ({len(items)/len(failed_key_list)*100:.2f}%)")

    # Detailed changes for valid paraphrases
    print(f"\n{'='*60}")
    print("正常な言い換えでの回答変化の詳細")
    print("=" * 60)

    print("\nwrong_to_correct:")
    print("| Subject | Q# | Original | Paraphrased | Correct |")
    print("|---------|-----|----------|-------------|---------|")
    for item in valid_patterns['wrong_to_correct']:
        print(f"| {item['subject']} | {item['question_number']} | {item['orig_answer']} | {item['para_answer']} | {item['correct_answer']} |")

    print("\ncorrect_to_wrong:")
    print("| Subject | Q# | Original | Paraphrased | Correct |")
    print("|---------|-----|----------|-------------|---------|")
    for item in valid_patterns['correct_to_wrong']:
        print(f"| {item['subject']} | {item['question_number']} | {item['orig_answer']} | {item['para_answer']} | {item['correct_answer']} |")

    # Analysis conclusion changes
    print(f"\n{'='*60}")
    print("言い換えによるAnalysis結論の変化")
    print("=" * 60)

    print(f"\n信頼できる結論変化: {len(conclusion_changes)}件")
    print("\n| Subject | Q# | Orig結論 | Para結論 | 正解 | 変化 |")
    print("|---------|-----|----------|----------|------|------|")
    for item in conclusion_changes:
        if item['orig_correct'] and not item['para_correct']:
            change = "correct→wrong"
        elif not item['orig_correct'] and item['para_correct']:
            change = "wrong→correct"
        elif item['orig_correct'] and item['para_correct']:
            change = "correct→correct"
        else:
            change = "wrong→wrong"
        print(f"| {item['subject']} | {item['question_number']} | {item['orig_conclusion']} | {item['para_conclusion']} | {item['correct']} | {change} |")

    # Summary
    c2w = sum(1 for item in conclusion_changes if item['orig_correct'] and not item['para_correct'])
    w2c = sum(1 for item in conclusion_changes if not item['orig_correct'] and item['para_correct'])
    w2w = sum(1 for item in conclusion_changes if not item['orig_correct'] and not item['para_correct'])
    c2c = sum(1 for item in conclusion_changes if item['orig_correct'] and item['para_correct'])

    print(f"\n結論変化のサマリー:")
    print(f"  wrong_to_correct: {w2c}件")
    print(f"  correct_to_wrong: {c2w}件")
    print(f"  wrong_to_wrong: {w2w}件")
    print(f"  correct_to_correct: {c2c}件")


def main():
    # Load all data
    original_data, paraphrased_data, paraphrase_batch_data = load_all_data(
        ORIGINAL_DIR, PARAPHRASED_DIR, PARAPHRASE_BATCH_DIR
    )

    # Detect failures
    failures, valid_keys, failed_keys = detect_failures(
        original_data, paraphrased_data, paraphrase_batch_data
    )

    # Get all keys
    all_keys = [k for k in original_data if k in paraphrased_data]

    # Analyze change patterns
    all_patterns = analyze_changes(all_keys, original_data, paraphrased_data)
    valid_patterns = analyze_changes(valid_keys, original_data, paraphrased_data)
    failed_patterns = analyze_changes(list(failed_keys), original_data, paraphrased_data)

    # Detect conclusion changes
    conclusion_changes = detect_conclusion_changes(
        valid_keys, original_data, paraphrased_data, paraphrase_batch_data
    )

    # Print report
    print_report(
        original_data, paraphrased_data, paraphrase_batch_data,
        failures, valid_keys, failed_keys, all_keys,
        all_patterns, valid_patterns, failed_patterns, conclusion_changes
    )


if __name__ == "__main__":
    main()
