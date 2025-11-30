#!/usr/bin/env python3
"""
Check both original and filler_tokens results for completeness.
"""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Directories
DATASET_DIR = Path("dataset/mmlu/data/test")
ORIGINAL_DIR = Path("results/mmlu/original")
FILLER_TOKENS_DIR = Path("results/mmlu/filler_tokens")


@dataclass
class SubjectStatus:
    """Status of a subject's generation."""
    subject: str
    expected_tasks: int
    actual_tasks: int
    file_exists: bool

    @property
    def is_complete(self) -> bool:
        return self.file_exists and self.expected_tasks == self.actual_tasks

    @property
    def is_missing_file(self) -> bool:
        return not self.file_exists

    @property
    def is_incomplete_tasks(self) -> bool:
        return self.file_exists and self.expected_tasks != self.actual_tasks


def count_lines(file_path: Path) -> int:
    """Count the number of lines in a file."""
    if not file_path.exists():
        return 0
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
        return 0


def get_dataset_subjects() -> Dict[str, int]:
    """Get all subjects from the dataset directory with their task counts."""
    subjects = {}
    if not DATASET_DIR.exists():
        print(f"Warning: Dataset directory not found: {DATASET_DIR}")
        return subjects

    for file in DATASET_DIR.glob("*_test.csv"):
        # Remove '_test.csv' suffix to get subject name
        subject = file.stem.replace("_test", "")
        task_count = count_lines(file)
        subjects[subject] = task_count

    return subjects


def check_subject_status(subject: str, expected_tasks: int, result_file: Path) -> SubjectStatus:
    """Check the status of a subject in a specific result file."""
    file_exists = result_file.exists()
    actual_tasks = count_lines(result_file) if file_exists else 0

    return SubjectStatus(
        subject=subject,
        expected_tasks=expected_tasks,
        actual_tasks=actual_tasks,
        file_exists=file_exists
    )


def check_original_results(dataset_subjects: Dict[str, int]) -> Dict[str, List[SubjectStatus]]:
    """Check results/mmlu/original directory."""
    print("=" * 70)
    print("ORIGINAL RESULTS (results/mmlu/original)")
    print("=" * 70)
    print()

    if not ORIGINAL_DIR.exists():
        print(f"Directory not found: {ORIGINAL_DIR}")
        return {}

    missing_files = []
    incomplete_tasks = []
    complete = []

    for subject, expected_tasks in sorted(dataset_subjects.items()):
        result_file = ORIGINAL_DIR / f"{subject}.jsonl"
        status = check_subject_status(subject, expected_tasks, result_file)

        if status.is_missing_file:
            missing_files.append(status)
        elif status.is_incomplete_tasks:
            incomplete_tasks.append(status)
        else:
            complete.append(status)

    # Summary
    print(f"Complete: {len(complete)}/{len(dataset_subjects)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Incomplete tasks: {len(incomplete_tasks)}")

    # Details
    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for status in missing_files:
            print(f"  - {status.subject} (expected {status.expected_tasks} tasks)")

    if incomplete_tasks:
        print(f"\nIncomplete tasks ({len(incomplete_tasks)}):")
        for status in incomplete_tasks:
            print(f"  - {status.subject}: {status.actual_tasks}/{status.expected_tasks} tasks")

    if not missing_files and not incomplete_tasks:
        print("\n✓ All subjects complete!")

    print()

    return {
        'missing_files': missing_files,
        'incomplete_tasks': incomplete_tasks,
        'complete': complete
    }


def check_filler_tokens_results(dataset_subjects: Dict[str, int]) -> Dict[str, Dict[str, List[SubjectStatus]]]:
    """Check results/mmlu/filler_tokens directory."""
    print("=" * 70)
    print("FILLER TOKENS RESULTS (results/mmlu/filler_tokens)")
    print("=" * 70)
    print()

    if not FILLER_TOKENS_DIR.exists():
        print(f"Directory not found: {FILLER_TOKENS_DIR}")
        return {}

    # Get all subdirectories
    subdirs = sorted([d for d in FILLER_TOKENS_DIR.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"No subdirectories found in {FILLER_TOKENS_DIR}")
        return {}

    all_issues = {}

    for subdir in subdirs:
        print(f"Directory: {subdir.name}")
        print("-" * 60)

        missing_files = []
        incomplete_tasks = []
        complete = []

        for subject, expected_tasks in sorted(dataset_subjects.items()):
            result_file = subdir / f"{subject}.jsonl"
            status = check_subject_status(subject, expected_tasks, result_file)

            if status.is_missing_file:
                missing_files.append(status)
            elif status.is_incomplete_tasks:
                incomplete_tasks.append(status)
            else:
                complete.append(status)

        # Summary
        print(f"  Complete: {len(complete)}/{len(dataset_subjects)}")
        print(f"  Missing files: {len(missing_files)}")
        print(f"  Incomplete tasks: {len(incomplete_tasks)}")

        # Details
        if missing_files:
            print(f"\n  Missing files ({len(missing_files)}):")
            for status in missing_files:
                print(f"    - {status.subject} (expected {status.expected_tasks} tasks)")

        if incomplete_tasks:
            print(f"\n  Incomplete tasks ({len(incomplete_tasks)}):")
            for status in incomplete_tasks:
                print(f"    - {status.subject}: {status.actual_tasks}/{status.expected_tasks} tasks")

        if not missing_files and not incomplete_tasks:
            print(f"  ✓ All subjects complete!")

        print()

        # Store issues
        if missing_files or incomplete_tasks:
            all_issues[subdir.name] = {
                'missing_files': missing_files,
                'incomplete_tasks': incomplete_tasks
            }

    return all_issues


def print_filler_tokens_summary(all_issues: Dict[str, Dict[str, List[SubjectStatus]]]):
    """Print summary for filler tokens results."""
    print("=" * 70)
    print("FILLER TOKENS SUMMARY")
    print("=" * 70)

    if not all_issues:
        print("\n✓ All subjects complete in all directories!")
        return

    print(f"\nDirectories with issues: {len(all_issues)}")

    for subdir_name, issues in sorted(all_issues.items()):
        missing = issues['missing_files']
        incomplete = issues['incomplete_tasks']

        print(f"\n{subdir_name}/")

        if missing:
            print(f"  Missing files ({len(missing)}):")
            for status in missing:
                print(f"    - {status.subject}")

        if incomplete:
            print(f"  Incomplete tasks ({len(incomplete)}):")
            for status in incomplete:
                print(f"    - {status.subject}: {status.actual_tasks}/{status.expected_tasks}")

    # Find common issues
    all_missing_files = [set(s.subject for s in issues['missing_files'])
                        for issues in all_issues.values()]
    all_incomplete = [set(s.subject for s in issues['incomplete_tasks'])
                     for issues in all_issues.values()]

    if all_missing_files:
        common_missing = set.intersection(*all_missing_files) if len(all_missing_files) > 1 else all_missing_files[0]
        if common_missing:
            print(f"\n\nSubjects with missing files in ALL affected directories ({len(common_missing)}):")
            for subject in sorted(common_missing):
                print(f"  - {subject}")

    if all_incomplete:
        common_incomplete = set.intersection(*all_incomplete) if len(all_incomplete) > 1 else all_incomplete[0]
        if common_incomplete:
            print(f"\n\nSubjects with incomplete tasks in ALL affected directories ({len(common_incomplete)}):")
            for subject in sorted(common_incomplete):
                print(f"  - {subject}")


def main():
    # Get all subjects from dataset
    dataset_subjects = get_dataset_subjects()
    print(f"Dataset: {len(dataset_subjects)} subjects, {sum(dataset_subjects.values())} total tasks")
    print()

    # Check original results
    original_issues = check_original_results(dataset_subjects)

    # Check filler tokens results
    filler_tokens_issues = check_filler_tokens_results(dataset_subjects)

    # Print summary
    print_filler_tokens_summary(filler_tokens_issues)


if __name__ == "__main__":
    main()
