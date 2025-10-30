import json
import os
from datetime import datetime
from typing import List, Dict, Any


class ResultSaver:
    """MMLU評価結果をJSON Lines形式で逐次保存するクラス"""

    def __init__(self, output_dir: str = "results", filename: str = None):
        """
        Args:
            output_dir: 結果を保存するディレクトリ
            filename: JSON Linesファイル名（Noneの場合はタイムスタンプを使用）
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mmlu_results_{timestamp}.jsonl"

        self.filepath = os.path.join(output_dir, filename)

        # JSON Linesファイルを作成（空ファイルとして初期化）
        with open(self.filepath, 'w', encoding='utf-8') as f:
            pass  # 空ファイルを作成

    def add_result(
        self,
        subject: str,
        question_number: int,
        question: str,
        choices: List[str],
        analysis: str,
        predicted_answer: str,
        correct_answer: str,
        is_correct: bool,
        ai_time: float
    ):
        """
        1つの質問の結果をJSON Lines形式で即座に書き込む

        Args:
            subject: サブジェクト名
            question_number: 質問番号
            question: 質問文（保存しない）
            choices: 選択肢のリスト（保存しない）
            analysis: モデルの分析
            predicted_answer: モデルの予測答え
            correct_answer: 正解
            is_correct: 正解かどうか
            ai_time: AI処理時間（秒）
        """
        result = {
            "subject": subject,
            "question_number": question_number,
            "analysis": analysis,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "ai_time_seconds": round(ai_time, 2)
        }

        # JSON Lines形式で追記（1行1JSON）
        with open(self.filepath, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    def save(self):
        """
        互換性のために残しているメソッド。
        逐次書き込みされているため、このメソッドは何もしない。
        """
        return self.filepath

    def get_filepath(self) -> str:
        """保存先のファイルパスを取得"""
        return self.filepath
