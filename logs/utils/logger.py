import logging
import os
from datetime import datetime


def setup_logger(log_dir: str = "logs", log_name: str = None):
    """
    ログ設定を初期化し、ファイルとコンソールの両方にログを出力する

    Args:
        log_dir: ログファイルを保存するディレクトリ
        log_name: ログファイル名（Noneの場合はタイムスタンプを使用）

    Returns:
        logging.Logger: 設定されたロガーインスタンス
    """
    # ログディレクトリを作成
    os.makedirs(log_dir, exist_ok=True)

    # ログファイル名を生成
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"mmlu_run_{timestamp}.log"

    log_path = os.path.join(log_dir, log_name)

    # ロガーの設定
    logger = logging.getLogger("mmlu_logger")
    logger.setLevel(logging.INFO)

    # 既存のハンドラをクリア（重複を防ぐ）
    if logger.handlers:
        logger.handlers.clear()

    # ログフォーマット
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ファイルハンドラ
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_path}")

    return logger


def get_logger():
    """
    既存のロガーインスタンスを取得

    Returns:
        logging.Logger: ロガーインスタンス
    """
    return logging.getLogger("mmlu_logger")
