"""ファイルとディレクトリ操作のユーティリティ"""
import os
from common import default_logger


def create_results_directory(strategy_name: str, timestamp: str) -> str:
    """結果保存用のディレクトリを作成する
    
    Args:
        strategy_name: 戦略名
        timestamp: タイムスタンプ
        
    Returns:
        作成されたディレクトリのパス
    """
    results_dir = "results"
    # 冪等に作成（存在チェック→作成の重複を排除）
    os.makedirs(results_dir, exist_ok=True)
    session_dir = os.path.join(results_dir, f"{strategy_name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    default_logger.info(f"Results directory prepared: {session_dir}")
    return session_dir
