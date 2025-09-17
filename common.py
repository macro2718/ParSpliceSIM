#!/usr/bin/env python3
"""
共通の定数、エラー処理、ユーティリティ関数を提供するモジュール

ParSpliceシミュレーションの全体で使用される共通機能を集約。
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
import time
import traceback


class SimulationState(Enum):
    """シミュレーション状態の列挙型"""
    IDLE = "idle"
    PARALLEL = "parallel"
    DECORRELATING = "decorrelating"
    FINISHED = "finished"
    ERROR = "error"


class LogLevel(Enum):
    """ログレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SimulationError(Exception):
    """シミュレーション全体で使用する基底例外クラス"""
    pass


class ProducerError(SimulationError):
    """Producer関連のエラー"""
    pass


class SchedulerError(SimulationError):
    """Scheduler関連のエラー"""
    pass


class SplicerError(SimulationError):
    """Splicer関連のエラー"""
    pass


class ValidationError(SimulationError):
    """バリデーションエラー"""
    pass


class Constants:
    """共通定数クラス"""
    
    # デフォルト値
    DEFAULT_MAX_TIME = 10
    
    # システム制限
    MAX_TRAJECTORY_LENGTH = 10000
    
    # ログフォーマット
    TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
    FILE_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class Logger:
    """簡単なロガークラス"""
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.level_values = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3
        }
    
    def _should_log(self, level: LogLevel) -> bool:
        return self.level_values[level] >= self.level_values[self.level]
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        timestamp = time.strftime(Constants.TIMESTAMP_FORMAT)
        return f"[{timestamp}] {level.value}: {message}"
    
    def debug(self, message: str) -> None:
        if self._should_log(LogLevel.DEBUG):
            print(self._format_message(LogLevel.DEBUG, message))
    
    def info(self, message: str) -> None:
        if self._should_log(LogLevel.INFO):
            print(self._format_message(LogLevel.INFO, message))
    
    def warning(self, message: str) -> None:
        if self._should_log(LogLevel.WARNING):
            print(self._format_message(LogLevel.WARNING, message))
    
    def error(self, message: str) -> None:
        if self._should_log(LogLevel.ERROR):
            print(self._format_message(LogLevel.ERROR, message))


# 共有で使うデフォルトロガー（INFO レベル）
default_logger = Logger(level=LogLevel.INFO)


class SafeOperationHandler:
    """例外を握りつぶさずに整形して返す安全実行ヘルパー"""

    @staticmethod
    def safe_execute(
        func: Callable[[], Any],
        error_cls: type,
        default_return: Any = None,
        logger: Optional[Logger] = None,
    ) -> Any:
        """
        関数を安全に実行し、例外時にはログしてフォールバック値を返す。

        Args:
            func: 実行する関数（引数なし想定）
            error_cls: 捕捉した例外をこの型に包んで扱う用途（ログメッセージ用途）
            default_return: 例外時に返す既定値
            logger: ログ出力に使用するロガー

        Returns:
            func() の戻り値、または例外時は default_return
        """
        lg = logger or default_logger
        try:
            return func()
        except Exception as e:
            # 例外種類を明示し、トレースバックも併記
            tb = traceback.format_exc()
            lg.error(f"{error_cls.__name__ if hasattr(error_cls, '__name__') else 'Error'}: {e}")
            lg.debug(tb)
            return default_return


class Validator:
    """共通バリデーションクラス"""
    
    @staticmethod
    def validate_positive_integer(value: Any, name: str) -> int:
        """正の整数であることを検証"""
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{name}は正の整数である必要があります。受け取った値: {value}")
        return value
    
    @staticmethod
    def validate_non_negative_integer(value: Any, name: str) -> int:
        """非負の整数であることを検証"""
        if not isinstance(value, int) or value < 0:
            raise ValidationError(f"{name}は非負の整数である必要があります。受け取った値: {value}")
        return value
    
    @staticmethod
    def validate_state_range(state: int, max_states: int, name: str = "state") -> int:
        """状態が有効な範囲内であることを検証"""
        if not isinstance(state, int) or state < 0 or state >= max_states:
            raise ValidationError(f"{name}は0から{max_states-1}の範囲である必要があります。受け取った値: {state}")
        return state
    
    @staticmethod
    def validate_worker_id(worker_id: int) -> int:
        """ワーカーIDが有効であることを検証"""
        if not isinstance(worker_id, int) or worker_id < 0:
            raise ValidationError(f"ワーカーIDは非負の整数である必要があります。受け取った値: {worker_id}")
        return worker_id
    
    @staticmethod
    def validate_dict_type(value: Any, name: str) -> Dict:
        """辞書型であることを検証"""
        if not isinstance(value, dict):
            raise ValidationError(f"{name}は辞書型である必要があります。受け取った型: {type(value)}")
        return value
    
    @staticmethod
    def validate_list_type(value: Any, name: str) -> List:
        """リスト型であることを検証"""
        if not isinstance(value, list):
            raise ValidationError(f"{name}はリスト型である必要があります。受け取った型: {type(value)}")
        return value


class ResultFormatter:
    """結果フォーマット用のユーティリティクラス"""
    
    @staticmethod
    def success_result(data: Dict[str, Any] = None) -> Dict[str, Any]:
        """成功結果のフォーマット"""
        result = {'status': 'success'}
        if data:
            result.update(data)
        return result
    
    @staticmethod
    def error_result(error: Union[str, Exception], data: Dict[str, Any] = None) -> Dict[str, Any]:
        """エラー結果のフォーマット"""
        result = {
            'status': 'error',
            'error': str(error) if isinstance(error, Exception) else error
        }
        if data:
            result.update(data)
        return result
    
    @staticmethod
    def warning_result(warning: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """警告結果のフォーマット"""
        result = {
            'status': 'warning',
            'warning': warning
        }
        if data:
            result.update(data)
        return result


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
    
    def start_timer(self, name: str) -> None:
        """タイマー開始"""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """タイマー停止して経過時間を返す"""
        if name not in self.timers:
            raise ValueError(f"タイマー '{name}' は開始されていません")
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """カウンターをインクリメント"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """カウンター値を取得"""
        return self.counters.get(name, 0)
    
    def reset_counter(self, name: str) -> None:
        """カウンターをリセット"""
        if name in self.counters:
            self.counters[name] = 0


def get_file_timestamp() -> str:
    """ファイル名向けのタイムスタンプを取得する（衝突しにくい形式）。"""
    return time.strftime(Constants.FILE_TIMESTAMP_FORMAT)
