#!/usr/bin/env python3
"""
共通の定数、エラー処理、ユーティリティ関数を提供するモジュール

ParSpliceシミュレーションの全体で使用される共通機能を集約。
"""

from typing import Dict, Any, List, Optional, Union
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
    DEFAULT_SEGMENT_BONUS_MULTIPLIER = 0.1
    DEFAULT_NEW_GROUP_BASE_VALUE = 0.8
    DEFAULT_EXISTING_GROUP_BASE_VALUE = 1.0
    
    # システム制限
    MAX_WORKERS = 100
    MAX_STATES = 50
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
    def validate_worker_id(worker_id: int, max_workers: int) -> int:
        """ワーカーIDが有効な範囲内であることを検証"""
        if not isinstance(worker_id, int) or worker_id < 0 or worker_id >= max_workers:
            raise ValidationError(f"ワーカーIDは0から{max_workers-1}の範囲である必要があります。受け取った値: {worker_id}")
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
    
    def get_all_counters(self) -> Dict[str, int]:
        """全カウンター値を取得"""
        return self.counters.copy()


class SafeOperationHandler:
    """安全な操作実行のためのハンドラークラス"""
    
    @staticmethod
    def safe_execute(operation, error_class: type = SimulationError, 
                    default_return: Any = None, logger: Optional[Logger] = None) -> Any:
        """
        安全に操作を実行し、エラーハンドリングを行う
        
        Args:
            operation: 実行する操作（関数やラムダ）
            error_class: 発生するエラーのクラス
            default_return: エラー時のデフォルト戻り値
            logger: ロガーインスタンス
        
        Returns:
            操作の結果またはdefault_return
        """
        try:
            return operation()
        except Exception as e:
            error_msg = f"操作実行中にエラーが発生: {str(e)}"
            
            if logger:
                logger.error(error_msg)
                logger.debug(f"スタックトレース: {traceback.format_exc()}")
            
            if error_class and not isinstance(e, error_class):
                raise error_class(error_msg) from e
            
            return default_return
    
    @staticmethod
    def safe_get_dict_value(dictionary: Dict, key: Any, default: Any = None,
                           logger: Optional[Logger] = None) -> Any:
        """辞書から安全に値を取得"""
        try:
            return dictionary.get(key, default)
        except Exception as e:
            if logger:
                logger.warning(f"辞書値取得エラー: key={key}, error={str(e)}")
            return default


# グローバルロガーインスタンス
default_logger = Logger(LogLevel.INFO)

# グローバルパフォーマンスモニター
performance_monitor = PerformanceMonitor()


def get_current_timestamp() -> str:
    """現在のタイムスタンプを取得"""
    return time.strftime(Constants.TIMESTAMP_FORMAT)


def get_file_timestamp() -> str:
    """ファイル名用のタイムスタンプを取得"""
    return time.strftime(Constants.FILE_TIMESTAMP_FORMAT)
