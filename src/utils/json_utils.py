"""JSONユーティリティ（共通のエンコーダとダンプ関数）"""
from typing import Any
import json
import gzip
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """numpy配列/スカラーをJSONにシリアライズ可能にするエンコーダ"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def convert_keys_to_strings(data: Any) -> Any:
    """辞書やリスト内のキーを再帰的に文字列化する（軽量な共通処理）"""
    if isinstance(data, dict):
        return {str(k): convert_keys_to_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys_to_strings(x) for x in data]
    return data


def sanitize_for_json(data: Any, _visited: set | None = None) -> Any:
    """値を再帰的にJSONシリアライズ可能な形に整形する。

    - dict: キーを文字列化し、値を再帰整形
    - list/tuple: listにして再帰整形
    - set: listにして再帰整形（順序保証なし）
    - numpy型: ndarrayはlistに、スカラーはPythonプリミティブに
    - Enum/その他: 代表的なものは値に、未知のオブジェクトはstr()にフォールバック
    - 循環参照を検出したら "<circular>" に置換
    """
    if _visited is None:
        _visited = set()

    # プリミティブはそのまま
    if data is None or isinstance(data, (bool, int, float, str)):
        return data

    obj_id = id(data)
    if obj_id in _visited:
        return "<circular>"
    _visited.add(obj_id)

    # numpy系
    if isinstance(data, np.ndarray):
        try:
            return data.tolist()
        finally:
            return data.tolist()
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.bool_,)):
        return bool(data)

    # コンテナ系
    if isinstance(data, dict):
        return {str(k): sanitize_for_json(v, _visited) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [sanitize_for_json(x, _visited) for x in data]
    if isinstance(data, set):
        return [sanitize_for_json(x, _visited) for x in data]

    # bytesは文字列化
    if isinstance(data, (bytes, bytearray)):
        try:
            return data.decode('utf-8', errors='replace')
        except Exception:
            return str(data)

    # Enumっぽいもの
    try:
        from enum import Enum
        if isinstance(data, Enum):
            return data.value
    except Exception:
        pass

    # dataclass等: asdictは使わず、__dict__ があれば辞書化を試みる
    if hasattr(data, '__dict__') and isinstance(getattr(data, '__dict__', None), dict):
        try:
            return sanitize_for_json(vars(data), _visited)
        except Exception:
            return str(data)

    # 最後の手段: 文字列化
    try:
        json.dumps(data)
        return data
    except Exception:
        return str(data)


def safe_dump_json(obj: Any, path: str, *, ensure_ascii: bool = False, indent: int = 2,
                   use_numpy_encoder: bool = True, compress: bool = False) -> None:
    """ファイルパスにJSONを書き出す（gzip対応・numpy対応）。

    Args:
        obj: シリアライズ対象
        path: 出力先パス
        ensure_ascii: json.dumpのensure_ascii
        indent: json.dumpのindent
        use_numpy_encoder: numpyを含む場合に専用エンコーダを使うか
        compress: gzip圧縮して書き出すか
    """
    encoder = NumpyJSONEncoder if use_numpy_encoder else None
    if compress:
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent, cls=encoder)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent, cls=encoder)

