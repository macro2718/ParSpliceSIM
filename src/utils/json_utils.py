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

