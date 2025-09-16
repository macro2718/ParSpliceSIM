"""トラジェクトリ長のみをストリーミング出力する軽量ライター"""
import os
from typing import Optional


class TrajectoryLengthStreamer:
    """各ステップのトラジェクトリ長だけを1行ずつ書き出す"""

    def __init__(self, output_dir: str, strategy_name: str, timestamp: str, flush_every: int = 20):
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.timestamp = timestamp
        self.flush_every = max(1, int(flush_every))
        self.fp: Optional[object] = None
        self._count = 0
        self.path = os.path.join(
            output_dir,
            f"trajectory_length_stream_{strategy_name}_{timestamp}.txt",
        )

    def start(self) -> str:
        if self.fp is None:
            # 行バッファリング有効のテキストモード
            self.fp = open(self.path, "w", encoding="utf-8", buffering=1)
        return self.path

    def append_length(self, length: int) -> None:
        if self.fp is None:
            raise RuntimeError("TrajectoryLengthStreamer not started")
        # 各行に長さのみを書き込む
        self.fp.write(f"{int(length)}\n")
        self._count += 1
        if (self._count % self.flush_every) == 0:
            self.fp.flush()

    def finalize(self) -> Optional[str]:
        try:
            if self.fp is not None:
                self.fp.flush()
                self.fp.close()
                return self.path
            return None
        finally:
            self.fp = None
