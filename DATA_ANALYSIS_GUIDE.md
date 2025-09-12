# ParSpliceSIM 生データ・解析システム

## 概要

ParSpliceSIMの実行結果を生データ（JSON）として保存し、後から詳細な解析・可視化を行うシステムです。これにより、シミュレーション実行と解析処理を分離し、様々な解析を効率的に行うことができます。

## 主な特徴

- **生データ保存**: 各ステップの`producer`、`splicer`、`scheduler`の完全な状態をJSONで保存
- **可視化分離**: 既存の可視化処理と同じ結果を生データから生成
- **完全復元**: 生データから元のシミュレーション状態を寸分違わず復元可能
- **柔軟な解析**: 後から任意のタイミングで解析・可視化を実行

## ファイル構成

### 新規追加ファイル

- `src/data/data_collector.py` - 生データ収集クラス
- `analyze_simulation_data.py` - 生データ解析・可視化スクリプト

### 修正されたファイル

- `src/simulation/par_splice_simulation.py` - 生データ収集機能を統合
- `src/config/simulation_config.py` - 新しい出力モード設定を追加

## 使用方法

### 1. 生データのみモードでシミュレーション実行

```python
from src.config import SimulationConfig
from src.simulation import ParSpliceSimulation

# 設定
config = SimulationConfig()
config.raw_data_only = True      # 生データのみ保存
config.save_legacy_format = False  # 従来形式は保存しない

# シミュレーション実行
simulation = ParSpliceSimulation(config)
simulation.run_simulation()
```

### 2. 両方のモードで実行（デフォルト）

```python
config = SimulationConfig()
config.raw_data_only = False     # 可視化も実行
config.save_legacy_format = True   # 従来形式も保存

simulation = ParSpliceSimulation(config)
simulation.run_simulation()
```

### 3. 生データから解析・可視化

```bash
# 基本的な使用方法
python analyze_simulation_data.py results/strategy_timestamp/raw_simulation_data_strategy_timestamp.json

# 出力ディレクトリを指定
python analyze_simulation_data.py raw_data.json -o custom_output_dir

# 生データファイル一覧表示
python analyze_simulation_data.py results/ --list-files
```

## 生データ形式

### JSONファイル構造

```json
{
  "metadata": {
    "config": {/* SimulationConfig の全設定 */},
    "system_matrices": {
      "transition_matrix": [/* 真の確率遷移行列 */],
      "stationary_distribution": [/* 定常分布 */],
      "t_phase_dict": {/* フェーズ時間 */},
      "t_corr_dict": {/* 補正時間 */}
    },
    "timestamp": "20250911_142002",
    "execution_time": "2025-09-11 14:20:02"
  },
  "step_data": [
    {
      "step": 1,
      "step_log": {/* ステップログ情報 */},
      "producer": {
        "groups": {/* 全グループの詳細状態 */},
        "unassigned_workers": [/* 未配置ワーカー */],
        "stored_segments_count": 0,
        "transition_statistics": {/* 遷移統計 */}
      },
      "splicer": {
        "trajectory": [/* 軌道状態列 */],
        "trajectory_length": 0,
        "final_state": 0,
        "segment_store_info": {/* セグメント貯蓄詳細 */},
        "segment_database_info": {/* セグメントDB情報 */},
        "transition_matrix_info": {/* 遷移行列情報 */}
      },
      "scheduler": {
        "statistics": {/* 統計情報 */},
        "estimated_transition_matrix": [/* 推定遷移行列 */],
        "true_transition_matrix": [/* 真の遷移行列 */],
        "observed_states": [/* 観測状態 */],
        "strategy_state": {/* 戦略固有状態 */},
        "total_value": 0.5,
        "last_splicer_state": 0
      }
    }
    /* 各ステップについて同様の構造 */
  ]
}
```

## 生成される解析ファイル

解析スクリプトは元のシミュレーション実行と同じファイルを生成します：

### グラフファイル

- `trajectory_graph_*.png` - 軌道長推移
- `trajectory_efficiency_*.png` - 軌道生成効率
- `total_value_per_worker_*.png` - ワーカー当たり価値
- `combined_value_efficiency_*.png` - 価値・効率統合
- `*_moving_avg_*.png` - 移動平均版グラフ
- `matrix_difference_*.png` - 行列差分推移

### アニメーション（有効時）

- `trajectory_animation_*.gif` - 軌道ランダムウォーク
- `segment_storage_animation_*.gif` - セグメント貯蓄状況

### テキストファイル

- `analysis_summary_*.txt` - 解析結果サマリー

## 設定オプション

### SimulationConfig の新しいオプション

```python
config = SimulationConfig()

# 生データのみ保存（可視化なし）
config.raw_data_only = True

# 従来形式の結果も保存
config.save_legacy_format = True
```

## コマンドライン使用例

```bash
# 小規模テスト実行
python gen-parsplice.py parsplice

# 大規模実行（生データのみ）
python -c "
from src.config import SimulationConfig
from src.simulation import ParSpliceSimulation
config = SimulationConfig()
config.max_simulation_time = 100
config.raw_data_only = True
simulation = ParSpliceSimulation(config)
simulation.run_simulation()
"

# 後で解析
python analyze_simulation_data.py results/parsplice_*/raw_simulation_data_*.json
```

## 利点

1. **効率性**: 大規模シミュレーションでは可視化を後回しにできる
2. **再現性**: 生データから何度でも同じ結果を生成
3. **拡張性**: 新しい解析手法を追加しやすい
4. **デバッグ**: 各ステップの詳細状態を確認可能
5. **比較**: 異なる設定の結果を統一的に解析

## 注意事項

- 生データファイルは大きくなる可能性があります（ステップ数に比例）
- 解析時には十分なメモリが必要です
- numpy、matplotlibが必要です（`pip install numpy matplotlib`）

## トラブルシューティング

### よくあるエラー

1. **ModuleNotFoundError**: 必要なライブラリをインストール
   ```bash
   pip install numpy matplotlib
   ```

2. **FileNotFoundError**: 正しいパスを指定
   ```bash
   python analyze_simulation_data.py $(find results -name "raw_simulation_data_*.json" | head -1)
   ```

3. **Memory Error**: 大きなファイルの場合はメモリを増やすか、データを分割

### ログレベル調整

詳細なログが必要な場合：
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```
