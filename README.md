# ParSpliceSIM

並列計算に基づく ParSplice/ParRep 風のセグメント生成・スプライシングをシミュレーションし、戦略別の効率や推移を可視化する Python 実装です。

- 状態遷移は詳細釣り合いを満たす転移行列を自動生成
- 複数ワーカーでセグメントを生成し Splicer が軌道を構築
- スケジューラが戦略に応じてワーカー再配置を最適化
- 結果と各種グラフ/アニメーションを `results/` に保存


## 主要コンポーネント

- `gen-parsplice.py`: エントリポイント。設定生成 → 実行 → 解析/保存までを統括
- `systemGenerater.py`: 定常分布から詳細釣り合いを満たす転移行列や `t_phase`, `t_corr` を生成
- `producer.py`: ワーカー群と各 `ParRepBox` を生成・管理
- `ParRepBox.py`: グループごとの実行状態管理（idle/parallel/decorrelating/finished）と最終セグメント収集
- `worker.py`: dephasing → run → decorrelation のフェーズを持つワーカー
- `splicer.py`: Producer からセグメントを取得し、軌道を順次スプライス
- `scheduler.py`: スケジューリング本体。戦略を切替可能
- `scheduling_strategies.py`, `strategies/*.py`: 戦略の定義（parrep, csparsplice, parsplice, epsplice など）
- `common.py`: 例外/定数/ロガー/ユーティリティ
- `theory/`: 実装の理論メモ（日本語）


## 必要要件

- Python 3.9+（推奨: 3.10–3.12）
- 依存ライブラリ:
  - `numpy`
  - `matplotlib`
  - `pillow`（GIF アニメ保存に使用）

インストール例:

```
pip install numpy matplotlib pillow
```


## クイックスタート

デフォルト設定（戦略: `parsplice`）で実行:

```
python gen-parsplice.py
```

利用可能なスケジューリング戦略を表示:

```
python gen-parsplice.py --list-strategies
```

戦略を指定して実行（例: `epsplice`）:

```
python gen-parsplice.py --strategy epsplice
```

実行後、`results/<strategy>_<timestamp>/` にテキスト、グラフ（PNG）、必要に応じて GIF が保存されます。


## 出力（`results/`）

- `parsplice_results_<strategy>_<timestamp>.txt`: 実行条件や概要統計のレポート
- `trajectory_graph_*.png`: ステップごとの軌道長推移
- `total_value_per_worker_*.png`, `combined_value_efficiency_*.png`: 戦略の価値や効率の推移
- `matrix_difference_*.png`: 真の転移行列と選択行列との差の推移（Frobenius ノルムなど）
- `trajectory_animation_*.gif`: 最終軌道の状態遷移をランダムウォークとして可視化（有効化時）
- `segment_storage_*.gif`: セグメント貯蓄状況アニメーション（有効化時）

既存の履歴が含まれるため、フォルダ名構成はコミット差分で若干異なる場合がありますが、最新実行は上記形式で出力されます。


## 設定（`SimulationConfig`）

設定は `gen-parsplice.py` の `SimulationConfig` で管理します。CLI からは「戦略名」のみ指定できます。他はコード側で変更してください。

- 乱数シード: `random_seed`
- 系生成: `num_states`, `self_loop_prob_mean`, `stationary_concentration`, `connectivity`
- 時間スケール: `t_phase_mean`, `t_phase_constant_mode`, `t_corr_mean`, `t_corr_constant_mode`
- 並列: `num_workers`
- 実行: `max_simulation_time`, `initial_splicer_state`
- 戦略: `scheduling_strategy`（`parrep|csparsplice|parsplice|epsplice`）, `strategy_params`
- 出力: `output_interval`, `trajectory_animation`, `segment_storage_animation`, `minimal_output`
- 軌道: `max_trajectory_length`

例（コード内で上書き）:

```python
config = SimulationConfig(
    num_states=12,
    num_workers=50,
    max_simulation_time=200,
    scheduling_strategy="epsplice",
    t_phase_constant_mode=True,
    t_corr_constant_mode=True,
    minimal_output=True,
)
```


## 戦略一覧と切替

現在実装済み（`scheduling_strategies.py` → `strategies/*.py`）:

- `parrep`: ParRep 戦略
- `csparsplice`: 現在状態特化 ParSplice
- `parsplice`: 一般 ParSplice 戦略（デフォルト）
- `epsplice`: ePSplice 戦略

確認コマンド:

```
python gen-parsplice.py --list-strategies
```


## 実行の流れ（概要）

1) 系生成
- `systemGenerater.py` で定常分布 → 詳細釣り合いを満たす転移行列を生成
- 各状態の `t_phase`, `t_corr` を生成（定数モード/確率分布）

2) 並列実行とスプライシング
- `Producer` がワーカーと `ParRepBox` を作成
- 各ボックスは dephasing → run → decorrelating を遷移し、最終セグメントを出力
- `Splicer` がセグメントを収集して軌道を構築（状態ごとのセグメント ID を順次利用）

3) スケジューリング
- `Scheduler` が `splicer/prod` 状態と遷移統計から価値を計算
- 選択戦略によりワーカー移動や新規ボックス作成を決定
- 真値行列と選択行列の差分などを計測し履歴化

4) 保存/可視化
- テキストサマリ、各種 PNG、（有効化時）GIF を `results/` に保存


## リポジトリ構成（抜粋）

- `gen-parsplice.py`（エントリ）
- `producer.py`, `ParRepBox.py`, `worker.py`（実行基盤）
- `splicer.py`（軌道構築）
- `scheduler.py`, `scheduling_strategies.py`, `strategies/`（戦略）
- `systemGenerater.py`（系生成）
- `common.py`（共通ユーティリティ）
- `theory/`（理論メモ）
- `results/`（実行結果）


## よくある注意点

- GIF 保存には `pillow` が必要です。環境により ImageMagick など別 writer の設定が必要な場合があります。
- `minimal_output=True` だと標準出力が簡潔になります（ログは内部で記録）。
- ステップ数/ワーカー数が大きいと計算・出力が重くなります。段階的に増やしてください。
- 生成結果は都度 `results/` に蓄積されます。不要な結果は手動で削除してください。


## 開発メモ

- 主要な公開 API は各クラスの `run_one_step`/`get_*_info` 系です。
- 追加戦略は `strategies/` にファイルを作成し、`scheduling_strategies.py` の `AVAILABLE_STRATEGIES` に登録してください。
- CLI オプションは現状「戦略名」のみです。拡張する場合は `gen-parsplice.py:main()` を編集します。


## ライセンス

このリポジトリには明示的なライセンスが含まれていません。公開/再配布ポリシーはリポジトリ所有者に確認してください。

