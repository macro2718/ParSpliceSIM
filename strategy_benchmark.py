#!/usr/bin/env python3
"""
スケジューリング戦略のベンチマークと比較実験

様々なスケジューリング戦略を比較実験するためのモジュール。
異なる戦略の性能を評価し、結果を可視化する。

使用方法:
1. strategy_benchmark.py を直接実行して全戦略を比較
2. 特定の戦略を指定して実験
3. 結果をCSVファイルとグラフで出力
"""

import time
import csv
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path

import importlib.util
import sys
from pathlib import Path

# gen-parsplice.pyを動的にインポート
gen_parsplice_path = Path(__file__).parent / "gen-parsplice.py"
spec = importlib.util.spec_from_file_location("gen_parsplice", gen_parsplice_path)
gen_parsplice = importlib.util.module_from_spec(spec)
sys.modules["gen_parsplice"] = gen_parsplice
spec.loader.exec_module(gen_parsplice)

SimulationConfig = gen_parsplice.SimulationConfig
ParSpliceSimulation = gen_parsplice.ParSpliceSimulation
from scheduling_strategies import list_available_strategies, AVAILABLE_STRATEGIES
from common import get_file_timestamp, default_logger


@dataclass
class BenchmarkConfig:
    """ベンチマーク実験の設定"""
    # 実験対象の戦略リスト
    strategies_to_test: List[str] = None
    
    # 各戦略に対する実験回数
    num_runs_per_strategy: int = 3
    
    # シミュレーション設定のベース
    base_simulation_config: SimulationConfig = None
    
    # 結果保存設定
    save_results: bool = True
    save_graphs: bool = True
    results_dir: str = "benchmark_results"
    
    def __post_init__(self):
        """初期化後処理"""
        if self.strategies_to_test is None:
            self.strategies_to_test = list(AVAILABLE_STRATEGIES.keys())
        
        if self.base_simulation_config is None:
            self.base_simulation_config = SimulationConfig()


@dataclass
class BenchmarkResult:
    """単一実験の結果"""
    strategy_name: str
    run_id: int
    execution_time: float
    total_steps: int
    total_worker_moves: int
    total_new_groups_created: int
    final_trajectory_length: int
    final_splicer_state: int
    observed_states_count: int
    strategy_calculations: int
    strategy_worker_moves: int
    
    # エラー情報
    success: bool = True
    error_message: str = ""


class StrategyBenchmark:
    """スケジューリング戦略のベンチマーク実験クラス"""
    
    def __init__(self, config: BenchmarkConfig):
        """
        ベンチマーク実験の初期化
        
        Parameters:
        config (BenchmarkConfig): ベンチマーク設定
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # 結果保存ディレクトリの作成
        if self.config.save_results or self.config.save_graphs:
            Path(self.config.results_dir).mkdir(exist_ok=True)
    
    def run_benchmark(self) -> List[BenchmarkResult]:
        """
        ベンチマーク実験を実行
        
        Returns:
        List[BenchmarkResult]: 実験結果のリスト
        """
        print("=== スケジューリング戦略ベンチマーク開始 ===")
        print(f"対象戦略: {self.config.strategies_to_test}")
        print(f"各戦略実行回数: {self.config.num_runs_per_strategy}")
        print(f"シミュレーション時間: {self.config.base_simulation_config.max_simulation_time}")
        print()
        
        total_experiments = len(self.config.strategies_to_test) * self.config.num_runs_per_strategy
        experiment_count = 0
        
        for strategy_name in self.config.strategies_to_test:
            print(f"--- 戦略 '{strategy_name}' の実験開始 ---")
            
            for run_id in range(self.config.num_runs_per_strategy):
                experiment_count += 1
                print(f"実験 {experiment_count}/{total_experiments}: {strategy_name} (Run {run_id + 1})")
                
                # 単一実験の実行
                result = self._run_single_experiment(strategy_name, run_id)
                self.results.append(result)
                
                if result.success:
                    print(f"  完了: {result.execution_time:.2f}秒, 移動数: {result.total_worker_moves}, "
                          f"軌道長: {result.final_trajectory_length}")
                else:
                    print(f"  エラー: {result.error_message}")
            
            print()
        
        print("=== ベンチマーク実験完了 ===")
        
        # 結果の保存と分析
        if self.config.save_results:
            self._save_results()
        
        if self.config.save_graphs:
            self._create_comparison_graphs()
        
        # 統計サマリーの表示
        self._print_summary_statistics()
        
        return self.results
    
    def _run_single_experiment(self, strategy_name: str, run_id: int) -> BenchmarkResult:
        """
        単一の実験を実行
        
        Parameters:
        strategy_name (str): 戦略名
        run_id (int): ラン番号
        
        Returns:
        BenchmarkResult: 実験結果
        """
        start_time = time.time()
        
        try:
            # シミュレーション設定をコピーして戦略を設定
            sim_config = SimulationConfig(
                random_seed=self.config.base_simulation_config.random_seed + run_id,  # シードを変更
                num_states=self.config.base_simulation_config.num_states,
                self_loop_prob_mean=self.config.base_simulation_config.self_loop_prob_mean,
                variance=self.config.base_simulation_config.variance,
                t_phase_mean=self.config.base_simulation_config.t_phase_mean,
                t_phase_constant_mode=self.config.base_simulation_config.t_phase_constant_mode,
                t_corr_mean=self.config.base_simulation_config.t_corr_mean,
                t_corr_constant_mode=self.config.base_simulation_config.t_corr_constant_mode,
                num_workers=self.config.base_simulation_config.num_workers,
                max_simulation_time=self.config.base_simulation_config.max_simulation_time,
                output_interval=self.config.base_simulation_config.output_interval,
                initial_splicer_state=self.config.base_simulation_config.initial_splicer_state,
                scheduling_strategy=strategy_name,
                strategy_params=self.config.base_simulation_config.strategy_params.copy()
            )
            
            # シミュレーションの実行
            simulation = ParSpliceSimulation(sim_config)
            simulation.run_simulation()
            
            # 結果の収集
            # 注: 実際にはシミュレーション結果を取得するAPIが必要
            # ここでは仮の値を設定（実装時に修正が必要）
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                strategy_name=strategy_name,
                run_id=run_id,
                execution_time=execution_time,
                total_steps=sim_config.max_simulation_time,
                total_worker_moves=0,  # 実際の値に置換
                total_new_groups_created=0,  # 実際の値に置換
                final_trajectory_length=0,  # 実際の値に置換
                final_splicer_state=0,  # 実際の値に置換
                observed_states_count=0,  # 実際の値に置換
                strategy_calculations=0,  # 実際の値に置換
                strategy_worker_moves=0,  # 実際の値に置換
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            default_logger.error(f"実験エラー ({strategy_name}, run {run_id}): {str(e)}")
            
            return BenchmarkResult(
                strategy_name=strategy_name,
                run_id=run_id,
                execution_time=execution_time,
                total_steps=0,
                total_worker_moves=0,
                total_new_groups_created=0,
                final_trajectory_length=0,
                final_splicer_state=0,
                observed_states_count=0,
                strategy_calculations=0,
                strategy_worker_moves=0,
                success=False,
                error_message=str(e)
            )
    
    def _save_results(self) -> None:
        """実験結果をCSVファイルに保存"""
        timestamp = get_file_timestamp()
        filename = f"{self.config.results_dir}/benchmark_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [field.name for field in BenchmarkResult.__dataclass_fields__.values()]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        
        print(f"実験結果を {filename} に保存しました")
    
    def _create_comparison_graphs(self) -> None:
        """比較グラフを作成"""
        timestamp = get_file_timestamp()
        
        # 成功した結果のみを取得
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("成功した実験がないため、グラフを作成できません")
            return
        
        # 戦略別の統計を計算
        strategy_stats = {}
        for result in successful_results:
            if result.strategy_name not in strategy_stats:
                strategy_stats[result.strategy_name] = {
                    'execution_times': [],
                    'worker_moves': [],
                    'trajectory_lengths': [],
                    'observed_states': []
                }
            
            stats = strategy_stats[result.strategy_name]
            stats['execution_times'].append(result.execution_time)
            stats['worker_moves'].append(result.total_worker_moves)
            stats['trajectory_lengths'].append(result.final_trajectory_length)
            stats['observed_states'].append(result.observed_states_count)
        
        # 1. 実行時間の比較
        self._create_bar_chart(
            strategy_stats, 'execution_times', 'Execution Time (seconds)',
            f'{self.config.results_dir}/execution_time_comparison_{timestamp}.png'
        )
        
        # 2. ワーカー移動数の比較
        self._create_bar_chart(
            strategy_stats, 'worker_moves', 'Total Worker Moves',
            f'{self.config.results_dir}/worker_moves_comparison_{timestamp}.png'
        )
        
        # 3. 軌道長の比較
        self._create_bar_chart(
            strategy_stats, 'trajectory_lengths', 'Final Trajectory Length',
            f'{self.config.results_dir}/trajectory_length_comparison_{timestamp}.png'
        )
        
        # 4. 観測状態数の比較
        self._create_bar_chart(
            strategy_stats, 'observed_states', 'Number of Observed States',
            f'{self.config.results_dir}/observed_states_comparison_{timestamp}.png'
        )
        
        print(f"比較グラフを {self.config.results_dir}/ に保存しました")
    
    def _create_bar_chart(self, strategy_stats: Dict, metric_key: str, 
                         ylabel: str, filename: str) -> None:
        """棒グラフを作成"""
        strategies = list(strategy_stats.keys())
        means = []
        stds = []
        
        for strategy in strategies:
            values = strategy_stats[strategy][metric_key]
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, means, yerr=stds, capsize=5, alpha=0.7)
        
        # 色を戦略ごとに変える
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Scheduling Strategy')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Comparison by Strategy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_summary_statistics(self) -> None:
        """統計サマリーを表示"""
        print("\n=== 実験結果サマリー ===")
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print(f"成功実験数: {len(successful_results)}")
        print(f"失敗実験数: {len(failed_results)}")
        
        if not successful_results:
            return
        
        # 戦略別統計
        strategy_names = list(set(r.strategy_name for r in successful_results))
        strategy_names.sort()
        
        print("\n戦略別平均結果:")
        print(f"{'Strategy':<15} {'ExecTime(s)':<12} {'Moves':<8} {'Trajectory':<12} {'States':<8}")
        print("-" * 60)
        
        for strategy in strategy_names:
            strategy_results = [r for r in successful_results if r.strategy_name == strategy]
            
            avg_time = np.mean([r.execution_time for r in strategy_results])
            avg_moves = np.mean([r.total_worker_moves for r in strategy_results])
            avg_trajectory = np.mean([r.final_trajectory_length for r in strategy_results])
            avg_states = np.mean([r.observed_states_count for r in strategy_results])
            
            print(f"{strategy:<15} {avg_time:<12.2f} {avg_moves:<8.1f} {avg_trajectory:<12.1f} {avg_states:<8.1f}")
    
    def get_best_strategy(self, metric: str = 'final_trajectory_length') -> Optional[str]:
        """
        指定されたメトリクスで最良の戦略を取得
        
        Parameters:
        metric (str): 評価メトリクス
        
        Returns:
        Optional[str]: 最良戦略名
        """
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return None
        
        strategy_avgs = {}
        for strategy_name in set(r.strategy_name for r in successful_results):
            strategy_results = [r for r in successful_results if r.strategy_name == strategy_name]
            
            if metric == 'execution_time':
                avg_value = np.mean([r.execution_time for r in strategy_results])
                # 実行時間は小さい方が良い
                strategy_avgs[strategy_name] = -avg_value
            elif metric == 'final_trajectory_length':
                avg_value = np.mean([r.final_trajectory_length for r in strategy_results])
                strategy_avgs[strategy_name] = avg_value
            elif metric == 'observed_states_count':
                avg_value = np.mean([r.observed_states_count for r in strategy_results])
                strategy_avgs[strategy_name] = avg_value
            else:
                avg_value = np.mean([getattr(r, metric) for r in strategy_results])
                strategy_avgs[strategy_name] = avg_value
        
        return max(strategy_avgs.keys(), key=lambda k: strategy_avgs[k])


def run_quick_benchmark() -> None:
    """クイックベンチマーク実行"""
    print("=== クイックベンチマーク実行 ===")
    
    # 軽量な設定でベンチマーク
    sim_config = SimulationConfig(
        num_states=3,
        num_workers=2,
        max_simulation_time=20,
        random_seed=42
    )
    
    benchmark_config = BenchmarkConfig(
        strategies_to_test=['default', 'greedy', 'random'],
        num_runs_per_strategy=2,
        base_simulation_config=sim_config,
        save_results=True,
        save_graphs=True
    )
    
    benchmark = StrategyBenchmark(benchmark_config)
    results = benchmark.run_benchmark()
    
    # 最良戦略の表示
    best_strategy = benchmark.get_best_strategy('final_trajectory_length')
    if best_strategy:
        print(f"\n最良戦略 (軌道長基準): {best_strategy}")


def run_full_benchmark() -> None:
    """フルベンチマーク実行"""
    print("=== フルベンチマーク実行 ===")
    
    # 標準設定でベンチマーク
    sim_config = SimulationConfig()
    
    benchmark_config = BenchmarkConfig(
        strategies_to_test=list(AVAILABLE_STRATEGIES.keys()),
        num_runs_per_strategy=5,
        base_simulation_config=sim_config,
        save_results=True,
        save_graphs=True
    )
    
    benchmark = StrategyBenchmark(benchmark_config)
    results = benchmark.run_benchmark()
    
    # 各メトリクスでの最良戦略を表示
    for metric in ['final_trajectory_length', 'execution_time', 'observed_states_count']:
        best_strategy = benchmark.get_best_strategy(metric)
        print(f"最良戦略 ({metric}): {best_strategy}")


def list_strategies_info() -> None:
    """利用可能な戦略の情報を表示"""
    print("=== 利用可能なスケジューリング戦略 ===")
    
    strategies = list_available_strategies()
    for strategy in strategies:
        print(f"名前: {strategy['name']}")
        print(f"クラス: {strategy['class_name']}")  
        print(f"説明: {strategy['description']}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_strategies_info()
        elif command == "quick":
            run_quick_benchmark()
        elif command == "full":
            run_full_benchmark()
        else:
            print("使用方法:")
            print("  python strategy_benchmark.py list    - 戦略一覧表示")
            print("  python strategy_benchmark.py quick   - クイックベンチマーク")
            print("  python strategy_benchmark.py full    - フルベンチマーク")
    else:
        # デフォルトはクイックベンチマーク
        run_quick_benchmark()
