"""Trajectoryの可視化とアニメーション生成を担当するクラス"""
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from common import get_file_timestamp
from src.config import SimulationConfig


class TrajectoryVisualizer:
    """Trajectoryの可視化とアニメーション生成を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def create_trajectory_animation(self, trajectory_states_history: List[List[int]], 
                                  transition_matrix: np.ndarray, filename_prefix: str = None) -> str:
        """
        Trajectoryの状態遷移を2Dランダムウォークとしてアニメーション化する
        
        Args:
            trajectory_states_history: ステップごとのtrajectory状態履歴
            transition_matrix: 遷移行列（グラフの構造決定に使用）
            filename_prefix: 出力ファイル名のプレフィックス
        
        Returns:
            str: 生成されたアニメーションファイルのパス
        """
        if not trajectory_states_history:
            if not self.config.minimal_output:
                print("警告: trajectory履歴が空のため、アニメーションを生成できません")
            return None
        
        # 状態の2D座標を生成（円形配置）
        state_positions = self._generate_state_positions(self.config.num_states)
        
        # 最終的なtrajectory（最後のステップの状態列）を取得
        final_trajectory = trajectory_states_history[-1] if trajectory_states_history else []
        
        if not final_trajectory:
            if not self.config.minimal_output:
                print("警告: 最終trajectoryが空のため、アニメーションを生成できません")
            return None
        
        # アニメーション設定
        fig, ax = plt.subplots(figsize=(12, 10))
        
        self._setup_figure(ax, state_positions, transition_matrix)
        line, point, title, step_text = self._setup_animation_elements(ax)
        
        # Add legend after labeled elements are created
        ax.legend(loc='upper right')
        
        # trajectory座標の準備
        trajectory_coords = self._prepare_trajectory_coords(final_trajectory, state_positions)
        
        # アニメーション関数の定義
        animate = self._create_animate_function(line, point, title, step_text, 
                                               trajectory_coords, final_trajectory)
        
        # アニメーション作成と保存
        return self._create_and_save_animation(fig, animate, trajectory_coords, filename_prefix)
    
    def _setup_figure(self, ax, state_positions: Dict, transition_matrix: np.ndarray) -> None:
        """図の基本設定を行う"""
        # 状態ノードを描画
        for state in range(self.config.num_states):
            x, y = state_positions[state]
            circle = plt.Circle((x, y), 0.15, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(state), ha='center', va='center', fontsize=14, fontweight='bold')
        
        # エッジ（遷移）を描画（薄いグレー）
        self._draw_transition_edges(ax, state_positions, transition_matrix)
        
        # 軸の設定
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        # Legend will be added after animation elements are created
    
    def _setup_animation_elements(self, ax) -> Tuple:
        """アニメーション要素を設定する"""
        # アニメーション用のラインとポイント
        line, = ax.plot([], [], 'r-', linewidth=3, alpha=0.7, label='Trajectory Path')
        point, = ax.plot([], [], 'ro', markersize=12, label='Current Position')
        
        # タイトルとステップカウンター
        title = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', 
                       fontsize=16, fontweight='bold')
        step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', 
                           fontsize=12)
        
        return line, point, title, step_text
    
    def _prepare_trajectory_coords(self, final_trajectory: List[int], 
                                  state_positions: Dict) -> List[Tuple[float, float]]:
        """trajectory座標を準備する"""
        trajectory_coords = []
        for state in final_trajectory:
            if state < len(state_positions):
                trajectory_coords.append(state_positions[state])
        return trajectory_coords
    
    def _create_animate_function(self, line, point, title, step_text, 
                                trajectory_coords: List, final_trajectory: List):
        """アニメーション更新関数を作成する"""
        def animate(frame):
            """アニメーションフレーム更新関数"""
            if frame >= len(trajectory_coords):
                frame = len(trajectory_coords) - 1
            
            # 現在までのパスを描画
            if frame > 0:
                x_coords = [coord[0] for coord in trajectory_coords[:frame+1]]
                y_coords = [coord[1] for coord in trajectory_coords[:frame+1]]
                line.set_data(x_coords, y_coords)
            
            # 現在位置を描画
            if trajectory_coords:
                current_x, current_y = trajectory_coords[frame]
                point.set_data([current_x], [current_y])
                current_state = final_trajectory[frame] if frame < len(final_trajectory) else final_trajectory[-1]
                
                # タイトルとステップ情報を更新
                title.set_text(f'ParSplice Trajectory Random Walk')
                step_text.set_text(f'Step: {frame+1}/{len(trajectory_coords)}\nCurrent State: {current_state}')
            
            return line, point, title, step_text
        
        return animate
    
    def _create_and_save_animation(self, fig, animate, trajectory_coords: List, 
                                  filename_prefix: str = None) -> str:
        """アニメーションを作成して保存する"""
        # アニメーション作成
        frames = len(trajectory_coords) if trajectory_coords else 1
        interval = max(100, 2000 // frames)  # フレーム間隔を調整（最大2秒の動画、高速再生）
        
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, 
                                     blit=False, repeat=True)
        
        # ファイルとして保存（GIFで直接保存）
        output_filename = self._build_output_filename('trajectory_animation', filename_prefix)
        
        try:
            # GIFとして保存
            anim.save(output_filename, writer='pillow', fps=min(10, max(3, frames//1)))
            if not self.config.minimal_output:
                print(f"✅ Trajectory animation saved as GIF: {output_filename}")
            
        except Exception as e:
            if not self.config.minimal_output:
                print(f"❌ GIF保存に失敗: {e}")
            output_filename = None
        
        plt.close(fig)
        return output_filename

    def _build_output_filename(self, kind: str, filename_prefix: str = None) -> str:
        """出力ファイル名を共通規則で生成"""
        results_dir = getattr(self, 'results_dir', 'results')
        timestamp = getattr(self, 'timestamp', get_file_timestamp())
        prefix = filename_prefix or self.config.scheduling_strategy
        return os.path.join(results_dir, f"{kind}_{prefix}_{timestamp}.gif")
    
    def _generate_state_positions(self, num_states: int) -> Dict[int, Tuple[float, float]]:
        """状態を円形に配置した2D座標を生成する"""
        positions = {}
        if num_states == 1:
            positions[0] = (0, 0)
        else:
            for i in range(num_states):
                angle = 2 * np.pi * i / num_states
                x = 1.8 * np.cos(angle)
                y = 1.8 * np.sin(angle)
                positions[i] = (x, y)
        return positions
    
    def _draw_transition_edges(self, ax, state_positions: Dict, transition_matrix: np.ndarray):
        """遷移行列に基づいてエッジを描画する"""
        num_states = len(state_positions)
        
        for i in range(num_states):
            for j in range(num_states):
                if i != j and transition_matrix[i, j] > 0.01:  # 閾値以上の遷移のみ描画
                    x1, y1 = state_positions[i]
                    x2, y2 = state_positions[j]
                    
                    # 矢印を描画（薄いグレー）
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', color='lightgray', 
                                            alpha=0.5, linewidth=1))
