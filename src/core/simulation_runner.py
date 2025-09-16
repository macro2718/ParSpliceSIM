"""シミュレーション実行管理クラス"""
from typing import Dict, List, Any
import numpy as np
from src.runtime.producer import Producer
from src.runtime.splicer import Splicer
from src.scheduling.scheduler import Scheduler
from src.config import SimulationConfig


class SimulationRunner:
    """シミュレーション実行を管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        # 超最小出力モードでは不要な情報は保持しない
        stream_only = getattr(config, 'stream_trajectory_only', False)
        self.trajectory_lengths = [] if not stream_only else None
        self.trajectory_states = [] if not stream_only else None
        self.total_values = [] if not stream_only else None
        self.step_logs = [] if not stream_only else None
        # 必要に応じて ParSpliceSimulation から注入される
        self.length_streamer = None
    
    def run_producer_one_step(self, producer: Producer, splicer: Splicer, 
                            scheduler: Scheduler, available_states: List[int], current_step: int = 0) -> List[int]:
        """
        アルゴリズム概要.txtに基づくProducerの1ステップを実行する
        """
        step_log = self._initialize_step_log(current_step) if not getattr(self.config, 'stream_trajectory_only', False) else {}
        
        # 操作1-1: splicer処理
        self._process_splicer_step(splicer, producer, step_log)
        
        # available_statesの更新
        available_states = self._update_available_states(producer, available_states)
        
        # 操作1-2: scheduler処理（ワーカー移動・設定を含む）
        self._process_scheduler_step(scheduler, producer, splicer, available_states, step_log)
        
        # 操作4: 全てのParRepBoxを1ステップ進める
        self._process_producer_step(producer, step_log)
        
        # trajectoryと詳細情報の記録
        self._record_trajectory_and_details(splicer, producer, step_log, current_step)
        
        if self.step_logs is not None:
            self.step_logs.append(step_log)
        return available_states
    
    def _initialize_step_log(self, current_step: int) -> Dict:
        """ステップログを初期化する"""
        return {
            'step': current_step + 1,
            'splicer_result': '',
            'scheduler_result': '',
            'trajectory_length': 0,
            'final_state': None,
            'segments_collected': 0,
            'parrepbox_details': []  # ParRepBox詳細情報
        }
    
    def _process_splicer_step(self, splicer: Splicer, producer: Producer, step_log: Dict) -> None:
        """Splicerの処理を実行し、結果をログに記録する"""
        splicer_result = self.run_splicer_one_step(splicer, producer)
        
        # splicer処理後にsegmentStoreをクリア（使用済みセグメントを削除）
        producer.clear_segment_store()
        
        # splicer結果をログに記録
        if splicer_result['status'] == 'success':
            load_result = splicer_result['load_result']
            splice_result = splicer_result['splice_result']
            if load_result['loaded_segments'] > 0 or splice_result['spliced_segments'] > 0:
                step_log['splicer_result'] = f"{load_result['loaded_segments']}読込, {splice_result['spliced_segments']}結合"
            else:
                step_log['splicer_result'] = "処理なし"
        else:
            step_log['splicer_result'] = "エラー"
    
    def _update_available_states(self, producer: Producer, available_states: List[int]) -> List[int]:
        """producerが新たな状態に到達した場合available_statesを更新する"""
        observed_transition_matrix = producer.get_observed_transition_statistics()["observed_transition_matrix"]
        try:
            new_targets = np.where(observed_transition_matrix.sum(axis=0) >= 1)[0].tolist()
        except Exception:
            # フォールバック（安全のため）
            new_targets = []
            num_states = observed_transition_matrix.shape[0]
            for j in range(num_states):
                for i in range(num_states):
                    if observed_transition_matrix[i][j] >= 1:
                        new_targets.append(j)
                        break
        
        for j in new_targets:
            if j not in available_states:
                available_states.append(j)
                if not self.config.minimal_output:
                    print(f"新しい状態 {j} をavailable_statesに追加: {available_states}")
        
        return available_states
    
    def _process_scheduler_step(self, scheduler: Scheduler, producer: Producer, 
                               splicer: Splicer, available_states: List[int], step_log: Dict) -> None:
        """Schedulerの処理を実行し、結果をログに記録する"""
        scheduler_result = self.run_scheduler_one_step(scheduler, producer, splicer, available_states)
        
        # scheduler結果をログに記録
        if scheduler_result['status'] == 'success':
            scheduling_result = scheduler_result.get('scheduling_result', {})
            worker_moves = len(scheduling_result.get('worker_moves', []))
            new_groups = len(scheduling_result.get('new_groups_config', []))
            if worker_moves > 0 or new_groups > 0:
                step_log['scheduler_result'] = f"{worker_moves}移動, {new_groups}新規グループ"
            else:
                step_log['scheduler_result'] = "処理なし"
        else:
            step_log['scheduler_result'] = "エラー"
    
    def _process_producer_step(self, producer: Producer, step_log: Dict) -> None:
        """Producerのステップ処理と完了したセグメントの収集を行う"""
        # 全てのParRepBoxを1ステップ進める
        step_result = producer.step_all_groups()
        state_dist = step_result['state_distribution']
        
        # 各ワーカーの所属情報を取得
        worker_assignments = producer.format_worker_assignments()
        worker_info = ", ".join([f"W{wid}:{assignment.replace('グループ', 'G').replace('未配置', 'unassigned')}" 
                                for wid, assignment in sorted(worker_assignments.items())])
        
        # 最小限出力モードでない場合のみ詳細情報を表示
        if not self.config.minimal_output:
            print(f"ステップ: idle={state_dist['idle']}, parallel={state_dist['parallel']}, decorr={state_dist['decorrelating']}, finished={state_dist['finished']} | {worker_info}")
        
        # finishedになったParRepBoxからsegmentを収集
        self._collect_finished_segments(producer, step_log)
    
    def _collect_finished_segments(self, producer: Producer, step_log: Dict) -> None:
        """完了したParRepBoxからセグメントを収集する"""
        finished_groups = [group_id for group_id in producer.get_all_group_ids() 
                          if producer.get_group_info(group_id)['group_state'] == 'finished']
        
        if finished_groups:
            collect_result = producer.collect_finished_segments()
            step_log['segments_collected'] = collect_result['collected_count']
        else:
            step_log['segments_collected'] = 0
    
    def _record_trajectory_and_details(self, splicer: Splicer, producer: Producer, 
                                     step_log: Dict, current_step: int) -> None:
        """trajectory情報とParRepBox詳細情報を記録する"""
        # trajectory長を記録（配列長から1を引いた値）
        current_trajectory_length = max(0, splicer.get_trajectory_length() - 1)
        current_final_state = splicer.get_final_state()
        if self.trajectory_lengths is not None:
            self.trajectory_lengths.append(current_trajectory_length)
        
        # 超最小出力モードでは状態遷移履歴は取得・保持しない
        if self.trajectory_states is not None:
            trajectory_states = splicer.get_trajectory_states()  # 全状態遷移履歴を取得
            self.trajectory_states.append(trajectory_states.copy() if trajectory_states else [])
        
        # ログに記録（保持しないモードでは何もしない）
        if step_log is not None:
            step_log['trajectory_length'] = current_trajectory_length
            step_log['final_state'] = current_final_state
        
        # ParRepBox詳細情報を収集（ストリーム専用モードでは収集しない）
        if self.step_logs is not None:
            self._collect_parrepbox_details(producer, step_log)
        
        # 出力処理
        self._handle_output(current_trajectory_length, current_final_state, current_step)
    
    def _collect_parrepbox_details(self, producer: Producer, step_log: Dict) -> None:
        """ParRepBoxの詳細情報を収集する"""
        # 念のためキーを初期化
        step_log.setdefault('parrepbox_details', [])
        for group_id in producer.get_all_group_ids():
            group_info = producer.get_group_info(group_id)
            group_state = group_info['group_state']
            
            # グループの初期状態を取得
            try:
                group = producer.get_group(group_id)
                initial_state = group.get_initial_state()
                if initial_state is None:
                    initial_state = "未設定"
            except Exception:
                initial_state = "不明"
            
            # ワーカー詳細を収集
            worker_details = self._collect_worker_details(producer, group_info['worker_ids'])
            worker_str = ", ".join(worker_details) if worker_details else "なし"
            
            step_log['parrepbox_details'].append({
                'group_id': group_id,
                'state': group_state,
                'initial_state': initial_state,
                'workers': worker_str
            })
    
    def _collect_worker_details(self, producer: Producer, worker_ids: List[int]) -> List[str]:
        """ワーカーの詳細情報を収集する（Producerの共通フォーマッタを使用）"""
        return producer.format_worker_phases(worker_ids)
    
    def _handle_output(self, trajectory_length: int, final_state: Any, current_step: int) -> None:
        """出力処理を行う"""
        # 超最小出力モード: ファイルに長さのみストリーミング書き込み + 従来のminimal出力をターミナル表示
        if getattr(self.config, 'stream_trajectory_only', False):
            if getattr(self, 'length_streamer', None) is not None:
                self.length_streamer.append_length(trajectory_length)
            # ターミナルには最小出力（従来と同形式）を表示
            final_step_index = self.config.max_simulation_time - 1
            should_emit = ((current_step + 1) % max(1, self.config.output_interval) == 0) or (current_step == final_step_index)
            if should_emit:
                print(f"Step {current_step + 1}: Trajectory Length {trajectory_length}, Current State {final_state}")
                if current_step == final_step_index:
                    print(f"最終状態: {final_state}")
            return

        # 詳細表示（verbose時）
        if not self.config.minimal_output:
            print(f"Trajectory: 長さ={trajectory_length}, 最終状態={final_state}")
            return

        # 最小限出力: 出力間隔ごと + 最終ステップのみ
        final_step_index = self.config.max_simulation_time - 1
        should_emit = ((current_step + 1) % max(1, self.config.output_interval) == 0) or (current_step == final_step_index)
        if should_emit:
            print(f"Step {current_step + 1}: Trajectory Length {trajectory_length}, Current State {final_state}")
            if current_step == final_step_index:
                print(f"最終状態: {final_state}")

    def run_splicer_one_step(self, splicer: Splicer, producer: Producer) -> Any:
        """Splicerの1ステップを実行する"""
        result = splicer.run_one_step(producer)
        
        # 結果の表示（最小限出力モードでない場合のみ）
        if not self.config.minimal_output and result['status'] == 'success':
            load_result = result['load_result']
            splice_result = result['splice_result']
            if load_result['loaded_segments'] > 0 or splice_result['spliced_segments'] > 0:
                print(f"Splicer: {load_result['loaded_segments']}読込, {splice_result['spliced_segments']}結合")
        
        return result

    def run_scheduler_one_step(self, scheduler: Scheduler, producer: Producer, splicer: Splicer, available_states: List[int]) -> Any:
        """Schedulerの1ステップを実行する (available_statesをknown_statesとして渡す)"""
        result = scheduler.run_one_step(producer, splicer, set(available_states))
        
        # スケジューリング戦略のtotal_valueを収集（基底クラスでデフォルト化）
        total_value_per_worker = scheduler.scheduling_strategy.total_value / self.config.num_workers
        if self.total_values is not None:
            self.total_values.append(total_value_per_worker)
        
        # スケジューリング結果に基づいてworkerの再配置を実行
        if result['status'] == 'success':
            scheduling_result = result.get('scheduling_result', {})
            worker_moves = scheduling_result.get('worker_moves', [])
            new_groups_config = scheduling_result.get('new_groups_config', [])
            
            # workerの移動を実行
            producer.execute_worker_moves_with_validation(worker_moves)
            
            # 新規ParRepBoxの設定を実行
            producer.configure_new_groups(new_groups_config)
            
            # 結果の表示（最小限出力モードでない場合のみ）
            if not self.config.minimal_output and (worker_moves or new_groups_config):
                print(f"Scheduler: {len(worker_moves)}移動, {len(new_groups_config)}新規グループ")
        
        return result
