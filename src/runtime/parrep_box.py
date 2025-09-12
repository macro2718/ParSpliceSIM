from .worker import worker
from typing import List, Dict, Any, Optional, Tuple
import random
import sys
from enum import Enum
from common import default_logger

class ParRepBoxState(Enum):
    """ParRepBoxの状態定数定義"""
    IDLE = "idle"
    PARALLEL = "parallel"
    DECORRELATING = "decorrelating"
    FINISHED = "finished"

class ParRepBox:
    """
    workerからなる集合を管理するクラス
    
    固有の初期状態を持ち、属するワーカーは同じ初期状態を持つもののみ追加可能。
    idle, parallel, decorrelating, finishedの4つの状態を持ち、最初はidle状態。
    ワーカーが追加されるとparallel状態となり、遷移が発生するとdecorrelating状態となる。
    parallel状態またはdecorrelating状態でワーカーが0になるとfinished状態となる。
    延べステップ数を記録し、最終的に初期状態の配列と計算結果を結合した配列を出力する。
    """
    
    # クラス変数でIDのカウンターを管理
    _next_id = 0
    
    def __init__(self, box_id: Optional[int] = None, initial_state: Optional[int] = None, max_time: Optional[int] = None, minimal_output: bool = False):
        """
        ParRepBoxクラスの初期化
        
        Parameters:
        box_id (Optional[int]): ParRepBoxの固有ID（省略可能、自動割り当て）
        initial_state (Optional[int]): グループ固有の初期状態（省略可能、後からset_initial_stateで設定）
        max_time (Optional[int]): 最大実行時間（省略可能、設定された場合は強制終了機能が有効）
        minimal_output (bool): 最小限の出力モード（Trueの場合、詳細なログを抑制）
        """
        # IDの設定（指定されない場合は自動割り当て）
        if box_id is not None:
            self.box_id = box_id
            # 指定されたIDが次のIDより大きい場合、カウンターを更新
            if box_id >= ParRepBox._next_id:
                ParRepBox._next_id = box_id + 1
        else:
            self.box_id = ParRepBox._next_id
            ParRepBox._next_id += 1
        
        self._workers: Dict[int, worker] = {}  # worker_id -> worker instance
        self.initial_state = initial_state
        self.group_state = ParRepBoxState.IDLE
        self.total_steps = 0  # ワーカーが削除された際に加算される延べステップ数
        self.final_segment = None
        self.step_stats: Dict[int, int] = {}  # 状態 -> その状態に遷移した回数
        self.transition_stats: Dict[Tuple[int, int], int] = {}  # (from_state, to_state) -> 遷移回数
        self.simulation_steps = 0  # シミュレーションした延べのステップ数
        self.max_time = max_time  # 最大実行時間
        self.minimal_output = minimal_output  # 最小限の出力モード
        self._default_producer_callback = None  # デフォルトのProducerコールバック
        self.segment_id = None  # このParRepBoxに割り当てられたセグメントID
        
    # ========================
    # メインの処理メソッド
    # ========================
    
    def set_initial_state(self, initial_state: int) -> None:
        """
        グループの初期状態を設定する
        
        Parameters:
        initial_state (int): グループの初期状態
        
        Raises:
        ValueError: ワーカーが既に追加されている場合
        """
        if len(self._workers) > 0:
            raise ValueError("ワーカーが既に追加されているため、初期状態を変更できません")
        
        self.initial_state = initial_state
    
    def set_default_producer_callback(self, callback):
        """
        デフォルトのProducerコールバックを設定する
        
        Parameters:
        callback (callable): ワーカー削除時に呼び出すコールバック関数
        """
        self._default_producer_callback = callback
    
    def set_max_time(self, max_time: Optional[int]) -> None:
        """
        最大実行時間を設定する
        
        Parameters:
        max_time (Optional[int]): 最大実行時間（Noneの場合は無制限）
        
        Raises:
        ValueError: 負の値が指定された場合
        """
        if max_time is not None and max_time < 0:
            raise ValueError("最大実行時間は0以上の値である必要があります")
        self.max_time = max_time
    
    def set_segment_id(self, segment_id: int) -> None:
        """
        セグメントIDを設定する
        
        Parameters:
        segment_id (int): このParRepBoxが作成するセグメントのID
        """
        self.segment_id = segment_id
    
    def get_segment_id(self) -> Optional[int]:
        """
        セグメントIDを取得する
        
        Returns:
        Optional[int]: セグメントID（設定されていない場合はNone）
        """
        return self.segment_id
    
    def step(self) -> Dict[str, Any]:
        """
        グループの状態に応じて1ステップ進める
        
        Returns:
        Dict[str, Any]: ステップ結果の詳細情報
        """
        if self.group_state == ParRepBoxState.IDLE:
            return {'status': 'idle', 'group_state': self.group_state.value, 'message': 'ワーカーが追加されていません'}
        elif self.group_state == ParRepBoxState.PARALLEL:
            return self._step_parallel()
        elif self.group_state == ParRepBoxState.DECORRELATING:
            return self._step_decorrelating()
        elif self.group_state == ParRepBoxState.FINISHED:
            return {'status': 'finished', 'group_state': self.group_state.value, 'message': '計算が完了しています'}
        else:
            raise ValueError(f"不正なグループ状態: {self.group_state}")
            
    def _step_parallel(self) -> Dict[str, Any]:
        """
        parallel状態でのステップ処理
        
        Returns:
        Dict[str, Any]: ステップ結果
        """
        worker_step_results = {}
        workers_with_transition = []
        
        # ステップごとにstep_statsとtransition_statsを初期化
        self.step_stats = {}
        self.transition_stats = {}
        
        # アクティブなワーカーのIDを取得
        active_workers = self.get_active_workers()
        
        if not active_workers:
            return {'status': 'no_active_workers', 'group_state': self.group_state.value}
        
        # 各アクティブワーカーを1ステップ進める
        for worker_id in active_workers:
            try:
                # ステップ実行前の状態を記録
                previous_state = self._workers[worker_id].get_state()
                
                step_result = self._workers[worker_id].step()
                worker_step_results[worker_id] = step_result
                
                # simulation_stepsを1加算（デフェージング中は除外）
                if self._workers[worker_id].get_current_phase() != 'dephasing':
                    self.simulation_steps += 1
                
                # step_statsを更新（step_resultの値をキーとして回数をカウント）
                if step_result is not None:
                    if step_result in self.step_stats:
                        self.step_stats[step_result] += 1
                    else:
                        self.step_stats[step_result] = 1
                
                # 遷移統計を更新（前の状態から現在の状態への遷移）
                if step_result is not None and previous_state is not None:
                    transition_pair = (previous_state, step_result)
                    if transition_pair in self.transition_stats:
                        self.transition_stats[transition_pair] += 1
                    else:
                        self.transition_stats[transition_pair] = 1

                # 遷移が発生したワーカーをチェック
                if self._workers[worker_id].get_transition_occurred():
                    workers_with_transition.append(worker_id)
                    
            except ValueError:
                # ステップ実行中にエラーが発生した場合
                worker_step_results[worker_id] = None
                
        # 遷移が発生したワーカーがいる場合
        if workers_with_transition:
            # 1つのワーカーを無作為に選択
            selected_worker_id = random.choice(workers_with_transition)
            worker_ids_to_remove = [worker_id for worker_id in active_workers if worker_id != selected_worker_id]
            
            # 他のワーカーを停止・削除
            stopped_workers_info = []
            for worker_id in worker_ids_to_remove:
                try:
                    # ワーカーを停止（デフォルトコールバックを使用してProducerに通知）
                    # 遷移処理でもワーカーを未配置リストに戻すため、コールバックを呼び出す
                    stop_result = self.stop_worker(worker_id, removal_type='transition_occurred')
                    stopped_workers_info.append(stop_result.get('stopped_worker', {'worker_id': worker_id}))
                    
                except (ValueError, KeyError):
                    # 既に停止済みまたはワーカーが存在しない場合は何もしない
                    pass
                    
            # 状態をdecorrelatingに変更
            self.group_state = ParRepBoxState.DECORRELATING
            
            return {
                'status': 'transition_to_decorrelating',
                'group_state': self.group_state.value,
                'selected_worker': selected_worker_id,
                'removed_workers': stopped_workers_info,
                'total_steps': self.total_steps,
                'step_results': worker_step_results
            }
        else:
            # 遷移が発生しなかった場合、最大時間をチェック
            if self.max_time is not None and self.simulation_steps >= self.max_time:
                # 最大時間に達した場合は強制終了
                if not self.minimal_output:
                    default_logger.warning(f"Group {self.box_id}: max_time({self.max_time})に到達のため強制終了 (simulation_steps={self.simulation_steps})")
                terminate_result = self.terminate_all_workers()
                
                # 最大時間による強制終了の情報を追加
                terminate_result['status'] = 'max_time_force_terminated'
                terminate_result['simulation_steps'] = self.simulation_steps
                terminate_result['max_time'] = self.max_time
                terminate_result['step_results'] = worker_step_results
                
                return terminate_result
            else:
                return {
                    'status': 'parallel_continue',
                    'group_state': self.group_state.value,
                    'step_results': worker_step_results
                }
            
    def _step_decorrelating(self) -> Dict[str, Any]:
        """
        decorrelating状態でのステップ処理
        
        Returns:
        Dict[str, Any]: ステップ結果
        """
        active_workers = self.get_active_workers()
        
        if len(active_workers) != 1:
            raise ValueError(f"decorrelating状態では1つのワーカーのみがアクティブである必要があります。現在: {len(active_workers)}")
            
        worker_id = active_workers[0]
        worker_instance = self._workers[worker_id]
        
        # ワーカーが既にデコリレートしている場合は計算終了
        if worker_instance.get_is_decorrelated():
            # stop_workerを用いて計算終了処理を実行
            stop_result = self.stop_worker(worker_id, removal_type='decorrelating_completed')
            
            return {
                'status': 'computation_completed',
                'group_state': stop_result['group_state'],
                'final_segment': stop_result['final_segment'],
                'total_steps': stop_result['total_steps']
            }
        else:
            # まだデコリレートしていない場合は1ステップ進める
            try:
                # ステップ実行前の状態を記録
                previous_state = worker_instance.get_state()
                
                step_result = worker_instance.step()
                
                # step_statsを更新（decorrelating状態でも遷移情報を記録）
                if step_result is not None:
                    if step_result in self.step_stats:
                        self.step_stats[step_result] += 1
                    else:
                        self.step_stats[step_result] = 1
                
                # 遷移統計を更新（decorrelating状態でも）
                if step_result is not None and previous_state is not None:
                    transition_pair = (previous_state, step_result)
                    if transition_pair in self.transition_stats:
                        self.transition_stats[transition_pair] += 1
                    else:
                        self.transition_stats[transition_pair] = 1
                
                return {
                    'status': 'decorrelating_continue',
                    'group_state': self.group_state.value,
                    'step_result': step_result,
                    'is_decorrelated': worker_instance.get_is_decorrelated()
                }
            except ValueError:
                return {
                    'status': 'decorrelating_error',
                    'group_state': self.group_state.value,
                    'error': 'ワーカーのステップ実行でエラーが発生しました'
                }
    
    def _check_state_for_operation(self, operation_name: str = "操作") -> None:
        """
        操作実行前の状態チェック（共通処理）
        
        Parameters:
        operation_name (str): 実行する操作の名前（エラーメッセージ用）
        
        Raises:
        ValueError: idle状態またはfinished状態の場合
        """
        if self.group_state == ParRepBoxState.IDLE:
            raise ValueError(f"idle状態では{operation_name}は実行できません")
        elif self.group_state == ParRepBoxState.FINISHED:
            raise ValueError(f"finished状態では{operation_name}は実行できません")
    
    def stop_worker(self, worker_id: int, producer_callback=None, removal_type='unknown') -> Dict[str, Any]:
        """
        ワーカーを停止し、セグメントを収集する
        
        Parameters:
        worker_id (str): 停止するワーカーのID
        producer_callback (callable, optional): Producer側で削除処理時に呼び出すコールバック関数
        
        Returns:
        Dict[str, Any]: 停止結果（final_segment含む）
        """
        if worker_id not in self._workers:
            raise ValueError(f"ワーカーID {worker_id} は存在しません")
            
        try:
            segment = self._workers[worker_id].get_segment()
            
            if len(self._workers) == 1:
                self.final_segment = [self.initial_state] * self.total_steps + segment

            steps = len(segment) - 1 if len(segment) > 0 else 0
            self.total_steps += steps
            
            # ワーカーを内部リストから削除
            stopped_worker = self._workers.pop(worker_id)
            
            # Producer側にワーカー削除を通知
            # コールバックが明示的にNoneでない限り、デフォルトコールバックを使用
            callback_to_use = producer_callback if producer_callback is not None else self._default_producer_callback
            
            if callback_to_use is not None:
                try:
                    callback_to_use(worker_id, self.box_id, removal_type)
                except Exception as e:
                    print(f"警告: Producer側のワーカー削除コールバックでエラーが発生しました: {e}", file=sys.stderr)
                    
            # ワーカーが0になった場合はfinished状態に遷移
            if len(self._workers) == 0:
                self.group_state = ParRepBoxState.FINISHED
                
            return {
                'stopped_worker': {
                    'worker_id': worker_id,
                    'phase': stopped_worker.get_current_phase(),
                    'total_steps': total_steps
                },
                'segment': segment,
                'final_segment': self.final_segment,
                'total_steps': self.total_steps,
                'group_state': self.group_state.value,
                'remaining_workers': len(self._workers)
            }
            
        except Exception as e:
            raise ValueError(f"ワーカー {worker_id} の停止処理中にエラーが発生しました: {e}")
            
    def terminate_all_workers(self) -> Dict[str, Any]:
        """
        任意の段階で全てのワーカーの計算を終了させる
        
        Returns:
        Dict[str, Any]: 終了処理の結果
        
        Raises:
        ValueError: idle状態またはfinished状態の場合
        """
        # 状態チェック（共通処理を使用）
        self._check_state_for_operation("終了処理")
        
        # 全てのワーカーを停止
        terminated_workers_info = []
        worker_ids_to_terminate = list(self._workers.keys())
        
        for worker_id in worker_ids_to_terminate:
            try:
                # 全終了処理では、Producerコンテキスト外の呼び出しが想定されるため、コールバックなしで呼び出し
                stop_result = self.stop_worker(worker_id, producer_callback=None, removal_type='force_terminated')
                terminated_workers_info.append(stop_result.get('stopped_worker', {'worker_id': worker_id}))
            except (ValueError, KeyError):
                # エラーが発生した場合もワーカー情報を記録
                worker_info = {
                    'worker_id': worker_id,
                    'steps_elapsed': 0
                }
                terminated_workers_info.append(worker_info)
        
        return {
            'status': 'all_workers_terminated',
            'group_state': self.group_state.value,
            'terminated_workers': terminated_workers_info,
            'total_steps': self.total_steps,
            'final_segment': self.final_segment
        }
            
    # ========================
    # ワーカー管理メソッド
    # ========================
    
    def add_worker(self, worker_instance: worker) -> int:
        """
        workerをグループに追加する（初期状態がグループの初期状態と一致するもののみ）
        IDLE状態またはPARALLEL状態でのみ追加可能
        
        Parameters:
        worker_instance (worker): 追加するworkerインスタンス
        
        Returns:
        int: 追加されたworkerのID
        
        Raises:
        ValueError: 初期状態が一致しない場合、またはIDLE/PARALLEL状態でない場合、またはIDが重複している場合
        TypeError: worker型でない場合
        """
        if self.group_state not in [ParRepBoxState.IDLE, ParRepBoxState.PARALLEL]:
            raise ValueError(f"ワーカーの追加はIDLE状態またはPARALLEL状態でのみ可能です。現在の状態: {self.group_state.value}")
            
        if not isinstance(worker_instance, worker):
            raise TypeError("worker型のインスタンスである必要があります")
        
        # グループの初期状態が未設定の場合、ワーカーの初期状態を採用
        if self.initial_state is None:
            self.initial_state = worker_instance.initial_state
        elif worker_instance.initial_state != self.initial_state:
            raise ValueError(f"ワーカーの初期状態 {worker_instance.initial_state} がグループの初期状態 {self.initial_state} と一致しません")
            
        worker_id = worker_instance.get_id()
        if worker_id in self._workers:
            raise ValueError(f"ID {worker_id} のワーカーは既に存在します")
            
        self._workers[worker_id] = worker_instance
        
        # ワーカーが追加されたらPARALLEL状態に変更
        if self.group_state == ParRepBoxState.IDLE:
            self.group_state = ParRepBoxState.PARALLEL
            
        return worker_id
        
    def get_worker(self, worker_id: int) -> worker:
        """
        指定されたIDのworkerを取得する
        
        Parameters:
        worker_id (int): 取得するworkerのID
        
        Returns:
        worker: 指定されたworkerインスタンス
        
        Raises:
        KeyError: 指定されたIDのworkerが存在しない場合
        """
        return self._get_worker_or_raise(worker_id)
        
    def _get_worker_or_raise(self, worker_id: int) -> worker:
        """
        ワーカーを取得する、存在しない場合は例外を発生させる
        
        Parameters:
        worker_id (int): ワーカーのID
        
        Returns:
        worker: ワーカーインスタンス
        
        Raises:
        KeyError: 指定されたIDのワーカーが存在しない場合
        """
        if worker_id not in self._workers:
            raise KeyError(f"ID {worker_id} のワーカーは存在しません")
        return self._workers[worker_id]
        
    def get_idle_workers(self) -> List[int]:
        """
        待機状態のworkerのIDリストを取得する
        
        Returns:
        List[int]: 待機状態のworkerのIDリスト
        """
        return [worker_id for worker_id, worker_instance in self._workers.items() 
                if worker_instance.get_is_idle()]
                
    def get_active_workers(self) -> List[int]:
        """
        実行状態のworkerのIDリストを取得する
        
        Returns:
        List[int]: 実行状態のworkerのIDリスト
        """
        return [worker_id for worker_id, worker_instance in self._workers.items() 
                if not worker_instance.get_is_idle()]
                
    def set_worker_initial_state(self, worker_id: int, initial_state: int, 
                                t_phase: int, t_corr: int) -> None:
        """
        指定されたworkerに初期状態を設定する
        
        Parameters:
        worker_id (int): ワーカーのID
        initial_state (int): 初期状態
        t_phase (int): フェーズ時間
        t_corr (int): 相関時間
        
        Raises:
        KeyError: 指定されたIDのワーカーが存在しない場合
        """
        worker_instance = self._get_worker_or_raise(worker_id)
        worker_instance.set_initial_state(initial_state, t_phase, t_corr)
            
    # ========================
    # 情報取得メソッド
    # ========================
    
    def get_box_id(self) -> int:
        """ParRepBoxの固有IDを取得"""
        return self.box_id
    
    def get_group_state(self) -> str:
        """現在のグループ状態を取得"""
        return self.group_state.value
    
    def get_initial_state(self) -> Optional[int]:
        """グループの初期状態を取得"""
        return self.initial_state
        
    def get_total_steps(self) -> int:
        """延べステップ数を取得"""
        return self.total_steps
        
    def get_final_segment(self) -> Optional[List[int]]:
        """最終結果を取得"""
        return self.final_segment
    
    def get_final_segment_with_id(self) -> Optional[Tuple[List[int], int]]:
        """
        最終結果とセグメントIDを同時に取得
        
        Returns:
        Optional[Tuple[List[int], int]]: (final_segment, segment_id) または None
        """
        if self.final_segment is not None and self.segment_id is not None:
            return (self.final_segment, self.segment_id)
        else:
            return None
        
    def get_step_stats(self) -> Dict[int, int]:
        """状態遷移の統計情報を取得"""
        return self.step_stats.copy()
    
    def get_transition_stats(self) -> Dict[Tuple[int, int], int]:
        """状態遷移ペアの統計情報を取得"""
        return self.transition_stats.copy()
        
    def get_simulation_steps(self) -> int:
        """シミュレーションした延べのステップ数を取得"""
        return self.simulation_steps
        
    def get_max_time(self) -> Optional[int]:
        """最大実行時間を取得"""
        return self.max_time
    
    def get_remaining_time(self) -> Optional[int]:
        """
        残りの実行時間を取得
        
        Returns:
        Optional[int]: 残りの実行時間（max_timeが設定されていない場合はNone）
        """
        if self.max_time is None:
            return None
        return max(0, self.max_time - self.simulation_steps)
        
    def is_computation_complete(self) -> bool:
        """計算が完了しているかどうかを確認"""
        return self.final_segment is not None
        
    def get_worker_count(self) -> int:
        """グループ内のworker数を取得"""
        return len(self._workers)
        
    def get_worker_ids(self) -> List[int]:
        """グループ内の全workerのIDリストを取得"""
        return list(self._workers.keys())
        
    def get_worker_initial_state(self, worker_id: int) -> Optional[int]:
        """指定されたworkerの初期状態を取得"""
        return self._get_worker_or_raise(worker_id).initial_state
        
    def get_worker_transition_occurred(self, worker_id: int) -> bool:
        """指定されたworkerで遷移が発生したかどうかを取得"""
        return self._get_worker_or_raise(worker_id).get_transition_occurred()
        
    def get_worker_steps_elapsed(self, worker_id: int) -> int:
        """指定されたworkerが何ステップ分シミュレーションしたかを取得"""
        return self._get_worker_or_raise(worker_id).get_steps_elapsed()
        
    def get_worker_info(self, worker_id: int) -> Dict[str, Any]:
        """
        指定されたworkerの詳細情報を取得する
        
        Parameters:
        worker_id (int): workerのID
        
        Returns:
        Dict[str, Any]: workerの詳細情報
        
        Raises:
        KeyError: 指定されたIDのworkerが存在しない場合
        """
        # 重複した存在チェックを削除し、共通処理を使用
        worker_instance = self._get_worker_or_raise(worker_id)
        
        return {
            'worker_id': worker_id,
            'initial_state': worker_instance.initial_state,
            'current_state': worker_instance.get_state(),
            'is_idle': worker_instance.get_is_idle(),
            'transition_occurred': worker_instance.get_transition_occurred(),
            'steps_elapsed': worker_instance.get_steps_elapsed(),
            'current_phase': worker_instance.get_current_phase(),
            'is_decorrelated': worker_instance.get_is_decorrelated(),
            'time_parameters': worker_instance.get_time_parameters(),
            'remaining_times': worker_instance.get_remaining_times()
        }
        
    def get_all_workers_info(self) -> Dict[int, Dict[str, Any]]:
        """
        すべてのworkerの詳細情報を取得する
        
        Returns:
        Dict[int, Dict[str, Any]]: worker_id -> 詳細情報のマッピング
        """
        return {worker_id: self.get_worker_info(worker_id) for worker_id in self._workers.keys()}
        
    # ========================
    # ユーティリティメソッド
    # ========================
    
    @classmethod
    def reset_id_counter(cls) -> None:
        """
        IDカウンターをリセットする（テスト用）
        """
        cls._next_id = 0
    
    @classmethod
    def get_next_id(cls) -> int:
        """
        次に割り当てられるIDを取得（参照用）
        """
        return cls._next_id
    
    def reset_group(self) -> None:
        """
        グループをリセットする
        """
        self.group_state = ParRepBoxState.IDLE
        self.initial_state = None  # 初期状態もリセット
        self.total_steps = 0
        self.final_segment = None
        self.step_stats = {}
        self.transition_stats = {}  # 遷移統計もリセット
        self.simulation_steps = 0
        # すべてのワーカーをリセット
        for worker_instance in self._workers.values():
            worker_instance.reset()
        
    def __len__(self) -> int:
        """
        グループ内のworker数を返す
        
        Returns:
        int: worker数
        """
        return len(self._workers)
        
    def __getitem__(self, worker_id: int) -> worker:
        """
        IDによるworkerアクセス
        
        Parameters:
        worker_id (int): workerのID
        
        Returns:
        worker: 指定されたworkerインスタンス
        """
        return self.get_worker(worker_id)
        
    def __str__(self) -> str:
        """
        文字列表現
        
        Returns:
        str: ParRepBoxの文字列表現
        """
        return (f"ParRepBox(id={self.box_id}, "
                f"initial_state={self.initial_state}, "
                f"workers={len(self._workers)}, "
                f"state={self.group_state.value}, "
                f"total_steps={self.total_steps}, "
                f"max_time={self.max_time}, "
                f"completed={self.is_computation_complete()})")
                
    def __repr__(self) -> str:
        """
        オブジェクトの表現
        
        Returns:
        str: ParRepBoxのオブジェクト表現
        """
        return self.__str__()


if __name__ == "__main__":
    import sys
    import numpy as np
    
    def create_test_workers(transition_matrix, initial_state=0, t_phase=5, t_corr=3):
        """テスト用ワーカーを作成するヘルパー関数"""
        worker1 = worker(worker_id=1, transition_matrix=transition_matrix)
        worker1.set_initial_state(initial_state, t_phase, t_corr)
        worker2 = worker(worker_id=2, transition_matrix=transition_matrix)
        worker2.set_initial_state(initial_state, t_phase, t_corr)
        return worker1, worker2
    
    def test_basic_functionality():
        """基本機能テスト"""
        print("=== 基本機能テスト ===")
        ParRepBox.reset_id_counter()
        
        # ParRepBox作成とID確認
        group = ParRepBox()
        assert group.get_group_state() == "idle"
        assert group.get_box_id() == 0
        
        # ワーカー追加とparallel状態への遷移
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        worker1, worker2 = create_test_workers(transition_matrix)
        
        group.set_initial_state(0)
        group.add_worker(worker1)
        group.add_worker(worker2)
        
        assert group.get_group_state() == "parallel"
        assert group.get_worker_count() == 2
        print("基本機能テスト: 成功\n")
    
    def test_step_execution():
        """ステップ実行テスト"""
        print("=== ステップ実行テスト ===")
        ParRepBox.reset_id_counter()
        
        group = ParRepBox()
        transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        worker1, worker2 = create_test_workers(transition_matrix, t_phase=2, t_corr=2)
        
        group.add_worker(worker1)
        group.add_worker(worker2)
        
        # ステップ実行
        for i in range(10):
            step_result = group.step()
            if step_result['status'] in ['finished', 'computation_completed']:
                print(f"完了: {group.get_final_segment()}")
                break
            elif step_result['status'] == 'transition_to_decorrelating':
                print("decorrelating状態に移行")
        
        print("ステップ実行テスト: 成功\n")
    
    def test_id_functionality():
        """ID機能テスト"""
        print("=== ID機能テスト ===")
        ParRepBox.reset_id_counter()
        
        # 自動ID割り当て
        box1 = ParRepBox()
        box2 = ParRepBox()
        assert box1.get_box_id() == 0
        assert box2.get_box_id() == 1
        
        # 手動ID指定
        box3 = ParRepBox(box_id=10)
        assert box3.get_box_id() == 10
        assert ParRepBox.get_next_id() == 11
        
        # 文字列表現確認
        assert "id=0" in str(box1)
        print("ID機能テスト: 成功\n")
    
    def test_error_handling():
        """エラーハンドリングテスト"""
        print("=== エラーハンドリングテスト ===")
        ParRepBox.reset_id_counter()
        
        group = ParRepBox()
        
        # 存在しないワーカーアクセス
        try:
            group.get_worker(999)
            assert False
        except KeyError:
            pass
        
        # 異なる初期状態のワーカー追加
        group.set_initial_state(0)
        transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        wrong_worker = worker(worker_id=1, transition_matrix=transition_matrix)
        wrong_worker.set_initial_state(1, 5, 3)
        
        try:
            group.add_worker(wrong_worker)
            assert False
        except ValueError:
            pass
        
        print("エラーハンドリングテスト: 成功\n")
    
    def test_max_time_functionality():
        """最大時間機能テスト"""
        print("=== 最大時間機能テスト ===")
        ParRepBox.reset_id_counter()
        
        # 最大時間設定
        group = ParRepBox(max_time=5)
        assert group.get_max_time() == 5
        
        # 遷移しにくい設定でワーカー追加
        transition_matrix = np.array([[0.99, 0.01], [0.02, 0.98]])
        worker1, worker2 = create_test_workers(transition_matrix, t_phase=20, t_corr=15)
        
        group.add_worker(worker1)
        group.add_worker(worker2)
        
        # 最大時間まで実行
        for i in range(10):
            step_result = group.step()
            if step_result['status'] == 'max_time_force_terminated':
                print("最大時間による強制終了が発生")
                break
        
        print("最大時間機能テスト: 成功\n")
    
    def run_all_tests():
        """すべてのテストを実行"""
        print("ParRepBox 簡易テストを開始します...\n")
        
        try:
            test_basic_functionality()
            test_step_execution()
            test_id_functionality()
            test_error_handling()
            test_max_time_functionality()
            
            print("=== すべてのテストが成功しました! ===")
            return True
            
        except Exception as e:
            print(f"テストエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # テストの実行
    if run_all_tests():
        sys.exit(0)
    else:
        sys.exit(1)
