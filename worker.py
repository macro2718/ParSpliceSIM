import numpy as np
from enum import Enum

class WorkerState(Enum):
    """フェーズの定数定義"""
    IDLE = "idle"
    DEPHASING = "dephasing"
    RUN = "run"
    DECORRELATION = "decorrelation"

# 定数定義
DEFAULT_TIME_VALUE = 0
MAX_SIMULATION_STEPS = 50

class worker:
    """
    ワーカークラス
    
    待機状態かどうかを示すブール変数と状態を持ち、確率遷移行列に従って状態を更新する
    """
    
    def __init__(self, worker_id, transition_matrix=None, t_phase_dict=None, t_corr_dict=None):
        """
        ワーカークラスの初期化
        
        Parameters:
        worker_id (int): ワーカーの固有ID
        transition_matrix (numpy.ndarray): 確率遷移行列 (省略可能)
        t_corr_dict (Dict[int, int]): 各状態に対応するt_corrの辞書 (省略可能)
        t_phase_dict (Dict[int, int]): 各状態に対応するt_phaseの辞書 (省略可能)
        """
        # 固有IDを設定
        self.id = worker_id
        
        self.transition_matrix = transition_matrix
        self.t_phase_dict = t_phase_dict if t_phase_dict is not None else {}
        self.t_corr_dict = t_corr_dict if t_corr_dict is not None else {}
        self._reset_all_variables()
        
    def _reset_all_variables(self):
        """すべての変数をリセットする統一メソッド"""
        self._reset_state_variables()
        self._reset_time_parameters()
        
    def _validate_initial_state_parameters(self, initial_state, t_phase, t_corr):
        """
        初期状態設定時のパラメータを検証する内部メソッド
        """
        if self.transition_matrix is None:
            raise ValueError("遷移行列が設定されていません")
            
        num_states = self.transition_matrix.shape[0]
        if not (0 <= initial_state < num_states):
            raise ValueError(f"初期状態は0から{num_states-1}の範囲である必要があります")
            
        if t_phase < 0 or t_corr < 0:
            raise ValueError("時間パラメータは非負の値である必要があります")
        
    def set_initial_state(self, initial_state, t_phase=None, t_corr=None):
        """
        初期状態を設定し、待機状態から実行状態に移行する
        
        Parameters:
        initial_state (int): 初期状態の値
        t_phase (int, optional): フェーズ時間。Noneの場合、t_phase_dictから取得
        t_corr (int, optional): 修正時間。Noneの場合、t_corr_dictから取得
        """
        # 辞書から値を取得
        if t_phase is None:
            t_phase = self.t_phase_dict.get(initial_state, DEFAULT_TIME_VALUE)
        if t_corr is None:
            t_corr = self.t_corr_dict.get(initial_state, DEFAULT_TIME_VALUE)
            
        self._validate_initial_state_parameters(initial_state, t_phase, t_corr)
            
        self.state = initial_state
        self.initial_state = initial_state
        self.is_idle = False
        self.current_phase = WorkerState.DEPHASING.value  # 初期フェーズから開始
        self.transition_occurred = False  # 遷移フラグをリセット
        self.previous_state = None  # 直前の状態をリセット
        self.is_decorrelated = False  # decorrelation状態をリセット
        self.segment = []
        
        # タイムフェーズ関連の変数を設定
        self.t_phase = t_phase
        self.t_corr = t_corr
        self.remaining_t_phase = t_phase
        self.remaining_t_corr = t_corr
        
        # decorrelation状態を更新
        self._update_decorrelation_status()
        
    def step(self):
        """
        1ステップ進める操作
        複雑なフェーズ制御を含む
        
        Returns:
        int: 次の状態
        """
        self._validate_step_preconditions()

        # remaining_t_phaseが0でない場合（dephasingフェーズ）
        if self.remaining_t_phase > 0:
            return self._step_dephasing_phase()
            
        # remaining_t_phaseが0になったら、セグメントに初期状態を追加
        if len(self.segment) == 0:
            self.transition_occurred = False
            self.segment.append(self.initial_state)
            self._update_decorrelation_status()
            
        # 通常の実行フェーズ
        return self._step_run_phase()
            
    def _validate_step_preconditions(self):
        """ステップ実行前の事前条件を検証"""
        if self.is_idle:
            raise ValueError("ノードが待機状態では状態遷移できません")
            
        if self.transition_matrix is None:
            raise ValueError("遷移行列が設定されていません")
            
        if self.state is None:
            raise ValueError("現在の状態が設定されていません")
        
    def _update_decorrelation_status(self):
        """
        デコリレーション状態を更新する内部メソッド
        セグメントの長さがt_corr以上で最後のt_corrステップが全て同じ値かどうかを判断
        """
        # t_corrが0の場合は常にデコリレーション状態
        if self.t_corr == 0:
            self.is_decorrelated = True
            return
            
        # 早期リターン条件
        if len(self.segment) == 0 or len(self.segment) <= self.t_corr:
            self.is_decorrelated = False
            return
            
        # 最後のt_corrステップを取得
        last_t_corr_steps = self.segment[-(self.t_corr + 1):]
        
        # 全て同じ値かチェック
        self.is_decorrelated = all(step == last_t_corr_steps[0] for step in last_t_corr_steps)
        
    def _perform_state_transition(self):
        """
        状態遷移を実行し、遷移発生フラグを更新する内部メソッド
        
        Returns:
        int: 次の状態
        """
        # 直前の状態を保存
        self.previous_state = self.state
        
        # 現在の状態から次の状態への遷移確率を取得
        transition_probs = self.transition_matrix[self.state]
        
        # 確率に従って次の状態を選択
        next_state = np.random.choice(len(transition_probs), p=transition_probs)
        
        # 遷移が発生したかチェック
        if not self.transition_occurred and self.previous_state != next_state:
            self.transition_occurred = True
            
        return next_state
        
    def _step_dephasing_phase(self):
        """
        デフェージングフェーズのステップ処理
        
        Returns:
        int: 次の状態
        """
        self.current_phase = WorkerState.DEPHASING.value
        
        next_state = self._perform_state_transition()
        
        if next_state == self.initial_state:
            # 次の状態が初期状態と等しい場合、remaining_t_phaseから1引く
            self.remaining_t_phase -= 1
            self.state = next_state
        else:
            # そうでない場合、remaining_t_phaseをt_phaseに戻して初期状態からやり直し
            self.remaining_t_phase = self.t_phase
            self.state = self.initial_state
            self.transition_occurred = False  # 遷移フラグをリセット
            
        return None
        
    def _step_run_phase(self):
        """
        通常実行フェーズのステップ処理
        
        Returns:
        int: 次の状態
        """
        self.current_phase = WorkerState.RUN.value
        
        next_state = self._perform_state_transition()
        
        # 状態を更新
        self.state = next_state
        self.t_corr = self.t_corr_dict.get(next_state, DEFAULT_TIME_VALUE)

        # セグメントに新しい状態を追加
        self.segment.append(next_state)
        self._update_decorrelation_status()
        
        return next_state
        
    def _reset_state_variables(self):
        """
        状態変数をリセットする内部メソッド
        """
        self.is_idle = True
        self.state = None
        self.initial_state = None
        self.current_phase = WorkerState.IDLE.value
        self.transition_occurred = False
        self.previous_state = None
        self.is_decorrelated = False
        self.segment = []
        
    def _reset_time_parameters(self):
        """
        時間パラメータをリセットする内部メソッド
        """
        self.t_phase = DEFAULT_TIME_VALUE
        self.t_corr = DEFAULT_TIME_VALUE
        self.remaining_t_phase = DEFAULT_TIME_VALUE
        self.remaining_t_corr = DEFAULT_TIME_VALUE
        
    def get_state(self):
        """
        現在の状態を取得
        
        Returns:
        int or None: 現在の状態（待機状態の場合はNone）
        """
        return self.state
        
    def get_is_idle(self):
        """
        待機状態かどうかを取得
        
        Returns:
        bool: 待機状態かどうか
        """
        return self.is_idle
        
    def get_id(self):
        """
        ワーカーの固有IDを取得
        
        Returns:
        int: ワーカーの固有ID
        """
        return self.id
        
    def get_segment(self):
        """
        現在のセグメント（状態の履歴）を取得
        
        Returns:
        list: 初期状態から現在までの状態の列
        """
        return self.segment.copy()  # コピーを返してセグメントの不正な変更を防ぐ
        
    def get_steps_elapsed(self):
        """
        計算開始から現在までに経過したステップ数を取得
        
        Returns:
        int: 経過したステップ数（待機状態の場合は0）
        """
        if len(self.segment) == 0:
            return DEFAULT_TIME_VALUE
        return len(self.segment) - 1  # 初期状態を除いたステップ数
        
    def get_time_parameters(self):
        """
        時間パラメータを取得
        
        Returns:
        dict: 時間パラメータの辞書
        """
        return {
            't_phase': self.t_phase,
            't_corr': self.t_corr,
            'remaining_t_phase': self.remaining_t_phase,
            'remaining_t_corr': self.remaining_t_corr
        }
        
    def get_remaining_times(self):
        """
        残り時間を取得
        
        Returns:
        tuple: (remaining_t_phase, remaining_t_corr)
        """
        return (self.remaining_t_phase, self.remaining_t_corr)
        
    def get_current_phase(self):
        """
        現在のフェーズを取得
        
        Returns:
        str: 現在のフェーズ（"idle", "dephasing", "run", "decorrelation"）
        """
        return self.current_phase
        
    def get_transition_occurred(self):
        """
        遷移が発生したかどうかを取得
        
        Returns:
        bool: 遷移が発生した場合True、そうでない場合False
        """
        return self.transition_occurred
        
    def reset_transition_flag(self):
        """
        遷移フラグをリセット
        """
        self.transition_occurred = False
        
    def get_is_decorrelated(self):
        """
        デコリレーション状態を取得
        セグメントの長さがt_corr以上で最後のt_corrステップが全て同じ値かどうか
        
        Returns:
        bool: デコリレーションが完了した場合True、そうでない場合False
        """
        return self.is_decorrelated
        
    def finish_calculation(self):
        """
        計算終了の指示を受け付け、状態を破棄して待機状態に戻る
        is_decorrelatedがTrueの場合、セグメント全体を出力する
        
        Returns:
        list or None: is_decorrelatedがTrueの場合、セグメント全体。それ以外はNone
        """
        if self.is_idle:
            raise ValueError("既に待機状態です")
        
        # is_decorrelatedがTrueの場合、セグメント全体を保存
        final_segment = self._get_final_segment()
        
        self._reset_all_variables()
        
        return final_segment
        
    def _get_final_segment(self):
        """
        最終セグメントを取得する内部メソッド
        
        Returns:
        list or None: is_decorrelatedがTrueの場合、セグメント全体を返す
        """
        if self.is_decorrelated:
            return self.segment.copy()
        return None
        
    def reset(self):
        """
        ノードをリセットし、待機状態に戻す
        """
        self._reset_all_variables()
        
    def __str__(self):
        """
        文字列表現
        """
        if self.is_idle:
            return f"worker(id={self.id}, idle=True, state=None, segment=[], phase={WorkerState.IDLE.value}, transition_occurred=False, is_decorrelated=False)"
        else:
            return f"worker(id={self.id}, idle=False, state={self.state}, segment={self.segment}, phase={self.current_phase}, transition_occurred={self.transition_occurred}, is_decorrelated={self.is_decorrelated}, remaining_t_phase={self.remaining_t_phase}, remaining_t_corr={self.remaining_t_corr})"
            
    def __repr__(self):
        """
        オブジェクトの表現
        """
        return self.__str__()


# 使用例
if __name__ == "__main__":
    # systemGenerater.pyから遷移行列を生成
    from systemGenerater import generate_random_transition_matrix
    
    # 3x3の遷移行列を生成
    transition_matrix = generate_random_transition_matrix(3, 0.9, 0.0001)
    print("生成された遷移行列:")
    print(transition_matrix)
    print()
    
    # t_phase_dictとt_corr_dictを定義
    t_phase_dict = {0: 3, 1: 2, 2: 4}
    t_corr_dict = {0: 2, 1: 1, 2: 3}
    
    # workerを初期化（IDと辞書を指定）
    node = worker(worker_id=1, transition_matrix=transition_matrix, 
                  t_phase_dict=t_phase_dict, t_corr_dict=t_corr_dict)
    print(f"初期状態: {node}")
    print(f"ワーカーID: {node.get_id()}")
    print(f"待機状態かどうか: {node.get_is_idle()}")
    print()
    
    # 初期状態を設定（辞書から自動取得）
    node.set_initial_state(0)  # t_phase=3, t_corr=2が自動で設定される
    print(f"初期状態設定後（辞書から自動取得）: {node}")
    print(f"待機状態かどうか: {node.get_is_idle()}")
    print(f"時間パラメータ: {node.get_time_parameters()}")
    print(f"残り時間: {node.get_remaining_times()}")
    print()
    
    # 明示的にt_phaseとt_corrを指定することも可能
    node.reset()
    node.set_initial_state(1, t_phase=5, t_corr=3)  # 明示的に指定
    print(f"初期状態設定後（明示的指定）: {node}")
    print(f"時間パラメータ: {node.get_time_parameters()}")
    print()
    
    # 辞書に存在しない状態でもデフォルト値が使用される例
    # （ただし、遷移行列のサイズ内である必要がある）
    node.reset()
    # 状態2は辞書に存在するが、別の例として状態2を使用
    node.set_initial_state(2)  # t_phase=4, t_corr=3が辞書から取得される
    print(f"初期状態設定後（状態2、辞書から取得）: {node}")
    print(f"時間パラメータ: {node.get_time_parameters()}")
    print()
    
    # 実際のシミュレーション例
    node.reset()
    node.set_initial_state(0)  # 辞書から自動取得
    print(f"初期状態設定後: {node}")
    print(f"待機状態かどうか: {node.get_is_idle()}")
    print(f"時間パラメータ: {node.get_time_parameters()}")
    print(f"残り時間: {node.get_remaining_times()}")
    print()
    
    # 複数ステップ実行
    print("ステップ実行:")
    step_count = 0
    while True:
        step_count += 1
        next_state = node.step()
        print(f"ステップ {step_count}: 状態 {next_state} | フェーズ: {node.get_current_phase()} | 遷移発生: {node.get_transition_occurred()} | 残り時間: {node.get_remaining_times()} | セグメント: {node.get_segment()}")
        
        # 無限ループ防止（手動で停止）
        if step_count > MAX_SIMULATION_STEPS:
            print("ステップ数が多すぎます。終了します。")
            break
        
    print()
    print(f"最終状態: {node}")
    print(f"セグメント: {node.get_segment()}")
    print(f"総経過ステップ数: {node.get_steps_elapsed()}")
    print(f"デコリレーション状態: {node.get_is_decorrelated()}")
    print(f"現在のフェーズ: {node.get_current_phase()}")
    print(f"遷移が発生したか: {node.get_transition_occurred()}")
    
    # 計算終了の指示
    final_result = node.finish_calculation()
    print(f"計算終了後: {node}")
    print(f"finish_calculationの戻り値: {final_result}")
    print(f"待機状態かどうか: {node.get_is_idle()}")
    print(f"終了後の経過ステップ数: {node.get_steps_elapsed()}")
    print(f"終了後のフェーズ: {node.get_current_phase()}")
    print()
    
    # 複数のワーカーを作成する例
    print("複数のワーカーを作成:")
    worker1 = worker(worker_id=10, transition_matrix=transition_matrix, 
                     t_phase_dict=t_phase_dict, t_corr_dict=t_corr_dict)
    worker2 = worker(worker_id=20, transition_matrix=transition_matrix, 
                     t_phase_dict=t_phase_dict, t_corr_dict=t_corr_dict)
    worker3 = worker(worker_id=30, transition_matrix=transition_matrix, 
                     t_phase_dict=t_phase_dict, t_corr_dict=t_corr_dict)
    
    print(f"Worker 1 ID: {worker1.get_id()}")
    print(f"Worker 2 ID: {worker2.get_id()}")
    print(f"Worker 3 ID: {worker3.get_id()}")
    print()
    
    # リセット例
    node.set_initial_state(1)  # 辞書から自動取得（t_phase=2, t_corr=1）
    print(f"再開後: {node}")
    print(f"再開後のセグメント: {node.get_segment()}")
    print(f"再開後の経過ステップ数: {node.get_steps_elapsed()}")
    print(f"再開後の時間パラメータ: {node.get_time_parameters()}")
    node.reset()
    print(f"リセット後: {node}")
    print(f"リセット後の経過ステップ数: {node.get_steps_elapsed()}")
    print(f"リセット後の時間パラメータ: {node.get_time_parameters()}")
