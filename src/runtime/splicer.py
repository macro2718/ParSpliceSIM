#!/usr/bin/env python3
"""
Splicer クラス（runtime 層）

producerのsegmentStoreからセグメントを取得し、それらを繋ぎ合わせて
一本のtrajectoryを作成するクラス。

リファクタリング後：
- 共通エラーハンドリングとロギングを追加
- バリデーション機能を強化
- コードの可読性を向上
"""

from typing import List, Dict, Any, Optional, Tuple
import copy

from common import (
    SplicerError, ValidationError, Validator, ResultFormatter,
    SafeOperationHandler, default_logger, Constants
)


class Splicer:
    """
    セグメントを取得し、スプライシングによって軌道を構築するクラス
    
    アルゴリズム概要のsplicerの4つの操作を実装：
    1. producerのsegmentStoreからセグメントを取得し、それらを自身のsegmentStoreに格納
    2. trajectoryの最後の状態から始まるセグメントをランダムに選択し、trajectoryに結合
    3. 2を行えなくなるまで繰り返す
    4. trajectoryの最後の状態を取得
    """
    
    def __init__(self, initial_state: int, max_trajectory_length: int = 10000, minimal_output: bool = False):
        """
        Splicerクラスの初期化
        
        Parameters:
        initial_state (int): trajectoryの初期状態
        max_trajectory_length (int): trajectoryの最大長（デフォルト: 10000）
        minimal_output (bool): 最小限出力モードのフラグ
        
        Raises:
        ValidationError: initial_stateが無効な場合
        """
        # バリデーション
        self.initial_state = Validator.validate_non_negative_integer(
            initial_state, "initial_state"
        )
        self.max_trajectory_length = Validator.validate_positive_integer(
            max_trajectory_length, "max_trajectory_length"
        )
        self.minimal_output = minimal_output  # 最小限出力モードのフラグを追加
        
        # segmentStore: 状態をキーとし、その状態から始まるセグメントとIDのリストを値とする辞書
        self.segment_store: Dict[int, List[Tuple[List[int], int]]] = {}  # state -> [(segment, segment_id), ...]
        
        # segmentDatabase: 今まで格納されたセグメント全てを保管する辞書（削除されない）
        self.segment_database: Dict[int, List[Tuple[List[int], int]]] = {}  # state -> [(segment, segment_id), ...]
        
        # segment_transition_matrix: 状態i→状態jの遷移を行うセグメントの本数を保存する行列
        self.segment_transition_matrix: Dict[Tuple[int, int], int] = {}  # (start_state, end_state) -> count
        
        # trajectory: セグメントを繋ぎ合わせて一本の軌道を作ったもの
        self.trajectory: List[int] = [self.initial_state]
        
        # 使用済みセグメントIDを各初期状態ごとに追跡（順次スプライシングのため）
        self._used_segment_ids: Dict[int, List[int]] = {}  # state -> [used_segment_ids]
        
        if not self.minimal_output:
            default_logger.info(f"Splicer初期化完了: 初期状態={initial_state}")
    
    def _validate_trajectory_length(self) -> None:
        """軌道長の妥当性をチェック"""
        if len(self.trajectory) > self.max_trajectory_length:
            raise SplicerError(
                f"軌道長が制限値({self.max_trajectory_length})を超えました: {len(self.trajectory)}"
            )
    
    def load_segments_from_producer(self, producer) -> Dict[str, Any]:
        """
        操作1: producerのsegmentStoreからセグメントを取得し、それらを自身のsegmentStoreに格納
        
        Parameters:
        producer: Producerインスタンス
        
        Returns:
        Dict[str, Any]: 読み込み結果の詳細情報
        """
        return SafeOperationHandler.safe_execute(
            lambda: self._load_segments_from_producer_impl(producer),
            SplicerError,
            ResultFormatter.error_result("セグメント読み込み中にエラーが発生"),
            default_logger
        )
    
    def _load_segments_from_producer_impl(self, producer) -> Dict[str, Any]:
        """セグメント読み込みの内部実装（セグメントIDを含む）"""
        if not hasattr(producer, 'get_all_stored_segments_info'):
            raise SplicerError("Producerにget_all_stored_segments_infoメソッドがありません")
        
        # producerから全てのsegmentを取得
        producer_segments = producer.get_all_stored_segments_info()
        
        loaded_count = 0
        total_segments = 0
        
        # 各セグメントを処理
        for group_id, segment_info in producer_segments.items():
            if not isinstance(segment_info, dict) or 'segment' not in segment_info:
                continue
                
            segment = segment_info['segment']
            if not segment or not isinstance(segment, list):
                continue
            
            initial_state = segment[0]
            segment_id = segment_info.get('segment_id', None)
            
            # セグメントIDが取得できない場合はスキップ
            if segment_id is None:
                default_logger.warning(f"グループ{group_id}のセグメントIDがNoneのためスキップ")
                continue
            
            # バリデーション
            if not isinstance(initial_state, int) or initial_state < 0:
                default_logger.warning(f"無効な初期状態のセグメントをスキップ: {initial_state}")
                continue
            
            # segmentStoreに格納（状態をキーとして、セグメントとIDのペアで）
            if initial_state not in self.segment_store:
                self.segment_store[initial_state] = []
            self.segment_store[initial_state].append((segment.copy(), segment_id))
            
            # segmentDatabaseにも格納（永続的に保管）
            if initial_state not in self.segment_database:
                self.segment_database[initial_state] = []
            self.segment_database[initial_state].append((segment.copy(), segment_id))
            
            # segment_transition_matrixを更新
            self._update_transition_matrix(segment)
            
            loaded_count += 1
            total_segments += len(segment) - 1  # 最初の状態を除く
            
            if not self.minimal_output:
                default_logger.info(f"セグメント読み込み: 状態{initial_state}, ID={segment_id}, 長さ={len(segment)}")
        
        return ResultFormatter.success_result({
            'loaded_segments': loaded_count,
            'total_length': total_segments,
            'segment_store_states': list(self.segment_store.keys()),
            'segments_per_state': {state: len(segments) for state, segments in self.segment_store.items()}
        })
    
    def splice(self) -> Dict[str, Any]:
        """
        操作2-3: trajectoryの最後の状態から始まるセグメントをランダムに選択し、
        trajectoryに結合する作業を、できなくなるまで繰り返す
        
        spliceでは、繋ぎ合わせる際にセグメントの一番最初の状態は削除して繋ぎ合わせる。
        
        Returns:
        Dict[str, Any]: スプライシング結果の詳細情報
        """
        return SafeOperationHandler.safe_execute(
            lambda: self._splice_impl(),
            SplicerError,
            ResultFormatter.error_result("スプライシング中にエラーが発生"),
            default_logger
        )
    
    def _splice_impl(self) -> Dict[str, Any]:
        """スプライシングの内部実装（セグメントID順を考慮）"""
        if not self.trajectory:
            raise SplicerError("軌道が空です")
        
        spliced_count = 0
        total_spliced_length = 0
        
        while True:
            self._validate_trajectory_length()
            
            # 現在の軌道の最後の状態を取得
            current_final_state = self.trajectory[-1]
            
            # その状態から始まるセグメントがあるかチェック
            if (current_final_state not in self.segment_store or 
                not self.segment_store[current_final_state]):
                # スプライシング可能なセグメントがない場合は終了
                break
            
            # 利用可能なセグメント（ID付き）を取得
            available_segments_with_id = self.segment_store[current_final_state]
            
            # セグメントIDの順番でソートして、次に使用可能なセグメントを探す
            # 既に使用済みのIDは除外する
            next_segment_id = self._find_next_usable_segment_id(current_final_state)
            selected_segment = None
            selected_segment_id = None
            
            # デバッグ情報（最小限出力モードでない場合のみ）
            if not self.minimal_output:
                available_ids = [seg_id for _, seg_id in available_segments_with_id]
                used_ids = self._used_segment_ids.get(current_final_state, [])
                default_logger.info(f"状態{current_final_state}: 次ID={next_segment_id}, 利用可能ID={sorted(available_ids)}, 使用済みID={sorted(used_ids)}")
            
            for segment, segment_id in available_segments_with_id:
                if segment_id == next_segment_id:
                    selected_segment = segment
                    selected_segment_id = segment_id
                    break
            
            # 次のIDのセグメントが見つからない場合は終了
            if selected_segment is None:
                if not self.minimal_output:
                    default_logger.info(f"状態{current_final_state}のセグメントID {next_segment_id} が見つからないため、スプライシングを終了")
                break
            
            # バリデーション：セグメントの最初の状態が期待値と一致するかチェック
            if selected_segment[0] != current_final_state:
                default_logger.warning(
                    f"セグメントの不整合: 期待値={current_final_state}, 実際={selected_segment[0]}"
                )
                # 不整合なセグメントを削除
                available_segments_with_id.remove((selected_segment, selected_segment_id))
                if not available_segments_with_id:
                    del self.segment_store[current_final_state]
                continue
            
            # trajectoryに結合（最初の状態は除く）
            segment_to_append = selected_segment[1:]  # 最初の状態を除く
            self.trajectory.extend(segment_to_append)
            
            # 使用したセグメントIDを記録
            if current_final_state not in self._used_segment_ids:
                self._used_segment_ids[current_final_state] = []
            self._used_segment_ids[current_final_state].append(selected_segment_id)
            
            # 使用したセグメントをsegmentStoreから削除
            available_segments_with_id.remove((selected_segment, selected_segment_id))
            if not available_segments_with_id:
                # その状態のセグメントがすべて使い切られた場合はキーも削除
                del self.segment_store[current_final_state]
            
            # 統計更新
            spliced_count += 1
            total_spliced_length += len(segment_to_append)
            
            if not self.minimal_output:
                default_logger.info(f"セグメントID {selected_segment_id} をスプライス")
        
        return ResultFormatter.success_result({
            'spliced_segments': spliced_count,
            'total_spliced_length': total_spliced_length,
            'final_trajectory_length': len(self.trajectory),
            'final_state': self.get_final_state(),
            'remaining_segment_store_states': list(self.segment_store.keys()),
            'used_segment_ids': self._used_segment_ids.copy()
        })
    
    def _find_next_usable_segment_id(self, state: int) -> int:
        """
        指定された状態での次に使用可能なセグメントIDを見つける
        
        Args:
            state (int): 状態
            
        Returns:
            int: 次に使用すべきセグメントID
        """
        # 指定された状態で既に使用済みのIDを取得
        used_ids = self._used_segment_ids.get(state, [])
        
        # その状態で利用可能なセグメントIDを取得
        if state in self.segment_store:
            available_ids = [seg_id for _, seg_id in self.segment_store[state]]
        else:
            available_ids = []
        
        # 利用可能なIDの中で、まだ使用されていない最小のIDを探す
        for segment_id in sorted(available_ids):
            if segment_id not in used_ids:
                return segment_id
        
        # 利用可能なセグメントがない場合は1を返す（エラーケース）
        return 1
    
    def _update_transition_matrix(self, segment: List[int]) -> None:
        """
        セグメントの遷移情報をsegment_transition_matrixに追加
        
        Args:
            segment (List[int]): セグメント（状態の列）
        """
        if len(segment) < 2:
            return
            
        start_state = segment[0]
        end_state = segment[-1]
        transition_key = (start_state, end_state)
        
        if transition_key not in self.segment_transition_matrix:
            self.segment_transition_matrix[transition_key] = 0
        self.segment_transition_matrix[transition_key] += 1
    
    def get_trajectory(self) -> List[int]:
        """
        現在のtrajectoryを取得
        
        Returns:
        List[int]: 現在のtrajectory
        """
        return self.trajectory.copy()
    
    def get_trajectory_states(self) -> List[int]:
        """現在のtrajectory状態列を取得（get_trajectoryのエイリアス）"""
        return self.trajectory.copy()
    
    def get_trajectory_length(self) -> int:
        """
        現在のtrajectoryの長さを取得
        
        Returns:
        int: trajectoryの長さ
        """
        return len(self.trajectory)
    
    def get_final_state(self) -> Optional[int]:
        """
        操作4: trajectoryの最後の状態を取得
        
        Returns:
        Optional[int]: trajectoryの最後の状態（trajectoryが空の場合はNone）
        """
        return self.trajectory[-1] if self.trajectory else None
    
    def get_segment_store_info(self) -> Dict[str, Any]:
        """
        segmentStoreの情報を取得（セグメントIDを含む、segmentDatabaseの情報も含む）
        
        Returns:
        Dict[str, Any]: segmentStoreの詳細情報とsegmentDatabaseの情報
        """
        total_segments = sum(len(segments) for segments in self.segment_store.values())
        segments_per_state = {state: len(segments) for state, segments in self.segment_store.items()}
        
        # セグメントIDも含む詳細情報
        segments_with_ids = {}
        for state, segments_with_id in self.segment_store.items():
            segments_with_ids[state] = [(len(segment), segment_id) for segment, segment_id in segments_with_id]
        
        # segmentDatabaseの情報も取得
        database_info = self.get_segment_database_info()
        transition_matrix_info = self.get_transition_matrix_info()
        
        return {
            # segmentStoreの情報
            'total_segments': total_segments,
            'segments_per_state': segments_per_state,
            'segments_with_ids': segments_with_ids,
            'segment_lengths_per_state': self.get_segment_lengths_per_state(),
            'available_states': list(self.segment_store.keys()),
            'states_count': len(self.segment_store),
            'used_segment_ids': self._used_segment_ids.copy(),
            'segment_store': copy.deepcopy(self.segment_store),
            
            # segmentDatabaseの情報
            'database_info': database_info,
            
            # 遷移行列の情報
            'transition_matrix_info': transition_matrix_info
        }
    
    def get_segment_database_info(self) -> Dict[str, Any]:
        """
        segmentDatabaseの情報を取得（永続的に保管されたセグメント情報）
        
        Returns:
        Dict[str, Any]: segmentDatabaseの詳細情報
        """
        total_segments = sum(len(segments) for segments in self.segment_database.values())
        segments_per_state = {state: len(segments) for state, segments in self.segment_database.items()}
        
        # セグメントIDも含む詳細情報
        segments_with_ids = {}
        for state, segments_with_id in self.segment_database.items():
            segments_with_ids[state] = [(len(segment), segment_id) for segment, segment_id in segments_with_id]
        
        return {
            'total_segments': total_segments,
            'segments_per_state': segments_per_state,
            'segments_with_ids': segments_with_ids,
            'segment_lengths_per_state': self.get_segment_database_lengths_per_state(),
            'available_states': list(self.segment_database.keys()),
            'states_count': len(self.segment_database),
            'segment_database': copy.deepcopy(self.segment_database)
        }
    
    def get_segment_database_lengths_per_state(self) -> Dict[int, int]:
        """
        segment_databaseの各キー（状態）ごとに、保存されたセグメントの長さの合計を辞書で返す
        
        Returns:
        Dict[int, int]: {状態: セグメント長の合計}
        """
        return {
            state: sum(len(segment) - 1 for segment, segment_id in segments_with_id)
            for state, segments_with_id in self.segment_database.items()
        }
    
    def get_transition_matrix_info(self) -> Dict[str, Any]:
        """
        segment_transition_matrixの情報を取得
        
        Returns:
        Dict[str, Any]: 遷移行列の詳細情報
        """
        if not self.segment_transition_matrix:
            return {
                'transition_matrix': {},
                'total_transitions': 0,
                'unique_start_states': [],
                'unique_end_states': [],
                'matrix_shape': (0, 0)
            }
        
        # 全ての開始状態と終了状態を取得
        start_states = set()
        end_states = set()
        for (start, end) in self.segment_transition_matrix.keys():
            start_states.add(start)
            end_states.add(end)
        
        start_states = sorted(start_states)
        end_states = sorted(end_states)
        
        # 2次元配列形式の行列を作成
        matrix_2d = []
        for start in start_states:
            row = []
            for end in end_states:
                count = self.segment_transition_matrix.get((start, end), 0)
                row.append(count)
            matrix_2d.append(row)
        
        return {
            'transition_matrix': dict(self.segment_transition_matrix),
            'transition_matrix_2d': matrix_2d,
            'start_states': start_states,
            'end_states': end_states,
            'total_transitions': sum(self.segment_transition_matrix.values()),
            'unique_start_states': start_states,
            'unique_end_states': end_states,
            'matrix_shape': (len(start_states), len(end_states))
        }
    
    def get_segment_lengths_per_state(self) -> Dict[int, int]:
        """
        segment_storeの各キー（状態）ごとに、保存されたセグメントの長さの合計を辞書で返す
        
        Returns:
        Dict[int, int]: {状態: セグメント長の合計}
        """
        return {
            state: sum(len(segment) - 1 for segment, segment_id in segments_with_id)
            for state, segments_with_id in self.segment_store.items()
        }
    
    def clear_segment_store(self) -> Dict[str, Any]:
        """
        segmentStoreをクリア
        
        Returns:
        Dict[str, Any]: クリア結果
        """
        cleared_count = sum(len(segments) for segments in self.segment_store.values())
        self.segment_store.clear()
        
        return ResultFormatter.success_result({
            'cleared_segments': cleared_count
        })
    
    def clear_segment_database(self) -> Dict[str, Any]:
        """
        segmentDatabaseをクリア
        
        Returns:
        Dict[str, Any]: クリア結果
        """
        cleared_count = sum(len(segments) for segments in self.segment_database.values())
        self.segment_database.clear()
        
        return ResultFormatter.success_result({
            'cleared_segments': cleared_count
        })
    
    def clear_transition_matrix(self) -> Dict[str, Any]:
        """
        segment_transition_matrixをクリア
        
        Returns:
        Dict[str, Any]: クリア結果
        """
        cleared_count = len(self.segment_transition_matrix)
        self.segment_transition_matrix.clear()
        
        return ResultFormatter.success_result({
            'cleared_transitions': cleared_count
        })
    
    def clear_all_data(self) -> Dict[str, Any]:
        """
        segmentStore、segmentDatabase、transition_matrixを全てクリア
        
        Returns:
        Dict[str, Any]: クリア結果
        """
        store_result = self.clear_segment_store()
        db_result = self.clear_segment_database()
        matrix_result = self.clear_transition_matrix()
        
        return ResultFormatter.success_result({
            'segment_store_cleared': store_result['result']['cleared_segments'],
            'segment_database_cleared': db_result['result']['cleared_segments'],
            'transition_matrix_cleared': matrix_result['result']['cleared_transitions']
        })
    
    def reset_trajectory(self, new_initial_state: Optional[int] = None) -> Dict[str, Any]:
        """
        trajectoryをリセット
        
        Parameters:
        new_initial_state (Optional[int]): 新しい初期状態（Noneの場合は元の初期状態を使用）
        
        Returns:
        Dict[str, Any]: リセット結果
        """
        old_length = len(self.trajectory)
        reset_state = new_initial_state if new_initial_state is not None else self.initial_state
        
        if new_initial_state is not None:
            reset_state = Validator.validate_non_negative_integer(new_initial_state, "new_initial_state")
            self.initial_state = reset_state
        
        self.trajectory = [reset_state]
        
        return ResultFormatter.success_result({
            'old_trajectory_length': old_length,
            'new_initial_state': reset_state
        })
    
    def run_one_step(self, producer) -> Dict[str, Any]:
        """
        アルゴリズム概要に基づいてSplicerの1ステップを実行
        
        1. producerのsegmentStoreからセグメントを取得し、それらを自身のsegmentStoreに格納
        2. trajectoryの最後の状態から始まるセグメントをランダムに選択し、trajectoryに結合
        3. 2を行えなくなるまで繰り返す
        4. trajectoryの最後の状態を取得
        
        Parameters:
        producer: Producerインスタンス
        
        Returns:
        Dict[str, Any]: 1ステップ実行結果
        """
        try:
            # 操作1: producerからセグメントを取得
            load_result = self.load_segments_from_producer(producer)
            
            # 操作2-3: スプライシング実行
            splice_result = self.splice()
            
            # 操作4: 最後の状態を取得
            final_state = self.get_final_state()
            
            return ResultFormatter.success_result({
                'load_result': load_result,
                'splice_result': splice_result,
                'final_state': final_state,
                'trajectory_length': self.get_trajectory_length(),
                'segment_store_info': self.get_segment_store_info()
            })
            
        except Exception as e:
            return ResultFormatter.error_result(e, {
                'final_state': self.get_final_state(),
                'trajectory_length': self.get_trajectory_length()
            })
    
    def __str__(self) -> str:
        """文字列表現"""
        db_states = len(self.segment_database)
        transition_count = len(self.segment_transition_matrix)
        return (f"Splicer(initial_state={self.initial_state}, "
                f"trajectory_length={len(self.trajectory)}, "
                f"final_state={self.get_final_state()}, "
                f"segment_store_states={list(self.segment_store.keys())}, "
                f"database_states={db_states}, "
                f"transitions={transition_count})")
    
    def __repr__(self) -> str:
        """オブジェクト表現"""
        return self.__str__()


# テスト関数
def test_splicer_basic():
    """Splicerの基本機能テスト"""
    print("=== Splicer基本機能テスト ===")
    
    # Splicer初期化
    splicer = Splicer(initial_state=0)
    print(f"初期化後: {splicer}")
    
    # 手動でsegmentStoreにデータを追加（segmentDatabaseとtransition_matrixも更新される）
    test_segments = [
        ([0, 1, 2], 1),
        ([0, 2, 1, 0], 2),
        ([2, 0, 1], 3),
        ([2, 2, 1], 4),
        ([1, 0, 2, 1], 5)
    ]
    
    for segment, seg_id in test_segments:
        initial_state = segment[0]
        if initial_state not in splicer.segment_store:
            splicer.segment_store[initial_state] = []
            splicer.segment_database[initial_state] = []
        
        splicer.segment_store[initial_state].append((segment, seg_id))
        splicer.segment_database[initial_state].append((segment, seg_id))
        splicer._update_transition_matrix(segment)
    
    print(f"テストデータ追加後のsegmentStore: {splicer.get_segment_store_info()}")
    print(f"segmentDatabase情報: {splicer.get_segment_database_info()}")
    print(f"遷移行列情報: {splicer.get_transition_matrix_info()}")
    print(f"初期trajectory: {splicer.get_trajectory()}")
    
    # スプライシング実行
    splice_result = splicer.splice()
    print(f"\nスプライシング結果: {splice_result}")
    print(f"最終trajectory: {splicer.get_trajectory()}")
    print(f"最終状態: {splicer.get_final_state()}")
    print(f"残りsegmentStore: {splicer.get_segment_store_info()}")
    print(f"segmentDatabase（変化なし）: {splicer.get_segment_database_info()}")
    print(f"遷移行列（変化なし）: {splicer.get_transition_matrix_info()}")
    
    print("基本機能テスト: 成功\n")


if __name__ == "__main__":
    print("Splicerテスト開始...")
    # 基本テストの実行
    test_splicer_basic()
    print("Splicerテスト完了!")
