#!/usr/bin/env python3
"""
VST-ParSpliceスケジューリング戦略

一般的なParSplice戦略を継承し、状態ごとに max_time を動的に選択する。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .parsplice_strategy import ParSpliceSchedulingStrategy
from .common_utils import (
    transform_transition_matrix as _tx_matrix,
    run_monte_carlo_simulation as _run_mc,
    calculate_simulation_steps_per_state_from_virtual as _steps_from_virtual,
    create_virtual_producer_data as _create_vp_data_util,
)


class VSTParSpliceSchedulingStrategy(ParSpliceSchedulingStrategy):
    """
    VST-ParSpliceのスケジューリング戦略
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "VST-ParSplice"
        self.description = "max_timeをボックスごとに可変化したParSplice戦略"
        self.target_exit_probability = 0.9
        self.unknown_discovery_probability_constant: float = 0.1
        self.previous_known_states: set = set()
        self.new_states: set = set()
        self.initial_self_loop_probability: Dict[int, float] = {}

    # ========================================
    # メインのスケジューリングロジック
    # ========================================

    def calculate_worker_moves(
        self,
        producer_info: Dict,
        splicer_info: Dict,
        known_states: set,
        transition_matrix: Optional[List[List[int]]] = None,
        stationary_distribution: Optional[np.ndarray] = None,
        use_modified_matrix: bool = True,
    ):
        current_known = set(known_states or [])
        self.new_states = current_known - self.previous_known_states
        self.previous_known_states = set(current_known)

        return super().calculate_worker_moves(
            producer_info,
            splicer_info,
            known_states,
            transition_matrix,
            stationary_distribution,
            use_modified_matrix,
        )

    # ========================================
    # 価値計算とモンテカルロシミュレーション
    # ========================================

    def _prepare_value_arrays(
        self,
        virtual_producer_data: Dict,
        known_states: set,
        is_acceptable: Dict[int, bool],
        value_calculation_info: Dict,
    ) -> Tuple[List[Dict], List[Dict]]:
        existing_value = []
        new_value = []

        initial_states = virtual_producer_data["initial_states"]

        for group_id, initial_state in initial_states.items():
            if is_acceptable.get(group_id, False) and initial_state is not None:
                value = self._calculate_existing_value(
                    group_id,
                    initial_state,
                    {},
                    value_calculation_info,
                    virtual_producer_data,
                )
                existing_value.append(
                    {
                        "group_id": group_id,
                        "state": initial_state,
                        "value": value,
                        "type": "existing",
                    }
                )

        for state in known_states:
            max_time = self._compute_max_time_via_root_solver(
                state, value_calculation_info, virtual_producer_data
            )
            value = self._calculate_value_with_fixed_max_time(
                state, max_time, value_calculation_info, virtual_producer_data
            )
            new_value.append(
                {
                    "state": state,
                    "value": value,
                    "max_time": max_time,
                    "type": "new",
                }
            )

        return existing_value, new_value

    def _optimize_worker_allocation(
        self,
        workers_to_move: List[int],
        virtual_producer_data: Dict,
        existing_value: List[Dict],
        new_value: List[Dict],
        known_states: set,
        value_calculation_info: Dict,
    ) -> Tuple[List[Dict], List[Dict]]:
        worker_moves = []
        new_groups_config = []
        used_new_group_states = set()

        next_producer = virtual_producer_data["next_producer"]
        initial_states = virtual_producer_data["initial_states"]
        simulation_steps_per_group = virtual_producer_data["simulation_steps"]
        remaining_steps_per_group = virtual_producer_data["remaining_steps"]
        max_time_per_group = virtual_producer_data.get("max_time_per_group", {})

        while workers_to_move:
            worker_id = workers_to_move.pop(0)

            best_existing_value = (
                max(existing_value, key=lambda x: x["value"])["value"]
                if existing_value
                else 0
            )
            best_existing_candidates = [
                x for x in existing_value if x["value"] == best_existing_value
            ]
            best_existing = (
                np.random.choice(best_existing_candidates)
                if best_existing_candidates
                else None
            )

            best_new_value = (
                max(new_value, key=lambda x: x["value"])["value"] if new_value else 0
            )
            best_new_candidates = [
                x for x in new_value if x["value"] == best_new_value
            ]
            best_new = (
                np.random.choice(best_new_candidates)
                if best_new_candidates
                else None
            )

            best_value = 0.0
            best_option = None

            if best_existing:
                best_value = max(best_value, best_existing["value"])
                if best_existing["value"] >= best_value:
                    best_option = best_existing

            if best_new:
                best_value = max(best_value, best_new["value"])
                if best_new["value"] >= best_value:
                    best_option = best_new

            if not best_option:
                continue

            if best_option["type"] == "existing":
                raise ValueError("既存のボックスに配置することはできません")

            target_state = best_option["state"]
            target_group_id = None

            for group_id in next_producer.keys():
                if not next_producer[group_id]:
                    target_group_id = group_id
                    break

            if target_group_id is None:
                raise ValueError("新規グループを作成できません。空のグループが見つかりませんでした。")

            next_producer[target_group_id] = [worker_id]
            initial_states[target_group_id] = target_state
            simulation_steps_per_group[target_group_id] = 0
            max_time = best_option.get("max_time", self.default_max_time)
            # 安全策: max_time が None/無効値のときは再計算で補う
            if not isinstance(max_time, (int, float)) or max_time <= 0:
                max_time = self._compute_max_time_via_root_solver(
                    target_state, value_calculation_info, virtual_producer_data
                )
            max_time = int(max(1, round(max_time)))
            remaining_steps_per_group[target_group_id] = max_time
            max_time_per_group[target_group_id] = max_time

            expected_remaining_time = value_calculation_info.setdefault(
                "expected_remaining_time", {}
            )
            expected_time = self._expected_time_for_state(
                target_state,
                max_time,
                value_calculation_info.get("selected_transition_matrix", []),
            )
            expected_remaining_time[target_group_id] = expected_time

            virtual_producer_data["next_producer"] = next_producer
            virtual_producer_data["initial_states"] = initial_states
            virtual_producer_data["simulation_steps"] = simulation_steps_per_group
            virtual_producer_data["remaining_steps"] = remaining_steps_per_group
            virtual_producer_data.setdefault("total_dephase_steps", {})[
                target_group_id
            ] = 0
            virtual_producer_data["max_time_per_group"] = max_time_per_group

            worker_states = virtual_producer_data.get("worker_states")
            if worker_states is not None:
                worker_states[target_group_id] = {worker_id: "dephasing"}
                virtual_producer_data["worker_states"] = worker_states

            new_groups_config.append(
                {
                    "group_id": target_group_id,
                    "initial_state": target_state,
                    "max_time": max_time,
                }
            )
            worker_moves.append(
                {
                    "worker_id": worker_id,
                    "action": "move_to_existing",
                    "target_group_id": target_group_id,
                    "target_state": target_state,
                    "value": best_option["value"],
                    "max_time": max_time,
                }
            )

            new_existing_entry = {
                "group_id": target_group_id,
                "state": target_state,
                "value": 0.0,
                "type": "existing",
            }
            existing_value.append(new_existing_entry)

            for item in existing_value:
                if item["type"] == "existing":
                    item["value"] = self._calculate_existing_value(
                        item["group_id"],
                        item["state"],
                        {},
                        value_calculation_info,
                        virtual_producer_data,
                    )

            for item in new_value:
                if item["state"] == target_state:
                    if item["state"] not in used_new_group_states:
                        updated_max_time = self._compute_max_time_via_root_solver(
                            item["state"], value_calculation_info, virtual_producer_data
                        )
                        updated_value = self._calculate_value_with_fixed_max_time(
                            item["state"],
                            updated_max_time,
                            value_calculation_info,
                            virtual_producer_data,
                        )
                        item["value"] = updated_value
                        item["max_time"] = updated_max_time
                    item["value"] = 0.0
                    item["max_time"] = None
                    break

            used_new_group_states.add(target_state)

        return worker_moves, new_groups_config

    # ========================================
    # max_time決定関連メソッド
    # ========================================

    def compute_unknown_transition_hazard_rate_from_mean(
        self,
        mean_prob_and_total_steps: Tuple[float, float],
    ) -> float:
        if (
            not isinstance(mean_prob_and_total_steps, (tuple, list))
            or len(mean_prob_and_total_steps) != 2
        ):
            return 0.0

        mean_prob, total_steps = mean_prob_and_total_steps
        try:
            mean_prob = float(mean_prob)
        except Exception:
            mean_prob = 0.0
        try:
            total_steps = float(total_steps)
        except Exception:
            total_steps = 0.0

        if not np.isfinite(mean_prob):
            mean_prob = 0.0
        if not np.isfinite(total_steps) or total_steps < 0.0:
            total_steps = 0.0

        return float(mean_prob * total_steps / 100.0)

    def compute_max_time_intermediate_h(
        self,
        transition_prob_matrix: List[List[float]],
        initial_state: int,
        target_state: int,
        k: int,
        unknown_discovery_probability_per_state: Dict[int, float],
        cache_bucket: Optional[Dict] = None,
        matrix_token: Optional[int] = None,
    ) -> Optional[float]:
        expected_steps_per_state = self.expected_steps_per_state_until_kth_target_arrival(
            transition_prob_matrix=transition_prob_matrix,
            initial_state=initial_state,
            target_state=target_state,
            k=k,
            cache_bucket=cache_bucket,
            matrix_token=matrix_token,
        )

        if expected_steps_per_state is None:
            return None

        mean_prob, total_steps = self.compute_mean_unknown_transition_probability_per_step(
            expected_steps_per_state=expected_steps_per_state,
            unknown_discovery_probability_per_state=unknown_discovery_probability_per_state,
        )

        h = self.compute_unknown_transition_hazard_rate_from_mean(
            (mean_prob, total_steps)
        )
        return float(h)

    def _compute_max_time_via_root_solver(
        self,
        state: int,
        value_calculation_info: Dict,
        virtual_producer_data: Dict,
    ) -> int:
        if "selected_transition_matrix" not in value_calculation_info:
            raise ValueError(
                "selected_transition_matrix が value_calculation_info に含まれていません"
            )

        transition_prob_matrix = value_calculation_info["selected_transition_matrix"]
        state_idx = int(state)
        try:
            row = transition_prob_matrix[state_idx]
            self_loop_probability = float(row[state_idx])
        except Exception as exc:
            raise ValueError(
                f"state {state_idx} の自己ループ確率を取得できません"
            ) from exc

        if not np.isfinite(self_loop_probability):
            raise ValueError(
                f"state {state_idx} の自己ループ確率が有限値ではありません: {self_loop_probability}"
            )
        if not (0.0 <= self_loop_probability <= 1.0):
            print(transition_prob_matrix)
            raise ValueError(
                f"state {state_idx} の自己ループ確率が (0,1) の範囲外です: {self_loop_probability}"
            )

        splicer_info = value_calculation_info.get("splicer_info", {}) or {}
        initial_state_raw = splicer_info.get("current_state", state_idx)
        try:
            initial_state = int(initial_state_raw)
        except Exception as exc:
            raise ValueError(
                f"current_state を整数に変換できません: {initial_state_raw}"
            ) from exc

        if "unknown_discovery_probability_per_state" not in value_calculation_info:
            raise ValueError(
                "unknown_discovery_probability_per_state が value_calculation_info に含まれていません"
            )
        unknown_prob_map = dict(
            value_calculation_info["unknown_discovery_probability_per_state"]
        )
        if state_idx not in unknown_prob_map:
            raise ValueError(
                f"未知状態発見確率マップに state {state_idx} の項目がありません"
            )

        current_n_i = self._calculate_current_segment_count(
            state, value_calculation_info, virtual_producer_data
        )
        try:
            n_i = int(current_n_i)
        except Exception as exc:
            raise ValueError(
                f"state {state} の現在のセグメント数 current_n_i が整数に変換できません: {current_n_i}"
            ) from exc

        k_order = n_i + 1
        if k_order <= 0:
            raise ValueError(
                f"state {state} に対して算出された k_order が正ではありません: {k_order}"
            )

        cache_bucket = value_calculation_info.setdefault("_expected_steps_cache", {})
        matrix_token = value_calculation_info.get("_matrix_cache_token")

        h_value = self.compute_max_time_intermediate_h(
            transition_prob_matrix=transition_prob_matrix,
            initial_state=initial_state,
            target_state=state_idx,
            k=k_order,
            unknown_discovery_probability_per_state=unknown_prob_map,
            cache_bucket=cache_bucket,
            matrix_token=matrix_token,
        )

        if h_value is None or h_value > 0.5:
            return max(1, int(self.default_max_time * 0.4))

        if not np.isfinite(h_value) or h_value < 0.0:
            raise ValueError(f"state {state_idx} に対する h が不正です: {h_value}")

        dephasing_times = value_calculation_info.get("dephasing_times")
        if dephasing_times is None or state_idx not in dephasing_times:
            raise ValueError(f"dephasing_times に state {state_idx} の情報がありません")
        try:
            dephasing_time = float(dephasing_times[state_idx])
        except Exception as exc:
            raise ValueError(
                f"state {state_idx} の dephasing_time を float に変換できません: {dephasing_times[state_idx]}"
            ) from exc
        if dephasing_time < 0.0:
            raise ValueError(f"state {state_idx} の dephasing_time が負です: {dephasing_time}")

        max_time_root = self.solve_positive_root_for_placeholder_equation(
            self_loop_probability=self_loop_probability,
            h=float(h_value),
            dephasing_time=dephasing_time,
        )

        if not np.isfinite(max_time_root) or max_time_root <= 0.0:
            raise ValueError(f"state {state_idx} の方程式解が不正です: {max_time_root}")

        max_time_int = max(1, int(round(float(max_time_root))))

        lower_bound = max(1, int(np.floor(self.default_max_time * 0.4)))
        upper_bound = max(lower_bound, int(np.ceil(self.default_max_time * 1.6)))

        max_time_clipped = max(lower_bound, min(max_time_int, upper_bound))
        return max_time_clipped

    def solve_positive_root_for_placeholder_equation(
        self,
        self_loop_probability: float,
        h: float,
        dephasing_time: float,
    ) -> float:
        try:
            p = float(self_loop_probability)
            h = float(h)
            dephasing_time = float(dephasing_time)
        except Exception as exc:
            raise ValueError("self_loop_probability / h / dephasing_time を float に変換できません") from exc

        if not np.isfinite(p):
            raise ValueError(f"self_loop_probability が有限値ではありません: {p}")
        if not np.isfinite(h):
            raise ValueError(f"h が有限値ではありません: {h}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"self_loop_probability が (0,1) の範囲外です: {p}")
        if h < 0.0:
            raise ValueError(f"h が正ではありません: {h}")
        if dephasing_time < 0.0:
            raise ValueError(f"dephasing_time が負です: {dephasing_time}")

        lam = -np.log(p)
        rate_h = -np.log(1.0 - max(1e-10, h))
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError(f"λ の計算結果が不正です: {lam}")

        def f(x: float) -> float:
            if not np.isfinite(x):
                raise ValueError("根探索中に非有限値が生成されました")
            if x <= 0.0:
                return -2.0
            term1 = (1.0 - np.exp(-(rate_h + lam) * x)) / (rate_h + lam)
            term2 = np.exp(-rate_h * x) * ((1.0 - np.exp(-lam * x)) / lam + dephasing_time)
            return term1 - term2

        left = 0.0
        f_left = f(left)
        right = 1.0
        f_right = f(right)
        expand_iter = 0
        while (not np.isfinite(f_right) or f_right <= 0.0) and expand_iter < 60:
            right *= 2.0
            f_right = f(right)
            expand_iter += 1

        if not (
            np.isfinite(f_left)
            and np.isfinite(f_right)
            and f_left < 0.0
            and f_right > 0.0
        ):
            raise ValueError("適切なブラケットが見つかりませんでした")

        try:
            from scipy.optimize import brentq
        except ImportError as exc:
            raise ImportError("SciPy がインストールされていないため根を計算できません") from exc

        root = brentq(f, left, right, maxiter=200, xtol=1e-12, rtol=1e-12)
        if not np.isfinite(root) or root <= 0.0:
            raise ValueError(f"brentq で得られた根が不正です: {root}")
        return float(root)

    def _calculate_value_with_fixed_max_time(
        self,
        state: int,
        max_time: int,
        value_calculation_info: Dict,
        virtual_producer_data: Dict,
    ) -> float:
        monte_carlo_results = value_calculation_info.get("monte_carlo_results", {})
        segment_counts_per_simulation = monte_carlo_results.get(
            "segment_counts_per_simulation", []
        )
        K = value_calculation_info.get("monte_carlo_K", 1000)

        if not segment_counts_per_simulation:
            raise ValueError("モンテカルロシミュレーションの結果が空です。")

        n_i = self._calculate_current_segment_count(
            state, value_calculation_info, virtual_producer_data
        )

        exceed_count = 0
        for segment_count in segment_counts_per_simulation:
            segments_from_state_i = segment_count.get(state, 0)
            if segments_from_state_i > n_i:
                exceed_count += 1

        probability = exceed_count / K if K else 0.0

        transition_prob_matrix = value_calculation_info.get(
            "selected_transition_matrix", []
        )
        dephasing_times = value_calculation_info.get("dephasing_times", {})
        if state in dephasing_times:
            tau = dephasing_times[state]
        else:
            raise ValueError(f"State {state}のdephasing時間が見つかりません。")

        expected_time = self._expected_time_for_state(
            state, max_time, transition_prob_matrix
        )
        denom = expected_time + tau
        value = probability * (expected_time / denom) if denom > 0 else 0.0
        return float(value)

    def _expected_time_for_state(
        self,
        state: int,
        max_time: int,
        transition_prob_matrix: List[List[float]],
    ) -> float:
        """自己ループ確率に基づき指定max_timeでの期待シミュレーション時間を算出"""
        if max_time <= 0:
            return 0.0

        if 0 <= state < len(transition_prob_matrix) and state < len(transition_prob_matrix[state]):
            p = float(transition_prob_matrix[state][state])
        else:
            p = 0.0

        if np.isclose(p, 1.0):
            return float(max_time)

        denom = 1.0 - p
        if np.isclose(denom, 0.0):
            return float(max_time)

        return float((1.0 - (p ** max_time)) / denom)

    def _gather_value_calculation_info(
        self,
        virtual_producer_data: Dict,
        splicer_info: Dict,
        transition_matrix: Optional[List[List[int]]],
        producer_info: Dict,
        stationary_distribution=None,
        known_states=None,
        use_modified_matrix: bool = True,
    ) -> Dict:
        info_transition_matrix = _tx_matrix(
            transition_matrix, stationary_distribution, known_states, use_modified_matrix
        )
        mle_transition_matrix = info_transition_matrix["mle_transition_matrix"]

        if use_modified_matrix and info_transition_matrix["modified_transition_matrix"] is not None:
            modified_transition_matrix = info_transition_matrix["modified_transition_matrix"]
            normalized_matrix = modified_transition_matrix
        else:
            modified_transition_matrix = None
            normalized_matrix = mle_transition_matrix

        if use_modified_matrix:
            try:
                matrix_size = len(normalized_matrix) if normalized_matrix is not None else 0
            except Exception:
                matrix_size = 0
            if matrix_size > 0:
                for s in list(getattr(self, "new_states", set()) or set()):
                    try:
                        si = int(s)
                    except Exception:
                        continue
                    if 0 <= si < matrix_size:
                        row = normalized_matrix[si]
                        if (
                            isinstance(row, (list, tuple))
                            and si < len(row)
                            and si not in self.initial_self_loop_probability
                        ):
                            p_ii = row[si]
                            try:
                                self.initial_self_loop_probability[si] = float(p_ii)
                            except Exception:
                                pass

        K = 50
        H = 50
        dephasing_times = producer_info.get("t_phase_dict", {})
        decorrelation_times = producer_info.get("t_corr_dict", {})

        current_state = splicer_info.get("current_state")
        if current_state is None:
            raise ValueError("スプライサーの現在状態が取得できません")

        monte_carlo_results = _run_mc(
            current_state,
            normalized_matrix,
            set(known_states),
            K,
            H,
            dephasing_times,
            decorrelation_times,
            self.default_max_time,
        )

        simulation_steps_per_state = _steps_from_virtual(
            virtual_producer_data["initial_states"],
            virtual_producer_data["simulation_steps"],
            splicer_info,
        )

        expected_remaining_time = {}
        initial_states = virtual_producer_data["initial_states"]
        remaining_steps = virtual_producer_data["remaining_steps"]

        for group_id, initial_state in initial_states.items():
            if initial_state is not None and remaining_steps.get(group_id) is not None:
                n = remaining_steps[group_id]

                if (
                    initial_state < len(normalized_matrix)
                    and initial_state < len(normalized_matrix[initial_state])
                ):
                    p = normalized_matrix[initial_state][initial_state]
                else:
                    p = 0.0

                if p == 1.0:
                    expected_time = n
                else:
                    expected_time = (1 - p**n) / (1 - p) if p != 1.0 else n

                expected_remaining_time[group_id] = expected_time
            else:
                expected_remaining_time[group_id] = None

        unknown_discovery_probability_per_state = (
            self._compute_unknown_discovery_probability_per_state(
                known_states, transition_matrix, normalized_matrix
            )
        )

        return {
            "transition_matrix_info": info_transition_matrix,
            "modified_transition_matrix": modified_transition_matrix,
            "selected_transition_matrix": normalized_matrix,
            "use_modified_matrix": use_modified_matrix,
            "simulation_steps_per_state": simulation_steps_per_state,
            "expected_remaining_time": expected_remaining_time,
            "dephasing_times": dephasing_times,
            "decorrelation_times": decorrelation_times,
            "stationary_distribution": stationary_distribution,
            "monte_carlo_results": monte_carlo_results,
            "monte_carlo_K": K,
            "monte_carlo_H": H,
            "unknown_discovery_probability_per_state": unknown_discovery_probability_per_state,
            "splicer_info": splicer_info,
            "_matrix_cache_token": id(normalized_matrix),
            "_expected_steps_cache": {},
        }

    def _compute_unknown_discovery_probability_per_state(
        self, known_states, transition_matrix, normalized_matrix
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        ks = set(known_states or [])

        try:
            tm_n = len(transition_matrix) if transition_matrix is not None else 0
        except Exception:
            tm_n = 0
        try:
            nm_n = len(normalized_matrix) if normalized_matrix is not None else 0
        except Exception:
            nm_n = 0

        for s in ks:
            try:
                i = int(s)
            except Exception:
                continue

            c_ii = 0.0
            if 0 <= i < tm_n:
                row = transition_matrix[i]
                if isinstance(row, (list, tuple)) and i < len(row):
                    try:
                        c_ii = float(row[i])
                    except Exception:
                        c_ii = 0.0

            p_ii = 0.0
            if 0 <= i < nm_n:
                row = normalized_matrix[i]
                if isinstance(row, (list, tuple)) and i < len(row):
                    try:
                        p_ii = float(row[i])
                    except Exception:
                        raise ValueError(f"State {i}の自己ループ確率が不正です: {row[i]}")

            p0_ii = self.initial_self_loop_probability.get(i, p_ii)
            try:
                p0_ii = float(p0_ii)
            except Exception:
                raise ValueError(f"State {i}の初回自己ループ確率が不正です: {p0_ii}")

            if not np.isfinite(p_ii) or p_ii <= 0.0:
                result[i] = 0.0
                continue

            alpha = (float(c_ii) + 1.0) * (1.0 - (p0_ii / p_ii))

            if not np.isfinite(alpha):
                alpha = 0.0
            alpha = max(0.0, min(1.0, alpha))

            denom = float(c_ii) + 1.0
            prob = (alpha * p_ii / denom) if denom > 0 else 0.0

            if not np.isfinite(prob):
                prob = 0.0
            prob = max(0.0, min(1.0, prob))

            result[i] = prob

        return result

    def expected_steps_per_state_until_kth_target_arrival(
        self,
        transition_prob_matrix: List[List[float]],
        initial_state: int,
        target_state: int,
        k: int,
        cache_bucket: Optional[Dict] = None,
        matrix_token: Optional[int] = None,
    ) -> Optional[Dict[int, float]]:
        cache_map = cache_bucket if isinstance(cache_bucket, dict) else None
        cache_key = None
        if cache_map is not None:
            cache_key = (matrix_token, int(initial_state), int(target_state), int(k))
            cached = cache_map.get(cache_key)
            if cached is not None:
                return dict(cached)
        P_full = np.asarray(transition_prob_matrix, dtype=float)
        if P_full.ndim != 2 or P_full.shape[0] != P_full.shape[1]:
            raise ValueError("transition_prob_matrix は正方行列である必要があります。")

        total_states = P_full.shape[0]
        if not (0 <= initial_state < total_states) or not (0 <= target_state < total_states):
            raise ValueError("initial_state / target_state が行列サイズの範囲外です。")
        if k <= 0:
            return {i: 0.0 for i in range(total_states)}

        adjacency = []
        for i in range(total_states):
            row = P_full[i]
            adjacency.append([int(j) for j, p in enumerate(row) if p > 0.0])

        index_counter = 0
        indices = [-1] * total_states
        lowlink = [0] * total_states
        on_stack = [False] * total_states
        stack = []
        component_id = [-1] * total_states
        component_count = 0

        def strongconnect(v: int) -> None:
            nonlocal index_counter, component_count
            indices[v] = index_counter
            lowlink[v] = index_counter
            index_counter += 1
            stack.append(v)
            on_stack[v] = True

            for w in adjacency[v]:
                if indices[w] == -1:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                while stack:
                    w = stack.pop()
                    on_stack[w] = False
                    component_id[w] = component_count
                    if w == v:
                        break
                component_count += 1

        for v in range(total_states):
            if indices[v] == -1:
                strongconnect(v)

        component_adj = [set() for _ in range(component_count)]
        for i in range(total_states):
            ci = component_id[i]
            for j in adjacency[i]:
                cj = component_id[j]
                if ci != cj:
                    component_adj[ci].add(cj)

        reachable_components = set()
        stack_components = [component_id[int(initial_state)]]
        while stack_components:
            comp = stack_components.pop()
            if comp in reachable_components:
                continue
            reachable_components.add(comp)
            stack_components.extend(component_adj[comp])

        reachable_states = {
            i for i in range(total_states) if component_id[i] in reachable_components
        }
        if int(target_state) not in reachable_states:
            return None

        state_list = sorted(reachable_states)
        if not state_list:
            return None

        state_index = {state: idx for idx, state in enumerate(state_list)}
        reduced_initial = state_index.get(int(initial_state))
        reduced_target = state_index.get(int(target_state))
        if reduced_initial is None or reduced_target is None:
            return None

        # Exact, accelerated computation without constructing the nk system.
        # Build reduced transition matrix and solve a single n x n system with two RHS.
        P = P_full[np.ix_(state_list, state_list)]
        n = P.shape[0]

        tgt = int(reduced_target)

        # Construct D: zero-out target column except its diagonal element, to keep
        # within-layer transitions while removing inter-layer jumps into target.
        D = np.array(P, dtype=float, copy=True)
        if n > 0:
            D[:, tgt] = 0.0
            D[tgt, tgt] = float(P[tgt, tgt])

        # Solve (I - D^T) x = b for two RHS: b = e_init and b = e_tgt
        A = np.eye(n, dtype=float) - D.T
        B = np.zeros((n, 2), dtype=float)
        B[int(reduced_initial), 0] = 1.0  # e_init
        B[tgt, 1] = 1.0                   # e_tgt

        try:
            X = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(X)):
            return None

        x0 = X[:, 0]  # visits vector for k=1 (single layer)
        xt = X[:, 1]  # response to injection at target

        # Compute c1 and gamma scalars controlling additional layers' contributions
        p_col = np.array(P[:, tgt], dtype=float)
        p_col[tgt] = 0.0
        c1 = float(np.dot(p_col, x0))
        gamma = float(np.dot(p_col, xt))

        # Total expected visits summed over layers m=0..k-1
        if k <= 1:
            expected_steps = x0
        else:
            # Sum_{m=1..k-1} c1 * gamma^{m-1}
            if not np.isfinite(gamma):
                return None
            if abs(1.0 - gamma) < 1e-12:
                sum_c = c1 * (k - 1)
            else:
                try:
                    sum_c = c1 * (1.0 - (gamma ** (k - 1))) / (1.0 - gamma)
                except OverflowError:
                    return None
            expected_steps = x0 + xt * sum_c

        if not np.all(np.isfinite(expected_steps)):
            return None

        result = {i: 0.0 for i in range(total_states)}
        for local_idx, state in enumerate(state_list):
            result[state] = float(expected_steps[local_idx])
        if cache_map is not None and cache_key is not None:
            cache_map[cache_key] = result
        return result

    def compute_mean_unknown_transition_probability_per_step(
        self,
        expected_steps_per_state: Dict[int, float],
        unknown_discovery_probability_per_state: Dict[int, float],
    ) -> Tuple[float, float]:
        if not expected_steps_per_state:
            return 0.0, 0.0

        num = 0.0
        den = 0.0

        states = set(expected_steps_per_state.keys()) | set(
            (unknown_discovery_probability_per_state or {}).keys()
        )
        for s in states:
            steps = expected_steps_per_state.get(s, 0.0)
            prob = (unknown_discovery_probability_per_state or {}).get(s, 0.0)

            try:
                steps = float(steps)
            except Exception:
                steps = 0.0
            try:
                prob = float(prob)
            except Exception:
                prob = 0.0

            if not np.isfinite(steps) or steps <= 0.0:
                continue
            if not np.isfinite(prob):
                prob = 0.0
            if prob < 0.0:
                prob = 0.0
            elif prob > 1.0:
                prob = 1.0

            num += steps * prob
            den += steps

        if den <= 0.0:
            return 0.0, 0.0

        mean_prob = num / den
        if not np.isfinite(mean_prob):
            return 0.0, float(den)
        if mean_prob < 0.0:
            return 0.0, float(den)
        if mean_prob > 1.0:
            return 1.0, float(den)
        return float(mean_prob), float(den)

    def _create_virtual_producer_data(self, producer_info: Dict) -> Dict:
        data = _create_vp_data_util(producer_info)
        max_time_per_group: Dict[int, Optional[int]] = {}
        for gid, ginfo in producer_info.get("groups", {}).items():
            max_time_per_group[gid] = ginfo.get("max_time")
        data["max_time_per_group"] = max_time_per_group
        return data
