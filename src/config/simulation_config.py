"""シミュレーション設定クラス"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Any
from common import Validator


@dataclass
class SimulationConfig:
    """シミュレーション設定を管理するクラス"""
    # 乱数シード設定
    random_seed: int = 42
    
    # システム設定
    num_states: int = 10  # 状態数
    self_loop_prob_mean: float = 0.99  # 自己ループの平均確率
    
    # 詳細釣り合い方式のパラメータ
    stationary_concentration: float = 1.0  # 定常分布生成時のディリクレ分布濃度パラメータ(大きいほど均等に近い)
    connectivity: float = 0.8  # 状態間接続性 (0.0-1.0), 1.0で全状態が接続

    # dephasing時間設定
    t_phase_mean: float = 2.0
    t_phase_constant_mode: bool = True
    
    # decorrelation時間設定
    t_corr_mean: float = 2.0
    t_corr_constant_mode: bool = True
    
    # 並列計算設定
    num_workers: int = 10
    
    # シミュレーション設定
    max_simulation_time: int = 10  # シミュレーションの最大時間ステップ数

    # 初期状態設定
    initial_splicer_state: int = 0  # Splicerとschedulerの初期状態（0～num_states-1の範囲で指定）
    
    # スケジューリング戦略設定
    scheduling_strategy: str = 'parsplice'  # 使用するスケジューリング戦略 ('parrep', 'csparsplice', 'parsplice', 'epsplice')
    strategy_params: Dict[str, Any] = None  # 戦略固有のパラメータ
    
    # 出力設定
    output_interval: int = 5
    minimal_output: bool = True  # 詳細出力を抑制するフラグ

    raw_data_only: bool = False  # 生データのみ出力モード
    save_legacy_format: bool = False  # 旧形式での生データ

    segment_storage_animation: bool = False  # セグメント貯蓄状況の動画化
    trajectory_animation: bool = False  # トラジェクトリの動画化
    
    # トラジェクトリ設定
    max_trajectory_length: int = 1000000  # トラジェクトリの最大長
    
    def __post_init__(self):
        """dataclassの初期化後処理"""
        if self.strategy_params is None:
            self.strategy_params = {}
    
    def validate(self) -> None:
        """設定値のバリデーション"""
        Validator.validate_positive_integer(self.num_states, "num_states")
        Validator.validate_positive_integer(self.num_workers, "num_workers")
        Validator.validate_positive_integer(self.max_simulation_time, "max_simulation_time")
        Validator.validate_positive_integer(self.output_interval, "output_interval")
        Validator.validate_positive_integer(self.max_trajectory_length, "max_trajectory_length")
        Validator.validate_state_range(self.initial_splicer_state, self.num_states, "initial_splicer_state")
    
    
    @classmethod
    def from_xml(cls, xml_path: str = None, create_if_missing: bool = True) -> 'SimulationConfig':
        """XMLファイルから設定を読み込み、SimulationConfigインスタンスを作成
        
        Args:
            xml_path: XMLファイルのパス。Noneの場合は'simulation_config.xml'を使用
            create_if_missing: ファイルが存在しない場合にデフォルト設定で新規作成するか
            
        Returns:
            SimulationConfig: XMLから読み込んだ設定を持つインスタンス
        """
        if xml_path is None:
            # デフォルトではプロジェクトルートのsimulation_config.xmlを使用
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            xml_path = os.path.join(project_root, 'simulation_config.xml')
        
        if not os.path.exists(xml_path):
            if create_if_missing:
                # ファイルが存在しない場合、デフォルト設定で新規作成
                print(f"設定ファイルが見つかりません。デフォルト設定で新規作成します: {xml_path}")
                instance = cls()  # デフォルト設定でインスタンス作成
                instance.to_xml(xml_path)  # ファイルに保存
                return instance
            else:
                raise FileNotFoundError(f"設定ファイルが見つかりません: {xml_path}")
        
        # XMLファイルを解析
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # XMLから設定値を抽出
        config_data = cls._parse_xml_config(root)
        
        # インスタンスを作成
        instance = cls(**config_data)
        instance.validate()
        return instance
    
    @staticmethod
    def _parse_xml_config(root: ET.Element) -> Dict[str, Any]:
        """XMLルート要素から設定値を抽出"""
        config_data = {}
        
        # 基本設定
        basic = root.find('basic')
        if basic is not None:
            config_data['random_seed'] = int(basic.find('random_seed').text)
            
            system = basic.find('system')
            if system is not None:
                config_data['num_states'] = int(system.find('num_states').text)
                config_data['self_loop_prob_mean'] = float(system.find('self_loop_prob_mean').text)
            
            detailed_balance = basic.find('detailed_balance')
            if detailed_balance is not None:
                config_data['stationary_concentration'] = float(detailed_balance.find('stationary_concentration').text)
                config_data['connectivity'] = float(detailed_balance.find('connectivity').text)
        
        # 時間設定
        timing = root.find('timing')
        if timing is not None:
            dephasing = timing.find('dephasing')
            if dephasing is not None:
                config_data['t_phase_mean'] = float(dephasing.find('t_phase_mean').text)
                config_data['t_phase_constant_mode'] = dephasing.find('t_phase_constant_mode').text.lower() == 'true'
            
            decorrelation = timing.find('decorrelation')
            if decorrelation is not None:
                config_data['t_corr_mean'] = float(decorrelation.find('t_corr_mean').text)
                config_data['t_corr_constant_mode'] = decorrelation.find('t_corr_constant_mode').text.lower() == 'true'
        
        # 並列計算設定
        parallel = root.find('parallel')
        if parallel is not None:
            config_data['num_workers'] = int(parallel.find('num_workers').text)
            config_data['max_simulation_time'] = int(parallel.find('max_simulation_time').text)
            config_data['initial_splicer_state'] = int(parallel.find('initial_splicer_state').text)
        
        # スケジューリング設定
        scheduling = root.find('scheduling')
        if scheduling is not None:
            config_data['scheduling_strategy'] = scheduling.find('strategy').text
            config_data['strategy_params'] = {}  # 現在は空の辞書
        
        # 出力設定
        output = root.find('output')
        if output is not None:
            config_data['output_interval'] = int(output.find('interval').text)
            config_data['minimal_output'] = output.find('minimal_output').text.lower() == 'true'
            config_data['raw_data_only'] = output.find('raw_data_only').text.lower() == 'true'
            config_data['save_legacy_format'] = output.find('save_legacy_format').text.lower() == 'true'
        
        # 動画化設定
        animation = root.find('animation')
        if animation is not None:
            config_data['segment_storage_animation'] = animation.find('segment_storage_animation').text.lower() == 'true'
            config_data['trajectory_animation'] = animation.find('trajectory_animation').text.lower() == 'true'
        
        # トラジェクトリ設定
        trajectory = root.find('trajectory')
        if trajectory is not None:
            config_data['max_trajectory_length'] = int(trajectory.find('max_trajectory_length').text)
        
        return config_data
    
    def to_xml(self, xml_path: str = None) -> None:
        """現在の設定をXMLファイルに保存
        
        Args:
            xml_path: 保存先のXMLファイルのパス。Noneの場合は'simulation_config.xml'を使用
        """
        if xml_path is None:
            # デフォルトではプロジェクトルートのsimulation_config.xmlを使用
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            xml_path = os.path.join(project_root, 'simulation_config.xml')
        
        # XMLツリーを構築
        root = ET.Element('simulation_config')
        root.append(ET.Comment(' ParSpliceSIM シミュレーション設定ファイル '))
        
        # 基本設定
        basic = ET.SubElement(root, 'basic')
        basic.append(ET.Comment(' 基本設定 '))
        
        # 乱数シード
        ET.SubElement(basic, 'random_seed').text = str(self.random_seed)
        basic.append(ET.Comment(' 再現可能な結果を得るためのランダムシード値 '))
        
        # システム設定
        system = ET.SubElement(basic, 'system')
        system.append(ET.Comment(' システム設定 '))
        ET.SubElement(system, 'num_states').text = str(self.num_states)
        system.append(ET.Comment(' マルコフ状態の数 '))
        ET.SubElement(system, 'self_loop_prob_mean').text = str(self.self_loop_prob_mean)
        system.append(ET.Comment(' 自己ループの平均確率 '))
        
        # 詳細釣り合い方式のパラメータ
        detailed_balance = ET.SubElement(basic, 'detailed_balance')
        detailed_balance.append(ET.Comment(' 詳細釣り合い方式のパラメータ '))
        ET.SubElement(detailed_balance, 'stationary_concentration').text = str(self.stationary_concentration)
        detailed_balance.append(ET.Comment(' 定常分布生成時のディリクレ分布濃度パラメータ（大きいほど均等に近い） '))
        ET.SubElement(detailed_balance, 'connectivity').text = str(self.connectivity)
        detailed_balance.append(ET.Comment(' 状態間接続性（0.0-1.0）、1.0で全状態が接続 '))
        
        # 時間設定
        timing = ET.SubElement(root, 'timing')
        timing.append(ET.Comment(' 時間設定 '))
        
        # Dephasing時間設定
        dephasing = ET.SubElement(timing, 'dephasing')
        dephasing.append(ET.Comment(' Dephasing時間設定 '))
        ET.SubElement(dephasing, 't_phase_mean').text = str(self.t_phase_mean)
        dephasing.append(ET.Comment(' dephasing時間の平均値 '))
        ET.SubElement(dephasing, 't_phase_constant_mode').text = str(self.t_phase_constant_mode).lower()
        dephasing.append(ET.Comment(' 定数モードの使用有無 '))
        
        # Decorrelation時間設定
        decorrelation = ET.SubElement(timing, 'decorrelation')
        decorrelation.append(ET.Comment(' Decorrelation時間設定 '))
        ET.SubElement(decorrelation, 't_corr_mean').text = str(self.t_corr_mean)
        decorrelation.append(ET.Comment(' decorrelation時間の平均値 '))
        ET.SubElement(decorrelation, 't_corr_constant_mode').text = str(self.t_corr_constant_mode).lower()
        decorrelation.append(ET.Comment(' 定数モードの使用有無 '))
        
        # 並列計算設定
        parallel = ET.SubElement(root, 'parallel')
        parallel.append(ET.Comment(' 並列計算設定 '))
        ET.SubElement(parallel, 'num_workers').text = str(self.num_workers)
        parallel.append(ET.Comment(' 使用するワーカープロセス数 '))
        ET.SubElement(parallel, 'max_simulation_time').text = str(self.max_simulation_time)
        parallel.append(ET.Comment(' シミュレーションの最大時間ステップ数 '))
        ET.SubElement(parallel, 'initial_splicer_state').text = str(self.initial_splicer_state)
        parallel.append(ET.Comment(' SplicerとSchedulerの初期状態（0～num_states-1の範囲で指定） '))
        
        # スケジューリング戦略設定
        scheduling = ET.SubElement(root, 'scheduling')
        scheduling.append(ET.Comment(' スケジューリング戦略設定 '))
        ET.SubElement(scheduling, 'strategy').text = self.scheduling_strategy
        scheduling.append(ET.Comment(' 使用するスケジューリング戦略 '))
        scheduling.append(ET.Comment(' 選択肢: parrep, csparsplice, parsplice, epsplice '))
        
        strategy_params = ET.SubElement(scheduling, 'strategy_params')
        strategy_params.append(ET.Comment(' 戦略固有のパラメータ '))
        strategy_params.append(ET.Comment(' 必要に応じて戦略固有のパラメータをここに追加 '))
        
        # 出力設定
        output = ET.SubElement(root, 'output')
        output.append(ET.Comment(' 出力設定 '))
        ET.SubElement(output, 'interval').text = str(self.output_interval)
        output.append(ET.Comment(' 出力間隔 '))
        ET.SubElement(output, 'minimal_output').text = str(self.minimal_output).lower()
        output.append(ET.Comment(' 詳細出力を抑制するフラグ '))
        ET.SubElement(output, 'raw_data_only').text = str(self.raw_data_only).lower()
        output.append(ET.Comment(' 生データのみ出力モード '))
        ET.SubElement(output, 'save_legacy_format').text = str(self.save_legacy_format).lower()
        output.append(ET.Comment(' 旧形式での生データ保存 '))
        
        # 動画化設定
        animation = ET.SubElement(root, 'animation')
        animation.append(ET.Comment(' 動画化設定 '))
        ET.SubElement(animation, 'segment_storage_animation').text = str(self.segment_storage_animation).lower()
        animation.append(ET.Comment(' セグメント貯蓄状況の動画化 '))
        ET.SubElement(animation, 'trajectory_animation').text = str(self.trajectory_animation).lower()
        animation.append(ET.Comment(' トラジェクトリの動画化 '))
        
        # トラジェクトリ設定
        trajectory = ET.SubElement(root, 'trajectory')
        trajectory.append(ET.Comment(' トラジェクトリ設定 '))
        ET.SubElement(trajectory, 'max_trajectory_length').text = str(self.max_trajectory_length)
        trajectory.append(ET.Comment(' トラジェクトリの最大長 '))
        
        # XMLファイルに保存
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)  # インデントを追加
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)