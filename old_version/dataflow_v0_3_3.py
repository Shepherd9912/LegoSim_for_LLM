# dataflow.py
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from utils_v0_3 import NPUConfig, PIMConfig, PerformanceResult

# ==============================================================================
# 1. 核心枚举与模型定义 (已修正和补完)
# ==============================================================================
class ExecutionTarget(Enum):
    NPU_ONLY = auto()
    PIM_ONLY = auto()
    SPLIT_EXECUTION = auto()
    MEMORY_BOUND = auto()

class SplitDimension(Enum):
    M = auto()
    K = auto()
    N = auto()

class PIMType(Enum):
    """描述通道内PIM的物理硬件类型"""
    COMMON = auto()
    ATTENTION_FFN = auto()

# --- 新增: 算子类别枚举 ---
class OpCategory(Enum):
    """描述算子自身的计算类别，用于调度"""
    GENERAL = auto()        # 通用计算，如矩阵乘
    ATTENTION = auto()      # Attention核心计算，如QK, SV
    FFN = auto()            # FFN计算
    MEMORY = auto()         # 访存密集型

@dataclass
class InterconnectModel:
    fixed_latency_cycles: int
    cycles_per_byte: float

# ==============================================================================
# 2. 硬件拓扑定义 (保持不变)
# ==============================================================================
@dataclass
class NPUCoreConfig:
    core_id: int
    serves_channel_ids: List[int]

@dataclass
class ChannelConfig:
    channel_id: int
    pim_type: Optional[PIMType]
    has_npu_slice: bool = True

# ==============================================================================
# 3. 数据流的核心数据结构 (已修正)
# ==============================================================================
@dataclass
class OperatorMapping:
    """定义单个算子的完整映射策略"""
    operator_name: str
    op_category: OpCategory  # --- 修改: 使用OpCategory ---
    execution_target: ExecutionTarget
    split_ratio_pim: float = 0.0
    split_dimension: Optional[SplitDimension] = None
    
    # pretty print for easier debugging
    def __str__(self):
        details = [f"Category: {self.op_category.name}", f"Target: {self.execution_target.name}"]
        if self.execution_target == ExecutionTarget.SPLIT_EXECUTION:
            details.append(f"Split(PIM: {self.split_ratio_pim:.2f}) along Dim '{self.split_dimension.name}'")
        return ", ".join(details)

@dataclass
class ExecutionStage:
    stage_description: str
    operator_mappings: Dict[str, OperatorMapping]

@dataclass
class LayerDataflow:
    stages: List[ExecutionStage]

@dataclass
class ExecutionPlan:
    description: str
    npu_config: NPUConfig
    pim_config: PIMConfig
    npu_topology: List[NPUCoreConfig]
    channel_topology: List[ChannelConfig]
    interconnect_model: InterconnectModel
    prefill_dataflow: LayerDataflow
    decoding_dataflow: LayerDataflow

# ==============================================================================
# 4. 蓝图模板 (Blueprint Templates) - 已更新以包含OpCategory
# ==============================================================================
DECODING_PIPELINE_BLUEPRINT = [
    {"description": "Input Norm & QKV Projections", "operators": {
        "Input_RMSNorm": OpCategory.MEMORY,
        "Attention::QKV_Projection": OpCategory.GENERAL
    }},
    {"description": "Attention Core Computation", "operators": {
        "Attention::Score_Computation_QK": OpCategory.ATTENTION,
        "Attention::Softmax": OpCategory.MEMORY
    }},
    {"description": "Attention SV & Output", "operators": {
        "Attention::Context_Computation_SV": OpCategory.ATTENTION,
        "Attention::Output_Projection": OpCategory.GENERAL
    }},
    {"description": "Post Norm & FFN Projections", "operators": {
        "Post_RMSNorm": OpCategory.MEMORY,
        "FFN::Gate_Up_Projection": OpCategory.FFN,
    }},
    {"description": "FFN Activation & Down Projection", "operators": {
        "FFN::Activation(SiLU)": OpCategory.MEMORY,
        "FFN::Down_Projection": OpCategory.FFN,
    }}
]

PREFILL_BLUEPRINT = [
    {"description": "Prefill Computation", "operators": {
        "Prefill_QKV": OpCategory.GENERAL,
        "Prefill_Attention": OpCategory.ATTENTION,
        "Prefill_FFN": OpCategory.FFN
    }}
]
# ==============================================================================
# 5. 确定性计划构造器 (已修正)
# ==============================================================================

def create_plan_from_configs(
    hw_config: Dict[str, Any],
    dataflow_config: Dict[str, Any],
    description: str = "DSE Generated Plan"
) -> ExecutionPlan:
    
    # --- 1. 实例化硬件和通信模型 (保持不变) ---
    npu_conf = NPUConfig(**hw_config['npu'])
    pim_conf = PIMConfig(**hw_config['pim'])
    interconnect = InterconnectModel(**hw_config['interconnect'])
    
    npu_topo = [NPUCoreConfig(**cfg) for cfg in hw_config['topology']['npu_cores']]
    
    channel_topo = []
    for cfg in hw_config['topology']['channels']:
        pim_type_str = cfg.get('pim_type')
        pim_type_enum = PIMType[pim_type_str] if pim_type_str else None
        channel_topo.append(ChannelConfig(channel_id=cfg['channel_id'], pim_type=pim_type_enum))
    
    # --- 2. 内部辅助函数 (已修正) ---
    def _build_layer_dataflow(blueprint: List[Dict], config: Dict) -> LayerDataflow:
        stages = []
        for stage_def in blueprint:
            mappings = {}
            # blueprint现在直接提供op_name和op_category
            for op_name, op_category in stage_def['operators'].items():
                op_config = config.get(op_name)
                if not op_config:
                    raise ValueError(f"Dataflow configuration missing for operator: {op_name}")

                target_enum = ExecutionTarget[op_config['target']]
                split_dim_str = op_config.get('split_dimension')
                split_dim_enum = SplitDimension[split_dim_str] if split_dim_str else None

                mappings[op_name] = OperatorMapping(
                    operator_name=op_name,
                    op_category=op_category, # --- 从蓝图中获取 ---
                    execution_target=target_enum,
                    split_ratio_pim=op_config.get('split_ratio_pim', 0.0),
                    split_dimension=split_dim_enum
                )
            stages.append(ExecutionStage(stage_def['description'], mappings))
        return LayerDataflow(stages=stages)

    # --- 3. 实例化数据流 (保持不变) ---
    prefill_flow = _build_layer_dataflow(PREFILL_BLUEPRINT, dataflow_config['prefill'])
    decoding_flow = _build_layer_dataflow(DECODING_PIPELINE_BLUEPRINT, dataflow_config['decoding'])

    # --- 4. 组装并返回 (保持不变) ---
    return ExecutionPlan(
        description=description,
        npu_config=npu_conf,
        pim_config=pim_conf,
        npu_topology=npu_topo,
        channel_topology=channel_topo,
        interconnect_model=interconnect,
        prefill_dataflow=prefill_flow,
        decoding_dataflow=decoding_flow
    )

# ==============================================================================
# 6. 主执行块 (已修正以匹配新架构)
# ==============================================================================
if __name__ == "__main__":
    
    print("--- Dataflow v3.1: This script defines data structures and a deterministic constructor for DSE. ---")
    print("--- Running main block to demonstrate usage with corrected logic. ---\n")

    # --- 1. 模拟DSE引擎采样得出的硬件配置 ---
    sampled_hw_config = {
        "npu": {"num_cores": 8, "num_channels": 32, "channels_per_core": 4, "clock_ghz": 2.0},
        "pim": {"total_channels": 32, "clock_ghz": 2.0, "global_buffer_size_per_channel_kb": 128},
        "interconnect": {"fixed_latency_cycles": 120, "cycles_per_byte": 0.05},
        "topology": {
            "npu_cores": [{"core_id": i, "serves_channel_ids": list(range(i*4, (i+1)*4))} for i in range(8)],
            # --- 使用新的PIMType枚举字符串 ---
            "channels": [{"channel_id": i, "pim_type": "ATTENTION_FFN"} for i in range(32)]
        }
    }

    # --- 2. 模拟DSE引擎采样得出的数据流配置 ---
    # --- 注意: required_pim_type 已被移除 ---
    sampled_dataflow_config = {
        "prefill": {
            "Prefill_QKV": {"target": "NPU_ONLY"},
            "Prefill_Attention": {"target": "NPU_ONLY"},
            "Prefill_FFN": {"target": "SPLIT_EXECUTION", "split_ratio_pim": 0.1, "split_dimension": "N"}
        },
        "decoding": {
            "Input_RMSNorm": {"target": "MEMORY_BOUND"},
            "Attention::QKV_Projection": {"target": "NPU_ONLY"},
            "Attention::Score_Computation_QK": {"target": "PIM_ONLY"},
            "Attention::Softmax": {"target": "MEMORY_BOUND"},
            "Attention::Context_Computation_SV": {"target": "PIM_ONLY"},
            "Attention::Output_Projection": {"target": "NPU_ONLY"},
            "Post_RMSNorm": {"target": "MEMORY_BOUND"},
            "FFN::Gate_Up_Projection": {"target": "SPLIT_EXECUTION", "split_ratio_pim": 0.4, "split_dimension": "N"},
            "FFN::Activation(SiLU)": {"target": "MEMORY_BOUND"},
            "FFN::Down_Projection": {"target": "SPLIT_EXECUTION", "split_ratio_pim": 0.5, "split_dimension": "N"}
        }
    }

    # --- 3. 调用构造器 ---
    print("DSE Engine: Calling constructor with new configs...")
    my_execution_plan = create_plan_from_configs(
        hw_config=sampled_hw_config,
        dataflow_config=sampled_dataflow_config,
        description="Plan with ATTENTION_FFN PIMs and corrected dataflow configs"
    )
    print("DSE Engine: Plan created successfully.\n")

    # --- 4. 打印验证 ---
    def pretty_print_plan(plan: ExecutionPlan):
        print("=" * 90)
        print(f"Execution Plan Details: {plan.description}")
        print("-" * 90)
        print(f"Hardware: {plan.npu_config.num_cores} NPU cores, {plan.channel_topology[0].pim_type.name} PIM architecture")
        print("-" * 90)
        print("Decoding Dataflow (Pipelined):")
        for i, stage in enumerate(plan.decoding_dataflow.stages):
            print(f"  --- Stage {i+1}: {stage.stage_description} ---")
            for op_name, mapping in stage.operator_mappings.items():
                print(f"    - {op_name:<35}: {mapping}")
        # ... (可以添加Prefill打印)
        print("=" * 90)

    pretty_print_plan(my_execution_plan)
