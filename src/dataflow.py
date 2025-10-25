# dataflow_v0_4_1.py
from enum import Enum, auto # 从 enum 模块导入 Enum 类和 auto 函数，用于创建枚举类型。
from typing import Dict, List, Optional, Any # 从 typing 模块导入类型提示。
from dataclasses import dataclass, field # 从 dataclasses 模块导入 dataclass 装饰器和 field 函数。

# ==============================================================================
# SECTION 1: UPGRADED DEPENDENCIES
# ==============================================================================
from utils import NPUConfig, PIMConfig # 从我们自定义的 utils_v0_4 模块中导入硬件配置类 NPUConfig 和 PIMConfig。

# ==============================================================================
# SECTION 2: CORE ENUMS AND DEFINITIONS
# ==============================================================================

class ExecutionTarget(Enum): # 定义一个名为 ExecutionTarget 的枚举类，表示一个操作将在哪种硬件上执行。
    NPU_ONLY = auto() # 表示操作仅在 NPU 上执行。
    PIM_ONLY = auto() # 表示操作仅在 PIM 上执行。
    SPLIT_EXECUTION = auto() # 表示操作将在 NPU 和 PIM 上分割执行。
    MEMORY_BOUND = auto() # 表示这是一个内存密集型操作（如Softmax），其性能主要由内存带宽决定。

class SplitDimension(Enum): # 定义一个枚举类，表示当操作被分割时，沿着哪个维度进行分割。
    M = auto() # 沿着矩阵的 M 维度（通常是批处理或序列长度维度）分割。
    K = auto() # 沿着矩阵的 K 维度（内积维度）分割。
    N = auto() # 沿着矩阵的 N 维度（输出通道维度）分割。

# PIM的架构。每个channel有1个通用PIM还是两个分别针对Attention和FFN的专用PIM
class PIMType(Enum): # 定义一个枚举类，表示 PIM 单元的类型或功能。
    COMMON = auto() # 通用型 PIM 单元。
    ATTENTION_FFN = auto() # 专门用于处理 Attention 和 FFN 操作的 PIM 单元。

class OpCategory(Enum): # 定义一个枚举类，用于对不同的操作进行分类。
    GENERAL = auto() # 通用操作类别。
    ATTENTION = auto() # 属于 Attention 模块的操作。
    FFN = auto() # 属于前馈网络 (Feed-Forward Network) 模块的操作。
    MEMORY = auto() # 属于内存密集型操作。

# 参考BlockPIM
# --- NEW: Enum for different workload balancing / scheduling strategies ---
class SchedulingPolicy(Enum): # 定义一个枚举类，表示不同的任务调度策略。
    HEAD_STATIC_MAPPING = auto()     # 粗粒度策略：将任务静态地映射到固定的硬件资源上（例如，某些头固定在某些PIM通道）。
    TASK_DYNAMIC_DISPATCH = auto()   # 细粒度策略：将所有任务放入一个全局池中，动态地分派给下一个可用的资源。
    BLOCK_GREEDY_MAPPING = auto()    # 中粒度策略：在本地资源块内（例如，一个NPU核心及其关联的PIM通道）进行贪婪调度。

# 参考BlockPIM
# --- NEW: Enum for different physical KV Cache memory layouts ---
class MemoryLayoutPolicy(Enum): # 定义一个枚举类，表示 KV Cache 在物理内存中的不同布局策略。
    REQ_PAR = auto()        # NeuPIM风格：一个请求的所有 KV Cache 数据都存放在单个内存通道中。
    REQ_HEAD_PAR = auto()   # AttAcc风格：一个请求的 KV Cache 数据按注意力头被分片到不同的内存通道中。
    BLOCK_PIM = auto()      # BlockPIM风格：一个请求的 KV Cache 数据被分块并交错存储在所有内存通道中。

@dataclass # 使用 dataclass 装饰器。
class InterconnectModel: # 定义一个数据类，用于模拟硬件单元之间的互连成本。
    fixed_latency_cycles: int # 固定的通信延迟（时钟周期）。
    cycles_per_byte_intra_group: float # 组内（如NPU核心与其直连通道之间）通信时，每字节传输所需的时钟周期。
    cycles_per_byte_inter_group: float # 组间（如NPU核心与其他核心的通道之间）通信时，每字节传输所需的时钟周期。

@dataclass # 使用 dataclass 装饰器。
class NPUCoreConfig: # 定义一个数据类，用于描述单个 NPU 核心的配置和拓扑关系。
    core_id: int # NPU 核心的唯一标识符。
    serves_channel_ids: List[int] # 该 NPU 核心直接服务的内存通道 ID 列表。

@dataclass # 使用 dataclass 装饰器。
class ChannelConfig: # 定义一个数据类，用于描述单个内存通道的配置。
    channel_id: int # 内存通道的唯一标识符。
    pim_type: Optional[PIMType] # 该通道上 PIM 单元的类型（如果有的话）。
    has_npu_slice: bool = True # 标志该通道是否与一个 NPU 切片相关联，默认为是。

@dataclass # 使用 dataclass 装饰器。
class OperatorMapping: # 定义一个数据类，用于描述单个操作如何映射到硬件上。
    operator_name: str # 操作的名称，例如 "Attention::Score_Computation_QK"。
    op_category: OpCategory # 操作的类别。
    execution_target: ExecutionTarget # 操作的执行目标（NPU, PIM, 或分割）。
    split_ratio_pim: float = 0.0 # 如果是分割执行，PIM 部分承担的计算比例，默认为0。
    split_dimension: Optional[SplitDimension] = None # 如果是分割执行，分割的维度，默认为None。
    def __str__(self): # 定义该类的字符串表示方法，方便打印和调试。
        details = [f"Category: {self.op_category.name}", f"Target: {self.execution_target.name}"] # 创建一个包含类别和执行目标的列表。
        if self.execution_target == ExecutionTarget.SPLIT_EXECUTION: # 如果是分割执行。
            details.append(f"Split(PIM: {self.split_ratio_pim:.2f}) along Dim '{self.split_dimension.name}'") # 添加分割比例和维度的详细信息。
        return ", ".join(details) # 将所有详细信息用逗号连接成一个字符串并返回。

@dataclass # 使用 dataclass 装饰器。
class ExecutionStage: # 定义一个数据类，表示计算图中的一个执行阶段。
    stage_description: str # 对该阶段的文字描述。
    operator_mappings: Dict[str, OperatorMapping] # 一个字典，存储该阶段中所有操作及其对应的映射策略。

@dataclass # 使用 dataclass 装饰器。
class LayerDataflow: # 定义一个数据类，描述单层 LLM 的完整数据流。
    stages: List[ExecutionStage] # 一个包含多个执行阶段的列表。
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.TASK_DYNAMIC_DISPATCH # 该层数据流采用的调度策略，默认为动态分派。

@dataclass # 使用 dataclass 装饰器。
class ExecutionPlan: # 定义一个顶层数据类，用于完整描述一个仿真方案。
    description: str # 对该执行计划的文字描述。
    npu_config: NPUConfig # NPU 的硬件配置。
    pim_config: PIMConfig # PIM 的硬件配置。
    npu_topology: List[NPUCoreConfig] # NPU 的拓扑结构。
    channel_topology: List[ChannelConfig] # 内存通道的拓扑结构。
    interconnect_model: InterconnectModel # 互连模型。
    decoding_dataflow: LayerDataflow # 解码（Decode）阶段的数据流策略。
    prefill_dataflow: Optional[LayerDataflow] = None # 预填充（Prefill）阶段的数据流策略，可选。
    memory_layout_policy: MemoryLayoutPolicy = MemoryLayoutPolicy.BLOCK_PIM # 采用的内存布局策略，默认为 BlockPIM 风格。

# ==============================================================================
# SECTION 3 & 4: BLUEPRINTS AND CONSTRUCTOR (Updated)
# ==============================================================================
PREFILL_PIPELINE_BLUEPRINT = [ # 定义 Prefill 阶段的计算流水线蓝图。
    {"description": "Prefill Stage", "operators": { # 这是一个包含单个阶段的列表。
        "Prefill::QKV_Projections": OpCategory.GENERAL, # 操作名及其类别。
        "Prefill::Attention_BMM": OpCategory.ATTENTION, # 将 QK 和 SV 的 BMM (批处理矩阵乘法) 合并为一个操作。
        "Prefill::FFN_Projections": OpCategory.FFN,
    }}
]
DECODING_PIPELINE_BLUEPRINT = [ # 定义 Decode 阶段的计算流水线蓝图。
    {"description": "Compute-Bound Stage", "operators": { # 这是一个包含单个阶段的列表，其中包含所有计算密集型操作。
        "Attention::QKV_Projection": OpCategory.GENERAL,
        "Attention::Score_Computation_QK": OpCategory.ATTENTION,
        "Attention::Context_Computation_SV": OpCategory.ATTENTION,
        "Attention::Output_Projection": OpCategory.GENERAL,
        "FFN::Gate_Up_Projection": OpCategory.FFN,
        "FFN::Down_Projection": OpCategory.FFN,
    }}
]

def create_plan_from_configs( # 定义一个工厂函数，用于从配置字典创建 ExecutionPlan 对象。
    hw_config: Dict[str, Any], # 包含硬件配置的字典。
    dataflow_config: Dict[str, Any], # 包含数据流配置的字典。
    description: str = "DSE Generated Plan" # 对生成的计划的描述。
) -> ExecutionPlan: # 函数返回一个 ExecutionPlan 对象。
    
    npu_conf = NPUConfig(**hw_config['npu']) # 使用字典解包的方式创建 NPUConfig 对象。
    pim_conf = PIMConfig(**hw_config['pim']) # 使用字典解包的方式创建 PIMConfig 对象。
    interconnect = InterconnectModel(**hw_config['interconnect']) # 创建 InterconnectModel 对象。
    npu_topo = [NPUCoreConfig(**cfg) for cfg in hw_config['topology']['npu_cores']] # 列表推导，创建 NPUCoreConfig 对象列表。
    channel_topo = [ChannelConfig(channel_id=cfg['channel_id'], pim_type=PIMType[cfg.get('pim_type')]) for cfg in hw_config['topology']['channels']] # 列表推导，创建 ChannelConfig 对象列表。

    def _build_layer_dataflow(blueprint: List[Dict], config: Dict) -> LayerDataflow: # 定义一个内部帮助函数，用于构建 LayerDataflow 对象。
        stages = [] # 初始化阶段列表。
        policy_str = config.get("scheduling_policy", "TASK_DYNAMIC_DISPATCH") # 从配置中获取调度策略字符串。
        scheduling_policy_enum = SchedulingPolicy[policy_str] # 将字符串转换为枚举成员。

        for stage_def in blueprint: # 遍历蓝图中的每个阶段定义。
            mappings = {} # 初始化当前阶段的操作映射字典。
            for op_name, op_category in stage_def['operators'].items(): # 遍历阶段中的每个操作。
                op_config = config.get(op_name) # 从数据流配置中获取该操作的具体配置。
                if not op_config: continue # 如果没有配置，则跳过。
                
                target_enum = ExecutionTarget[op_config['target']] # 将执行目标的字符串转换为枚举成员。
                split_dim_str = op_config.get('split_dimension') # 获取分割维度的字符串。
                split_dim_enum = SplitDimension[split_dim_str] if split_dim_str else None # 如果存在，则转换为枚举成员。
                mappings[op_name] = OperatorMapping( # 创建 OperatorMapping 对象。
                    operator_name=op_name, op_category=op_category,
                    execution_target=target_enum,
                    split_ratio_pim=op_config.get('split_ratio_pim', 0.0),
                    split_dimension=split_dim_enum
                )
            stages.append(ExecutionStage(stage_def['description'], mappings)) # 将构建好的阶段添加到列表中。
        
        return LayerDataflow(stages=stages, scheduling_policy=scheduling_policy_enum) # 返回创建的 LayerDataflow 对象。

    decoding_flow = _build_layer_dataflow(DECODING_PIPELINE_BLUEPRINT, dataflow_config['decoding']) # 为 Decode 阶段构建数据流。
    prefill_flow = None # 初始化 Prefill 数据流为 None。
    if 'prefill' in dataflow_config: # 检查配置中是否存在 Prefill 的数据流定义。
        prefill_flow = _build_layer_dataflow(PREFILL_PIPELINE_BLUEPRINT, dataflow_config['prefill']) # 如果存在，则构建它。
    
    # --- NEW: Read memory layout policy from the config ---
    mem_layout_str = dataflow_config.get("memory_layout_policy", "BLOCK_PIM") # 从配置中获取内存布局策略字符串。
    mem_layout_enum = MemoryLayoutPolicy[mem_layout_str] # 将字符串转换为枚举成员。

    return ExecutionPlan( # 创建并返回最终的 ExecutionPlan 对象。
        description=description, 
        npu_config=npu_conf, pim_config=pim_conf,
        npu_topology=npu_topo, channel_topology=channel_topo,
        interconnect_model=interconnect, decoding_dataflow=decoding_flow,
        prefill_dataflow=prefill_flow, memory_layout_policy=mem_layout_enum
    )
