# evaluator_v0_3_1.py
import math
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

# 导入我们已经定义好的所有模块
# 假设这些文件与此文件在同一目录下或在Python路径中
from dataflow_v0_3_3 import (
    ExecutionPlan, ExecutionTarget, PIMType, SplitDimension,
    OperatorMapping, ExecutionStage, LayerDataflow, ChannelConfig, OpCategory
)
from pim_v0_3 import ApplicationLayerEvaluator as PIMEvaluator
from npu_v0_3 import ApplicationLayerEvaluator as NPUEvaluator
from utils_v0_3 import PerformanceResult # 假设在utils.py中

# ==============================================================================
# 1. 辅助数据结构
# ==============================================================================
@dataclass
class ModelConfig:
    """ 存储具体模型和推理任务的参数 """
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    prompt_len: int = 4096
    generation_len: int = 4096

# 使用元组作为硬件资源的唯一标识符，例如 (0, "PIM_ATTENTION")
HardwareResource = Tuple[int, str]

@dataclass
class SingleTokenPathResult:
    """ 存储模拟单个token完整路径的结果 """
    total_latency_cycles: float  # 填满流水线的时间
    unit_usage_cycles: Dict[HardwareResource, float] # 每个资源在此过程中的累计繁忙时间

# ==============================================================================
# 2. 统一评估器 (Unified Evaluator)
# ==============================================================================
class UnifiedEvaluator:
    """
    v2.0: 解释并执行ExecutionPlan，能够模拟通道内异构PIM架构和Token间的流水线。
    """
    def __init__(self, plan: ExecutionPlan, model_config: ModelConfig):
        self.plan = plan
        self.model = model_config
        self.element_size_bytes = 2

        self.pim_eval = PIMEvaluator(plan.pim_config, self.element_size_bytes)
        self.npu_eval = NPUEvaluator(plan.npu_config, self.element_size_bytes)
        
        # --- 核心改动: 资源抽象 ---
        self.resources: List[HardwareResource] = []
        self._initialize_resources()
        
        # 用于快速查找的拓扑地图
        self.channel_map: Dict[int, ChannelConfig] = {cfg.channel_id: cfg for cfg in plan.channel_topology}

    def _initialize_resources(self):
        """根据硬件拓扑，创建整个系统的资源池"""
        for ch_cfg in self.plan.channel_topology:
            channel_id = ch_cfg.channel_id
            if ch_cfg.has_npu_slice:
                self.resources.append((channel_id, "NPU_SLICE"))
            
            if ch_cfg.pim_type == PIMType.COMMON:
                self.resources.append((channel_id, "PIM_COMMON"))
            elif ch_cfg.pim_type == PIMType.ATTENTION_FFN:
                self.resources.append((channel_id, "PIM_ATTENTION"))
                self.resources.append((channel_id, "PIM_FFN"))

    def _get_required_resource_type(self, mapping: OperatorMapping, channel_id: int, target_unit: str) -> str:
        """
        v2.2: 能够处理SPLIT任务，并根据目标单元(NPU/PIM)返回资源类型。
        
        Args:
            mapping: The operator mapping.
            channel_id: The channel being scheduled.
            target_unit: "NPU" or "PIM", specifying which part of a split task we want the resource for.
        """
        if target_unit == "NPU":
            return "NPU_SLICE"
        
        # --- 以下逻辑用于确定PIM资源类型 ---
        
        channel_hw_type = self.channel_map[channel_id].pim_type
        if channel_hw_type is None:
            raise ValueError(f"Task {mapping.operator_name} requires PIM, but channel {channel_id} has none.")

        task_category = mapping.op_category
        
        if channel_hw_type == PIMType.COMMON:
            return "PIM_COMMON"
        elif channel_hw_type == PIMType.ATTENTION_FFN:
            if task_category == OpCategory.ATTENTION: return "PIM_ATTENTION"
            if task_category == OpCategory.FFN: return "PIM_FFN"
            # For GENERAL tasks on dedicated PIMs, we can have a default policy
            # Here, we default to using the ATTENTION PIM.
            if task_category == OpCategory.GENERAL: return "PIM_ATTENTION"

        raise ValueError(f"Cannot find a suitable PIM resource on channel {channel_id} for a {task_category.name} task.")

    def _get_op_shape(self, op_name: str, kv_cache_len: int, is_prefill: bool) -> Dict[str, int]:
        """动态获取算子形状，现在依赖kv_cache_len"""
        M = self.model.prompt_len if is_prefill else 1
        
        shapes = {
            "Attention::QKV_Projection": {"M": M, "K": self.model.hidden_size, "N": self.model.hidden_size * 3},
            "Attention::Score_Computation_QK": {"M": 1, "K": self.model.head_dim, "N": kv_cache_len},
            "Attention::Context_Computation_SV": {"M": self.model.head_dim, "K": kv_cache_len, "N": self.model.head_dim},
            "Attention::Output_Projection": {"M": M, "K": self.model.hidden_size, "N": self.model.hidden_size},
            "FFN::Gate_Up_Projection": {"M": M, "K": self.model.hidden_size, "N": self.model.intermediate_size * 2},
            "FFN::Down_Projection": {"M": M, "K": self.model.intermediate_size, "N": self.model.hidden_size},
        }
        # Prefill BMM (Batch GEMM) shapes are more complex, simplified here
        if is_prefill:
            shapes["Attention::Score_Computation_QK"] = {"M": M, "K": self.model.head_dim, "N": M}
            shapes["Attention::Context_Computation_SV"] = {"M": M, "K": M, "N": self.model.head_dim}
            
        return shapes.get(op_name, {})
        
    def _evaluate_single_task(self, mapping: OperatorMapping, shape: Dict, num_channels: int, num_heads: int, is_prefill: bool) -> PerformanceResult:
        """调用底层评估器，与之前_evaluate_single_op逻辑类似"""
        # ... (此部分与上一版 evaluator 的 _evaluate_single_op 逻辑几乎完全相同)
        M, K, N = shape.get("M", 1), shape.get("K", 1), shape.get("N", 1)
        # ... [此处省略与上一版本几乎相同的代码, 为了简洁]
        # ... [它会处理 SPLIT_EXECUTION, PIM_ONLY, NPU_ONLY]
        # ... [并返回一个 PerformanceResult 对象]
        # 这是一个简化实现
        if mapping.execution_target == ExecutionTarget.NPU_ONLY:
            op_type = "Attention_QK" if "QK" in mapping.operator_name else "GEMM"
            return self.npu_eval.evaluate(M, K, N, op_type=op_type, num_heads=num_heads)
        elif mapping.execution_target == ExecutionTarget.PIM_ONLY:
            op_type = "GEMM" if is_prefill else "GEMV"
            return self.pim_eval.evaluate(M, K, N, op_type=op_type, num_channels_used=num_channels, num_ops_parallel=num_heads)
        elif mapping.execution_target == ExecutionTarget.MEMORY_BOUND:
            return PerformanceResult(0,0,0,0, "Memory_Bound")
        return PerformanceResult(1e12, 0,0,0, "ERROR") # Fallback for SPLIT etc. for now
    
    """
    _simulate_single_token_path 函数逻辑详解
    初始化 (Step 1): 创建两个字典。resource_finish_times 是时序模拟的核心，记录每个资源的“忙闲状态”。unit_usage_cycles 是瓶颈分析的核心，只记录“纯工作时长”。
    遍历阶段 (Step 2): 代码按 dataflow 中定义的顺序，一个一个地处理流水线阶段，这保证了数据依赖的正确性。
    阶段同步与通信 (Step 3): stage_start_time 的计算是模拟真实流水线的关键。它确保了后一个阶段必须在前一个阶段完全结束并且数据传输完成后才能开始。
    并行调度 (Step 4):
    并行度 (4.1): 判断一个算子是应该作为一个整体执行，还是应该拆分成多个并行单元（按Head）。
    任务分配 (4.3): for i in range(num_parallel_units): 这个循环就是“一个Channel一个Head”策略的直接体现。
    资源匹配 (4.4 & 4.5): 这是代码的“智能”所在。它调用 _get_required_resource_type 来确定任务的目标资源。例如，对于 COMMON PIM 硬件，Attention 和 FFN 任务都会被正确地导向 "PIM_COMMON" 资源。如果所需的资源在硬件中不存在，就会触发错误处理。
    时序计算 (4.6): task_start_time = max(...) 这一点非常重要。它模拟了资源争用。如果一个资源（比如 (0, "PIM_COMMON")) 在一个阶段内需要连续执行两个任务，第二个任务就必须等待第一个任务完成，即使整个阶段已经开始了。
    记录更新 (4.7): 每次任务调度后，都会更新资源的时间表和工作量记录，为下一个任务的调度和最终的瓶颈分析提供依据。
    阶段收尾 (Step 5): 当一个阶段的所有并行任务都调度完毕后，last_stage_finish_time 被更新为这个阶段最晚完成的那个任务的时间点。这成为了下一阶段的“起跑线”。
    返回结果 (Step 6): 函数返回两个最重要的结果：第一个Token跑完整个流水线需要的时间 (total_latency_cycles)，以及在此过程中每个硬件单元到底工作了多久 (unit_usage_cycles)。这两个值将直接用于 evaluate 方法中计算最终的吞吐量。
    """

    def _simulate_single_token_path(self, start_kv_len: int) -> SingleTokenPathResult:
        """
        v2.2: 增加了对SPLIT_EXECUTION任务的正确时序模拟。
        """
        resource_finish_times: Dict[HardwareResource, float] = {res: 0.0 for res in self.resources}
        unit_usage_cycles: Dict[HardwareResource, float] = {res: 0.0 for res in self.resources}
        last_stage_finish_time = 0.0
        
        for stage in self.plan.decoding_dataflow.stages:
            comm_latency = self.plan.interconnect_model.fixed_latency_cycles
            stage_start_time = last_stage_finish_time + comm_latency
            current_stage_resource_times = resource_finish_times.copy()

            for op_name, mapping in stage.operator_mappings.items():
                # ... (并行单位和形状获取逻辑保持不变) ...
                op_category = mapping.op_category
                num_parallel_units = self.model.num_heads if op_category in [OpCategory.ATTENTION, OpCategory.FFN] else 1
                shape = self._get_op_shape(op_name, start_kv_len, is_prefill=False)
                if not shape: continue
                
                for i in range(num_parallel_units):
                    channel_id = i
                    M, K, N = shape.get("M", 1), shape.get("K", 1), shape.get("N", 1)
                    
                    # --- 分发任务到不同的执行单元 ---
                    
                    if mapping.execution_target == ExecutionTarget.SPLIT_EXECUTION:
                        # --- 处理分裂任务 ---
                        try:
                            npu_res_type = self._get_required_resource_type(mapping, channel_id, "NPU")
                            pim_res_type = self._get_required_resource_type(mapping, channel_id, "PIM")
                        except ValueError as e:
                            return SingleTokenPathResult(1e12, {}) # 硬件不匹配

                        npu_resource: HardwareResource = (channel_id, npu_res_type)
                        pim_resource: HardwareResource = (channel_id, pim_res_type)

                        if npu_resource not in self.resources or pim_resource not in self.resources:
                            return SingleTokenPathResult(1e12, {}) # 资源不存在
                        
                        # 计算分裂后的形状
                        N_pim = math.ceil(N * mapping.split_ratio_pim)
                        N_npu = N - N_pim
                        
                        # 评估PIM部分
                        pim_perf = self.pim_eval.evaluate(M, K, N_pim, op_type="GEMV", num_channels_used=1, num_ops_parallel=1)
                        # 评估NPU部分
                        npu_perf = self.npu_eval.evaluate(M, K, N_npu, op_type="GEMM", num_heads=1)
                        
                        # 计算时序
                        pim_start_time = max(stage_start_time, resource_finish_times[pim_resource])
                        npu_start_time = max(stage_start_time, resource_finish_times[npu_resource])
                        
                        pim_finish_time = pim_start_time + pim_perf.latency_cycles
                        npu_finish_time = npu_start_time + npu_perf.latency_cycles
                        
                        # 更新时间戳和使用量
                        current_stage_resource_times[pim_resource] = pim_finish_time
                        current_stage_resource_times[npu_resource] = npu_finish_time
                        unit_usage_cycles[pim_resource] += pim_perf.latency_cycles
                        unit_usage_cycles[npu_resource] += npu_perf.latency_cycles

                    else: # 处理非分裂任务 (NPU_ONLY, PIM_ONLY, MEMORY_BOUND)
                        if mapping.execution_target == ExecutionTarget.MEMORY_BOUND: continue

                        try:
                            res_type = self._get_required_resource_type(mapping, channel_id, "PIM" if mapping.execution_target == ExecutionTarget.PIM_ONLY else "NPU")
                        except ValueError as e:
                            return SingleTokenPathResult(1e12, {})

                        resource: HardwareResource = (channel_id, res_type)
                        if resource not in self.resources: return SingleTokenPathResult(1e12, {})

                        # 评估任务
                        if mapping.execution_target == ExecutionTarget.PIM_ONLY:
                            perf = self.pim_eval.evaluate(M, K, N, op_type="GEMV", num_channels_used=1, num_ops_parallel=1)
                        else: # NPU_ONLY
                            perf = self.npu_eval.evaluate(M, K, N, op_type="GEMM", num_heads=1)
                        
                        # 计算时序
                        start_time = max(stage_start_time, resource_finish_times[resource])
                        finish_time = start_time + perf.latency_cycles

                        # 更新时间戳和使用量
                        current_stage_resource_times[resource] = finish_time
                        unit_usage_cycles[resource] += perf.latency_cycles
            
            resource_finish_times = current_stage_resource_times
            last_stage_finish_time = max(resource_finish_times.values()) if resource_finish_times else 0
        """
        # 临时的诊断代码
        print("\n--- Path Simulation Diagnostics ---")
        # 按通道聚合资源使用情况
        channel_usages: Dict[int, Dict[str, float]] = {i: {} for i in range(self.plan.pim_config.total_channels)}
        for (channel_id, unit_type), usage in unit_usage_cycles.items():
            channel_usages[channel_id][unit_type] = usage
        
        # 打印前几个通道的详细情况
        for i in range(min(4, self.plan.pim_config.total_channels)):
            print(f"Channel {i} Usage (cycles): {channel_usages[i]}")
        
        # 计算并打印两种架构下的理论瓶颈
        common_pim_total_usage = sum(v for k, v in unit_usage_cycles.items() if 'PIM' in k[1])
        npu_total_usage = sum(v for k, v in unit_usage_cycles.items() if 'NPU' in k[1])
        
        dedicated_bottleneck = max(
            max((v for k, v in unit_usage_cycles.items() if k[1] == 'PIM_ATTENTION'), default=0),
            max((v for k, v in unit_usage_cycles.items() if k[1] == 'PIM_FFN'), default=0),
            max((v for k, v in unit_usage_cycles.items() if k[1] == 'NPU_SLICE'), default=0)
        )
        common_bottleneck = max(common_pim_total_usage, npu_total_usage)
        
        print(f"Theoretical Common Arch Bottleneck: {common_bottleneck:.0f} cycles")
        print(f"Theoretical Dedicated Arch Bottleneck: {dedicated_bottleneck:.0f} cycles")
        print("---------------------------------\n")
        """
        return SingleTokenPathResult(
            total_latency_cycles=last_stage_finish_time,
            unit_usage_cycles=unit_usage_cycles
        )

    def evaluate(self) -> Dict[str, Any]:
        """
        执行完整的端到端性能评估。
        """
        # --- 1. Prefill阶段评估 (保持简化模型：无流水线，所有资源并行) ---
        # (这部分逻辑可以复用之前的 _evaluate_stage, 但为了清晰分开)
        prefill_result = self._simulate_single_token_path(self.model.prompt_len) # 借用此函数模拟
        total_prefill_cycles = prefill_result.total_latency_cycles * self.model.num_layers
        
        # --- 2. Decoding阶段评估 (采用新的流水线吞吐量模型) ---
        # 2.1 模拟一个token的路径来获取填充时间和瓶颈
        # 我们使用一个典型的kv_cache长度（例如，prompt_len）来做代表性计算
        path_result = self._simulate_single_token_path(self.model.prompt_len)
        
        if path_result.total_latency_cycles > 1e11: # 无效Plan
            return {
                "total_latency_ms": 1e12,
                "prefill_cycles": 0,
                "decoding_cycles_total": 1e12,
                "pipeline_fill_cycles": 1e12,
                "system_bottleneck_cycles": 1e12,
                "plan_description": self.plan.description + " [INVALID PLAN]"
            }

        T_pipeline_fill = path_result.total_latency_cycles * self.model.num_layers
        
        # 2.2 计算系统瓶颈
        # 将资源按通道分组
        channel_usages: Dict[int, List[float]] = {i: [] for i in range(self.plan.pim_config.total_channels)}
        for (channel_id, unit_type), usage in path_result.unit_usage_cycles.items():
            channel_usages[channel_id].append(usage)
            
        # 找出每个通道内部最繁忙的资源
        channel_bottlenecks = [max(usages) if usages else 0 for usages in channel_usages.values()]
        
        # 系统瓶颈是所有通道中最慢的那个
        T_system_bottleneck = max(channel_bottlenecks) * self.model.num_layers if channel_bottlenecks else 0
        
        # 2.3 应用流水线公式
        if self.model.generation_len > 0:
            total_decoding_cycles = T_pipeline_fill + (self.model.generation_len - 1) * T_system_bottleneck
        else:
            total_decoding_cycles = 0

        # --- 3. 汇总结果 ---
        clock_ghz = self.plan.npu_config.clock_ghz
        total_cycles = total_prefill_cycles + total_decoding_cycles
        total_latency_ms = total_cycles / (clock_ghz * 1e6)
        
        return {
            "total_latency_ms": total_latency_ms,
            "prefill_cycles": total_prefill_cycles,
            "decoding_cycles_total": total_decoding_cycles,
            "pipeline_fill_cycles": T_pipeline_fill,
            "system_bottleneck_cycles": T_system_bottleneck,
            "plan_description": self.plan.description
        }
    
    """
    def evaluate(self) -> Dict[str, Any]:
        
        # v2.3: 修正了流水线瓶颈的计算逻辑，以正确区分COMMON和DEDICATED PIM架构。
        
        # --- 1. Prefill阶段评估 (保持简化) ---
        # 实际项目中，prefill也需要一个更精细的评估函数
        # 此处为了聚焦decoding问题，暂时简化
        total_prefill_cycles = 1.0 

        # --- 2. Decoding阶段评估 ---
        # 2.1 模拟单个token的路径来获取填充时间和各资源的基础工作量
        path_result = self._simulate_single_token_path(self.model.prompt_len)
        
        # 如果Plan无效，提前返回结构完整的惩罚字典
        if path_result.total_latency_cycles > 1e11:
            return {
                "total_latency_ms": 1e12, "prefill_cycles": 0, "decoding_cycles_total": 1e12,
                "pipeline_fill_cycles": 1e12, "system_bottleneck_cycles": 1e12,
                "plan_description": self.plan.description + " [INVALID PLAN]"
            }

        # 第一个token的延迟 (填满流水线的时间)
        T_pipeline_fill = path_result.total_latency_cycles * self.model.num_layers
        
        # --- 2.2 核心修正: 重新计算系统瓶颈 ---
        
        # 将所有资源的纯工作时间按 (channel_id, resource_category) 分组
        # 例如: (0, 'NPU'), (0, 'ATTENTION'), (0, 'FFN')
        # 注意：对于COMMON架构，我们只有一个'PIM'类别
        channel_usages_by_category: Dict[int, Dict[str, float]] = {i: {} for i in range(self.plan.pim_config.total_channels)}
        
        for (channel_id, unit_type), usage in path_result.unit_usage_cycles.items():
            category = "NPU"
            if "NPU" not in unit_type:
                # 对PIM资源进行分类
                if self.channel_map[channel_id].pim_type == PIMType.COMMON:
                    category = "PIM_SHARED" # 这是一个共享的PIM资源
                else: # ATTENTION_FFN
                    if "ATTENTION" in unit_type: category = "PIM_ATTENTION"
                    if "FFN" in unit_type: category = "PIM_FFN"
            
            # 累加同类资源的工作时间
            channel_usages_by_category[channel_id][category] = channel_usages_by_category[channel_id].get(category, 0.0) + usage

        # 现在，根据架构计算每个通道的真实瓶颈
        channel_bottlenecks = []
        for i in range(self.plan.pim_config.total_channels):
            usages = channel_usages_by_category[i]
            npu_usage = usages.get("NPU", 0.0)
            
            pim_arch_type = self.channel_map[i].pim_type
            
            if pim_arch_type == PIMType.COMMON:
                # 在COMMON架构下，所有PIM任务串行，所以PIM的总负载是所有PIM工作之和
                pim_total_usage = usages.get("PIM_SHARED", 0.0)
                channel_bottleneck = max(npu_usage, pim_total_usage)
            elif pim_arch_type == PIMType.ATTENTION_FFN:
                # 在专用架构下，Attention和FFN是独立的PIM单元，它们与NPU并行
                # 瓶颈是这三个独立工作单元中最慢的那个
                attn_pim_usage = usages.get("PIM_ATTENTION", 0.0)
                ffn_pim_usage = usages.get("PIM_FFN", 0.0)
                channel_bottleneck = max(npu_usage, attn_pim_usage, ffn_pim_usage)
            else: # 无PIM
                channel_bottleneck = npu_usage
            
            channel_bottlenecks.append(channel_bottleneck)
            
        # 系统瓶颈是所有通道中最慢的那个
        T_system_bottleneck = max(channel_bottlenecks) * self.model.num_layers if channel_bottlenecks else 0
        
        # 2.3 应用流水线公式
        if self.model.generation_len > 0:
            total_decoding_cycles = T_pipeline_fill + (self.model.generation_len - 1) * T_system_bottleneck
        else:
            total_decoding_cycles = 0

        # --- 3. 汇总结果 (保持不变) ---
        clock_ghz = self.plan.npu_config.clock_ghz
        total_cycles = total_prefill_cycles + total_decoding_cycles
        total_latency_ms = total_cycles / (clock_ghz * 1e6)
        
        return {
            "total_latency_ms": total_latency_ms,
            "prefill_cycles": total_prefill_cycles,
            "decoding_cycles_total": total_decoding_cycles,
            "pipeline_fill_cycles": T_pipeline_fill,
            "system_bottleneck_cycles": T_system_bottleneck,
            "plan_description": self.plan.description
        }
    """

# ==============================================================================
# 主执行块 (演示)
# ==============================================================================
if __name__ == '__main__':
    from dataflow_v0_3_3 import create_plan_from_configs, PIMType

    llama7b_task = ModelConfig(prompt_len=39, generation_len=100)
    
    # --- 演示两种不同的硬件拓扑 ---
    # 拓扑1: 每个通道都有一个COMMON PIM
    hw_config_common = {
        "npu": {"num_cores": 8, "num_channels": 32, "channels_per_core": 4, "clock_ghz": 2.0},
        "pim": {"total_channels": 32, "clock_ghz": 2.0, "global_buffer_size_per_channel_kb": 128},
        "interconnect": {"fixed_latency_cycles": 120, "cycles_per_byte": 0.05},
        "topology": {
            "npu_cores": [{"core_id": i, "serves_channel_ids": list(range(i*4, (i+1)*4))} for i in range(8)],
            "channels": [{"channel_id": i, "pim_type": "COMMON"} for i in range(32)]
        }
    }
    
    # 拓扑2: 每个通道都有独立的 ATTENTION 和 FFN PIM
    hw_config_dedicated = {
        "npu": {"num_cores": 8, "num_channels": 32, "channels_per_core": 4, "clock_ghz": 2.0},
        "pim": {"total_channels": 32, "clock_ghz": 2.0, "global_buffer_size_per_channel_kb": 128},
        "interconnect": {"fixed_latency_cycles": 120, "cycles_per_byte": 0.05},
        "topology": {
            "npu_cores": [{"core_id": i, "serves_channel_ids": list(range(i*4, (i+1)*4))} for i in range(8)],
            "channels": [{"channel_id": i, "pim_type": "ATTENTION_FFN"} for i in range(32)]
        }
    }
    
    # 一个将PIM任务卸载的dataflow
    dataflow_config_pim_offload = {
        "prefill": {
            # Prefill阶段几乎全用NPU
            "Prefill_QKV": {"target": "NPU_ONLY"},
            "Prefill_Attention": {"target": "NPU_ONLY"},
            "Prefill_FFN": {"target": "SPLIT_EXECUTION", "split_ratio_pim": 0.1, "split_dimension": "N", "required_pim_type": "FFN"}
        },
        "decoding": {
            # Decoding阶段策略复杂
            "Input_RMSNorm": {"target": "MEMORY_BOUND"},
            "Attention::QKV_Projection": {"target": "NPU_ONLY"},
            "Attention::Score_Computation_QK": {"target": "PIM_ONLY", "required_pim_type": "ATTENTION"},
            "Attention::Softmax": {"target": "MEMORY_BOUND"},
            "Attention::Context_Computation_SV": {"target": "PIM_ONLY", "required_pim_type": "ATTENTION"},
            "Attention::Output_Projection": {"target": "NPU_ONLY"},
            "Post_RMSNorm": {"target": "MEMORY_BOUND"},
            "FFN::Gate_Up_Projection": {"target": "SPLIT_EXECUTION", "split_ratio_pim": 0.4, "split_dimension": "N", "required_pim_type": "FFN"},
            "FFN::Activation(SiLU)": {"target": "MEMORY_BOUND"},
            "FFN::Down_Projection": {"target": "SPLIT_EXECUTION", "split_ratio_pim": 0.5, "split_dimension": "N", "required_pim_type": "FFN"}
        }
    }

    dataflow_config_pim_heavy = {
        "prefill": { # Prefill remains steady
            "Prefill_QKV": {"target": "NPU_ONLY"},
            "Prefill_Attention": {"target": "NPU_ONLY"},
            "Prefill_FFN": {"target": "NPU_ONLY"}
        },
        "decoding": {
            #
            "Input_RMSNorm": {"target": "MEMORY_BOUND"},
            "Attention::QKV_Projection": {"target": "PIM_ONLY"}, # <--- change
            "Attention::Score_Computation_QK": {"target": "PIM_ONLY"},
            "Attention::Softmax": {"target": "MEMORY_BOUND"},
            "Attention::Context_Computation_SV": {"target": "PIM_ONLY"},
            "Attention::Output_Projection": {"target": "PIM_ONLY"}, # <--- change
            "Post_RMSNorm": {"target": "MEMORY_BOUND"},
            "FFN::Gate_Up_Projection": {"target": "PIM_ONLY"},
            "FFN::Activation(SiLU)": {"target": "MEMORY_BOUND"},
            "FFN::Down_Projection": {"target": "PIM_ONLY"},
        }
    }
    
    # --- 评估 COMMON PIM 架构 ---
    plan_common = create_plan_from_configs(hw_config_common, dataflow_config_pim_heavy, "Plan on COMMON PIM hardware")
    evaluator_common = UnifiedEvaluator(plan=plan_common, model_config=llama7b_task)
    perf_common = evaluator_common.evaluate()
    
    # --- 评估 ATTENTION_FFN PIM 架构 ---
    plan_dedicated = create_plan_from_configs(hw_config_dedicated, dataflow_config_pim_heavy, "Plan on Dedicated ATTN+FFN PIM hardware")
    evaluator_dedicated = UnifiedEvaluator(plan=plan_dedicated, model_config=llama7b_task)
    perf_dedicated = evaluator_dedicated.evaluate()

    # --- 打印对比结果 ---
    def print_summary(perf, title):
        print("\n" + "="*80)
        print(f"Performance Summary for: {title}")
        print("="*80)
        print(f"Total Latency: {perf['total_latency_ms']:.4f} ms")
        fill_ms = perf['pipeline_fill_cycles'] / (2.0 * 1e6)
        bottleneck_ms = perf['system_bottleneck_cycles'] / (2.0 * 1e6)
        print(f"  - Pipeline Fill Time (1st token): {fill_ms:.4f} ms")
        print(f"  - System Bottleneck (per token):  {bottleneck_ms:.4f} ms")
        print(f"  - Implied Throughput: {1000/bottleneck_ms if bottleneck_ms > 0 else 'inf'} token/s")
        print("="*80)

    print_summary(perf_common, plan_common.description)
    print_summary(perf_dedicated, plan_dedicated.description)
