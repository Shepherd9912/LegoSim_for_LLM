# ==============================================================================
# unified_evaluator_final_strategy_driven.py
#
# DESCRIPTION: The definitive, high-fidelity, strategy-driven simulator.
# This version understands physical memory layouts (from BlockPIM), executes
# different scheduling policies, and simulates both Prefill and Decode stages,
# making it a true DSE tool.
# ==============================================================================
import math # 导入 math 模块，用于数学计算。
import numpy as np # 导入 numpy 库，用于高效的数值数组操作。
import argparse # 导入 argparse 模块，用于解析命令行参数。
import pandas as pd # 导入 pandas 库，用于数据处理，尤其是在主程序块中。
from typing import Dict, Any, List, Tuple, Optional # 从 typing 模块导入类型提示。
import copy # 导入 copy 模块，用于创建对象的深拷贝。
from dataclasses import dataclass # 从 dataclasses 模块导入 dataclass 装饰器。

# Import all our library components
from dataflow import ( # 从自定义的 dataflow 模块中导入所有需要的类和枚举。
    ExecutionPlan, ExecutionTarget, PIMType, OperatorMapping, 
    create_plan_from_configs, SchedulingPolicy, MemoryLayoutPolicy, OpCategory, LayerDataflow
)
from utils import NPUConfig, PIMConfig, PerformanceResult # 从自定义的 utils 模块中导入需要的类。
from npu import ApplicationLayerEvaluator as NPUEvaluator, HybridCoreCostModel # 从 npu 模型模块导入评估器和成本模型，并重命名为 NPUEvaluator。
from pim import ApplicationLayerEvaluator as PIMEvaluator # 从 pim 模型模块导入评估器，并重命名为 PIMEvaluator。

# ==============================================================================
# SECTION 1: MODEL, WORKLOAD, AND SIMULATION DATA STRUCTURES
# ==============================================================================
@dataclass # 使用 dataclass 装饰器。
class ModelConfig: # 定义一个数据类来存储语言模型的配置参数。
    hidden_size: int = 4096; intermediate_size: int = 11008 # 定义隐藏层大小和前馈网络中间层大小。
    num_heads: int = 32; num_layers: int = 32 # 定义注意力头数量和模型总层数。
    prompt_len: int = 128; add_tokens: int = 128 # 定义初始提示长度和要生成的 token 数量。
    batch_size: int = 4 # 定义批处理大小。
    @property # 使用 property 装饰器将一个方法变成一个只读属性。
    def head_dim(self): return self.hidden_size // self.num_heads # 计算并返回每个注意力头的维度。

@dataclass # 使用 dataclass 装饰器。
class TaskUnit: # 定义一个数据类来表示一个需要被调度的最小计算任务单元。
    mapping: OperatorMapping; shape: Dict[str, int] # 任务的操作映射策略和形状（维度）。
    unit_id: int; op_type: str # 任务的唯一标识符（例如，头ID）和操作类型（如GEMM/GEMV）。

HardwareResource = Tuple[int, str] # 为硬件资源定义一个类型别名，它是一个元组，包含 (通道ID, 资源类型字符串)。

class ProfileProvider: # 定义 ProfileProvider 类。
    def __init__(self, plan: ExecutionPlan, npu_evaluator: NPUEvaluator, pim_evaluator: PIMEvaluator, workload: ModelConfig): # 类的构造函数。
        self.npu_eval = npu_evaluator; self.pim_eval = pim_evaluator # 保存 NPU 和 PIM 评估器库的引用。
        print("INFO: High-fidelity simulation mode. ProfileProvider serves as a library holder.") # 打印信息，说明在高保真模式下，此类仅作为评估器库的容器。


# ==============================================================================
# SECTION 2: THE STRATEGY-DRIVEN UNIFIED EVALUATOR
# ==============================================================================
class UnifiedEvaluator: # 定义统一评估器主类。
    def __init__(self, plan: ExecutionPlan, workload: ModelConfig, profile_provider: ProfileProvider): # 类的构造函数。
        self.plan = plan; self.workload = workload # 保存执行计划和工作负载配置。
        self.npu_eval = profile_provider.npu_eval; self.pim_eval = profile_provider.pim_eval # 从 profile_provider 获取 NPU 和 PIM 评估器。
        self.element_size_bytes = 2 # 定义数据元素的大小为2字节（例如 FP16）。
        
        self.num_channels = len(plan.channel_topology) # 获取总的物理通道数。
        self.num_npu_cores = len(plan.npu_topology) # 获取总的 NPU 核心数。

        self.resources: List[HardwareResource] = [] # 初始化一个列表，用于存储系统中所有可调度的硬件资源。
        self.resource_finish_times: Dict[HardwareResource, float] = {} # 初始化一个字典，用于跟踪每个逻辑计算资源（如 NPU slice）的完成时间。
        self.physical_channel_finish_times = np.zeros(self.num_channels, dtype=np.float64) # 初始化一个 numpy 数组，用于跟踪每个物理内存通道的完成时间。
        
        # MODIFIED: Separate stats for prefill and decode
        self.prefill_stats: Dict[str, Dict[str, float]] = {} # 初始化一个字典，用于存储 Prefill 阶段的性能统计数据。
        self.decode_stats: Dict[HardwareResource, Dict[str, float]] = {} # 初始化一个字典，用于存储 Decode 阶段每个资源的性能统计数据。
        self.total_prefill_cycles: float = 0.0 # 初始化 Prefill 阶段的总周期数为0。

        self.core_to_channels_map: Dict[int, List[int]] = {c.core_id: c.serves_channel_ids for c in plan.npu_topology} # 创建一个从 NPU 核心 ID 到其服务的通道 ID 列表的映射。
        
        self._initialize_resources() # 调用内部方法来初始化所有资源和统计数据结构。

    def _initialize_resources(self): # 定义资源初始化方法。
        self.resources.clear() # 清空现有资源列表。
        for ch in self.plan.channel_topology: # 遍历拓扑中的每个通道。
            self.resources.append((ch.channel_id, "NPU_SLICE")) # 为每个通道添加一个 NPU 切片资源。
            if ch.pim_type == PIMType.ATTENTION_FFN: # 如果 PIM 是专用类型。
                self.resources.append((ch.channel_id, "PIM_ATTENTION")) # 添加一个专门处理 Attention 的 PIM 资源。
                self.resources.append((ch.channel_id, "PIM_FFN")) # 添加一个专门处理 FFN 的 PIM 资源。
        
        # MODIFIED: Initialize separate stat dictionaries
        stat_keys = ["total_op_flops", "total_bytes_moved", "active_io_cycles", "active_compute_cycles", "total_active_cycles"] # 定义统计指标的键。
        self.prefill_stats = {"NPU": {k: 0.0 for k in stat_keys}, "PIM": {k: 0.0 for k in stat_keys}} # 为 Prefill 阶段初始化统计字典。
        self.decode_stats = {r: {k: 0.0 for k in stat_keys} for r in self.resources} # 为 Decode 阶段的每个资源初始化统计字典。

        print(f"Evaluator initialized for plan: '{self.plan.description}'") # 打印评估器初始化信息。
        print(f"  - Memory Layout: {self.plan.memory_layout_policy.name}, Scheduling Policy: {self.plan.decoding_dataflow.scheduling_policy.name}") # 打印内存和调度策略。
        print(f"  - Topology: {self.num_npu_cores} Cores, {self.num_channels} Channels. Bindings: {self.core_to_channels_map}") # 打印硬件拓扑信息。

    # --- Helper Functions (Unchanged) ---
    def _get_op_shape(self, op_name: str, kv_cache_len: int) -> Dict[str, int]: # 定义一个辅助函数来获取操作的维度（形状）。
        shapes = { "Attention::QKV_Projections": {"M": self.workload.batch_size, "K": self.workload.hidden_size, "N": self.workload.hidden_size * 3}, "Attention::Score_Computation_QK": {"M": 1, "K": self.workload.head_dim, "N": kv_cache_len}, "Attention::Context_Computation_SV": {"M": self.workload.head_dim, "K": kv_cache_len, "N": self.workload.head_dim}, "Attention::Output_Projection": {"M": self.workload.batch_size, "K": self.workload.hidden_size, "N": self.workload.hidden_size}, "FFN::Gate_Up_Projections": {"M": self.workload.batch_size, "K": self.workload.hidden_size, "N": self.workload.intermediate_size * 2}, "FFN::Down_Projection": {"M": self.workload.batch_size, "K": self.workload.intermediate_size, "N": self.workload.hidden_size}, } # 定义一个包含所有操作形状的字典。
        return shapes.get(op_name, {}) # 根据操作名返回对应的形状，如果找不到则返回空字典。
    def _get_required_resource_type(self, mapping: OperatorMapping) -> str: # 定义辅助函数来确定任务所需的资源类型。
        if mapping.execution_target == ExecutionTarget.NPU_ONLY: return "NPU_SLICE" # 如果在 NPU 上执行，则需要 NPU_SLICE。
        task_cat = mapping.op_category # 获取任务的类别。
        pim_type = self.plan.channel_topology[0].pim_type # 获取 PIM 的类型。
        if pim_type == PIMType.ATTENTION_FFN: return "PIM_ATTENTION" if task_cat == OpCategory.ATTENTION else "PIM_FFN" # 根据 PIM 类型和任务类别返回专用 PIM 资源类型。
        return "PIM_COMMON" # 否则返回通用 PIM 资源类型。
    def _get_requesting_core(self, task: TaskUnit) -> int: return task.unit_id % self.num_npu_cores # 定义辅助函数来确定发起请求的 NPU 核心 ID。
    def _get_task_data_size(self, task: TaskUnit) -> int: # 定义辅助函数来计算任务需要传输的数据大小。
        M, K, N = task.shape.get("M",1), task.shape.get("K",1), task.shape.get("N",1) # 获取任务的维度。
        if task.mapping.execution_target == ExecutionTarget.NPU_ONLY: return (M * K + K * N) * self.element_size_bytes # 如果是 NPU 任务，数据量为输入+权重。
        if task.mapping.execution_target == ExecutionTarget.PIM_ONLY: return (M * K) * self.element_size_bytes # 如果是 PIM 任务，数据量为输入激活。
        return 0 # 其他情况返回0。
    def _get_noc_delay(self, req_core_id: int, target_ch_id: int, data_bytes: int) -> float: # 定义辅助函数计算片上网络（NoC）的延迟。
        if data_bytes == 0: return 0.0 # 如果没有数据传输，则延迟为0。
        model = self.plan.interconnect_model # 获取互连模型。
        return data_bytes * (model.cycles_per_byte_intra_group if target_ch_id in self.core_to_channels_map.get(req_core_id, []) else model.cycles_per_byte_inter_group) # 根据是组内还是组间通信，计算总的 NoC 延迟。


    def _simulate_prefill_stage(self): # 定义模拟 Prefill 阶段的方法。
        """
        FINAL CORRECTED VERSION: This version fixes two critical issues.
        1.  Modeling Bug: The total latency of a layer is now correctly modeled as the
            sum of its COMPUTE cycles, reflecting a deep pipeline dominated by the
            compute bottleneck, not the flawed serial addition of total latencies.
        2.  Statistics Bug: All statistics, including the previously omitted
            'active_compute_cycles' and 'total_active_cycles', are now correctly
            accumulated, fixing the "0% utilization" error.
        """
        prompt_len = self.workload.prompt_len # 获取提示长度。
        batch_size = self.workload.batch_size # 获取批处理大小。
        
        prefill_ops = { # 定义 Prefill 阶段的所有操作及其维度。
            "Prefill::QKV_Projections":    {"M": prompt_len * batch_size, "K": self.workload.hidden_size, "N": self.workload.hidden_size, "num_ops": 3},
            "Prefill::Attention_BMM_QK":   {"M": prompt_len * batch_size, "K": self.workload.head_dim, "N": prompt_len, "num_ops": self.workload.num_heads},
            "Prefill::Attention_BMM_SV":   {"M": prompt_len * batch_size, "K": prompt_len, "N": self.workload.head_dim, "num_ops": self.workload.num_heads},
            "Prefill::Output_Projection":  {"M": prompt_len * batch_size, "K": self.workload.hidden_size, "N": self.workload.hidden_size, "num_ops": 1},
            "Prefill::FFN_Projections":    {"M": prompt_len * batch_size, "K": self.workload.hidden_size, "N": self.workload.intermediate_size, "num_ops": 2},
            "Prefill::FFN_Down_Projection":{"M": prompt_len * batch_size, "K": self.workload.intermediate_size, "N": self.workload.hidden_size, "num_ops": 1},
        }
        
        # --- FIX: Unified accumulation of all stats for one layer ---
        stats = self.prefill_stats["NPU"] # 获取 Prefill 阶段 NPU 的统计数据字典的引用。
        # Ensure stats are reset if this function were ever called multiple times
        for key in stats: stats[key] = 0.0 # 重置所有统计数据为0，以防函数被多次调用。

        for op_name, shape in prefill_ops.items(): # 遍历 Prefill 阶段的每一个操作。
            params = { # 准备调用 NPU 评估器所需的参数字典。
                "M": shape['M'], "K": shape['K'], "N": shape['N'],
                "num_ops_parallel": shape['num_ops'],
                "is_data_parallel": False, # Prefill 阶段的操作被建模为张量并行。
                "op_type": "GEMM", "num_channels_used": self.num_channels, "op_name": op_name
            }
            
            perf_result = self.npu_eval.evaluate(**params) # 调用 NPU 评估器获取该操作的性能结果。
            
            # Correctly accumulate ALL statistics from the performance result.
            stats["total_op_flops"] += perf_result.op_flops # 累加浮点运算次数。
            stats["total_bytes_moved"] += (perf_result.bytes_from_dram + perf_result.bytes_to_dram) # 累加数据移动量。
            stats["active_compute_cycles"] += perf_result.compute_cycles # 累加活跃的计算周期。
            
            # In our pipelined model for this compute-bound stage, the total active time
            # is equivalent to the compute time, as I/O is hidden.
            stats["total_active_cycles"] += perf_result.compute_cycles # 累加总活跃周期（在流水线模型中，这等于计算周期）。
            
        # The total latency for one layer IS the total active time we just calculated.
        total_prefill_cycles_one_layer = stats["total_active_cycles"] # 单层的总延迟等于我们累加的总活跃周期。
        
        # Scale the per-layer results to the full model.
        self.total_prefill_cycles = total_prefill_cycles_one_layer * self.workload.num_layers # 将单层延迟乘以总层数得到整个模型的 Prefill 延迟。
        
        # Also scale all accumulated stats for the final report.
        for key in stats: # 遍历统计字典中的所有键。
            stats[key] *= self.workload.num_layers # 将所有统计数据也乘以总层数。
        
        # Update all hardware resources as a sync barrier.
        current_max_time = np.max(self.physical_channel_finish_times) # 找到当前所有物理通道的最晚完成时间。
        new_finish_time = current_max_time + self.total_prefill_cycles # 计算 Prefill 阶段结束后的新完成时间。
        
        for res in self.resources: # 遍历所有硬件资源。
            self.resource_finish_times[res] = new_finish_time # 将每个逻辑资源的完成时间更新为新时间（同步点）。
        self.physical_channel_finish_times.fill(new_finish_time) # 将所有物理通道的完成时间也更新为新时间。

    # --- MODIFIED: Strategy-Driven DECODING Simulation Kernel ---
    def _simulate_decoding_step(self, kv_cache_len: int): # 定义模拟单个 Decode 步骤的方法。
        for stage in self.plan.decoding_dataflow.stages: # 遍历 Decode 数据流中的每一个阶段。
            task_pool: List[TaskUnit] = [] # 初始化一个任务池列表。
            for op_name, mapping in stage.operator_mappings.items(): # 遍历阶段中的每个操作映射。
                if mapping.execution_target == ExecutionTarget.MEMORY_BOUND: continue # 如果是内存密集型操作，则跳过（在此处不模拟）。
                shape = self._get_op_shape(op_name, kv_cache_len) # 获取操作的形状。
                if not shape: continue # 如果没有形状信息，则跳过。
                op_type = "GEMV" if "Computation" in op_name else "GEMM" # 判断操作类型是 GEMV 还是 GEMM。
                num_units = self.workload.batch_size * self.workload.num_heads if "Computation" in op_name else self.workload.batch_size # 计算需要创建的任务单元数量。
                for i in range(num_units): # 为每个单元创建 TaskUnit 对象。
                    task_pool.append(TaskUnit(mapping=mapping, shape=shape, unit_id=i, op_type=op_type)) # 将任务添加到池中。
            
            candidate_map = self._determine_candidate_resources(task_pool) # 根据内存布局策略确定每个任务的候选资源。
            
            policy = self.plan.decoding_dataflow.scheduling_policy # 获取当前计划的调度策略。
            if policy == SchedulingPolicy.TASK_DYNAMIC_DISPATCH: self._schedule_dynamic_dispatch(task_pool, candidate_map) # 如果是动态分派策略，则调用相应函数。
            elif policy == SchedulingPolicy.HEAD_STATIC_MAPPING: self._schedule_static_mapping(task_pool, candidate_map) # 如果是静态映射策略，则调用相应函数。
            elif policy == SchedulingPolicy.BLOCK_GREEDY_MAPPING: self._schedule_block_greedy(task_pool, candidate_map) # 如果是块内贪婪策略，则调用相应函数。
            else: raise NotImplementedError(f"Policy {policy.name} not implemented.") # 如果策略未实现，则抛出错误。

    def _determine_candidate_resources(self, task_pool: List[TaskUnit]) -> Dict[int, List[HardwareResource]]: # 定义确定候选资源的方法。
        candidate_map = {} # 初始化候选资源映射字典。
        layout = self.plan.memory_layout_policy # 获取内存布局策略。
        
        for task in task_pool: # 遍历任务池中的每个任务。
            res_type = self._get_required_resource_type(task.mapping) # 获取任务需要的资源类型。
            if layout == MemoryLayoutPolicy.BLOCK_PIM: candidates = [r for r in self.resources if r[1] == res_type] # BlockPIM策略：所有同类型资源都是候选。
            elif layout == MemoryLayoutPolicy.REQ_PAR: # REQ_PAR策略：
                req_id = task.unit_id // self.workload.num_heads # 计算请求 ID。
                target_channel = req_id % self.num_channels # 确定目标通道。
                candidates = [(target_channel, res_type)] # 只有一个候选资源。
            elif layout == MemoryLayoutPolicy.REQ_HEAD_PAR: # REQ_HEAD_PAR策略：
                target_channel = task.unit_id % self.num_channels # 根据任务（头）ID 确定目标通道。
                candidates = [(target_channel, res_type)] # 只有一个候选资源。
            candidate_map[id(task)] = candidates # 将任务的候选资源列表存入字典。
        return candidate_map # 返回映射字典。

    def _schedule_task(self, task: TaskUnit, resource: HardwareResource): # 定义调度单个任务到特定资源上的方法。
        params = {**task.shape, "num_ops_parallel": 1, "is_data_parallel": False, "op_type": task.op_type, "num_channels_used": 1, "op_name": task.mapping.operator_name} # 准备调用评估器的参数。
        perf = self.npu_eval.evaluate(**params) if task.mapping.execution_target == ExecutionTarget.NPU_ONLY else self.pim_eval.evaluate(**params) # 根据目标硬件调用相应的评估器。
        
        target_ch_id, req_core_id = resource[0], self._get_requesting_core(task) # 获取目标通道 ID 和请求核心 ID。
        data_size, noc_delay = self._get_task_data_size(task), self._get_noc_delay(req_core_id, target_ch_id, self._get_task_data_size(task)) # 计算数据大小和 NoC 延迟。
        
        earliest_start = max(self.resource_finish_times.get(resource, 0.0), self.physical_channel_finish_times[target_ch_id]) + self.plan.interconnect_model.fixed_latency_cycles # 计算任务最早可以开始的时间。

        if task.mapping.execution_target == ExecutionTarget.NPU_ONLY: # 如果是 NPU 任务。
            io_start = earliest_start + noc_delay # I/O 开始时间需要加上 NoC 延迟。
            self.physical_channel_finish_times[target_ch_id] = io_start + perf.io_cycles # 更新物理通道的完成时间。
            self.resource_finish_times[resource] = io_start + perf.latency_cycles # 更新逻辑计算资源的完成时间。
        else: # 如果是 PIM 任务。
            finish_time = earliest_start + perf.latency_cycles # 计算完成时间。
            self.resource_finish_times[resource], self.physical_channel_finish_times[target_ch_id] = finish_time, finish_time # 同时更新逻辑资源和物理通道的完成时间。
        
        # MODIFIED: Update decode_stats dictionary
        stats = self.decode_stats[resource] # 获取该资源的统计数据字典。
        stats["total_op_flops"] += perf.op_flops; stats["total_bytes_moved"] += perf.bytes_from_dram + perf.bytes_to_dram # 累加 FLOPs 和数据移动量。
        stats["active_io_cycles"] += perf.io_cycles; stats["active_compute_cycles"] += perf.compute_cycles # 累加 I/O 和计算周期。
        stats["total_active_cycles"] += perf.latency_cycles # 累加总活跃周期。

    def _schedule_dynamic_dispatch(self, task_pool: List[TaskUnit], candidate_map: Dict[int, List[HardwareResource]]): # 定义动态分派调度策略的实现。
        for task in task_pool: # 遍历任务池中的每个任务。
            candidates = candidate_map[id(task)] # 获取该任务的候选资源。
            best_resource, min_finish_time = None, float('inf') # 初始化最佳资源和最早完成时间。
            for res in candidates: # 遍历所有候选资源。
                finish_time = max(self.resource_finish_times.get(res, 0.0), self.physical_channel_finish_times[res[0]]) # 计算如果任务调度到该资源上的完成时间。
                if finish_time < min_finish_time: # 如果找到了一个能更早完成的资源。
                    min_finish_time, best_resource = finish_time, res # 更新最佳资源和最早完成时间。
            if best_resource: self._schedule_task(task, best_resource) # 如果找到了最佳资源，则将任务调度上去。

    def _schedule_static_mapping(self, task_pool: List[TaskUnit], candidate_map: Dict[int, List[HardwareResource]]): # 定义静态映射调度策略的实现。
        for task in task_pool: # 遍历任务池中的每个任务。
            candidates = candidate_map[id(task)] # 获取该任务的候选资源（在静态映射下，通常只有一个）。
            if len(candidates) == 1 and candidates[0] in self.resources: # 确认只有一个候选资源且该资源有效。
                self._schedule_task(task, candidates[0]) # 将任务调度到该唯一的资源上。
    
    def _schedule_block_greedy(self, task_pool: List[TaskUnit], candidate_map: Dict[int, List[HardwareResource]]): # 定义块内贪婪调度策略的实现。
        core_to_block_map = {core_id: [res for res in self.resources if res[0] in channels] for core_id, channels in self.core_to_channels_map.items()} # 创建一个从核心ID到其本地资源块的映射。
        for task in task_pool: # 遍历任务池中的每个任务。
            req_core = self._get_requesting_core(task) # 获取请求核心。
            local_candidates = list(set(candidate_map[id(task)]).intersection(set(core_to_block_map[req_core]))) # 找到既是任务候选资源又是本地资源的交集。
            best_resource, min_finish_time = None, float('inf') # 初始化最佳资源和最早完成时间。
            for res in local_candidates: # 遍历本地候选资源。
                finish_time = max(self.resource_finish_times.get(res, 0.0), self.physical_channel_finish_times[res[0]]) # 计算完成时间。
                if finish_time < min_finish_time: # 寻找最早完成的资源。
                    min_finish_time, best_resource = finish_time, res # 更新最佳资源。
            if best_resource: self._schedule_task(task, best_resource) # 将任务调度到找到的本地最佳资源上。

    # --- Analytics and Reporting ---
    def _calculate_metrics_for_stage(self, stats_source: Dict, total_time_s: float) -> Dict[str, Any]: # 定义计算最终性能指标的方法。
        aggr = {"NPU": {}, "PIM": {}} # 初始化聚合统计数据的字典。
        if any(isinstance(k, tuple) for k in stats_source.keys()): # 判断是 Decode 阶段的按资源统计数据。
             stat_keys = list(list(stats_source.values())[0].keys()) # 获取统计键。
             aggr = {"NPU": {k: 0.0 for k in stat_keys}, "PIM": {k: 0.0 for k in stat_keys}} # 初始化聚合字典。
             for r, s in stats_source.items(): # 遍历每个资源的统计数据。
                 t = "NPU" if "NPU" in r[1] else "PIM" # 判断资源类型是 NPU 还是 PIM。
                 for k, v in s.items(): aggr[t][k] += v # 将数据累加到对应的 NPU 或 PIM 类别下。
        else: # 如果是 Prefill 阶段的统计数据。
             aggr = stats_source # 直接使用。

        npu_conf, pim_conf = self.plan.npu_config, self.plan.pim_config # 获取 NPU 和 PIM 的硬件配置。
        npu_peak_tops = npu_conf.peak_tops_per_core * self.num_npu_cores # 计算 NPU 的理论峰值算力。
        npu_peak_bw = npu_conf.effective_memory_bw_gb_s # 获取 NPU 的有效内存带宽。
        pim_peak_tops = pim_conf.peak_tops_per_channel * self.num_channels # 计算 PIM 的理论峰值算力。
        pim_peak_bw = pim_conf.main_memory_bw_per_channel_gb_s * self.num_channels # 计算 PIM 的理论峰值带宽。
        
        m = {"npu_effective_tops": 0, "pim_effective_tops": 0, "npu_compute_utilization_pct": 0, "pim_compute_utilization_pct": 0, "npu_bw_utilization_pct": 0, "pim_bw_utilization_pct": 0} # 初始化一个存储最终指标的字典。
        if total_time_s > 0: # 确保总时间大于0，避免除零错误。
            m["npu_effective_tops"] = (aggr["NPU"].get("total_op_flops", 0) / total_time_s) / 1e12 # 计算 NPU 的有效算力（TFLOPS）。
            m["pim_effective_tops"] = (aggr["PIM"].get("total_op_flops", 0) / total_time_s) / 1e12 # 计算 PIM 的有效算力（TFLOPS）。
            npu_achieved_bw = (aggr["NPU"].get("total_bytes_moved", 0) / total_time_s) / 1e9 # 计算 NPU 实现的带宽（GB/s）。
            m["npu_bw_utilization_pct"] = (npu_achieved_bw / npu_peak_bw) * 100 if npu_peak_bw > 0 else 0 # 计算 NPU 的带宽利用率。
            pim_achieved_bw = (aggr["PIM"].get("total_bytes_moved", 0) / total_time_s) / 1e9 # 计算 PIM 实现的带宽（GB/s）。
            m["pim_bw_utilization_pct"] = (pim_achieved_bw / pim_peak_bw) * 100 if pim_peak_bw > 0 else 0 # 计算 PIM 的带宽利用率。
            npu_active_time = aggr["NPU"].get("active_compute_cycles", 0) / (npu_conf.clock_ghz * 1e9) # 计算 NPU 的活跃计算时间（秒）。
            m["npu_compute_utilization_pct"] = (aggr["NPU"].get("total_op_flops", 0) / (npu_peak_tops * 1e12 * npu_active_time)) * 100 if npu_active_time > 0 and npu_peak_tops > 0 else 0.0 # 计算 NPU 的计算单元利用率。
            pim_active_time = aggr["PIM"].get("total_active_cycles", 0) / (pim_conf.clock_ghz * 1e9) # 计算 PIM 的活跃时间（秒）。
            m["pim_compute_utilization_pct"] = (aggr["PIM"].get("total_op_flops", 0) / (pim_peak_tops * 1e12 * pim_active_time)) * 100 if pim_active_time > 0 and pim_peak_tops > 0 else 0.0 # 计算 PIM 的计算单元利用率。
        return m # 返回包含所有计算指标的字典。

    def evaluate(self) -> Dict[str, Any]: # 定义顶层评估方法，启动整个仿真流程。
        self.resource_finish_times = {res: 0.0 for res in self.resources}; self.physical_channel_finish_times.fill(0.0) # 重置所有资源完成时间为0。
        self._initialize_resources(); self.total_prefill_cycles = 0.0 # 重新初始化资源和统计数据。
        
        if self.plan.prefill_dataflow: self._simulate_prefill_stage() # 如果计划中定义了 Prefill 阶段，则运行 Prefill 仿真。
        
        decode_start_cycles = self.total_prefill_cycles # 记录 Decode 阶段开始的时间点。
        for i in range(self.workload.add_tokens): self._simulate_decoding_step(self.workload.prompt_len + i) # 循环模拟每一个 Decode 步骤。
        
        total_cycles = np.max(self.physical_channel_finish_times) # 整个仿真的总周期数是所有物理通道中最晚的完成时间。
        decode_cycles = total_cycles - decode_start_cycles # 计算 Decode 阶段的总周期数。
        
        clock_hz = self.plan.npu_config.clock_ghz * 1e9 # 获取时钟频率（Hz）。
        prefill_latency_ms = (self.total_prefill_cycles / clock_hz * 1000) # 计算 Prefill 延迟（毫秒）。
        decode_latency_ms = (decode_cycles / clock_hz * 1000) # 计算 Decode 延迟（毫秒）。
        
        prefill_time_s = self.total_prefill_cycles / clock_hz # 计算 Prefill 时间（秒）。
        decode_time_s = decode_cycles / clock_hz # 计算 Decode 时间（秒）。
        
        return { # 构建并返回一个包含所有仿真结果的字典。
            "plan_description": self.plan.description, # 计划描述。
            "prefill": { # Prefill 阶段结果。
                "latency_ms": prefill_latency_ms,
                "throughput_tokens_per_sec": (self.workload.prompt_len * self.workload.batch_size) / prefill_time_s if prefill_time_s > 0 else 0,
                "metrics": self._calculate_metrics_for_stage(self.prefill_stats, prefill_time_s)
            },
            "decode": { # Decode 阶段结果。
                "latency_ms": decode_latency_ms,
                "throughput_tokens_per_sec": (self.workload.add_tokens * self.workload.batch_size) / decode_time_s if decode_time_s > 0 else 0,
                "metrics": self._calculate_metrics_for_stage(self.decode_stats, decode_time_s)
            },
            "end_to_end": { # 端到端结果。
                "total_latency_ms": prefill_latency_ms + decode_latency_ms,
                "avg_throughput_tps": ( (self.workload.prompt_len + self.workload.add_tokens) * self.workload.batch_size) / (prefill_time_s + decode_time_s) if (prefill_time_s + decode_time_s) > 0 else 0
            }
        }

# ==============================================================================
# SECTION 4: MAIN EXECUTION FLOW FOR DSE
# ==============================================================================
# (此为主程序块的旧版本)
"""
if __name__ == '__main__':
    ...
"""
# ==============================================================================
# SECTION 4: MAIN EXECUTION FLOW FOR DSE
# ==============================================================================
if __name__ == '__main__': # 程序的入口点。
    # --- Data Pre-processing Helper ---
    # This helper function prepares the multi-core dataset.
    # It should ideally be run once, not every time the simulation starts.
    def prepare_multicore_dataset(files_with_cores: Dict[str, int], output_file: str): # 定义一个数据预处理辅助函数。
        import os # 导入 os 模块。
        if os.path.exists(output_file): # 检查合并后的数据集是否已存在。
            print(f"INFO: Multi-core dataset '{output_file}' already exists. Skipping preparation.") # 如果存在，则跳过。
            return # 结束函数。
        
        print(f"INFO: Preparing multi-core dataset '{output_file}'...") # 打印准备信息。
        df_list = [] # 初始化一个空列表来存放 DataFrame。
        for file_path, num_cores in files_with_cores.items(): # 遍历输入的多个数据文件。
            try: # 使用 try-except 处理文件未找到错误。
                df = pd.read_csv(file_path) # 读取 CSV 文件。
                df['num_cores'] = num_cores # 添加一个 'num_cores' 列来标记数据来源。
                df_list.append(df) # 将处理后的 DataFrame 添加到列表中。
            except FileNotFoundError: # 如果文件未找到。
                print(f"WARNING: Data file '{file_path}' not found during pre-processing. It will be skipped.") # 打印警告。
        
        if not df_list: # 如果列表为空（所有文件都未找到）。
            raise FileNotFoundError("None of the multi-core data files were found. Cannot create combined dataset.") # 抛出异常。
            
        combined_df = pd.concat(df_list, ignore_index=True) # 将列表中的所有 DataFrame 合并成一个。
        combined_df.to_csv(output_file, index=False) # 将合并后的 DataFrame 保存到输出文件中。
        print("INFO: Multi-core dataset successfully prepared.") # 打印成功信息。

    # --- Import the new TensorParallelCostModel ---
    from npu_v0_4_2_model import TensorParallelCostModel # 再次导入 TensorParallelCostModel (虽然已在顶部导入，但此处可能是为了明确性)。

    print("="*40); print("PHASE 1: NPU CORE MODEL TRAINING"); print("="*40) # 打印阶段标题。
    
    # --- MODIFIED: Prepare and train dual models ---
    try: # 使用 try-except 块来捕获模型训练过程中的错误。
        # 1a. Prepare the combined multi-core dataset
        multicore_files = { # 定义多核数据集的文件和对应的核心数。
            "gemm_rich_perf_data_extended_2.csv": 2,
            "gemm_rich_perf_data_extended_4.csv": 4
        }
        prepare_multicore_dataset(multicore_files, "multi_core_data.csv") # 调用辅助函数准备合并后的数据集。

        # 1b. Train the single-core model for data-parallel tasks
        single_core_model = HybridCoreCostModel(csv_path="gemm_rich_perf_data_extended_1.csv") # 训练单核性能模型。
        
        # 1c. Train the tensor-parallel model using the combined multi-core data
        tensor_parallel_model = TensorParallelCostModel(csv_path="multi_core_data.csv") # 训练张量并行性能模型。

    except (FileNotFoundError, ValueError) as e: # 捕获可能的文件未找到或值错误。
        print(f"\nFATAL ERROR during model training: {e}"); exit() # 打印致命错误并退出程序。

    print("\n" + "="*40); print("PHASE 2: DEFINING DSE SCENARIOS"); print("="*40) # 打印阶段标题。
    # This section is unchanged
    hw_config = { # 定义硬件配置字典。
        "npu": {"num_cores": 2, "num_channels": 4, "channels_per_core": 2, "clock_ghz": 2.0},
        "pim": {"total_channels": 4, "clock_ghz": 2.0}, 
        "interconnect": {"fixed_latency_cycles": 50, "cycles_per_byte_intra_group": 0.1, "cycles_per_byte_inter_group": 0.2},
        "topology": {
            "npu_cores": [{"core_id": 0, "serves_channel_ids": [0, 1]}, {"core_id": 1, "serves_channel_ids": [2, 3]}],
            "channels": [{"channel_id": i, "pim_type": "ATTENTION_FFN"} for i in range(4)]
        }
    }
    
    base_dataflow_config = { # 定义一个基础的数据流配置字典。
        "prefill": {
            "Prefill::QKV_Projections": {"target": "NPU_ONLY"}, "Prefill::Attention_BMM": {"target": "NPU_ONLY"},
            "Prefill::FFN_Projections": {"target": "NPU_ONLY"}, "Prefill::Output_Projection": {"target": "NPU_ONLY"},
            "Prefill::FFN_Down_Projection": {"target": "NPU_ONLY"},
        },
        "decoding": { "Attention::QKV_Projections": {"target": "NPU_ONLY"}, "Attention::Score_Computation_QK": {"target": "PIM_ONLY"},
                      "Attention::Context_Computation_SV": {"target": "PIM_ONLY"}, "Attention::Output_Projection":  {"target": "NPU_ONLY"},
                      "FFN::Gate_Up_Projections": {"target": "NPU_ONLY"}, "FFN::Down_Projection": {"target": "NPU_ONLY"} }
    }

    dataflow_config_1 = copy.deepcopy(base_dataflow_config) # 创建第一个DSE场景的配置。
    dataflow_config_1.update({"memory_layout_policy": "BLOCK_PIM", "decoding": {**base_dataflow_config["decoding"], "scheduling_policy": "TASK_DYNAMIC_DISPATCH"}}) # 更新其内存和调度策略。
    
    dataflow_config_2 = copy.deepcopy(base_dataflow_config) # 创建第二个DSE场景的配置。
    dataflow_config_2.update({"memory_layout_policy": "BLOCK_PIM", "decoding": {**base_dataflow_config["decoding"], "scheduling_policy": "BLOCK_GREEDY_MAPPING"}}) # 更新其调度策略。

    dataflow_config_3 = copy.deepcopy(base_dataflow_config) # 创建第三个DSE场景的配置。
    dataflow_config_3.update({"memory_layout_policy": "REQ_HEAD_PAR", "decoding": {**base_dataflow_config["decoding"], "scheduling_policy": "HEAD_STATIC_MAPPING"}}) # 更新其内存和调度策略。

    workload = ModelConfig(prompt_len=2048, add_tokens=4096, batch_size=4, num_layers=32) # 创建工作负载配置实例。
    
    # --- MODIFIED: Initialize reusable components with dual models ---
    plan_proto = create_plan_from_configs(hw_config, dataflow_config_1, "") # 使用一个配置创建原型计划，用于初始化评估器。
    
    # Create the NPUEvaluator library using its new constructor
    npu_eval_lib = NPUEvaluator( # 创建 NPU 评估器库实例。
        npu_config=plan_proto.npu_config, # 传入 NPU 配置。
        single_core_model=single_core_model, # 传入单核模型。
        tensor_parallel_model=tensor_parallel_model # 传入张量并行模型。
    )
    
    pim_eval_lib = PIMEvaluator(plan_proto.pim_config) # 创建 PIM 评估器库实例。
    profile_provider = ProfileProvider(plan_proto, npu_eval_lib, pim_eval_lib, workload) # 创建 ProfileProvider 实例。

    # --- UNCHANGED: The simulation and reporting loop ---
    for i, (df_config, desc) in enumerate([ # 循环遍历定义的 DSE 场景。
        (dataflow_config_1, "Layout: BLOCK_PIM, Schedule: DYNAMIC_DISPATCH (Fine-grained)"),
        (dataflow_config_2, "Layout: BLOCK_PIM, Schedule: BLOCK_GREEDY (Medium-grained)"),
        (dataflow_config_3, "Layout: REQ_HEAD_PAR, Schedule: STATIC_MAPPING (Coarse-grained)")
    ]):
        print("\n" + "="*40 + f" DSE RUN #{i+1} " + "="*40) # 打印 DSE 运行标题。
        plan = create_plan_from_configs(hw_config, df_config, desc) # 为当前场景创建具体的执行计划。
        # The UnifiedEvaluator's constructor and methods are not changed.
        simulator = UnifiedEvaluator(plan, workload, profile_provider) # 创建统一评估器（仿真器）实例。
        results = simulator.evaluate() # 运行仿真并获取结果。
        
        # The printing logic remains exactly the same.
        print("\n" + "="*80); print(f"--- SIMULATION RESULTS FOR: {results.get('plan_description', 'N/A')} ---") # 打印结果的顶层标题。
        prefill_res, decode_res, e2e_res = results.get("prefill", {}), results.get("decode", {}), results.get("end_to_end", {}) # 从结果字典中提取各阶段的结果。
        
        print("\n" + "-"*35 + " PREFILL STAGE " + "-"*32) # 打印 Prefill 阶段的标题。
        print(f"  - Latency: {prefill_res.get('latency_ms', 0):.4f} ms"); print(f"  - Throughput: {prefill_res.get('throughput_tokens_per_sec', 0):.2f} tokens/sec") # 打印延迟和吞吐量。
        metrics = prefill_res.get("metrics", {}) # 获取 Prefill 的详细指标。
        if metrics: # 如果有指标。
            print(f"  NPU:"); print(f"    - Effective Compute:   {metrics.get('npu_effective_tops', 0):.3f} TFLOPS"); print(f"    - Compute Utilization: {metrics.get('npu_compute_utilization_pct', 0):.2f} %"); print(f"    - BW Utilization:      {metrics.get('npu_bw_utilization_pct', 0):.2f} %") # 打印 NPU 的性能指标。
        
        print("\n" + "-"*35 + " DECODE STAGE " + "-"*33) # 打印 Decode 阶段的标题。
        print(f"  - Latency: {decode_res.get('latency_ms', 0):.4f} ms (for {workload.add_tokens} tokens)"); print(f"  - Throughput: {decode_res.get('throughput_tokens_per_sec', 0):.2f} tokens/sec") # 打印延迟和吞吐量。
        metrics = decode_res.get("metrics", {}) # 获取 Decode 的详细指标。
        if metrics: # 如果有指标。
            print(f"  NPU:"); print(f"    - Effective Compute:   {metrics.get('npu_effective_tops', 0):.3f} TFLOPS"); print(f"    - Compute Utilization: {metrics.get('npu_compute_utilization_pct', 0):.2f} %"); print(f"    - BW Utilization:      {metrics.get('npu_bw_utilization_pct', 0):.2f} %") # 打印 NPU 的性能指标。
            print(f"  PIM:"); print(f"    - Effective Compute:   {metrics.get('pim_effective_tops', 0):.3f} TFLOPS"); print(f"    - Compute Utilization: {metrics.get('pim_compute_utilization_pct', 0):.2f} %"); print(f"    - BW Utilization:      {metrics.get('pim_bw_utilization_pct', 0):.2f} %") # 打印 PIM 的性能指标。
            
        print("\n" + "-"*35 + " END-TO-END SUMMARY " + "-"*26) # 打印端到端总结的标题。
        print(f"  - Total Latency: {e2e_res.get('total_latency_ms', 0):.4f} ms"); print(f"  - Avg System Throughput: {e2e_res.get('avg_throughput_tps', 0):.2f} tokens/sec") # 打印总延迟和平均吞吐量。
        print("="*80) # 打印结束分隔线
