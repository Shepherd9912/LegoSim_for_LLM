# pim_v0_4.py
import math # 导入 math 模块，用于执行数学运算，如向上取整 (ceil)。
from typing import Dict, Optional # 从 typing 模块导入类型提示 Dict (字典) 和 Optional (可选类型)。
from dataclasses import dataclass # 从 dataclasses 模块导入 dataclass 装饰器，用于自动生成类的方法。
import argparse # 导入 argparse 模块，用于解析命令行参数。

# ==============================================================================
# SECTION 1: UPGRADED DEPENDENCIES
# ==============================================================================

# --- MODIFIED: Import from the new utils file ---
from utils import PIMConfig, PerformanceResult, PIMEnergyCostModel # 从我们自定义的 utils_v0_4 模块中导入 PIMConfig 和 PerformanceResult 类。


# ==============================================================================
# SECTION 2: UNCHANGED CORE MODELS
# The underlying microarchitecture and operation models remain the same.
# ==============================================================================

@dataclass # 使用 dataclass 装饰器来自动生成类的构造函数等。
class MicroarchitectureCostModel: # Unchanged # 定义微架构成本模型类，用于存储底层原子操作的成本。
    """
    存储从cycle-accurate仿真中标定出的原子操作成本。
    """
    GPR_SIZE_BYTES: int = 32 # 定义通用寄存器 (GPR) 的大小（字节），默认为32。
    ELEMENTS_PER_MAC: int = 16 # 定义每次 MAC (乘加) 操作处理的元素数量，默认为16。
    T_pipeline_fast: float = 226.0 # 定义快速路径下（如GB缓存命中）流水线的周期时间，默认为226.0。
    T_pipeline_slow: float = 384.0 # 定义慢速路径下（如DRAM访问）流水线的周期时间，默认为384.0。
    T_stall_read_after_write: float = 54.0 # 定义写后读操作导致的流水线停顿周期数，默认为54.0。

    def get_gb_fill_latency(self, num_elements: int, element_size_bytes: int = 2) -> float: # 定义一个方法来计算填充全局缓冲区 (Global Buffer) 的延迟。
        if num_elements == 0: return 0 # 如果没有元素需要填充，则延迟为0。
        total_bytes = num_elements * element_size_bytes # 计算需要填充的总字节数。
        num_gprs = math.ceil(total_bytes / self.GPR_SIZE_BYTES) # 根据总字节数计算需要多少次通用寄存器 (GPR) 的加载操作。
        return 102.0 + 2.74 * num_gprs # 根据一个线性模型返回总的填充延迟。

class PIMOperationModel: # Unchanged # 定义 PIM 操作模型类，用于将高层运算分解为底层指令并计算延迟。
    """
    将高阶数学运算分解为具体的指令流模板，并根据MCM的成本计算总延迟。
    """
    def __init__(self, mcm: MicroarchitectureCostModel, config: PIMConfig, element_size_bytes: int = 2, ecm: PIMEnergyCostModel = None): # 类的构造函数。
        self.mcm = mcm # 保存传入的微架构成本模型对象。
        self.config = config # 保存传入的 PIM 硬件配置对象。
        self.element_size_bytes = element_size_bytes # 保存每个数据元素的大小（字节），默认为2。
        # self.ecm = PIMEnergyCostModel() # NEW: 实例化能耗成本模型
        self.ecm = ecm if ecm is not None else PIMEnergyCostModel()
    
    def _calculate_gemv_gb_cached(self, M: int, K: int) -> float: # 定义一个内部方法，计算在全局缓冲区缓存命中的情况下的 GEMV (矩阵向量乘) 延迟。
        """
        MODIFIED: The latency model is updated to better reflect I/O and compute overlap.
        Instead of simple addition (I/O + Compute), we now model that the initial I/O
        (filling the Global Buffer) is largely hidden behind the much longer compute pipeline.
        The total latency is therefore dominated by the compute loop itself.
        """
        # I/O latency to fill the Global Buffer. This is the part that will be "hidden".
        latency_activation_fill = self.mcm.get_gb_fill_latency(K, self.element_size_bytes) # 计算填充激活向量到全局缓冲区的I/O延迟（这部分将被隐藏）。
        
        # Compute loop latency. This is the dominant factor.
        GEMV_OPSIZE = 64.0 # 定义 GEMV 操作的一个参数。
        elements_per_instruction = self.mcm.ELEMENTS_PER_MAC * GEMV_OPSIZE # 计算单条指令能处理的元素数量。
        num_instructions_per_output = math.ceil(K / elements_per_instruction) # 计算得到一个输出结果需要多少条指令。
        num_parallel_batches = math.ceil(M / self.config.num_banks_per_channel) # 计算并行批处理的数量，这取决于输出向量维度 M 和 PIM 的 bank 数量。
        latency_compute_loop = num_parallel_batches * num_instructions_per_output * self.mcm.T_pipeline_fast # 计算核心的计算循环延迟。
        
        # Finalization/readout latency.
        latency_readout_stall = self.mcm.T_stall_read_after_write # 获取最终读出结果时的停顿延迟。

        # --- CORE FIX: Return a latency dominated by compute, not a simple sum. ---
        # (以下注释解释了模型修正的核心思想)
        # A conservative model is that the total time is the compute loop plus the initial
        # I/O, as the I/O must complete before the final results are available from the compute loop.
        # However, a more aggressive and often more realistic model is that I/O is pipelined.
        # Let's adopt a model where the latency is the time to fill the buffer plus the compute time,
        # acknowledging that this is still conservative but better than a simple sum that double-counts.
        # A simple, effective fix is to model latency as being dominated by the compute part.
        # For a long pipeline, the initial fill is a one-time cost amortized over the execution.
        # The key is that the main DRAM bus is free after the initial fill.
        
        # Let's use a model where total latency is MAX(io, compute) + overheads,
        # but a simpler, more robust fix is to acknowledge compute is the bottleneck.
        # We will return the sum of the compute loop and the final stall, assuming the
        # initial I/O is pipelined/hidden.
        
        return latency_compute_loop + latency_readout_stall # 返回计算循环延迟和读出停顿之和，假设初始I/O填充延迟被计算过程隐藏。
    
    # NEW: 新增一个专门计算GEMV能耗的内部方法
    def _calculate_gemv_energy_pj(self, M: int, K: int) -> float:
        """
        NEW: 根据GEMV的指令流模板计算总能耗 (单位: pJ)。
        """
        total_energy_pj = 0.0
        
        # 1. 能耗: 填充全局缓冲区 (WR_GB)
        # 假设激活向量通过 GPR -> Global Buffer
        activation_bytes = K * self.element_size_bytes
        num_gpr_transfers_in = math.ceil(activation_bytes / self.mcm.GPR_SIZE_BYTES)
        # 假设一次 WR_GB 对应一次 GPR 传输
        energy_fill_gb = num_gpr_transfers_in * (
            self.ecm.e_cmd_wr_gb_pj + \
            self.ecm.e_bus_transfer_pj_per_byte * self.mcm.GPR_SIZE_BYTES
        )
        total_energy_pj += energy_fill_gb
        
        # 2. 能耗: 计算循环 (MAC_ABK)
        GEMV_OPSIZE = 64.0
        elements_per_instruction = self.mcm.ELEMENTS_PER_MAC * GEMV_OPSIZE
        num_instructions_per_output = math.ceil(K / elements_per_instruction)
        num_parallel_batches = math.ceil(M / self.config.num_banks_per_channel)
        num_mac_instructions = num_parallel_batches * num_instructions_per_output
        energy_compute = num_mac_instructions * (
            self.ecm.e_mac_abk_compute_pj + self.ecm.e_cmd_mac_abk_pj
        )
        total_energy_pj += energy_compute
        
        # 3. 能耗: 结果读出 (RD_MAC)
        output_bytes = M * self.element_size_bytes
        num_gpr_transfers_out = math.ceil(output_bytes / self.mcm.GPR_SIZE_BYTES)
        # 假设一次 RD_MAC 对应一次 GPR 传输
        energy_readout = num_gpr_transfers_out * (
            self.ecm.e_cmd_rd_mac_pj + \
            self.ecm.e_bus_transfer_pj_per_byte * self.mcm.GPR_SIZE_BYTES
        )
        total_energy_pj += energy_readout
        
        return total_energy_pj

    def _calculate_gemv_dram_bound(self, M: int, K: int) -> float: # 定义一个内部方法，计算DRAM带宽受限情况下的 GEMV 延迟。
        GEMV_OPSIZE = 64.0 # 定义 GEMV 操作的一个参数。
        elements_per_instruction = self.mcm.ELEMENTS_PER_MAC * GEMV_OPSIZE # 计算单条指令能处理的元素数量。
        num_parallel_batches = math.ceil(M / self.config.num_banks_per_channel) # 计算并行批处理的数量。
        num_instructions_per_output = math.ceil(K / elements_per_instruction) # 计算得到一个输出结果需要多少条指令。
        num_total_loops = num_parallel_batches * num_instructions_per_output # 计算总的循环次数。
        latency_compute_loop = num_total_loops * self.mcm.T_pipeline_slow # 使用慢速路径的流水线周期计算核心计算延迟。
        latency_readout_stall = self.mcm.T_stall_read_after_write # 获取最终读出结果时的停顿延迟。
        return latency_compute_loop + latency_readout_stall # 返回计算循环延迟和读出停顿之和。

    def evaluate_gemv(self, M: int, K: int) -> Dict: # 定义评估 GEMV 操作性能的公共方法。
        activation_size_bytes = K * self.element_size_bytes # 计算激活向量（输入向量）的总大小（字节）。
        gb_size_bytes = self.config.global_buffer_size_per_channel_kb * 1024 # 计算全局缓冲区的总大小（字节）。
        path_taken = "" # 初始化路径选择字符串为空。
        total_cycles = 0.0 # 初始化总周期数为0。
        # NEW: 调用新的能耗计算方法
        # 假设快慢路径的能耗模型是相同的，因为它们都执行相同的指令序列，只是时序不同
        total_energy_pj = self._calculate_gemv_energy_pj(M, K)
        total_energy_joules = total_energy_pj * 1e-12 # 转换为焦耳
        if activation_size_bytes <= gb_size_bytes: # 检查激活向量是否能完全放入全局缓冲区。
            path_taken = "PIM_Fast_Path_(GB_Cached)" # 如果可以，则选择快速路径（缓存命中）。
            total_cycles = self._calculate_gemv_gb_cached(M, K) # 调用缓存命中模型计算延迟。
        else: # 如果不能放入全局缓冲区。
            path_taken = "PIM_Slow_Path_(DRAM_Bound)" # 则选择慢速路径（DRAM受限）。
            total_cycles = self._calculate_gemv_dram_bound(M, K) # 调用DRAM受限模型计算延迟。
        return {"total_cycles": total_cycles, "path": path_taken, "energy_joules": total_energy_joules} # 以字典形式返回总周期数和所选路径。
    
    def evaluate_gemm(self, M: int, K: int, N: int) -> Dict: # 定义评估 GEMM (矩阵乘法) 操作性能的公共方法。
        # PIM通过将GEMM分解为M次独立的GEMV操作来执行。
        single_gemv_result = self.evaluate_gemv(M=N, K=K) # 计算一次 GEMV(N, K) 的性能，在PIM模型中，GEMV的M维度对应GEMM的N维度。
        total_cycles = M * single_gemv_result["total_cycles"] # 总周期数等于单次GEMV的周期数乘以M。
        total_energy_joules = M * single_gemv_result["energy_joules"]
        return {"total_cycles": total_cycles, "path": single_gemv_result["path"], "energy_joules": total_energy_joules} # 返回总周期数和GEMV所选的路径。


# ==============================================================================
# SECTION 3: MODIFIED APPLICATION LAYER EVALUATOR
# ==============================================================================
# ==============================================================================
# pim_v0_4.py
# 请用以下完整的 ApplicationLayerEvaluator 类替换您文件中的同名类
# ==============================================================================
class ApplicationLayerEvaluator: # 定义应用层评估器类。
    """
    MODIFIED: v0.4, to align with the new PerformanceResult data structure.
    FIXED: The evaluate method is corrected to handle all operations as GEMM,
           ensuring the 'N' dimension (kv_cache_len) is properly used.
    """
    def __init__(self, pim_config: PIMConfig, element_size_bytes: int = 2): # 类的构造函数。
        self.config = pim_config # 保存传入的 PIM 硬件配置对象。
        self.element_size_bytes = element_size_bytes # 保存每个数据元素的大小（字节）。
        self.mcm = MicroarchitectureCostModel() # 创建一个微架构成本模型实例。
        self.pom = PIMOperationModel(self.mcm, self.config, self.element_size_bytes) # 创建一个 PIM 操作模型实例。
        self.ecm = PIMEnergyCostModel() # NEW: 直接在此处创建并持有能耗模型实例
        # 模型参数保持不变
        self.hidden_size = 4096 # 定义LLM模型的隐藏层大小。
        self.intermediate_size = 11008 # 定义LLM模型的前馈网络中间层大小。
        self.num_heads = 32 # 定义LLM模型的注意力头数量。
        self.head_dim = self.hidden_size // self.num_heads # 计算每个注意力头的维度。
    
    def evaluate(self, # 定义统一的评估方法，用于计算任意操作的性能。
                 M: int, # 输入矩阵的行数或批处理大小。
                 K: int, # 输入矩阵的列数/内积维度。
                 N: int, # 输出矩阵的列数。
                 num_ops_parallel: int, # 并行执行的操作数量。
                 is_data_parallel: bool,  # 注意：此参数被PIM模型有意忽略 # 标志是否为数据并行。
                 op_type: str, # 操作类型（如 "GEMM", "GEMV"）。
                 num_channels_used: int, # 使用的 PIM 通道数量。
                 op_name: str  # 注意：此参数被PIM模型有意忽略 # 操作的具体名称。
                ) -> PerformanceResult: # 该方法返回一个 PerformanceResult 对象。
        
        # MODIFIED: 统一的顶层评估接口，采用了标准化的函数签名。所有操作统一按GEMM处理，解决了之前GEMV错误忽略N维度的问题。现在可以正确反映解码阶段随 kv_cache_len 增长的计算负载。
        
        # --- 1. 计算总延迟和FLOPs (统一按GEMM处理) ---
        
        # 将总的并行操作数/任务数，平均分配到使用的PIM通道上
        # 对于解码阶段的注意力计算，num_ops_parallel就是头的数量
        ops_per_channel = math.ceil(num_ops_parallel / num_channels_used) # 计算每个通道需要处理的操作数。

        # 核心修复：无论是GEMV还是GEMM，我们都调用pom.evaluate_gemm来计算性能。
        # pom.evaluate_gemm内部会调用evaluate_gemv，但它会正确处理M维度，
        # 即使M=1，也能得到正确的结果。
        # 我们将总任务(num_ops_parallel)看作是对一个 (M, K)x(K, N) GEMM的批处理(Batch)，
        # 批大小为num_ops_parallel。
        # 在PIM上，这些批处理任务被分配到不同通道上并行执行。
        
        # 计算单个GEMM (M,K,N) 在一个通道上的延迟
        # 注意：PIM的GEMM实现是按M维度拆分的，所以我们将(M * ops_per_channel)作为等效的M传入
        # 这是为了模拟在一个通道上连续执行ops_per_channel次GEMV(M=M, K=K)操作
        # (因为evaluate_gemm的实现是 M * evaluate_gemv(M=N, K=K))
        # 更好的方式是直接 M * evaluate_gemv(M=M, K=K)
        
        # 为了与原始的GEMM实现兼容，我们采用以下逻辑：
        # Prefill (M > 1) -> GEMM(M, K, N)
        # Decode (M = 1) -> num_ops * GEMV(M_gemv, K_gemv)
        # 我们用一个更通用、更正确的逻辑来统一
        
        total_latency_cycles = 0.0 # 初始化总延迟周期为0。
        total_energy_joules = 0.0 # 初始化总能耗为0。
        path_taken = "" # 初始化路径选择字符串为空。
        
        # 对于PIM，N维度被划分到所有通道上
        N_per_channel = math.ceil(N / num_channels_used) # 计算每个通道负责的 N 维度的大小。
        
        # (此注释块为旧的逻辑，已被下面的逻辑替换)
        """
        # 现在，我们不再信任op_type，而是基于M的值来判断
        if M > 1: # Prefill阶段或非标准GEMM
            ...
        else: # Decode阶段 (M=1), 此时 num_ops_parallel 代表 batch*num_heads
            ...

        # --- 2. 计算数据移动量 ---
        # 这个计算逻辑可以简化和统一
        input_bytes = (M * K) * self.element_size_bytes * num_ops_parallel
        """
        if M > 1: # Prefill 或 GEMM 阶段。
            # PIM's GEMM is M sequential GEMVs. Total ops on a channel = M * ops_per_channel
            result = self.pom.evaluate_gemm(M=(M * ops_per_channel), K=K, N=N_per_channel) # 调用GEMM评估，等效M为 M * 每个通道的操作数。
            total_latency_cycles = result["total_cycles"] # 获取总延迟。
            total_energy_joules = result["energy_joules"]
            path_taken = result["path"] # 获取执行路径。
            op_flops = 2 * M * K * N * num_ops_parallel # 计算总浮点运算次数。
        else: # Decode 或 GEMV 阶段 (M=1)。
            result = self.pom.evaluate_gemv(M=N_per_channel, K=K) # 调用GEMV评估。
            total_latency_cycles = result["total_cycles"] * ops_per_channel # 总延迟是单次延迟乘以每个通道的操作数。
            total_energy_joules = result["energy_joules"] * ops_per_channel
            path_taken = result["path"] # 获取执行路径。
            op_flops = 2 * K * N * num_ops_parallel # 计算总浮点运算次数。

        # --- 2. Byte Calculation (Aligned with Latency Path) ---
        if "GB_Cached" in path_taken: # 如果走的是快速路径（全局缓存命中）。
            # Fast Path: Only the activation vector (K) is read from DRAM once per operation.
            input_bytes = (K * self.element_size_bytes) * num_ops_parallel # 输入字节数仅为激活向量的大小。
        else: # "DRAM_Bound" # 如果走的是慢速路径（DRAM受限）。
            if M > 1: # GEMM
                # Slow Path GEMM: Entire input matrix (M*K) is streamed.
                input_bytes = (M * K * self.element_size_bytes) * num_ops_parallel # 输入字节数是整个输入矩阵的大小。
            else: # GEMV
                # Slow Path GEMV: Input matrix is effectively (N_per_channel x K).
                # The latency model (evaluate_gemv) already accounts for this.
                # We assume the entire matrix must be streamed for each of the parallel ops.
                # 权重矩阵大小为 N * K，对于所有并行操作都需要流式传输。
                input_bytes = (N * K * self.element_size_bytes) * num_ops_parallel # 输入字节数是权重矩阵的大小。


        output_bytes = (M * N) * self.element_size_bytes * num_ops_parallel # 计算输出数据的总字节数。
        # --- 3. 延迟分解 (PIM的简化模型) ---
        # 对于PIM，其整个操作过程都占用内存通道，因此io_cycles等于总延迟。
        io_cycles = total_latency_cycles # 在PIM模型中，I/O周期等于总延迟周期。
        # PIM没有独立于内存通道的“外部”计算单元，因此compute_cycles为0。
        compute_cycles = 0.0 # PIM没有独立的计算周期，所以为0。

        # --- 4. 创建并返回新版的PerformanceResult对象 ---
        return PerformanceResult( # 创建并返回一个填充了所有性能指标的 PerformanceResult 对象。
            latency_cycles=total_latency_cycles, # 填充总延迟。
            energy_joules=total_energy_joules, # 能量模型仍为占位符, 能量消耗暂时设为0。
            input_bytes=int(input_bytes), # 填充输入字节数。
            output_bytes=int(output_bytes), # 填充输出字节数。
            path_taken=path_taken, # 填充执行路径。
            io_cycles=io_cycles, # 填充I/O周期。
            compute_cycles=compute_cycles, # 填充计算周期。
            op_flops=op_flops, # 填充总浮点运算次数。
            bytes_from_dram=int(input_bytes),  # 对于PIM，输入激活是从DRAM流向计算单元的 # 从DRAM读取的字节数。
            bytes_to_dram=int(output_bytes)    # 结果需要写回DRAM # 写入DRAM的字节数。
        )
    

    def evaluate_prefill_latency(self, prompt_len: int) -> Dict: # 定义一个方法来评估 Prefill 阶段的延迟。
        """
        评估一个完整层在Prefill阶段的延迟。 (此函数逻辑无需修改)
        """
        total_cycles = 0 # 初始化总周期数为0。
        total_energy_joules = 0.0
        op_cycles = {} # 初始化一个字典来存储每个操作的周期数。
        op_energy = {}
        pim_channels = self.config.total_channels # 获取PIM的总通道数。
        
        # Prefill阶段所有操作都是GEMM (M=prompt_len)
        # 1. Projections & FFNs
        fixed_ops = { # 定义固定的、与序列长度无关的投影和FFN操作。
            "Attention::QKV_Projections": (self.hidden_size, self.hidden_size, 3), # 操作名: (N, K, 操作次数)
            "Attention::Output_Projection": (self.hidden_size, self.hidden_size, 1),
            "FFN::Gate_Up_Projections":   (self.intermediate_size, self.hidden_size, 2),
            "FFN::Down_Projection":       (self.hidden_size, self.intermediate_size, 1)
        }
        for op_name, (N, K, num_ops) in fixed_ops.items(): # 遍历这些操作。
            # GEMM (M=prompt_len, K=K, N=N)
            # N维度被划分到所有PIM通道上
            N_per_channel = math.ceil(N / pim_channels) # 计算每个通道负责的N维度大小。
            result = self.pom.evaluate_gemm(M=prompt_len, K=K, N=N_per_channel) # 调用操作模型评估GEMM性能。
            cycles = result.get("total_cycles", 0) * num_ops # 获取总周期数并乘以操作次数。
            energy = result.get("energy_joules", 0) * num_ops
            total_cycles += cycles # 累加到总周期数。
            total_energy_joules += energy
            op_cycles[op_name] = cycles # 在字典中记录该操作的周期数。
            op_energy[op_name] = energy

        # 2. Attention BMM (Batch GEMM)
        # Heads 被分配到通道上
        heads_per_channel = math.ceil(self.num_heads / pim_channels) # 计算每个通道负责的注意力头数量。
        
        # QK BMM: (B=num_heads, M=prompt_len, K=head_dim, N=prompt_len)
        qk_result = self.pom.evaluate_gemm(M=prompt_len, K=self.head_dim, N=prompt_len) # 评估 QK 矩阵乘法的性能。
        qk_cycles = qk_result.get("total_cycles", 0) * heads_per_channel # 计算所有头的总周期数。
        qk_energy = qk_result.get("energy_joules", 0) * heads_per_channel
        total_cycles += qk_cycles # 累加到总周期数。
        total_energy_joules += qk_energy
        op_cycles["Attention::Score_Computation_QK"] = qk_cycles # 记录该操作的周期数。
        op_energy["Attention::Score_Computation_QK"] = qk_energy
        
        # SV BMM: (B=num_heads, M=prompt_len, K=prompt_len, N=head_dim)
        sv_result = self.pom.evaluate_gemm(M=prompt_len, K=prompt_len, N=self.head_dim) # 评估 SV 矩阵乘法的性能。
        sv_energy = sv_result.get("energy_joules", 0) * heads_per_channel
        sv_cycles = sv_result.get("total_cycles", 0) * heads_per_channel # 计算所有头的总周期数。
        total_cycles += sv_cycles # 累加到总周期数。
        total_energy_joules += sv_energy
        op_cycles["Attention::Context_Computation_SV"] = sv_cycles # 记录该操作的周期数。
        op_energy["Attention::Context_Computation_SV"] = sv_energy
        
        return {"total_cycles": total_cycles, "op_breakdown": op_cycles, "total_energy_joules": total_energy_joules, "energy_breakdown": op_energy} # 返回包含总周期数和各操作分解的字典。


    def evaluate_single_token_latency(self, kv_cache_len: int) -> Dict: # 定义一个方法来评估 Decode 阶段单个 token 的延迟。
        """
        计算在给定KV Cache长度下，生成一个新token所需的总延迟（周期）。(此函数逻辑无需修改)
        """
        total_cycles = 0 # 初始化总周期数为0。
        total_energy_joules = 0.0
        op_cycles = {} # 初始化一个字典来存储每个操作的周期数。
        op_energy = {}
        pim_channels = self.config.total_channels # 获取PIM的总通道数。
        
        # --- 模拟一个解码层的所有操作 (均为GEMV) ---

        # 1. Projections & FFNs (延迟不随kv_cache_len变化)
        # 注意：N维度(输出维度)被划分到所有PIM通道上
        fixed_ops = { # 定义固定的投影和FFN操作。
            "Attention::QKV_Projections": (self.hidden_size, self.hidden_size, 3), # N, K, num_ops
            "Attention::Output_Projection": (self.hidden_size, self.hidden_size, 1),
            "FFN::Gate_Up_Projections":   (self.intermediate_size, self.hidden_size, 2),
            "FFN::Down_Projection":       (self.hidden_size, self.intermediate_size, 1)
        }

        for op_name, (N_total, K, num_ops) in fixed_ops.items(): # 遍历这些操作。
            N_per_channel = math.ceil(N_total / pim_channels) # 计算每个通道负责的N维度大小。
            # GEMV(M=N_per_channel, K=K)
            result = self.pom.evaluate_gemv(M=N_per_channel, K=K) # 调用操作模型评估GEMV性能。
            cycles = result.get("total_cycles", 0) * num_ops # 获取总周期数并乘以操作次数。
            energy = result.get("energy_joules", 0) * num_ops
            total_cycles += cycles # 累加到总周期数。
            total_energy_joules += energy
            op_cycles[op_name] = op_cycles.get(op_name, 0) + cycles # 在字典中记录该操作的周期数。
            op_energy[op_name] = op_energy.get(op_name, 0) + energy

        # 2. KV Cache 相关算子 (延迟随kv_cache_len增长)
        heads_per_channel = math.ceil(self.num_heads / pim_channels) # 计算每个通道负责的注意力头数量。
        
        # a) QK GEMV: M = kv_cache_len, K = head_dim
        # evaluate_gemv的M是输出维度, K是输入维度
        # 在QK操作 (1, K_dim) x (K_dim, kv_len) 中, 输出维度是kv_len, 输入是K_dim
        qk_result = self.pom.evaluate_gemv(M=kv_cache_len, K=self.head_dim) # 评估QK操作（GEMV）的性能。
        qk_cycles = qk_result.get("total_cycles", 0) * heads_per_channel # 计算所有头的总周期数。
        qk_energy = qk_result.get("energy_joules", 0) * heads_per_channel
        total_cycles += qk_cycles # 累加到总周期数。
        op_cycles["Attention::Score_Computation_QK"] = qk_cycles # 记录该操作的周期数。
        op_energy["Attention::Score_Computation_QK"] = qk_energy

        # b) SV GEMV: M = head_dim, K = kv_cache_len
        # 在SV操作 (d_dim, kv_len) x (kv_len, 1) 中, 输出维度是d_dim, 输入是kv_len
        sv_result = self.pom.evaluate_gemv(M=self.head_dim, K=kv_cache_len) # 评估SV操作（GEMV）的性能。
        sv_energy = sv_result.get("energy_joules", 0) * heads_per_channel
        sv_cycles = sv_result.get("total_cycles", 0) * heads_per_channel # 计算所有头的总周期数。
        total_cycles += sv_cycles # 累加到总周期数。
        total_energy_joules += sv_energy
        op_cycles["Attention::Context_Computation_SV"] = sv_cycles # 记录该操作的周期数。
        op_energy["Attention::Context_Computation_SV"] = sv_energy

        # 3. 访存密集型算子 (保持不变)
        # Memory-bound ops - for now, we assume their energy is negligible compared to GEMV/GEMM
        # This can be refined later if needed.
        bw_total_gb_s = self.config.main_memory_bw_per_channel_gb_s * pim_channels # 计算总的内存带宽 (GB/s)。
        clock_hz = self.config.clock_ghz * 1e9 # 将时钟频率从GHz转换为Hz。
        
        softmax_bytes = self.num_heads * kv_cache_len * 2 # 计算Softmax操作需要传输的字节数。
        softmax_cycles = (softmax_bytes / (bw_total_gb_s * 1e9)) * clock_hz # 根据带宽计算Softmax操作的延迟周期数。
        softmax_energy_pj = (softmax_bytes * 2) * self.ecm.e_bus_transfer_pj_per_byte
        softmax_energy_j = softmax_energy_pj * 1e-12
        total_energy_joules += softmax_energy_j
        total_cycles += softmax_cycles # 累加到总周期数。
        op_cycles["Attention::Softmax"] = softmax_cycles # 记录该操作的周期数。
        op_energy["Attention::Softmax"] = softmax_energy_j

        rms_bytes = self.hidden_size * 2 * 2 # 计算RMSNorm操作需要传输的字节数。
        rms_cycles = (rms_bytes / (bw_total_gb_s * 1e9)) * clock_hz # 根据带宽计算RMSNorm操作的延迟周期数。
        rms_energy_pj = (rms_bytes * 2) * self.ecm.e_bus_transfer_pj_per_byte
        rms_energy_j = rms_energy_pj * 1e-12
        total_cycles += rms_cycles # 累加到总周期数。
        total_energy_joules += rms_energy_j
        op_cycles["RMSNorm"] = rms_cycles # 记录该操作的周期数。
        op_energy["RMSNorm"] = rms_energy_j
        
        silu_bytes = self.intermediate_size * 2 # 计算SiLU激活函数需要传输的字节数。
        silu_cycles = (silu_bytes / (bw_total_gb_s * 1e9)) * clock_hz # 根据带宽计算SiLU操作的延迟周期数。
        silu_energy_pj = (silu_bytes * 2) * self.ecm.e_bus_transfer_pj_per_byte
        silu_energy_j = silu_energy_pj * 1e-12
        total_cycles += silu_cycles # 累加到总周期数。
        total_energy_joules += silu_energy_j
        op_cycles["FFN::Activation(SiLU)"] = silu_cycles # 记录该操作的周期数。
        op_energy["FFN::Activation(SiLU)"] = silu_energy_j
        
        return {"total_cycles": total_cycles, "op_breakdown": op_cycles, "total_energy_joules": total_energy_joules, "energy_breakdown": op_energy} # 返回包含总周期数和各操作分解的字典。

# ==============================================================================
# 主分析流程 (已更新)
# ==============================================================================
def simulate_long_text_generation( # 定义一个函数来模拟长文本生成的整个过程。
    pim_config: PIMConfig, # PIM硬件配置。
    prompt_len: int, # 初始提示的长度。
    add_tokens: int, # 需要新生成的 token 数量。
    layers: int # 模型的总层数。
):
    print("\n" + "="*80) # 打印分隔线。
    print("LLM Inference Performance & Energy Simulation (PIM v0.4 - Energy Integrated)") # MODIFIED: Title # 打印仿真标题。
    print("="*80) # 打印分隔线。
    print("Hardware Configuration:") # 打印硬件配置信息。
    print(f"  - PIM Channels: {pim_config.total_channels} @ {pim_config.clock_ghz} GHz") # 打印PIM通道数和时钟频率。
    print(f"  - GB Size per PIM Channel: {pim_config.global_buffer_size_per_channel_kb} KB") # 打印每个通道的全局缓冲区大小。
    print("\nSimulation Parameters:") # 打印仿真参数信息。
    print(f"  - Model Layers: {layers}") # 打印模型层数。
    print(f"  - Initial Prompt Length: {prompt_len}") # 打印初始提示长度。
    print(f"  - New Tokens to Generate: {add_tokens}") # 打印生成 token 的数量。
    print("-"*80) # 打印分隔线。

    evaluator = ApplicationLayerEvaluator(pim_config) # 创建一个应用层评估器实例。
    
    # --- 1. Prefill Stage ---
    prefill_result = evaluator.evaluate_prefill_latency(prompt_len) # 调用评估器计算单层 Prefill 的延迟。
    total_prefill_cycles = prefill_result["total_cycles"] * layers # 将单层延迟乘以总层数，得到整个模型的 Prefill 延迟。
    total_prefill_energy = prefill_result["total_energy_joules"] * layers

    # --- 2. Decoding Stage ---
    total_decoding_cycles = 0 # 初始化总解码周期数为0。
    total_decoding_energy = 0.0
    cumulative_op_cycles = prefill_result["op_breakdown"].copy() # 复制 Prefill 阶段各操作的周期数，用于累积。
    cumulative_op_energy = {k: v * layers for k, v in prefill_result["energy_breakdown"].items()}
    """
    for key in cumulative_op_cycles: # 遍历字典中的每个操作。
        cumulative_op_cycles[key] *= layers # 将 Prefill 阶段的周期数乘以总层数。
    
    for i in range(add_tokens): # 循环生成每一个新的 token。
        current_kv_cache_len = prompt_len + i # 计算当前步的 KV Cache 长度。
        step_result = evaluator.evaluate_single_token_latency( # 调用评估器计算生成当前 token 的单层延迟。
            kv_cache_len=current_kv_cache_len
        )
        total_decoding_cycles += step_result["total_cycles"] * layers # 将单层延迟乘以总层数，并累加到总解码周期中。
        for op_name, cycles in step_result["op_breakdown"].items(): # 遍历当前步中每个操作的周期数。
            cumulative_op_cycles[op_name] = cumulative_op_cycles.get(op_name, 0) + (cycles * layers) # 将其累加到总的操作周期分解中。
    """
    for i in range(add_tokens):
        current_kv_cache_len = prompt_len + i
        step_result = evaluator.evaluate_single_token_latency(
            kv_cache_len=current_kv_cache_len
        )
        # Aggregate per-step results
        layer_cycles = step_result["total_cycles"] * layers
        layer_energy = step_result["total_energy_joules"] * layers # NEW
        total_decoding_cycles += layer_cycles
        total_decoding_energy += layer_energy # NEW
        
        # Aggregate breakdowns
        for op_name, cycles in step_result["op_breakdown"].items():
            cumulative_op_cycles[op_name] = cumulative_op_cycles.get(op_name, 0) + (cycles * layers)
        for op_name, energy in step_result["energy_breakdown"].items(): # NEW
            cumulative_op_energy[op_name] = cumulative_op_energy.get(op_name, 0) + (energy * layers)
    
    # --- 3. Report Generation ---
    total_cycles_all_tokens = total_prefill_cycles + total_decoding_cycles # 计算总的周期数。
    total_energy_all_tokens = total_prefill_energy + total_decoding_energy
    clock_hz = pim_config.clock_ghz * 1e9 # 将时钟频率从GHz转换为Hz。
    prefill_latency_ms = (total_prefill_cycles / clock_hz * 1000) if clock_hz > 0 else 0 # 计算 Prefill 阶段的延迟（毫秒）。
    decoding_latency_ms = (total_decoding_cycles / clock_hz * 1000) if clock_hz > 0 else 0 # 计算 Decode 阶段的总延迟（毫秒）。
    total_latency_ms = prefill_latency_ms + decoding_latency_ms # 计算端到端的总延迟（毫秒）。
    avg_latency_per_token_ms = decoding_latency_ms / add_tokens if add_tokens > 0 else 0 # 计算每个 token 的平均生成延迟（毫秒）。
    avg_energy_per_token_mj = (total_decoding_energy / add_tokens * 1000) if add_tokens > 0 else 0
    power_watts = (total_energy_all_tokens / (total_latency_ms / 1000)) if total_latency_ms > 0 else 0
    throughput_tps = 1000 / avg_latency_per_token_ms if avg_latency_per_token_ms > 0 else 0 # 计算生成吞吐量（tokens/sec）。

    print(f"\n[Summary]")
    print(f"  - Prefill Latency:         {prefill_latency_ms:>10.4f} ms  |  Energy: {total_prefill_energy * 1000:>8.2f} mJ")
    print(f"  - Total Decoding Latency:  {decoding_latency_ms:>10.4f} ms  |  Energy: {total_decoding_energy * 1000:>8.2f} mJ (for {add_tokens} tokens)")
    print(f"  - Total End-to-End Latency:{total_latency_ms:>10.4f} ms  |  Energy: {total_energy_all_tokens * 1000:>8.2f} mJ")
    print(f"  ----------------------------------------------------------------------") # MODIFIED: line length
    print(f"  - Avg Latency per Token:   {avg_latency_per_token_ms:>10.4f} ms/token")
    print(f"  - Avg Energy per Token:    {avg_energy_per_token_mj:>10.4f} mJ/token") # NEW
    print(f"  - Generation Throughput:   {throughput_tps:>10.2f} tokens/sec")
    print(f"  - Average Power:           {power_watts:>10.2f} W") # NEW

    print("\n--- Per-Operator Latency & Energy Breakdown (Aggregated) ---") # MODIFIED: Title
    # Sort by cycles, but we will display both
    sorted_ops_by_cycles = sorted(cumulative_op_cycles.items(), key=lambda item: item[1], reverse=True)
    
    print(f"{'Operator':<35} {'Total Cycles':>20} {'% Latency':>10} | {'Total Energy (mJ)':>20} {'% Energy':>10}")
    print("-" * 105) # MODIFIED: line length
    for op_name, cycles in sorted_ops_by_cycles:
        energy_j = cumulative_op_energy.get(op_name, 0.0)
        latency_perc = (cycles / total_cycles_all_tokens * 100) if total_cycles_all_tokens > 0 else 0
        energy_perc = (energy_j / total_energy_all_tokens * 100) if total_energy_all_tokens > 0 else 0
        print(f"{op_name:<35} {int(cycles):>20,d} {latency_perc:>9.2f}% | {energy_j * 1000:>20.2f} {energy_perc:>9.2f}%")
    print("="*105) # MODIFIED: line length


# ==============================================================================
# SECTION 4: UNCHANGED MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__": # 程序的入口点。
    parser = argparse.ArgumentParser(description="Run a microarchitecture-based LLM inference performance simulation (v3.1).") # 创建一个命令行参数解析器。
    parser.add_argument('--prompt-len', type=int, default=39, help="Initial prompt length (context size).") # 添加 prompt 长度参数。
    parser.add_argument('--layers', type=int, default=32, help="Number of layers in the model.") # 添加模型层数参数。
    parser.add_argument('--add-tokens', type=int, default=100, help="Number of new tokens to generate.") # 添加生成 token 数量的参数。
    parser.add_argument('--clock-ghz', type=float, default=2.0, help="PIM clock frequency in GHz.") # 添加 PIM 时钟频率参数。
    parser.add_argument('--gb-size-kb', type=int, default=256, help="Global Buffer size per PIM channel in KB.") # 添加全局缓冲区大小参数。
    parser.add_argument('--pim-channels', type=int, default=32, help="Number of PIM channels to use.") # 添加 PIM 通道数参数。
    args = parser.parse_args() # 解析命令行传入的参数。

    # 1. 定义硬件配置
    pim_hw_config = PIMConfig( # 根据解析的参数创建一个 PIMConfig 对象。
        total_channels=args.pim_channels, # 设置总通道数。
        clock_ghz=args.clock_ghz, # 设置时钟频率。for GDDR6, 2GHZ; for LPDDR5, 0.8GHZ (from AiM simulator)
        global_buffer_size_per_channel_kb=args.gb_size_kb # 设置全局缓冲区大小。
    )

    # 2. 运行主仿真
    simulate_long_text_generation( # 调用主仿真函数。
        pim_config=pim_hw_config, # 传入硬件配置。
        prompt_len=args.prompt_len, # 传入 prompt 长度。
        add_tokens=args.add_tokens, # 传入生成 token 数量。
        layers=args.layers # 传入模型层数。
    )

    # 3. (可选) 运行一个对比场景来展示GB容量的影响
    print("\n\n" + "*"*30 + " RUNNING COMPARISON SCENARIO " + "*"*30) # 打印一个分隔符，表示开始对比场景。
    pim_hw_small_gb = PIMConfig( # 创建一个新的 PIMConfig 对象，使用较小的全局缓冲区。
        total_channels=args.pim_channels, # 设置总通道数。
        clock_ghz=args.clock_ghz, # 设置时钟频率。
        global_buffer_size_per_channel_kb=16 # 容量受限，设置为 16 KB。
    )
    simulate_long_text_generation( # 再次调用主仿真函数，运行对比场景。
        pim_config=pim_hw_small_gb, # 传入新的硬件配置。
        prompt_len=args.prompt_len, # 传入 prompt 长度。
        add_tokens=args.add_tokens, # 传入生成 token 数量。
        layers=args.layers # 传入模型层数。
    )
