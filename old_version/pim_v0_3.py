import math
from typing import Dict, Optional
from dataclasses import dataclass
import argparse

from utils_v0_3 import PIMConfig, PerformanceResult

# ==============================================================================
# 层 1: 微架构成本模型 (Microarchitecture Cost Model - MCM)
# 职责: 封装从AiM实验中校准出的底层指令成本。
# ==============================================================================

@dataclass
class MicroarchitectureCostModel:
    """
    存储从cycle-accurate仿真中标定出的原子操作成本。
    这是我们整个预测模型的物理基础。
    """
    # 硬件微架构常量
    GPR_SIZE_BYTES: int = 32
    ELEMENTS_PER_MAC: int = 16  # FP16 elements per MAC_ABK instruction

    # --- 校准出的流水线延迟 (cycles) ---
    
    # 来自实验一: 1024组 WR_ABK + MAC_ABK = 669,655 cycles
    # 669655 / 1024 ≈ 654 cycles/op
    # --- 核心修正: 假设权重加载可以被计算完美流水化 ---
    # 新的成本基于流水线瓶颈 (max)，而不是串行相加。
    # T_standalone(MAC_ABK) ≈ 226 cycles, T_standalone(WR_ABK) ≈ 101 cycles.
    # T_pipeline_fast = max(226, 101)
    # T_pipeline_fast: float = 654.0
    # 假设WR_ABK和MAC_ABK强制串行+冲突->两者可以流水线并行，是否可以要看硬件架构
    T_pipeline_fast: float = 226.0

    # 来自实验二: 1024组 COPY_BKGB + WR_ABK + MAC_ABK = 393,311 cycles
    # 393311 / 1024 ≈ 384 cycles/op
    T_pipeline_slow: float = 384.0

    # 来自实验四: 1024组 MAC+RD_MAC+MAC = 521,175 cycles
    # 理论无冲突 = T(MAC)+T(RD_MAC)+T(MAC) = 226+3+226 = 455
    # Stall = 509 (实测/组) - 455 (理论) ≈ 54 cycles
    T_stall_read_after_write: float = 54.0

    def get_gb_fill_latency(self, num_elements: int, element_size_bytes: int = 2) -> float:
        """
        根据实验三拟合的成本函数，计算填充GB的延迟。
        数据: (8KB -> 803 cycles), (32KB -> 2909 cycles)
        拟合结果: Latency = 102 + 2.74 * Num_GPRs
        """
        if num_elements == 0:
            return 0
        
        total_bytes = num_elements * element_size_bytes
        num_gprs = math.ceil(total_bytes / self.GPR_SIZE_BYTES)
        
        C1_fixed_overhead = 102.0
        C2_cost_per_gpr = 2.74
        
        return C1_fixed_overhead + C2_cost_per_gpr * num_gprs

# ==============================================================================
# 层 2: PIM 操作模型 (PIM Operation Model - POM)
# 职责: 模拟高阶操作 (如GEMV) 的指令流，并使用MCM计算其延迟。
# ==============================================================================

class PIMOperationModel:
    """
    将高阶数学运算分解为具体的指令流模板，并根据MCM的成本计算总延迟。
    """
    def __init__(self, mcm: MicroarchitectureCostModel, config: PIMConfig, element_size_bytes: int = 2):
        self.mcm = mcm
        self.config = config
        self.element_size_bytes = element_size_bytes

    def _calculate_gemv_gb_cached(self, M: int, K: int) -> float:
        latency_activation_fill = self.mcm.get_gb_fill_latency(K, self.element_size_bytes)
        GEMV_OPSIZE = 64.0 
        elements_per_instruction = self.mcm.ELEMENTS_PER_MAC * GEMV_OPSIZE
        num_instructions_per_output = math.ceil(K / elements_per_instruction)
        num_parallel_batches = math.ceil(M / self.config.num_banks_per_channel)
        latency_compute_loop = num_parallel_batches * num_instructions_per_output * self.mcm.T_pipeline_fast
        latency_readout_stall = self.mcm.T_stall_read_after_write
        return latency_activation_fill + latency_compute_loop + latency_readout_stall

    def _calculate_gemv_dram_bound(self, M: int, K: int) -> float:
        GEMV_OPSIZE = 64.0
        elements_per_instruction = self.mcm.ELEMENTS_PER_MAC * GEMV_OPSIZE
        num_parallel_batches = math.ceil(M / self.config.num_banks_per_channel)
        num_instructions_per_output = math.ceil(K / elements_per_instruction)
        num_total_loops = num_parallel_batches * num_instructions_per_output
        latency_compute_loop = num_total_loops * self.mcm.T_pipeline_slow
        latency_readout_stall = self.mcm.T_stall_read_after_write
        return latency_compute_loop + latency_readout_stall

    def evaluate_gemv(self, M: int, K: int) -> Dict:
        """
        评估一个GEMV(M, K)操作的延迟。
        核心是基于GB容量选择正确的指令流路径。
        """
        activation_size_bytes = K * self.element_size_bytes
        gb_size_bytes = self.config.global_buffer_size_per_channel_kb * 1024
        
        path_taken = ""
        total_cycles = 0.0
        
        if activation_size_bytes <= gb_size_bytes:
            path_taken = "PIM_Fast_Path_(GB_Cached)"
            total_cycles = self._calculate_gemv_gb_cached(M, K)
        else:
            path_taken = "PIM_Slow_Path_(DRAM_Bound)"
            total_cycles = self._calculate_gemv_dram_bound(M, K)
            
        return {"total_cycles": total_cycles, "path": path_taken}
    
    # ==========================================================================
    # 新增: GEMM评估函数 (通过分解为GEMV实现)
    # ==========================================================================
    def evaluate_gemm(self, M: int, K: int, N: int) -> Dict:
        """
        评估一个GEMM(M, K, N)操作的延迟。
        通过将其分解为 M 个独立的 GEMV(1, K, N) 来建模。
        这为Prefill阶段的评估提供了支持。
        """
        # 1. 对单个GEMV进行评估
        # 注意: M=N, K=K, N=M_original (因为权重矩阵是 KxN)
        # 这里的 M,K,N 是指GEMV的 M,K。GEMM的M维度被分解了。
        single_gemv_result = self.evaluate_gemv(M=N, K=K)
        
        # 2. 假设M个GEMV串行执行 (这是一个保守的估计)
        total_cycles = M * single_gemv_result["total_cycles"]
        
        return {"total_cycles": total_cycles, "path": single_gemv_result["path"]}


# ==============================================================================
# 层 3: 应用层评估器 (Application Layer Evaluator - ALE)
# ==============================================================================

class ApplicationLayerEvaluator:
    """
    v3.1: 基于微架构模型，评估LLM完整推理流程(Prefill+Decode)的性能。
    """
    def __init__(self, pim_config: PIMConfig, element_size_bytes: int = 2):
        self.config = pim_config
        self.element_size_bytes = element_size_bytes
        self.mcm = MicroarchitectureCostModel()
        self.pom = PIMOperationModel(self.mcm, self.config, self.element_size_bytes)
        
        # Llama-7B 模型参数
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.num_heads = 32
        self.head_dim = self.hidden_size // self.num_heads
    
    # ==========================================================================
    # 新增: 顶层评估函数
    # ==========================================================================
    def evaluate(self, M: int, K: int, N: int, op_type: str = "GEMV", num_channels_used: int = 1, num_ops_parallel: int = 1) -> PerformanceResult:
        """
        统一的顶层评估接口，评估单个操作在PIM上的性能。

        Args:
            M, K, N: 操作的核心维度.
            op_type: "GEMM" 或 "GEMV".
            num_channels_used: 执行此操作总共使用了多少PIM通道.
            num_ops_parallel: 在这些通道上，有多少个独立的操作在并行执行 (例如, num_heads).
        
        Returns:
            一个PerformanceResult对象.
        """
        latency_cycles = 0.0
        path_taken = ""
        input_bytes = 0
        output_bytes = 0

        # PIM的并行模型: 假设并行任务被均匀分配到所有可用通道上
        ops_per_channel = math.ceil(num_ops_parallel / num_channels_used)

        if op_type == "GEMV":
            # GEMV: (M, K) - M是输出维度, K是输入维度
            result = self.pom.evaluate_gemv(M=M, K=K)
            latency_cycles = result["total_cycles"] * ops_per_channel
            path_taken = result["path"]
            
            # 计算数据量
            if "GB_Cached" in path_taken:
                input_bytes = K * self.element_size_bytes * num_ops_parallel
            else: # DRAM Bound, 激活被反复读取，但外部看来只加载一次
                input_bytes = K * self.element_size_bytes * num_ops_parallel
            output_bytes = M * self.element_size_bytes * num_ops_parallel

        elif op_type == "GEMM":
            # GEMM: (M, K, N) -> 分解为 M个 (K, N)的GEMV
            # 在PIM上，一个(M,K,N)的GEMM通常是把N维度分配到通道上
            N_per_channel = math.ceil(N / num_channels_used)
            result = self.pom.evaluate_gemm(M=M, K=K, N=N_per_channel)
            latency_cycles = result["total_cycles"] # M的循环已经在evaluate_gemm内部计算
            path_taken = result["path"]
            
            # 计算数据量
            input_bytes = (M * K) * self.element_size_bytes
            output_bytes = (M * N) * self.element_size_bytes

        # 能量模型 (占位符)
        energy_joules = 0.0

        return PerformanceResult(
            latency_cycles=latency_cycles,
            energy_joules=energy_joules,
            path_taken=path_taken,
            input_bytes=input_bytes,
            output_bytes=output_bytes
        )

    # ==========================================================================
    # 新增: Prefill阶段评估函数
    # ==========================================================================
    def evaluate_prefill_latency(self, prompt_len: int) -> Dict:
        """
        评估一个完整层在Prefill阶段的延迟。
        """
        total_cycles = 0
        op_cycles = {}
        pim_channels = self.config.total_channels
        
        # Prefill阶段所有操作都是GEMM (M=prompt_len)
        # 1. Projections & FFNs
        fixed_ops = {
            "Attention::QKV_Projections": (self.hidden_size, self.hidden_size, 3),
            "Attention::Output_Projection": (self.hidden_size, self.hidden_size, 1),
            "FFN::Gate_Up_Projections":   (self.intermediate_size, self.hidden_size, 2),
            "FFN::Down_Projection":       (self.hidden_size, self.intermediate_size, 1)
        }
        for op_name, (N, K, num_ops) in fixed_ops.items():
            # GEMM (M=prompt_len, K=K, N=N)
            # N维度被划分到所有PIM通道上
            N_per_channel = math.ceil(N / pim_channels)
            result = self.pom.evaluate_gemm(M=prompt_len, K=K, N=N_per_channel)
            cycles = result.get("total_cycles", 0) * num_ops
            total_cycles += cycles
            op_cycles[op_name] = cycles

        # 2. Attention BMM (Batch GEMM)
        # Heads 被分配到通道上
        heads_per_channel = math.ceil(self.num_heads / pim_channels)
        
        # QK BMM: (B=num_heads, M=prompt_len, K=head_dim, N=prompt_len)
        qk_result = self.pom.evaluate_gemm(M=prompt_len, K=self.head_dim, N=prompt_len)
        qk_cycles = qk_result.get("total_cycles", 0) * heads_per_channel
        total_cycles += qk_cycles
        op_cycles["Attention::Score_Computation_QK"] = qk_cycles
        
        # SV BMM: (B=num_heads, M=prompt_len, K=prompt_len, N=head_dim)
        sv_result = self.pom.evaluate_gemm(M=prompt_len, K=prompt_len, N=self.head_dim)
        sv_cycles = sv_result.get("total_cycles", 0) * heads_per_channel
        total_cycles += sv_cycles
        op_cycles["Attention::Context_Computation_SV"] = sv_cycles
        
        return {"total_cycles": total_cycles, "op_breakdown": op_cycles}


    def evaluate_single_token_latency(self, kv_cache_len: int) -> Dict:
        """
        计算在给定KV Cache长度下，生成一个新token所需的总延迟（周期）。
        """
        total_cycles = 0
        op_cycles = {}
        pim_channels = self.config.total_channels
        
        # --- 模拟一个解码层的所有操作 (均为GEMV) ---

        # 1. Projections & FFNs (延迟不随kv_cache_len变化)
        # 注意：N维度(输出维度)被划分到所有PIM通道上
        fixed_ops = {
            "Attention::QKV_Projections": (self.hidden_size, self.hidden_size, 3), # N, K, num_ops
            "Attention::Output_Projection": (self.hidden_size, self.hidden_size, 1),
            "FFN::Gate_Up_Projections":   (self.intermediate_size, self.hidden_size, 2),
            "FFN::Down_Projection":       (self.hidden_size, self.intermediate_size, 1)
        }

        for op_name, (N_total, K, num_ops) in fixed_ops.items():
            N_per_channel = math.ceil(N_total / pim_channels)
            # GEMV(M=N_per_channel, K=K)
            result = self.pom.evaluate_gemv(M=N_per_channel, K=K)
            cycles = result.get("total_cycles", 0) * num_ops
            total_cycles += cycles
            op_cycles[op_name] = op_cycles.get(op_name, 0) + cycles

        # 2. KV Cache 相关算子 (延迟随kv_cache_len增长)
        heads_per_channel = math.ceil(self.num_heads / pim_channels)
        
        # a) QK GEMV: M = kv_cache_len, K = head_dim
        qk_result = self.pom.evaluate_gemv(M=kv_cache_len, K=self.head_dim)
        qk_cycles = qk_result.get("total_cycles", 0) * heads_per_channel
        total_cycles += qk_cycles
        op_cycles["Attention::Score_Computation_QK"] = qk_cycles

        # b) SV GEMV: M = head_dim, K = kv_cache_len
        sv_result = self.pom.evaluate_gemv(M=self.head_dim, K=kv_cache_len)
        sv_cycles = sv_result.get("total_cycles", 0) * heads_per_channel
        total_cycles += sv_cycles
        op_cycles["Attention::Context_Computation_SV"] = sv_cycles

        # 3. 访存密集型算子 (保持不变)
        bw_total_gb_s = self.config.main_memory_bw_per_channel_gb_s * pim_channels
        clock_hz = self.config.clock_ghz * 1e9
        
        softmax_bytes = self.num_heads * kv_cache_len * 2
        softmax_cycles = (softmax_bytes / (bw_total_gb_s * 1e9)) * clock_hz
        total_cycles += softmax_cycles
        op_cycles["Attention::Softmax"] = softmax_cycles

        rms_bytes = self.hidden_size * 2 * 2
        rms_cycles = (rms_bytes / (bw_total_gb_s * 1e9)) * clock_hz
        total_cycles += rms_cycles
        op_cycles["RMSNorm"] = rms_cycles
        
        silu_bytes = self.intermediate_size * 2
        silu_cycles = (silu_bytes / (bw_total_gb_s * 1e9)) * clock_hz
        total_cycles += silu_cycles
        op_cycles["FFN::Activation(SiLU)"] = silu_cycles
        
        return {"total_cycles": total_cycles, "op_breakdown": op_cycles}


# ==============================================================================
# 主分析流程 (已更新)
# ==============================================================================
def simulate_long_text_generation(
    pim_config: PIMConfig,
    prompt_len: int,
    add_tokens: int,
    layers: int
):
    print("\n" + "="*80)
    print("LLM Inference Performance Simulation (PIM v3.1 - Prefill+Decode)")
    print("="*80)
    print("Hardware Configuration:")
    print(f"  - PIM Channels: {pim_config.total_channels} @ {pim_config.clock_ghz} GHz")
    print(f"  - GB Size per PIM Channel: {pim_config.global_buffer_size_per_channel_kb} KB")
    print("\nSimulation Parameters:")
    print(f"  - Model Layers: {layers}")
    print(f"  - Initial Prompt Length: {prompt_len}")
    print(f"  - New Tokens to Generate: {add_tokens}")
    print("-"*80)

    evaluator = ApplicationLayerEvaluator(pim_config)
    
    # --- 1. Prefill Stage ---
    prefill_result = evaluator.evaluate_prefill_latency(prompt_len)
    total_prefill_cycles = prefill_result["total_cycles"] * layers

    # --- 2. Decoding Stage ---
    total_decoding_cycles = 0
    cumulative_op_cycles = prefill_result["op_breakdown"].copy()
    for key in cumulative_op_cycles:
        cumulative_op_cycles[key] *= layers
    
    for i in range(add_tokens):
        current_kv_cache_len = prompt_len + i
        step_result = evaluator.evaluate_single_token_latency(
            kv_cache_len=current_kv_cache_len
        )
        total_decoding_cycles += step_result["total_cycles"] * layers
        for op_name, cycles in step_result["op_breakdown"].items():
            cumulative_op_cycles[op_name] = cumulative_op_cycles.get(op_name, 0) + (cycles * layers)

    # --- 3. Report Generation ---
    total_cycles_all_tokens = total_prefill_cycles + total_decoding_cycles

    clock_hz = pim_config.clock_ghz * 1e9
    
    prefill_latency_ms = (total_prefill_cycles / clock_hz * 1000) if clock_hz > 0 else 0
    decoding_latency_ms = (total_decoding_cycles / clock_hz * 1000) if clock_hz > 0 else 0
    total_latency_ms = prefill_latency_ms + decoding_latency_ms
    avg_latency_per_token_ms = decoding_latency_ms / add_tokens if add_tokens > 0 else 0
    throughput_tps = 1000 / avg_latency_per_token_ms if avg_latency_per_token_ms > 0 else 0

    print(f"\n[Summary]")
    print(f"  - Prefill Latency:         {prefill_latency_ms:>10.4f} ms")
    print(f"  - Total Decoding Latency:  {decoding_latency_ms:>10.4f} ms (for {add_tokens} tokens)")
    print(f"  - Total End-to-End Latency:{total_latency_ms:>10.4f} ms")
    print(f"  --------------------------------------------------")
    print(f"  - Avg Latency per Token:   {avg_latency_per_token_ms:>10.4f} ms/token")
    print(f"  - Generation Throughput:   {throughput_tps:>10.2f} tokens/sec")
    
    print("\n--- Per-Operator Latency Breakdown (Aggregated for Prefill + Decode) ---")
    sorted_ops = sorted(cumulative_op_cycles.items(), key=lambda item: item[1], reverse=True)
    print(f"{'Operator':<40} {'Total Cycles':>22} {'Percentage':>15}")
    print("-" * 80)
    for op_name, cycles in sorted_ops:
        percentage = (cycles / total_cycles_all_tokens) * 100 if total_cycles_all_tokens > 0 else 0
        print(f"{op_name:<40} {int(cycles):>22,d} {percentage:>14.2f}%")
    print("="*80)


# ==============================================================================
# Main执行块
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a microarchitecture-based LLM inference performance simulation (v3.1).")
    parser.add_argument('--prompt-len', type=int, default=39, help="Initial prompt length (context size).")
    parser.add_argument('--layers', type=int, default=32, help="Number of layers in the model.")
    parser.add_argument('--add-tokens', type=int, default=100, help="Number of new tokens to generate.")
    parser.add_argument('--clock-ghz', type=float, default=2.0, help="PIM clock frequency in GHz.")
    parser.add_argument('--gb-size-kb', type=int, default=256, help="Global Buffer size per PIM channel in KB.")
    parser.add_argument('--pim-channels', type=int, default=32, help="Number of PIM channels to use.")
    args = parser.parse_args()

    # 1. 定义硬件配置
    pim_hw_config = PIMConfig(
        total_channels=args.pim_channels,
        clock_ghz=args.clock_ghz,
        global_buffer_size_per_channel_kb=args.gb_size_kb
    )

    # 2. 运行主仿真
    simulate_long_text_generation(
        pim_config=pim_hw_config,
        prompt_len=args.prompt_len,
        add_tokens=args.add_tokens,
        layers=args.layers
    )

    # 3. (可选) 运行一个对比场景来展示GB容量的影响
    print("\n\n" + "*"*30 + " RUNNING COMPARISON SCENARIO " + "*"*30)
    pim_hw_small_gb = PIMConfig(
        total_channels=args.pim_channels,
        clock_ghz=args.clock_ghz,
        global_buffer_size_per_channel_kb=16 # 容量受限
    )
    simulate_long_text_generation(
        pim_config=pim_hw_small_gb,
        prompt_len=args.prompt_len,
        add_tokens=args.add_tokens,
        layers=args.layers
    )
