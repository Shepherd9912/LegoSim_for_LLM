import math
import io
import re
import argparse
from typing import Dict, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from utils_v0_3 import NPUConfig, PerformanceResult


# ==============================================================================
# 原始单核性能日志数据
# ==============================================================================
SINGLE_CORE_LOG_DATA = """
# ... (Full log data from previous turn is included here for completeness)
2025-10-06 13:03:34,425 - INFO -   Test: Matched_Size (M=128, K=128, N=128) -> Cycles: 3699
2025-10-06 13:03:34,470 - INFO -   Test: Small_GEMV (M=1, K=128, N=128) -> Cycles: 1574
2025-10-06 13:03:34,511 - INFO -   Test: Small_GEMV_T (M=128, K=128, N=1) -> Cycles: 1684
2025-10-06 13:03:34,563 - INFO -   Test: K_Sweep_64 (M=128, K=64, N=128) -> Cycles: 2701
2025-10-06 13:03:34,622 - INFO -   Test: K_Sweep_128 (M=128, K=128, N=128) -> Cycles: 3699
2025-10-06 13:03:34,702 - INFO -   Test: K_Sweep_256 (M=128, K=256, N=128) -> Cycles: 5829
2025-10-06 13:03:34,801 - INFO -   Test: K_Sweep_512 (M=128, K=512, N=128) -> Cycles: 10089
2025-10-06 13:03:34,958 - INFO -   Test: K_Sweep_1024 (M=128, K=1024, N=128) -> Cycles: 18714
2025-10-06 13:03:35,234 - INFO -   Test: K_Sweep_2048 (M=128, K=2048, N=128) -> Cycles: 35665
2025-10-06 13:03:35,719 - INFO -   Test: K_Sweep_4096 (M=128, K=4096, N=128) -> Cycles: 70160
2025-10-06 13:03:35,792 - INFO -   Test: N_Sweep_64 (M=128, K=128, N=64) -> Cycles: 2684
2025-10-06 13:03:35,875 - INFO -   Test: N_Sweep_128 (M=128, K=128, N=128) -> Cycles: 3699
2025-10-06 13:03:35,971 - INFO -   Test: N_Sweep_256 (M=128, K=128, N=256) -> Cycles: 5226
2025-10-06 13:03:36,102 - INFO -   Test: N_Sweep_512 (M=128, K=128, N=512) -> Cycles: 11821
2025-10-06 13:03:36,295 - INFO -   Test: N_Sweep_1024 (M=128, K=128, N=1024) -> Cycles: 23983
2025-10-06 13:03:36,632 - INFO -   Test: N_Sweep_2048 (M=128, K=128, N=2048) -> Cycles: 49607
2025-10-06 13:03:37,342 - INFO -   Test: N_Sweep_4096 (M=128, K=128, N=4096) -> Cycles: 121592
2025-10-06 13:03:37,821 - INFO -   Test: M_Sweep_1 (M=1, K=1024, N=1024) -> Cycles: 66089
2025-10-06 13:03:38,309 - INFO -   Test: M_Sweep_8 (M=8, K=1024, N=1024) -> Cycles: 67407
2025-10-06 13:03:38,835 - INFO -   Test: M_Sweep_32 (M=32, K=1024, N=1024) -> Cycles: 72125
2025-10-06 13:03:39,391 - INFO -   Test: M_Sweep_64 (M=64, K=1024, N=1024) -> Cycles: 79507
2025-10-06 13:03:40,047 - INFO -   Test: M_Sweep_128 (M=128, K=1024, N=1024) -> Cycles: 94606
2025-10-06 13:03:40,888 - INFO -   Test: M_Sweep_256 (M=256, K=1024, N=1024) -> Cycles: 124819
MESSAGE TRUNCATED"""

PERFORMANCE_BOOST_DATA_CSV = """
Test_Name,0.5,0.25,0.125,0.0625,0.03125
N_Split_64,1.66,2.43,3.14,3.16,3.16
N_Split_128,1.80,2.93,4.25,4.28,4.28
N_Split_256,1.90,3.35,5.40,5.43,5.43
N_Split_512,1.96,3.63,6.36,6.39,6.39
N_Split_1024,1.97,3.79,6.99,7.01,7.01
K_Split_64,1.93,3.63,3.67,3.67,3.67
K_Split_128,1.98,3.93,7.06,7.10,7.09
K_Split_256,1.97,3.97,7.33,7.34,7.33
K_Split_512,1.98,3.88,7.45,7.46,7.46
K_Split_1024,2.02,3.94,7.54,7.55,7.55
M_Split_1,1.99,3.92,7.59,7.60,7.59
M_Split_2,2.00,4.00,7.64,7.64,7.63
M_Split_4,2.01,3.97,7.60,7.59,7.61
M_Split_8,1.99,3.94,7.54,7.57,7.57
M_Split_16,2.01,3.98,7.64,7.67,7.68
M_Split_32,2.01,3.96,7.63,7.68,7.68
M_Split_64,2.02,4.01,7.70,7.79,7.82
Micro_1x64x64,1.17,1.31,1.37,1.40,1.40
Micro_1x64x128,1.31,1.55,1.70,1.72,1.74
Micro_1x128x64,1.32,1.55,1.68,1.72,1.74
Micro_1x128x128,1.50,1.96,2.25,2.31,2.31
Micro_64x64x64,1.38,1.68,1.88,1.97,1.99
Micro_64x64x128,1.50,1.96,2.31,2.42,2.46
Micro_64x128x64,1.48,1.97,2.33,2.40,2.43
Micro_64x128x128,1.63,2.33,2.88,3.00,3.04
"""

# ==============================================================================
# 层 0: 核心模型
# ==============================================================================

class AnalyticalMemoryModel:
    # ... (No changes)
    def __init__(self, config: NPUConfig, element_size_bytes: int = 2):
        self.config = config
        self.element_size_bytes = element_size_bytes
    def _get_cycles(self, shape: list, equivalent_passes: int) -> float:
        if self.config.bytes_per_cycle == 0: return 0
        num_elements = np.prod(shape)
        bytes_to_transfer = num_elements * self.element_size_bytes * equivalent_passes
        return bytes_to_transfer / self.config.bytes_per_cycle
    def model_rmsnorm(self, shape: list) -> float: return self._get_cycles(shape, 4)
    def model_softmax(self, shape: list) -> float: return self._get_cycles(shape, 5)
    def model_element_wise(self, shape: list) -> float: return self._get_cycles(shape, 2)


class SingleCoreCostModel:
    # ... (No changes)
    def __init__(self, log_data: str): self._build_model(log_data)
    def _parse_log_data(self, log_data: str) -> pd.DataFrame:
        pattern = re.compile(r"Test:.*?\(M=(\d+), K=(\d+), N=(\d+)\) -> Cycles: (\d+)")
        records = []
        for line in log_data.strip().split('\n'):
            match = pattern.search(line)
            if match:
                m, k, n, cycles = map(int, match.groups())
                records.append({'M': m, 'K': k, 'N': n, 'Cycles': cycles})
        return pd.DataFrame(records).drop_duplicates()
    def _build_model(self, log_data: str):
        df = self._parse_log_data(log_data)
        df['MK'] = df['M'] * df['K']; df['KN'] = df['K'] * df['N']; df['MN'] = df['M'] * df['N']; df['MKN'] = df['M'] * df['K'] * df['N']
        self.features = ['M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
        self.model = LinearRegression().fit(df[self.features], df['Cycles'])
    def predict(self, M: int, K: int, N: int) -> float:
        data = {'M': [M], 'K': [K], 'N': [N], 'MK': [M*K], 'KN': [K*N], 'MN': [M*N], 'MKN': [M*K*N]}
        feature_vector = pd.DataFrame(data)[self.features]
        return max(0.0, self.model.predict(feature_vector)[0])

class PerformanceBoostModel:
    """
    封装了从实测数据中提取的性能提升表，
    能够根据GEMM尺寸和核/通道比，预测性能提升倍率。
    """
    def __init__(self, csv_data: str):
        self.df = pd.read_csv(io.StringIO(csv_data)).set_index('Test_Name')
        # 将列名字符串转为浮点数，并创建用于插值的x轴
        self.ratios = sorted([float(c) for c in self.df.columns])

    def _classify_gemm(self, M: int, K: int, N: int) -> str:
        """ 将任意GEMM尺寸分类到最接近的测试名称 """
        # 规则1: 微内核
        if M <= 64 and K <= 128 and N <= 128:
            # 找到最接近的微内核尺寸
            micro_dims = [(1,64,64), (1,64,128), (1,128,64), (1,128,128),
                          (64,64,64), (64,64,128), (64,128,64), (64,128,128)]
            target = np.array([M, K, N])
            closest_dim = min(micro_dims, key=lambda dim: np.linalg.norm(target - np.array(dim)))
            return f"Micro_{closest_dim[0]}x{closest_dim[1]}x{closest_dim[2]}"
        
        # 规则2: 大型GEMM/GEMV的分类
        if K > 2048 and N > 2048: return f"M_Split_{min(64, M)}" # M_Split
        if M == 1 and K > 2048: return f"N_Split_{min(1024, N)}" # N_Split
        if M == 1 and N > 2048: return f"K_Split_{min(1024, K)}" # K_Split

        # 默认回退到一个泛化的类别
        return "M_Split_64"

    def get_boost_factor(self, M: int, K: int, N: int, core_per_channel_ratio: float) -> float:
        """ 获取指定GEMM和硬件配比下的性能提升倍率 """
        if core_per_channel_ratio >= 1.0:
            return 1.0

        category = self._classify_gemm(M, K, N)
        if category not in self.df.index:
            # 如果分类结果无效，使用一个安全的默认值
            category = "M_Split_64"

        # 准备插值点 (x=ratios, y=boost_factors)
        # 确保插值点按x轴升序排列
        x_points = [1.0] + self.ratios[::-1]
        y_points = [1.0] + self.df.loc[category, [str(r) for r in self.ratios]].tolist()[::-1]

        # 使用np.interp进行线性插值
        return float(np.interp(core_per_channel_ratio, x_points, y_points))

# ==============================================================================
# 层 1: NPU 系统级操作模型 (SOM)
# ==============================================================================

class SystemLevelOperationModel:
    """
    基于单核和性能提升模型，应用系统级并行化策略。
    """
    def __init__(self, sccm: SingleCoreCostModel, boost_model: PerformanceBoostModel, config: NPUConfig):
        self.sccm = sccm
        self.boost_model = boost_model
        self.config = config
    
    def _get_core_capability_scaling_factor(self) -> float:
        """
        计算由于一个物理核心服务多个通道而导致的性能缩放因子。
        如果一个核心服务4个通道，它的计算能力被均分为4份，因此执行时间会变为大约4倍。
        """
        return float(self.config.channels_per_core)
    
    def evaluate_gemm_system_tensor_parallel(self, M: int, K: int, N: int) -> float:
        """
        评估应用了Tensor Parallel（N维度切分）策略的GEMM操作。
        """
        # 计算每个核心的子任务尺寸
        # N_per_core = math.ceil(N / self.config.num_cores)
        # 获取子任务在单核/单通道下的基准性能
        # T_baseline_sub = self.sccm.predict(M, K, N_per_core)
        # 计算核/通道比
        # core_per_channel = self.config.num_cores / self.config.num_channels
        # 获取性能提升倍率 (如果ratio<1)
        # boost_factor = self.boost_model.get_boost_factor(M, K, N_per_core, core_per_channel)
        # 应用性能提升
        # T_final_sub = T_baseline_sub / boost_factor

        # 1. 计算每个“逻辑核心”（即每个通道）分配到的任务
        N_per_logical_core = math.ceil(N / self.config.num_logical_cores)
        # 2. 获取子任务在“完整”单核下的基准性能
        T_baseline_sub = self.sccm.predict(M, K, N_per_logical_core)
        # 3. 计算由于单核能力被切分导致的性能损失
        scaling_factor = self._get_core_capability_scaling_factor()
        T_scaled_sub = T_baseline_sub * scaling_factor
        # 4. 计算核/通道比 (现在基于物理核心数)
        core_per_channel_ratio = self.config.num_cores / self.config.num_channels
        # 5. 获取多核通信带来的性能提升
        boost_factor = self.boost_model.get_boost_factor(M, K, N_per_logical_core, core_per_channel_ratio)
        # 6. 应用性能提升
        T_final_sub = T_scaled_sub / boost_factor
        
        return T_final_sub
        
    def evaluate_gemm_system_data_parallel(self, M: int, K: int, N: int, num_heads: int) -> float:
        """
        评估应用了Data Parallel（Head维度并行）策略的GEMM/GEMV操作。
        """
        # 计算需要多少轮次(wave)才能处理完所有heads
        # num_waves = math.ceil(num_heads / self.config.num_cores)
        # 单个head的基准性能
        # T_baseline_single_head = self.sccm.predict(M=M, K=K, N=N)
        # 计算核/通道比
        # core_per_channel = self.config.num_cores / self.config.num_channels
        # 获取性能提升倍率
        # boost_factor = self.boost_model.get_boost_factor(M, K, N, core_per_channel)
        # 应用性能提升并乘以轮次数
        # T_final = num_waves * (T_baseline_single_head / boost_factor)

        # 1. 计算处理所有heads需要的轮次数（基于逻辑核心/通道数）
        num_waves = math.ceil(num_heads / self.config.num_logical_cores)        
        # 2. 获取单个head在“完整”单核下的基准性能
        T_baseline_single_head = self.sccm.predict(M=M, K=K, N=N)        
        # 3. 计算由于单核能力被切分导致的性能损失
        scaling_factor = self._get_core_capability_scaling_factor()
        T_scaled_single_head = T_baseline_single_head * scaling_factor        
        # 4. 计算核/通道比
        core_per_channel_ratio = self.config.num_cores / self.config.num_channels        
        # 5. 获取性能提升
        boost_factor = self.boost_model.get_boost_factor(M, K, N, core_per_channel_ratio)        
        # 6. 应用性能提升并乘以轮次数
        T_final = num_waves * (T_scaled_single_head / boost_factor)
        
        return T_final

# ==============================================================================
# 层 2: 应用层评估器 (ALE)
# ==============================================================================

class ApplicationLayerEvaluator:
    def __init__(self, npu_config: NPUConfig, element_size_bytes: int = 2):
        self.config = npu_config
        self.element_size_bytes = element_size_bytes
        self.sccm = SingleCoreCostModel(SINGLE_CORE_LOG_DATA)
        self.boost_model = PerformanceBoostModel(PERFORMANCE_BOOST_DATA_CSV)
        # SOM现在会接收包含新拓扑信息的config
        self.som = SystemLevelOperationModel(self.sccm, self.boost_model, self.config)
        self.mem_model = AnalyticalMemoryModel(self.config, self.element_size_bytes)
        
        self.hidden_size, self.intermediate_size = 4096, 11008
        self.num_heads, self.head_dim = 32, 128

    # ==========================================================================
    # 新增: 顶层评估函数
    # ==========================================================================
    def evaluate(self, M: int, K: int, N: int, batch_size: int = 1, op_type: str = "GEMM", num_heads: int = 32) -> PerformanceResult:
        """
        统一的顶层评估接口，根据操作类型选择合适的并行策略。
        
        Args:
            M, K, N: GEMM/GEMV的核心维度.
            batch_size: 对于Data Parallel, 这通常是heads的数量.
            op_type: 'GEMM' 或 'GEMV'. 决定了并行策略的选择.
        
        Returns:
            一个PerformanceResult对象.
        """
        # 假设FP16
        element_size = self.element_size_bytes

        # 1. 计算延迟 (cycles)
        latency_cycles = 0.0
        
        # 根据操作类型和形状选择并行策略
        # 启发式规则: M > 1 的通常是Prefill阶段的大型GEMM, 适合Tensor Parallel
        # M = 1 的通常是Decoding阶段的GEMV, 如果K, N较大也适合Tensor Parallel
        # 特例: Attention QK/SV 是在Head维度上Data Parallel
        # if op_type == "Attention_QK" or op_type == "Attention_SV":
        #      latency_cycles = self.som.evaluate_gemm_system_data_parallel(M, K, N, num_heads)
        # else: # 默认使用Tensor Parallel
        #      latency_cycles = self.som.evaluate_gemm_system_tensor_parallel(M, K, N)
        # --- 修改的逻辑: 根据op_type选择正确的SOM函数 ---
        # 假设我们通过op_type来区分Attention的Data Parallel和其他的Tensor Parallel
        if op_type in ["Attention_QK", "Attention_SV"]:
             latency_cycles = self.som.evaluate_gemm_system_data_parallel(M, K, N, num_heads)
        else: # 默认使用Tensor Parallel for projections and FFNs
             latency_cycles = self.som.evaluate_gemm_system_tensor_parallel(M, K, N)

        # 2. 计算输入/输出数据量 (bytes)
        # 注意：这里我们只计算一次完整操作所需的数据量，不考虑并行化带来的重复读取
        # 因为带宽限制的影响已经包含在了PerformanceBoostModel中
        input_bytes = (M * K + K * N) * element_size
        output_bytes = M * N * element_size
        
        # 3. 能量模型 (当前为占位符)
        energy_joules = 0.0 # TODO: 未来可以添加能量模型

        return PerformanceResult(
            latency_cycles=latency_cycles,
            energy_joules=energy_joules,
            path_taken="NPU_Execution", # NPU只有一种执行路径
            input_bytes=input_bytes,
            output_bytes=output_bytes
        )

    # ==========================================================================
    # 新增: Prefill阶段评估函数
    # ==========================================================================
    def evaluate_prefill_latency(self, prompt_len: int) -> Dict:
        """
        评估一个完整层在Prefill阶段的延迟。
        Prefill的特点是 M = prompt_len > 1。
        """
        op_cycles = {}
        
        # 1. 访存密集型
        op_cycles["RMSNorm"] = self.mem_model.model_rmsnorm([prompt_len, self.hidden_size]) * 2
        op_cycles["Attention::Softmax"] = self.mem_model.model_softmax([1, self.num_heads, prompt_len, prompt_len])
        op_cycles["FFN::Activation(SiLU)"] = self.mem_model.model_element_wise([prompt_len, self.intermediate_size])

        # 2. 大型GEMM (Tensor Parallel)
        # QKV, Output Projections
        op_cycles["Attention::QKV_Projections"] = self.som.evaluate_gemm_system_tensor_parallel(prompt_len, self.hidden_size, self.hidden_size) * 3
        op_cycles["Attention::Output_Projection"] = self.som.evaluate_gemm_system_tensor_parallel(prompt_len, self.hidden_size, self.hidden_size) * 1
        
        # FFN Projections
        op_cycles["FFN::Gate_Up_Projections"] = self.som.evaluate_gemm_system_tensor_parallel(prompt_len, self.hidden_size, self.intermediate_size) * 2
        op_cycles["FFN::Down_Projection"] = self.som.evaluate_gemm_system_tensor_parallel(prompt_len, self.intermediate_size, self.hidden_size) * 1
        
        # 3. Attention BMM (Batch GEMM), 作为Data Parallel处理
        # Score_Computation_QK: (B=num_heads, M=prompt_len, K=head_dim, N=prompt_len)
        op_cycles["Attention::Score_Computation_QK"] = self.som.evaluate_gemm_system_data_parallel(
            M=prompt_len, K=self.head_dim, N=prompt_len, num_heads=self.num_heads
        )
        # Context_Computation_SV: (B=num_heads, M=prompt_len, K=prompt_len, N=head_dim)
        op_cycles["Attention::Context_Computation_SV"] = self.som.evaluate_gemm_system_data_parallel(
            M=prompt_len, K=prompt_len, N=self.head_dim, num_heads=self.num_heads
        )
        
        total_cycles = sum(op_cycles.values())
        return {"total_cycles": total_cycles, "op_breakdown": op_cycles}

    def evaluate_single_token_latency(self, kv_cache_len: int) -> Dict:
        op_cycles = {}
        # 1. 访存密集型
        op_cycles["RMSNorm"] = self.mem_model.model_rmsnorm([1, self.hidden_size]) * 2
        op_cycles["Attention::Softmax"] = self.mem_model.model_softmax([1, self.num_heads, 1, kv_cache_len])
        op_cycles["FFN::Activation(SiLU)"] = self.mem_model.model_element_wise([1, self.intermediate_size])

        # 2. 大型GEMM/GEMV (Tensor Parallel)
        op_cycles["Attention::QKV_Projections"] = self.som.evaluate_gemm_system_tensor_parallel(1, self.hidden_size, self.hidden_size) * 3
        op_cycles["Attention::Output_Projection"] = self.som.evaluate_gemm_system_tensor_parallel(1, self.hidden_size, self.hidden_size) * 1
        op_cycles["FFN::Gate_Up_Projections"] = self.som.evaluate_gemm_system_tensor_parallel(1, self.hidden_size, self.intermediate_size) * 2
        op_cycles["FFN::Down_Projection"] = self.som.evaluate_gemm_system_tensor_parallel(1, self.intermediate_size, self.hidden_size) * 1
        
        # 3. Attention QK/SV (Data Parallel)
        op_cycles["Attention::Score_Computation_QK"] = self.som.evaluate_gemm_system_data_parallel(
            M=1, K=self.head_dim, N=kv_cache_len, num_heads=self.num_heads
        )
        op_cycles["Attention::Context_Computation_SV"] = self.som.evaluate_gemm_system_data_parallel(
            M=self.head_dim, K=kv_cache_len, N=self.head_dim, num_heads=self.num_heads
        )
        
        total_cycles = sum(op_cycles.values())
        return {"total_cycles": total_cycles, "op_breakdown": op_cycles}


# ==============================================================================
# 主分析流程与命令行接口 (已更新)
# ==============================================================================
def simulate_long_text_generation(config: NPUConfig, prompt_len: int, add_tokens: int, num_layers: int):
    """
    print("\n" + "="*80)
    print("LLM Inference Performance Simulation (NPU Data-Driven Model v3.1 - Prefill+Decode)")
    print("="*80)
    print("Hardware Configuration:")
    print(f"  - NPU Cores / Channels: {config.num_cores} / {config.num_channels}")
    print(f"  - Core/Channel Ratio:   {config.num_cores/config.num_channels:.4f}")
    print(f"  - NPU Clock: {config.clock_ghz:.2f} GHz")
    print(f"  - Effective Memory BW: {config.effective_memory_bw_gb_s:.2f} GB/s")
    print("\nSimulation Parameters:")
    print(f"  - Model Layers: {num_layers}")
    print(f"  - Initial Prompt Length: {prompt_len}")
    print(f"  - New Tokens to Generate: {add_tokens}")
    print("-"*80)
    """
    print("\n" + "="*80)
    print("LLM Inference Performance Simulation (NPU Data-Driven Model v3.2)")
    print("="*80)
    print("Hardware Configuration:")
    print(f"  - Physical NPU Cores:   {config.num_cores}")
    print(f"  - DRAM Channels:          {config.num_channels}")
    print(f"  - Channels per Core:      {config.channels_per_core}") # 新增
    print(f"  - Core/Channel Ratio:     {config.num_cores/config.num_channels:.4f}")
    print(f"  - NPU Clock: {config.clock_ghz:.2f} GHz")
    print(f"  - Effective Memory BW: {config.effective_memory_bw_gb_s:.2f} GB/s")
    print("\nSimulation Parameters:")
    print(f"  - Model Layers: {num_layers}")
    print(f"  - Initial Prompt Length: {prompt_len}")
    print(f"  - New Tokens to Generate: {add_tokens}")
    print("-"*80)

    evaluator = ApplicationLayerEvaluator(config)
    
    # --- 1. Prefill Stage ---
    prefill_result = evaluator.evaluate_prefill_latency(prompt_len)
    total_prefill_cycles = prefill_result["total_cycles"] * num_layers
    
    # --- 2. Decoding Stage ---
    total_decoding_cycles = 0
    cumulative_op_cycles = prefill_result["op_breakdown"].copy()
    for key in cumulative_op_cycles:
        cumulative_op_cycles[key] *= num_layers

    for i in range(add_tokens):
        current_kv_cache_len = prompt_len + i
        single_layer_result = evaluator.evaluate_single_token_latency(current_kv_cache_len)
        total_layer_cycles = single_layer_result["total_cycles"] * num_layers
        total_decoding_cycles += total_layer_cycles

        for op, cycles in single_layer_result["op_breakdown"].items():
            cumulative_op_cycles[op] = cumulative_op_cycles.get(op, 0) + (cycles * num_layers)
            
    # --- 3. Report Generation ---
    total_cycles_all_tokens = total_prefill_cycles + total_decoding_cycles

    clock_hz = config.clock_ghz * 1e9
    
    # Latency breakdown
    prefill_latency_ms = (total_prefill_cycles / clock_hz * 1000) if clock_hz > 0 else 0
    decoding_latency_ms = (total_decoding_cycles / clock_hz * 1000) if clock_hz > 0 else 0
    total_latency_ms = prefill_latency_ms + decoding_latency_ms
    avg_latency_per_token_ms = decoding_latency_ms / add_tokens if add_tokens > 0 else 0
    
    # Throughput (conventionally measures generation speed)
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
        if math.isnan(cycles): continue
        percentage = (cycles / total_cycles_all_tokens) * 100 if total_cycles_all_tokens > 0 else 0
        print(f"{op_name:<40} {int(cycles):>22,d} {percentage:>14.2f}%")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a data-driven LLM decoding performance simulation for NPU (v3.1).")
    parser.add_argument('--prompt-len', type=int, default=39, help="Initial prompt length.")
    parser.add_argument('--add-tokens', type=int, default=100, help="Number of new tokens to generate.")
    parser.add_argument('--layers', type=int, default=32, help="Number of model layers.")
    parser.add_argument('--npu-cores', type=int, default=32, help="Number of NPU cores.")
    parser.add_argument('--npu-channels', type=int, default=32, help="Number of DRAM channels.")
    parser.add_argument('--clock-ghz', type=float, default=2.0, help="NPU clock frequency in GHz.")
    parser.add_argument('--channels-per-core', type=int, default=4, help="Number of channels served by a single physical core.")
    args = parser.parse_args()

    """
    # 1. Baseline: 1 core per channel
    print("\n\n" + "*"*30 + " RUNNING BASELINE SCENARIO: 1 CORE PER CHANNEL " + "*"*30)
    config_baseline = NPUConfig(num_cores=32, num_channels=32, clock_ghz=args.clock_ghz)
    simulate_long_text_generation(config_baseline, args.prompt_len, args.add_tokens, args.layers)

    # 2. Comparison: 0.25 core per channel (e.g., 8 cores, 32 channels)
    print("\n\n" + "*"*30 + " RUNNING COMPARISON SCENARIO: 0.25 CORE PER CHANNEL " + "*"*30)
    config_mem_heavy = NPUConfig(num_cores=8, num_channels=32, clock_ghz=args.clock_ghz)
    simulate_long_text_generation(config_mem_heavy, args.prompt_len, args.add_tokens, args.layers)

    # 3. Comparison: 0.125 core per channel (e.g., 4 cores, 32 channels)
    print("\n\n" + "*"*30 + " RUNNING COMPARISON SCENARIO: 0.125 CORE PER CHANNEL " + "*"*30)
    config_mem_heavy = NPUConfig(num_cores=4, num_channels=32, clock_ghz=args.clock_ghz)
    simulate_long_text_generation(config_mem_heavy, args.prompt_len, args.add_tokens, args.layers)
    """
    # 1. 运行您指定的场景: 32个物理核心，每个服务1个通道 (传统1:1映射)
    print("\n\n" + "*"*30 + " RUNNING CUSTOM SCENARIO " + "*"*30)
    config_custom = NPUConfig(
        num_cores=args.npu_cores, 
        num_channels=args.npu_channels, 
        channels_per_core=args.channels_per_core,
        clock_ghz=args.clock_ghz
    )
    simulate_long_text_generation(config_custom, args.prompt_len, args.add_tokens, args.layers)

    # 2. 对比场景: 8个物理核心，每个服务4个通道
    print("\n\n" + "*"*30 + " RUNNING COMPARISON SCENARIO: 1 CORE PER CHANNEL " + "*"*30)
    config_baseline = NPUConfig(
        num_cores=8, 
        num_channels=32, 
        channels_per_core=4, 
        clock_ghz=args.clock_ghz
    )
    simulate_long_text_generation(config_baseline, args.prompt_len, args.add_tokens, args.layers)
