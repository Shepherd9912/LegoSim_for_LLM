# ==============================================================================
# npu_model_v4_dual_model.py
#
# DESCRIPTION: This file serves as the core NPU performance library.
# It now features a dual-model approach:
#   1. HybridCoreCostModel: Trained on single-core data for data-parallel tasks.
#   2. TensorParallelCostModel: Trained on multi-core data for tensor-parallel tasks.
# This provides a higher-fidelity simulation for heterogeneous workloads.
# ==============================================================================

import math # 导入 math 模块，用于数学计算，例如向上取整。
import argparse # 导入 argparse 模块，用于解析命令行参数（在此文件中未使用，但可能为通用模板）。
from typing import Dict, Tuple, List, Optional # 从 typing 模块导入类型提示。
from functools import lru_cache # 从 functools 模块导入 lru_cache 装饰器，用于缓存函数结果，提高性能。
import pandas as pd # 导入 pandas 库，用于数据处理和 CSV 文件操作，通常简写为 pd。
import numpy as np # 导入 numpy 库，用于数值计算，通常简写为 np。
from sklearn.ensemble import RandomForestRegressor # 从 scikit-learn 库中导入 RandomForestRegressor，一个用于回归的机器学习模型。

from utils import NPUConfig, PerformanceResult # 从我们自定义的 utils_v0_4 模块中导入 NPUConfig 和 PerformanceResult 类。

# ==============================================================================
# SECTION 1: CORE PERFORMANCE ENGINES (RandomForest-based)
# ==============================================================================

# 现在改用“对数变换”+特征的训练方式
class HybridCoreCostModel:
    """
    此模型现在专用于单核性能预测，作为数据并行操作建模的基础。
    [MODIFIED v3]: Uses log transform AND advanced feature engineering.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        # --- ADVANCED FEATURE ENGINEERING START ---
        # 扩展特征集，加入对数、比率和计算密度特征
        self.features_ = [
            'M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN', # Base Features
            'log_M', 'log_K', 'log_N', 'log_MKN',   # Logarithmic Features
            'ratio_M_N', 'ratio_K_N', 'ratio_M_K', # Shape (Ratio) Features
            'compute_density'                      # Compute Density Feature
        ]
        # --- ADVANCED FEATURE ENGINEERING END ---
        self.total_cycle_model_ = None
        self.compute_cycle_model_ = None
        self._train()
        print(f"INFO: Single-Core Cost Model trained successfully from '{csv_path}'.")

    def _train(self):
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"FATAL ERROR: The data file '{self.csv_path}' was not found.")
            raise
        df = df[df['Cycles'] > 0].dropna(subset=['Cycles', 'MAC_Utilization_Percent'])
        y_total = df['Cycles']
        
        if 'Memory_Stall_Cycles' in df.columns:
            y_compute = df['Cycles'] - df['Memory_Stall_Cycles']
        else:
            y_compute = df['Cycles'] * (df['MAC_Utilization_Percent'] / 100.0)

        # --- ADVANCED FEATURE ENGINEERING START ---
        # Base Features
        df['MK'] = df['M'] * df['K']; df['KN'] = df['K'] * df['N']
        df['MN'] = df['M'] * df['N']; df['MKN'] = df['M'] * df['K'] * df['N']
        
        # Logarithmic Features (using log1p for safety against log(0))
        df['log_M'] = np.log1p(df['M'])
        df['log_K'] = np.log1p(df['K'])
        df['log_N'] = np.log1p(df['N'])
        df['log_MKN'] = np.log1p(df['MKN'])

        # Shape (Ratio) Features (adding a small epsilon to prevent division by zero)
        epsilon = 1e-9
        df['ratio_M_N'] = df['M'] / (df['N'] + epsilon)
        df['ratio_K_N'] = df['K'] / (df['N'] + epsilon)
        df['ratio_M_K'] = df['M'] / (df['K'] + epsilon)

        # Compute Density Feature
        df['compute_density'] = (2 * df['MKN']) / (df['MK'] + df['KN'] + df['MN'] + epsilon)
        # --- ADVANCED FEATURE ENGINEERING END ---

        X = df[self.features_]
        
        # Apply log transform to the target variables
        y_total_log = np.log1p(y_total)
        y_compute_log = np.log1p(y_compute)
        
        self.total_cycle_model_ = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.total_cycle_model_.fit(X, y_total_log) 
        
        self.compute_cycle_model_ = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.compute_cycle_model_.fit(X, y_compute_log)

    @lru_cache(maxsize=1024)
    def predict(self, M: int, K: int, N: int) -> Tuple[float, float]:
        if self.total_cycle_model_ is None or self.compute_cycle_model_ is None:
            raise RuntimeError("Model has not been trained.")
            
        # --- ADVANCED FEATURE ENGINEERING START ---
        # Must create all features for the prediction vector, mirroring the training logic
        epsilon = 1e-9
        data = {
            'M': [M], 'K': [K], 'N': [N],
            'MK': [M*K], 'KN': [K*N], 'MN': [M*N], 'MKN': [M*K*N],
            'log_M': [np.log1p(M)], 'log_K': [np.log1p(K)], 'log_N': [np.log1p(N)],
            'log_MKN': [np.log1p(M*K*N)],
            'ratio_M_N': [M / (N + epsilon)],
            'ratio_K_N': [K / (N + epsilon)],
            'ratio_M_K': [M / (K + epsilon)],
            'compute_density': [(2*M*K*N) / (M*K + K*N + M*N + epsilon)]
        }
        # --- ADVANCED FEATURE ENGINEERING END ---
        
        feature_vector = pd.DataFrame(data)[self.features_]
        
        predicted_log_total = self.total_cycle_model_.predict(feature_vector)[0]
        predicted_log_compute = self.compute_cycle_model_.predict(feature_vector)[0]
        
        predicted_total = np.expm1(predicted_log_total)
        predicted_compute = np.expm1(predicted_log_compute)
        
        predicted_compute = min(predicted_total, predicted_compute)
        return max(0.0, predicted_total), max(0.0, predicted_compute)

class TensorParallelCostModel:
    """
    一个专用于张量并行性能预测的新模型。
    [MODIFIED v3]: Uses log transform AND advanced feature engineering.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        # --- ADVANCED FEATURE ENGINEERING START ---
        # 扩展特征集，加入对数、比率和计算密度特征
        self.features_ = [
            'M', 'K', 'N', 'num_cores',             # Base Features
            'MK', 'KN', 'MN', 'MKN',
            'log_M', 'log_K', 'log_N', 'log_MKN',   # Logarithmic Features
            'ratio_M_N', 'ratio_K_N', 'ratio_M_K', # Shape (Ratio) Features
            'compute_density'                      # Compute Density Feature
        ]
        # --- ADVANCED FEATURE ENGINEERING END ---
        self.total_cycle_model_ = None
        self.compute_cycle_model_ = None
        self._train()
        print(f"INFO: Tensor-Parallel Cost Model trained successfully from '{csv_path}'.")

    def _train(self):
        try:
            df = pd.read_csv(self.csv_path)
            if 'num_cores' not in df.columns:
                raise ValueError(f"The multi-core data file '{self.csv_path}' is missing the required 'num_cores' column.")
        except FileNotFoundError:
            print(f"FATAL ERROR: The data file '{self.csv_path}' was not found.")
            raise

        df = df[df['Cycles'] > 0].dropna(subset=['Cycles', 'MAC_Utilization_Percent', 'num_cores'])
        y_total = df['Cycles']

        if 'Memory_Stall_Cycles' in df.columns:
            y_compute = df['Cycles'] - df['Memory_Stall_Cycles']
        else:
            y_compute = df['Cycles'] * (df['MAC_Utilization_Percent'] / 100.0)

        # --- ADVANCED FEATURE ENGINEERING START ---
        # Base Features
        df['MK'] = df['M'] * df['K']; df['KN'] = df['K'] * df['N']
        df['MN'] = df['M'] * df['N']; df['MKN'] = df['M'] * df['K'] * df['N']
        
        # Logarithmic Features
        df['log_M'] = np.log1p(df['M'])
        df['log_K'] = np.log1p(df['K'])
        df['log_N'] = np.log1p(df['N'])
        df['log_MKN'] = np.log1p(df['MKN'])

        # Shape (Ratio) Features
        epsilon = 1e-9
        df['ratio_M_N'] = df['M'] / (df['N'] + epsilon)
        df['ratio_K_N'] = df['K'] / (df['N'] + epsilon)
        df['ratio_M_K'] = df['M'] / (df['K'] + epsilon)

        # Compute Density Feature
        df['compute_density'] = (2 * df['MKN']) / (df['MK'] + df['KN'] + df['MN'] + epsilon)
        # --- ADVANCED FEATURE ENGINEERING END ---

        X = df[self.features_]

        # Apply log transform to the target variables
        y_total_log = np.log1p(y_total)
        y_compute_log = np.log1p(y_compute)

        self.total_cycle_model_ = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.total_cycle_model_.fit(X, y_total_log)

        self.compute_cycle_model_ = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.compute_cycle_model_.fit(X, y_compute_log)

    @lru_cache(maxsize=1024)
    def predict(self, M: int, K: int, N: int, num_cores: int) -> Tuple[float, float]:
        if self.total_cycle_model_ is None:
            raise RuntimeError("Model has not been trained.")
        
        # --- ADVANCED FEATURE ENGINEERING START ---
        # Must create all features for the prediction vector, mirroring the training logic
        epsilon = 1e-9
        data = {
            'M': [M], 'K': [K], 'N': [N], 'num_cores': [num_cores],
            'MK': [M*K], 'KN': [K*N], 'MN': [M*N], 'MKN': [M*K*N],
            'log_M': [np.log1p(M)], 'log_K': [np.log1p(K)], 'log_N': [np.log1p(N)],
            'log_MKN': [np.log1p(M*K*N)],
            'ratio_M_N': [M / (N + epsilon)],
            'ratio_K_N': [K / (N + epsilon)],
            'ratio_M_K': [M / (K + epsilon)],
            'compute_density': [(2*M*K*N) / (M*K + K*N + M*N + epsilon)]
        }
        # --- ADVANCED FEATURE ENGINEERING END ---
        
        feature_vector = pd.DataFrame(data)[self.features_]
        
        predicted_log_total = self.total_cycle_model_.predict(feature_vector)[0]
        predicted_log_compute = self.compute_cycle_model_.predict(feature_vector)[0]
        
        predicted_total = np.expm1(predicted_log_total)
        predicted_compute = np.expm1(predicted_log_compute)
        
        predicted_compute = min(predicted_total, predicted_compute)
        return max(0.0, predicted_total), max(0.0, predicted_compute)
# ==============================================================================
# SECTION 2: SYSTEM & APPLICATION LAYERS
# ==============================================================================

class SystemLevelOperationModel: # 定义系统级操作模型类。
    """
    MODIFIED: Now holds references to both single-core and tensor-parallel models
    and dispatches calls to the appropriate one.
    (修改：现在持有对单核和张量并行模型的引用，并将调用分派给适当的模型。)
    """
    def __init__(self, config: NPUConfig, sccm: HybridCoreCostModel, tpcm: Optional[TensorParallelCostModel] = None): # 构造函数。
        self.config = config # 保存 NPU 硬件配置。
        self.sccm = sccm  # 保存单核成本模型 (Single-Core Cost Model)。
        self.tpcm = tpcm  # 保存张量并行成本模型 (Tensor-Parallel Cost Model)，可能为 None。

    def evaluate_gemm_tensor_parallel(self, M: int, K: int, N: int) -> Tuple[float, float]: # 评估张量并行 GEMM 操作的性能。
        """
        MODIFIED: Now uses the high-fidelity TensorParallelCostModel if available.
        (修改：现在如果张量并行模型可用，则使用它。)
        """
        if self.tpcm: # 检查张量并行模型是否存在。
            # High-fidelity path: Use the model trained on multi-core data.
            return self.tpcm.predict(M, K, N, self.config.num_cores) # 如果存在，则调用其 predict 方法进行高保真度预测。
        else: # 如果不存在。
            # Fallback path: Use the old logic as a less accurate approximation.
            print("WARNING: TensorParallelCostModel not provided. Falling back to less accurate single-core-based estimation for tensor parallelism.") # 打印警告信息。
            N_per_core = math.ceil(N / self.config.num_cores) # 将 N 维度划分到每个核心上。
            return self.sccm.predict(M, K, N_per_core) # 使用单核模型对划分后的子问题进行预测作为近似。

    def evaluate_gemm_data_parallel(self, M: int, K: int, N: int, num_ops: int) -> Tuple[float, float]: # 评估数据并行 GEMM 操作的性能。
        """
        UNCHANGED: This logic remains correct as it's based on single-core performance.
        (未改变：此逻辑仍然正确，因为它基于单核性能。)
        """
        if num_ops == 0: return 0.0, 0.0 # 如果没有操作，则返回0。
        # Predict the performance of a single operation on a single core.
        single_op_total_cycles, single_op_compute_cycles = self.sccm.predict(M, K, N) # 使用单核模型预测单个操作的性能。
        if single_op_total_cycles <= 0: return 0.0, 0.0 # 如果预测周期无效，则返回0。
        
        # Calculate how many waves of operations are needed.
        num_waves = math.ceil(num_ops / self.config.num_cores) # 计算需要多少“波”操作才能完成所有任务（每波最多处理 num_cores 个任务）。
        pipeline_interval = max(1.0, single_op_compute_cycles) # 定义流水线间隔，通常是单个操作的计算部分，确保不小于1。
        
        total_cycles = single_op_total_cycles + (num_waves - 1) * pipeline_interval # 应用流水线模型计算总延迟。
        total_compute_cycles = single_op_compute_cycles * num_waves # 总计算周期是单个操作的计算周期乘以波数。
        return total_cycles, total_compute_cycles # 返回计算出的总周期和总计算周期。

class ApplicationLayerEvaluator: # 定义应用层评估器类。
    """
    MODIFIED: The constructor now accepts both models and passes them to the
    SystemLevelOperationModel. The 'evaluate' method remains unchanged as the
    dispatch logic is handled one level below.
    (修改：构造函数现在接受两个模型并将其传递给 SystemLevelOperationModel。'evaluate' 方法保持不变，因为分派逻辑在下一层处理。)
    """
    def __init__(self, # 类的构造函数。
                 npu_config: NPUConfig, # 接收 NPU 硬件配置。
                 single_core_model: HybridCoreCostModel, # 接收单核性能模型。
                 tensor_parallel_model: Optional[TensorParallelCostModel] = None, # 接收可选的张量并行模型。
                 element_size_bytes: int = 2): # 接收数据元素大小（字节）。
        self.element_size_bytes = element_size_bytes # 保存元素大小。
        self.config = npu_config # 保存 NPU 配置。
        
        # Store references to the models if needed, though they are primarily used by SOM
        self.sccm = single_core_model # 保存单核模型引用。
        self.tpcm = tensor_parallel_model # 保存张量并行模型引用。
        
        # Pass both models to the SystemLevelOperationModel
        self.som = SystemLevelOperationModel(self.config, self.sccm, self.tpcm) # 创建一个系统级操作模型实例，并将两个性能模型传递给它。
        
        self.hidden_size, self.intermediate_size = 4096, 11008 # 定义 LLM 的隐藏层和中间层大小。
        self.num_heads, self.head_dim = 32, 128 # 定义 LLM 的注意力头数和头维度。

    def evaluate(self, # 定义统一的评估方法。
                 M: int, # 输入维度 M。
                 K: int, # 输入维度 K。
                 N: int, # 输入维度 N。
                 num_ops_parallel: int, # 并行操作数。
                 is_data_parallel: bool, # 是否为数据并行的标志。
                 op_type: str, # 操作类型（未直接使用，但为标准接口）。
                 num_channels_used: int, # 使用的通道数（未直接使用，但为标准接口）。
                 op_name: str # 操作名称。
                ) -> PerformanceResult: # 返回一个 PerformanceResult 对象。
        """
        UNCHANGED: This method's logic is sound. It correctly uses the 'is_data_parallel'
        flag to call the appropriate function in the SystemLevelOperationModel, which now
        handles the final dispatch to the correct underlying ML model.
        (未改变：此方法的逻辑是合理的。它正确使用 'is_data_parallel' 标志来调用 SystemLevelOperationModel 中适当的函数。)
        """
        total_cycles, compute_cycles = 0.0, 0.0 # 初始化总周期和计算周期为0。
        
        if is_data_parallel: # 如果是数据并行任务。
            total_cycles, compute_cycles = self.som.evaluate_gemm_data_parallel(M, K, N, num_ops_parallel) # 调用系统模型的相应方法进行评估。
        else: # 如果是张量并行任务。
            # (这是一个旧的、有缺陷的实现逻辑，已被下面的流水线模型取代)
            """
            # For a sequence of distinct tensor-parallel ops (e.g., Q, K, V projections)
            total_op_cycles, total_compute_cycles = 0.0, 0.0
            for _ in range(num_ops_parallel):
                # Each call models one large GEMM executed in a tensor-parallel fashion.
                op_total, op_compute = self.som.evaluate_gemm_tensor_parallel(M, K, N)
                total_op_cycles += op_total
                total_compute_cycles += op_compute
            total_cycles, compute_cycles = total_op_cycles, total_compute_cycles
            """
            # --- CORE FIX: Implement the pipeline model for Prefill ---
            if num_ops_parallel == 0: # 如果没有并行操作。
                return PerformanceResult(0,0,0,0) # 返回一个空的性能结果。

            # 1. Get the performance profile of a single large GEMM operation (e.g., for Q-projection).
            single_op_total, single_op_compute = self.som.evaluate_gemm_tensor_parallel(M, K, N) # 获取单个张量并行操作的性能。

            if num_ops_parallel > 1: # 如果有多个操作可以流水化。
                # 2. Define the pipeline interval. A good approximation is the compute portion
                #    of the operation, as I/O for the next op can overlap with the current op's compute.
                pipeline_interval = max(1.0, single_op_compute) # 定义流水线间隔为计算周期部分（确保不小于1）。
                
                # 3. Apply the pipeline formula.
                total_cycles = single_op_total + (num_ops_parallel - 1) * pipeline_interval # 应用流水线公式计算总延迟。
            else: # 如果只有一个操作。
                total_cycles = single_op_total # 总延迟就是单个操作的延迟。

            # Total compute cycles is the sum of compute portions of all operations.
            compute_cycles = single_op_compute * num_ops_parallel # 总计算周期是单个操作的计算周期乘以操作数。

        io_cycles = total_cycles - compute_cycles # I/O周期等于总周期减去计算周期。
        
        op_flops = 2.0 * M * K * N * num_ops_parallel # 计算总的浮点运算次数。
        # (旧的数据移动量计算方式)
        # bytes_read = (M * K + K * N) * num_ops_parallel * self.element_size_bytes
        # bytes_written = M * N * num_ops_parallel * self.element_size_bytes
        
        bytes_read = io_cycles * self.config.bytes_per_cycle # 根据I/O周期和每周期字节数估算读取的字节数。
        bytes_written = M * N * num_ops_parallel * self.element_size_bytes # 计算写入的字节数（输出结果的大小）。
        
        return PerformanceResult( # 创建并返回一个填充了所有性能指标的 PerformanceResult 对象。
            latency_cycles=total_cycles, # 总延迟周期。
            energy_joules=0.0, # 能量消耗（占位符）。
            input_bytes=int(bytes_read), # 输入字节数。
            output_bytes=int(bytes_written), # 输出字节数。
            path_taken=op_name, # 执行路径（这里用操作名代替）。
            io_cycles=io_cycles, # I/O 周期。
            compute_cycles=compute_cycles, # 计算周期。
            op_flops=op_flops, # 浮点运算次数。
            bytes_from_dram=int(bytes_read), # 从DRAM读取的字节数。
            bytes_to_dram=int(bytes_written) # 写入DRAM的字节数。
        )

# ==============================================================================
# SECTION 3: ANALYTICAL APPROXIMATION MODEL (FOR FAST SIMULATION)
# (此部分为快速但精度较低的分析模型，用于快速仿真)
# ==============================================================================
class ProfileProvider: # 定义一个 ProfileProvider 类，用于提供快速的性能剖析。
    def __init__(self, evaluator_npu: ApplicationLayerEvaluator, prompt_len: int, add_tokens: int, num_samples: int = 10): # 构造函数。
        self.evaluator_npu = evaluator_npu # 保存一个高保真 NPU 评估器实例，用于采样。
        self.prompt_len = prompt_len # 保存初始提示长度。
        self.add_tokens = add_tokens # 保存要生成的 token 数量。
        self.num_samples = min(num_samples, add_tokens) # 定义采样点数量。
        
        self.models = {} # 初始化一个字典来存储拟合出的线性模型。
        self.static_profiles = {} # 初始化一个字典来存储静态操作的性能剖析结果。
        
        # Define which ops are static vs dynamic
        self.op_info = { # 定义操作信息，区分静态和动态（性能是否依赖kv_cache_len）。
            "Attention::QKV_Projections":    {'is_dynamic': False, 'M': 1, 'K': 4096, 'N': 4096, 'num_ops': 3, 'is_dp': False},
            "Attention::Output_Projection":  {'is_dynamic': False, 'M': 1, 'K': 4096, 'N': 4096, 'num_ops': 1, 'is_dp': False},
            "FFN::Gate_Up_Projections":      {'is_dynamic': False, 'M': 1, 'K': 4096, 'N': 11008, 'num_ops': 2, 'is_dp': False},
            "FFN::Down_Projection":          {'is_dynamic': False, 'M': 1, 'K': 11008, 'N': 4096, 'num_ops': 1, 'is_dp': False},
            "Attention::Score_Computation_QK": {'is_dynamic': True, 'M': 1, 'K': 128, 'N_is_kv': True, 'num_ops': 32, 'is_dp': True},
            "Attention::Context_Computation_SV": {'is_dynamic': True, 'M': 128, 'K_is_kv': True, 'N': 128, 'num_ops': 32, 'is_dp': True},
        }
        
        print("INFO: Building fast profile provider by sampling performance curves...") # 打印提示信息。
        self._build_models() # 调用内部方法来构建快速模型。
        print("INFO: Profile provider ready.") # 打印准备就绪信息。

    def _build_models(self): # 定义构建快速模型的方法。
        # Sample static ops once
        for op_name, info in self.op_info.items(): # 遍历所有操作。
            if not info['is_dynamic']: # 如果是静态操作。
                # (注释：这里直接调用了高保真评估器，并将结果存起来)
                self.static_profiles[op_name] = self.evaluator_npu.evaluate(**{k:v for k,v in info.items() if k not in ['is_dynamic','is_dp', 'op_name']}, op_name=op_name, is_data_parallel=info['is_dp'], num_channels_used=self.evaluator_npu.config.num_channels, op_type="GEMM")

        # Sample dynamic ops across the kv_cache_len range
        start_len, end_len = self.prompt_len, self.prompt_len + self.add_tokens - 1 # 定义动态操作的采样范围。
        sample_points = np.linspace(start_len, end_len, self.num_samples, dtype=int) if self.num_samples > 1 else np.array([start_len]) # 生成采样点。

        for op_name, info in self.op_info.items(): # 遍历所有操作。
            if info['is_dynamic']: # 如果是动态操作。
                sampled_total_cycles, sampled_compute_cycles = [], [] # 初始化列表来存储采样结果。
                for kv_len in sample_points: # 在每个采样点进行评估。
                    K = kv_len if info.get('K_is_kv') else info['K'] # 根据操作信息确定 K 维度。
                    N = kv_len if info.get('N_is_kv') else info['N'] # 根据操作信息确定 N 维度。
                    
                    profile = self.evaluator_npu.evaluate(info['M'], K, N, info['num_ops'], info['is_dp'], "GEMV", self.evaluator_npu.config.num_channels, op_name) # 调用高保真评估器获取性能。
                    sampled_total_cycles.append(profile.latency_cycles) # 记录总周期。
                    sampled_compute_cycles.append(profile.compute_cycles) # 记录计算周期。

                # Fit linear models for total and compute cycles
                self.models[op_name] = { # 为该动态操作创建模型字典。
                    'total': np.poly1d(np.polyfit(sample_points, sampled_total_cycles, 1)), # 使用 numpy 拟合一个一次多项式（线性模型）来预测总周期。
                    'compute': np.poly1d(np.polyfit(sample_points, sampled_compute_cycles, 1)) # 拟合一个线性模型来预测计算周期。
                }

    def get_profile(self, op_name: str, kv_cache_len: int) -> PerformanceResult: # 定义获取性能剖析的方法。
        if op_name not in self.op_info: # 如果操作名未知。
            return PerformanceResult(0,0,0,0) # 返回空的性能结果。

        info = self.op_info[op_name] # 获取操作信息。
        if not info['is_dynamic']: # 如果是静态操作。
            return self.static_profiles[op_name] # 直接返回预先计算好的性能剖析结果。
        
        # For dynamic ops, use the fitted model to predict cycles
        total_cycles = self.models[op_name]['total'](kv_cache_len) # 使用拟合的线性模型预测总周期。
        compute_cycles = self.models[op_name]['compute'](kv_cache_len) # 使用拟合的线性模型预测计算周期。
        
        K = kv_cache_len if info.get('K_is_kv') else info['K'] # 确定 K 维度。
        N = kv_cache_len if info.get('N_is_kv') else info['N'] # 确定 N 维度。
        
        # Re-construct the full PerformanceResult object
        # (注意：此处为了方便返回完整的 PerformanceResult 对象，再次调用了高保真评估器。
        # 在一个纯粹的快速模式下，可能会直接用预测的周期数和其他估算值来构造对象。)
        return self.evaluator_npu.evaluate(info['M'], K, N, info['num_ops'], info['is_dp'], "GEMV", self.evaluator_npu.config.num_channels, op_name)
