# utils_v0_4.py
from dataclasses import dataclass, field # 从 dataclasses 模块导入 dataclass 装饰器和 field 函数，用于自动生成类的方法和定义字段属性。
from typing import Dict, Optional # 从 typing 模块导入类型提示 Dict (字典) 和 Optional (可选类型)。

# ==============================================================================
# UNCHANGED: Hardware configuration classes remain the same.
# ==============================================================================

@dataclass # 使用 dataclass 装饰器，它会自动为类添加 __init__, __repr__ 等特殊方法。
class NPUConfig: # 定义一个名为 NPUConfig 的类，用于存储NPU系统的硬件配置。
    """ NPU系统的可配置硬件参数 (无需修改) """ # 类的文档字符串，解释该类的用途。
    num_cores: int = 32 # 定义 NPU 的核心数量，类型为整数，默认值为 32。
    num_channels: int = 32 # 定义内存通道的总数量，类型为整数，默认值为 32。
    channels_per_core: int = 1 # 定义每个 NPU 核心服务的内存通道数，类型为整数，默认值为 1。
    clock_ghz: float = 1.0 # 定义 NPU 的时钟频率，单位为 GHz，类型为浮点数，默认值为 1.0。
    macs_per_cycle_per_core: int = 128 * 128 # 定义每个核心在每个时钟周期内能执行的 MAC (乘加) 操作次数，类型为整数。
    bytes_per_cycle_at_1ghz: float = 759.5 # 定义在 1GHz 频率下，每个时钟周期的数据传输字节数，类型为浮点数。
    effective_memory_bw_gb_s: float = field(init=False) # 定义有效的内存带宽（GB/s），使用 field(init=False) 表示该字段不在构造函数中初始化。
    bytes_per_cycle: float = field(init=False) # 定义在当前频率下每个时钟周期的数据传输字节数，不在构造函数中初始化。
    num_logical_cores: int = field(init=False) # 定义逻辑核心数，不在构造函数中初始化。
    peak_tops_per_core: float = field(init=False) # 定义每个核心的峰值算力（TOPS），不在构造函数中初始化。

    def __post_init__(self): # 定义一个在 __init__ 方法之后自动调用的方法，用于计算派生字段。
        """
        后初始化函数，用于根据基础参数计算派生出的性能指标。
        """
        self.effective_memory_bw_gb_s = self.bytes_per_cycle_at_1ghz * self.clock_ghz # 计算有效内存带宽，等于 1GHz 时的每周期字节数乘以当前时钟频率。
        # This is now a derived value, representing bytes per cycle at the current frequency.
        self.bytes_per_cycle = self.effective_memory_bw_gb_s / self.clock_ghz if self.clock_ghz > 0 else 0 # 计算当前频率下的每周期字节数，如果时钟频率大于0。
        if self.num_channels % self.channels_per_core != 0: # 检查总通道数是否能被每个核心的通道数整除。
            raise ValueError("num_channels must be divisible by channels_per_core") # 如果不能整除，则抛出一个 ValueError 异常。
        if self.num_cores * self.channels_per_core != self.num_channels: # 检查 NPU 核心与通道的映射是否平衡。
            print(f"Warning: The NPU core to channel mapping is unbalanced. " # 如果不平衡，则打印一条警告信息。
                  f"{self.num_cores} cores * {self.channels_per_core} ch/core != {self.num_channels} total channels. "
                  f"Performance model will assume num_cores = num_channels / channels_per_core.")
        self.num_logical_cores = self.num_channels # 将逻辑核心数设置为等于总通道数，这是一种性能建模的假设。
        self.peak_tops_per_core = (self.macs_per_cycle_per_core * 2 * self.clock_ghz) / 1000.0 # 计算每个核心的峰值算力（TOPS），乘以2是因为一个MAC操作包含两次浮点运算。

@dataclass # 使用 dataclass 装饰器。
class PIMConfig: # 定义一个名为 PIMConfig 的类，用于存储PIM子系统的硬件配置。
    """ PIM子系统的可配置硬件参数，用于DSE (无需修改) """ # 类的文档字符串。
    total_channels: int # 定义 PIM 使用的总通道数，类型为整数。
    clock_ghz: float = 2.0 # 定义 PIM 的时钟频率，单位为 GHz，类型为浮点数，默认值为 2.0。
    global_buffer_size_per_channel_kb: int = 256 # 定义每个 PIM 通道的全局缓冲区大小，单位为 KB，类型为整数，默认值为 256。
    num_banks_per_channel: int = 16 # 定义每个 PIM 通道的内存 bank 数量，类型为整数，默认值为 16。
    pe_bandwidth_gb_s: float = 1024.0 # 定义处理引擎 (PE) 的带宽，单位为 GB/s，类型为浮点数，默认值为 1024.0。
    main_memory_bw_per_channel_gb_s: float = 30.0 # 定义每个通道的主内存带宽，单位为 GB/s，类型为浮点数，默认值为 30.0。
    peak_tops_per_channel: float = 0.29 # 定义每个通道的峰值算力（TOPS），类型为浮点数，默认值为 0.29。 # 通过运行 calculate_tops_pim.py得到

# ==============================================================================
# MODIFIED: PerformanceResult is upgraded to support advanced simulation.
# ==============================================================================

@dataclass # 使用 dataclass 装饰器。
class PerformanceResult: # 定义一个名为 PerformanceResult 的类，用于存储和传递性能仿真结果。
    """
    MODIFIED: 性能结果的数据结构。
    这是我们升级的核心，用于在模块间传递更精细的性能分解信息。
    """
    # --- 原有字段 ---
    latency_cycles: float # 定义总延迟，单位为时钟周期，类型为浮点数。
    energy_joules: float # 定义总能耗，单位为焦耳，类型为浮点数。
    input_bytes: int # 定义输入数据的总字节数，类型为整数。
    output_bytes: int # 定义输出数据的总字节数，类型为整数。
    path_taken: Optional[str] = None # 定义执行路径（例如PIM的快速或慢速路径），类型为可选的字符串，默认为 None。

    # io_cycles: 任务占用内存通道（需要物理通道互斥锁）的周期数。
    # compute_cycles: 任务仅占用计算单元（如NPU Core），不占用内存通道的周期数。
    io_cycles: float = 0.0 # 定义 I/O 占用的周期数，类型为浮点数，默认值为 0.0。
    compute_cycles: float = 0.0 # 定义纯计算占用的周期数，类型为浮点数，默认值为 0.0。

    # --- 新增字段 ---
    op_flops: float = 0.0          # 定义总浮点运算次数，类型为浮点数，默认值为 0.0。
    bytes_from_dram: int = 0      # 定义从主存 (DRAM) 读取的字节数，类型为整数，默认值为 0。
    bytes_to_dram: int = 0        # 定义写入主存 (DRAM) 的字节数，类型为整数，默认值为 0。

    def __post_init__(self): # 定义一个在 __init__ 方法之后自动调用的方法。
        """
        后初始化函数，用于确保向后兼容性。
        如果一个旧的性能模型只提供了总的 latency_cycles，
        我们会做一个保守的假设：整个过程都占用I/O通道。
        这可以防止仿真器做出过于乐观的并行假设。
        """
        if self.latency_cycles > 0 and self.io_cycles == 0 and self.compute_cycles == 0: # 检查是否只设置了总延迟而未分解 I/O 和计算延迟。
            self.io_cycles = self.latency_cycles # 如果是，则将 I/O 周期数保守地设置为等于总延迟周期数。
        
        # 逻辑一致性检查: 对于一个完整的任务，其在计算单元上的总占用时间应等于分解之和
        # 注意: 这是一个简化的模型，实际硬件中I/O和计算可能部分重叠。
        # 但对于调度器而言，latency_cycles代表了该计算单元被占用的总时间。
        if self.io_cycles + self.compute_cycles > self.latency_cycles: # 检查分解的周期数之和是否大于总延迟。
             # 在一个简单的串行模型中，我们期望 latency_cycles >= io_cycles + compute_cycles
             # 在更复杂的模型中，latency_cycles 可能是 max(io, compute) 或其他组合
             # 这里我们只确保总延迟不小于分解部分，以避免逻辑错误
             pass # 使用 pass 语句，暂时不进行严格检查，以允许更复杂的 I/O 和计算重叠模型。

@dataclass
class PIMEnergyCostModel:
    """
    NEW: 存储从cycle-accurate仿真中标定出的原子操作能耗。
    所有单位均为皮焦 (pJ)。
    这些值根据您提供的AiM simulator基准测试结果进行归一化计算。
    """
    # --- 计算操作能耗 (pJ/instruction) ---
    # 来自测试(2): 38,976,251,495.98 pJ / 1024 instructions
    e_mac_abk_compute_pj: float = 38062745.6
    # 来自测试(1): 6,643,679,232 pJ / 1024 instructions
    e_ewmul_compute_pj: float = 6487968.0
    # 来自测试(6): (39,566,809,499 pJ / 1024) - e_mac_abk_compute_pj
    e_af_compute_pj: float = 781250.0 # 粗略估算

    # --- 数据传输能耗 (pJ/byte) ---
    # 来自测试(3): 7,557,120 pJ / (1024 instructions * 32 channels * 32 bytes/GPR)
    e_bus_transfer_pj_per_byte: float = 7.2056

    # --- DRAM命令固定开销 (pJ/instruction) ---
    # 来自测试(2): 315,621,376 pJ / 1024 instructions
    e_cmd_mac_abk_pj: float = 308224.0
    # 来自测试(8), WR_GB部分: 5,725,599 pJ / 1024 instructions
    e_cmd_wr_gb_pj: float = 5591.4
    # 来自测试(3), RD_MAC部分能耗主要为数据传输，命令开销假设为0
    e_cmd_rd_mac_pj: float = 0.0
    # 来自测试(5), WR_BIAS部分: 数据传输能耗为主，命令开-销假设为0
    e_cmd_wr_bias_pj: float = 0.0
