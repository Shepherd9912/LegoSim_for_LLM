from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class NPUConfig:
    """ NPU系统的可配置硬件参数 """
    num_cores: int = 32
    num_channels: int = 32
    # --- 新增参数 ---
    # 描述一个NPU物理核心负责多少个DRAM通道。
    # 例如: num_cores=8, num_channels=32, channels_per_core=4
    # 这意味着8个物理核心中的每一个都驱动4个通道。
    channels_per_core: int = 1
    clock_ghz: float = 1.0
    macs_per_cycle_per_core: int = 128 * 128
    bytes_per_cycle_at_1ghz: float = 759.5
    effective_memory_bw_gb_s: float = field(init=False)
    bytes_per_cycle: float = field(init=False)
    # --- 新增的衍生参数 ---
    # 逻辑核心数，与通道数相等，代表了并行处理流的数量。
    num_logical_cores: int = field(init=False)

    def __post_init__(self):
        self.bytes_per_cycle = self.bytes_per_cycle_at_1ghz * self.clock_ghz
        self.effective_memory_bw_gb_s = self.bytes_per_cycle * self.clock_ghz
        # --- 新增的合法性检查 ---
        if self.num_channels % self.channels_per_core != 0:
            raise ValueError("num_channels must be divisible by channels_per_core")
        if self.num_cores * self.channels_per_core != self.num_channels:
            print(f"Warning: The NPU core to channel mapping is unbalanced. "
                  f"{self.num_cores} cores * {self.channels_per_core} ch/core != {self.num_channels} total channels. "
                  f"Performance model will assume num_cores = num_channels / channels_per_core.")
        
        # --- 关键衍生参数 ---
        # 逻辑核心数总是等于通道数，因为我们的并行模型是基于通道的。
        self.num_logical_cores = self.num_channels

@dataclass
class PIMConfig:
    """ PIM子系统的可配置硬件参数，用于DSE """
    """ channel信息和pim信息绑定在了一起，统称为pim config """
    total_channels: int
    clock_ghz: float = 2.0
    global_buffer_size_per_channel_kb: int = 256
    num_banks_per_channel: int = 16
    
    # PE Bandwidth - DSE参数的接口占位符
    # 为了简化当前版本，我们假设其影响已包含在MCM的标定延迟中
    pe_bandwidth_gb_s: float = 1024.0 # 假设值  # Bank * burst数

    # 主内存带宽，用于非GEMV的访存密集型算子
    main_memory_bw_per_channel_gb_s: float = 30.0

@dataclass
class PerformanceResult:
    latency_cycles: float
    energy_joules: float    
    input_bytes: int   # 这次计算需要从DRAM读取的总字节数。对于NPU，input_bytes = (M*K + K*N) * element_size
    output_bytes: int  # 这次计算需要写回DRAM的总字节数。对于NPU，output_bytes = M*N * element_size
    path_taken: Optional[str] = None
