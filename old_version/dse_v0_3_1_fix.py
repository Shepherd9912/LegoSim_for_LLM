import random
import math
import copy
from typing import Dict, List, Any, Tuple, Optional

# 导入我们已经构建好的所有模块
from dataflow_v0_3_3 import (
    ExecutionPlan, OpCategory, ExecutionTarget, SplitDimension, PIMType,
    create_plan_from_configs
)
from evaluator_v0_3_1 import UnifiedEvaluator, ModelConfig

# ==============================================================================
# 1. 搜索空间定义 (Search Space Definition)
# ==============================================================================
class SearchSpace:
    TOTAL_CHANNELS = 32
    
    def __init__(self):
        self.hardware_space = {
            "pim.global_buffer_size_per_channel_kb": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
            "npu.num_cores": [2, 4, 8, 16, 32],
            # <<< 在这里指定你想要的PIM类型，例如只探索COMMON
            "topology.channels.pim_type": ["COMMON"],
            "interconnect.fixed_latency_cycles": list(range(50, 151, 10)),
        }
        self.explorable_ops = [
            "Attention::QKV_Projection",
            "Attention::Score_Computation_QK",
            "Attention::Context_Computation_SV",
            "Attention::Output_Projection",
            "FFN::Gate_Up_Projection",
            "FFN::Down_Projection",
        ]
        self.dataflow_op_space = {
            "target": ["NPU_ONLY", "PIM_ONLY", "SPLIT_EXECUTION"],
            "split_ratio_pim": (0.1, 0.9)
        }
    
    def get_random_hardware_config(self) -> Dict[str, Any]:
        # (此函数无需修改)
        hw_config = { "npu": {"clock_ghz": 2.0}, "pim": {"clock_ghz": 2.0} }
        num_cores = random.choice(self.hardware_space["npu.num_cores"])
        channels_per_core = self.TOTAL_CHANNELS // num_cores
        hw_config["npu"]["num_cores"] = num_cores
        hw_config["npu"]["channels_per_core"] = channels_per_core
        hw_config["npu"]["num_channels"] = self.TOTAL_CHANNELS
        hw_config["pim"]["total_channels"] = self.TOTAL_CHANNELS
        hw_config["pim"]["global_buffer_size_per_channel_kb"] = random.choice(self.hardware_space["pim.global_buffer_size_per_channel_kb"])
        hw_config["interconnect"] = { "fixed_latency_cycles": random.choice(self.hardware_space["interconnect.fixed_latency_cycles"]), "cycles_per_byte": 0.05 }
        pim_type = random.choice(self.hardware_space["topology.channels.pim_type"])
        hw_config["topology"] = {
            "npu_cores": [{"core_id": i, "serves_channel_ids": list(range(i * channels_per_core, (i + 1) * channels_per_core))} for i in range(num_cores)],
            "channels": [{"channel_id": i, "pim_type": pim_type} for i in range(self.TOTAL_CHANNELS)]
        }
        return hw_config

    def get_random_dataflow_config(self) -> Dict[str, Any]:
        # (此函数无需修改)
        df_config = {"decoding": {}, "prefill": {}}
        for op in ["Prefill_QKV", "Prefill_Attention", "Prefill_FFN"]: df_config["prefill"][op] = {"target": "NPU_ONLY"}
        for op in ["Input_RMSNorm", "Attention::Softmax", "Post_RMSNorm", "FFN::Activation(SiLU)"]: df_config["decoding"][op] = {"target": "MEMORY_BOUND"}
        for op in self.explorable_ops:
            target = random.choice(self.dataflow_op_space["target"])
            op_conf = {"target": target}
            if target == "SPLIT_EXECUTION":
                min_r, max_r = self.dataflow_op_space["split_ratio_pim"]
                op_conf["split_ratio_pim"] = random.uniform(min_r, max_r)
                op_conf["split_dimension"] = "N"
            df_config["decoding"][op] = op_conf
        return df_config

# ==============================================================================
# 2. DSE 引擎 (Simulated Annealing)
# ==============================================================================
class DSE_Engine:
    def __init__(self, model_config: ModelConfig, search_space: SearchSpace):
        self.model_config = model_config
        self.search_space = search_space
        self.best_plan: Optional[ExecutionPlan] = None
        self.best_cost: float = float('inf')
        self.history: List[Tuple[float, float]] = []

    def _get_neighbor_state(self, current_hw: Dict, current_df: Dict) -> Tuple[Dict, Dict]:
        neighbor_hw = copy.deepcopy(current_hw)
        neighbor_df = copy.deepcopy(current_df)

        if random.random() < 0.7:
            # (软件修改部分无需修改)
            op_to_change = random.choice(self.search_space.explorable_ops)
            op_conf = neighbor_df["decoding"][op_to_change]
            possible_mutations = ['change_target']
            if op_conf.get("target") == "SPLIT_EXECUTION":
                possible_mutations.append('tweak_split_ratio')
            chosen_mutation = random.choice(possible_mutations)
            if chosen_mutation == 'change_target':
                new_target = random.choice(self.search_space.dataflow_op_space["target"])
                op_conf["target"] = new_target
                if new_target == "SPLIT_EXECUTION":
                    min_r, max_r = self.search_space.dataflow_op_space["split_ratio_pim"]
                    op_conf["split_ratio_pim"] = random.uniform(min_r, max_r)
                    op_conf["split_dimension"] = "N"
                else:
                    op_conf.pop("split_ratio_pim", None)
                    op_conf.pop("split_dimension", None)
            elif chosen_mutation == 'tweak_split_ratio':
                min_r, max_r = self.search_space.dataflow_op_space["split_ratio_pim"]
                current_ratio = op_conf["split_ratio_pim"]
                new_ratio = current_ratio + random.uniform(-0.1, 0.1)
                op_conf["split_ratio_pim"] = max(min_r, min(max_r, new_ratio))
        else:
            # --- 改变硬件配置 ---
            hw_keys_to_change = ["npu.num_cores", "topology.channels.pim_type", "pim.global_buffer_size_per_channel_kb"]
            chosen_key = random.choice(hw_keys_to_change)

            if chosen_key == "npu.num_cores":
                current_cores = neighbor_hw["npu"]["num_cores"]
                possible_new_cores = [c for c in self.search_space.hardware_space["npu.num_cores"] if c != current_cores]
                if possible_new_cores:
                    new_cores = random.choice(possible_new_cores)
                    new_cpc = self.search_space.TOTAL_CHANNELS // new_cores
                    neighbor_hw["npu"]["num_cores"] = new_cores
                    neighbor_hw["npu"]["channels_per_core"] = new_cpc
                    neighbor_hw["topology"]["npu_cores"] = \
                        [{"core_id": i, "serves_channel_ids": list(range(i * new_cpc, (i + 1) * new_cpc))} for i in range(new_cores)]
            
            # --------------------- <<< MODIFICATION START >>> --------------------
            elif chosen_key == "topology.channels.pim_type":
                #
                # <<< OLD BUGGY LOGIC (FOR REFERENCE) >>>
                # This logic ignored the search space and created illegal states.
                # current_pim_type = neighbor_hw["topology"]["channels"][0]["pim_type"]
                # new_pim_type = "COMMON" if current_pim_type == "ATTENTION_FFN" else "ATTENTION_FFN"
                #
                
                # <<< NEW CORRECT LOGIC >>>
                # 1. 获取当前PIM类型
                current_pim_type = neighbor_hw["topology"]["channels"][0]["pim_type"]
                
                # 2. 从SearchSpace定义的“合法列表”中，选择一个与当前不同的新类型
                possible_new_types = [p for p in self.search_space.hardware_space["topology.channels.pim_type"] if p != current_pim_type]

                # 3. 只有在存在其他可选类型时，才进行修改
                if possible_new_types:
                    new_pim_type = random.choice(possible_new_types)
                    # 将新类型应用到所有通道
                    for channel_conf in neighbor_hw["topology"]["channels"]:
                        channel_conf["pim_type"] = new_pim_type
                # 如果possible_new_types为空 (例如，你只指定了["COMMON"])，
                # 则不会进行任何修改，自然也就不会产生非法状态。

            # ---------------------- <<< MODIFICATION END >>> ---------------------

            elif chosen_key == "pim.global_buffer_size_per_channel_kb":
                 # (此部分逻辑正确，无需修改)
                 current_gb_size = neighbor_hw["pim"]["global_buffer_size_per_channel_kb"]
                 possible_new_sizes = [s for s in self.search_space.hardware_space["pim.global_buffer_size_per_channel_kb"] if s != current_gb_size]
                 if possible_new_sizes:
                    new_gb_size = random.choice(possible_new_sizes)
                    neighbor_hw["pim"]["global_buffer_size_per_channel_kb"] = new_gb_size

        return neighbor_hw, neighbor_df

    def run(self, initial_temp: float, cooling_rate: float, max_iterations: int):
        # (无需修改)
        print("DSE Started: Initializing with a random plan...")
        current_hw = self.search_space.get_random_hardware_config()
        current_df = self.search_space.get_random_dataflow_config()
        plan = create_plan_from_configs(current_hw, current_df, "Initial Plan")
        evaluator = UnifiedEvaluator(plan, self.model_config)
        current_cost = evaluator.evaluate()['total_latency_ms']
        self.best_plan = plan
        self.best_cost = current_cost
        temp = initial_temp
        print(f"Initial Cost: {current_cost:.4f} ms, Initial Temp: {temp:.2f}\n")
        for i in range(max_iterations):
            neighbor_hw, neighbor_df = self._get_neighbor_state(current_hw, current_df)
            try:
                neighbor_plan = create_plan_from_configs(neighbor_hw, neighbor_df, f"Plan Iter {i}")
                evaluator = UnifiedEvaluator(neighbor_plan, self.model_config)
                neighbor_cost = evaluator.evaluate()['total_latency_ms']
            except (ValueError, KeyError) as e:
                print(f"Warning: Invalid plan generated at iter {i}. Skipping. Error: {e}")
                neighbor_cost = float('inf')
            delta = neighbor_cost - current_cost
            acceptance_prob = math.exp(-delta / temp) if temp > 0 else 0.0
            if delta < 0 or random.random() < acceptance_prob:
                current_hw, current_df, current_cost = neighbor_hw, neighbor_df, neighbor_cost
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_plan = create_plan_from_configs(current_hw, current_df, f"Best Plan at Iter {i}")
            temp *= cooling_rate
            self.history.append((current_cost, self.best_cost))
            if (i+1) % 10 == 0:
                print(f"Iter {i+1}/{max_iterations} | Temp: {temp:.2f} | "
                      f"Current Cost: {current_cost:.4f} ms | Best Cost: {self.best_cost:.4f} ms")
        print("\nDSE Finished.")
        return self.best_plan, self.best_cost

# ==============================================================================
# 3. 主执行块 (无需修改)
# ==============================================================================
if __name__ == '__main__':
    llama7b_task = ModelConfig(prompt_len=2048, generation_len=4096)
    space = SearchSpace()
    dse_engine = DSE_Engine(model_config=llama7b_task, search_space=space)
    INITIAL_TEMPERATURE = 100.0
    COOLING_RATE = 0.99
    MAX_ITERATIONS = 1000 # 增加迭代次数以获得更稳定的结果
    best_plan_found, best_latency_found = dse_engine.run(
        initial_temp=INITIAL_TEMPERATURE,
        cooling_rate=COOLING_RATE,
        max_iterations=MAX_ITERATIONS
    )
    # ... (打印部分无需修改) ...
    print("\n" + "="*80)
    print("           >>> Optimal Plan Found by DSE (with 32-Channel Constraint) <<<")
    print("="*80)
    print(f"Description: {best_plan_found.description}")
    print(f"Minimum Latency Found: {best_latency_found:.4f} ms\n")
    print("--- Optimal Hardware Configuration ---")
    print(f"  - NPU: {best_plan_found.npu_config.num_cores} cores, {best_plan_found.npu_config.channels_per_core} ch/core -> Total Channels: {best_plan_found.npu_config.num_channels}")
    print(f"  - PIM Arch: {best_plan_found.channel_topology[0].pim_type.name}")
    print(f"  - PIM GB Size: {best_plan_found.pim_config.global_buffer_size_per_channel_kb} KB")
    print(f"  - Interconnect Latency: {best_plan_found.interconnect_model.fixed_latency_cycles} cycles\n")
    print("--- Optimal Decoding Dataflow ---")
    for stage in best_plan_found.decoding_dataflow.stages:
        print(f"  - Stage: {stage.stage_description}")
        for op_name, mapping in stage.operator_mappings.items():
            print(f"    - {op_name:<35}: {mapping}")
    print("="*80)
