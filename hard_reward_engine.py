class HardRewardEngine:
    def compute(self, sample, requested_tools, scratchpad):
        reward = 0.0

        reward_logic = sample.get("reward_logic", {})
        for rule in reward_logic.get("dense_rewards", []):
            if scratchpad.get(rule.get("condition"), False):
                reward += rule.get("reward", 0.0)

        for rule in reward_logic.get("sparse_rewards", []):
            if scratchpad.get(rule.get("condition"), False):
                reward += rule.get("reward", 0.0)

        for rule in reward_logic.get("penalties", []):
            if scratchpad.get(rule.get("condition"), False):
                reward += rule.get("reward", 0.0)

        optimal_len = len(sample.get("ground_truth", {}).get("optimal_action_sequence", []))
        extra = max(0, len(requested_tools) - optimal_len)
        efficiency_penalty = reward_logic.get("efficiency_penalty_per_extra_tool", 0.0)
        reward -= extra * efficiency_penalty

        return round(reward, 4)