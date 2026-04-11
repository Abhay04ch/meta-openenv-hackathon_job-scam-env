class HardTaskGrader:
    def grade(self, sample, requested_tools, scratchpad):
        gt = sample.get("ground_truth", {})
        w = sample.get("grading_logic", {})

        optimal_sequence = gt.get("optimal_action_sequence", [])
        tool_score = len(set(optimal_sequence) & set(requested_tools)) / max(1, len(optimal_sequence))
        trajectory = 1.0 if requested_tools == optimal_sequence else 0.7
        final_action = 1.0 if scratchpad.get("final_action") in gt.get("expected_final_actions", []) else 0.0
        evidence_quality = min(1.0, len(scratchpad.get("evidence_used", [])) / 3)
        efficiency = max(
            0.0,
            1 - max(0, len(requested_tools) - len(optimal_sequence)) / max(1, len(optimal_sequence)),
        )

        final_score = (
            w.get("tool_correctness_weight", 0.0) * tool_score
            + w.get("trajectory_weight", 0.0) * trajectory
            + w.get("final_action_weight", 0.0) * final_action
            + w.get("evidence_quality_weight", 0.0) * evidence_quality
            + w.get("efficiency_weight", 0.0) * efficiency
        )

        return {
            "final_score": round(final_score, 4)
        }