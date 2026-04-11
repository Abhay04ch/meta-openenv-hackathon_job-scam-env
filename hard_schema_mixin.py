try:
    from .hard_tools import FIELD_MAP, HardTool
except ImportError:
    from hard_tools import FIELD_MAP, HardTool


class HardSchemaMixin:
    def _hard_reset_obs(self, episode):
        return {
            "episode_id": episode["episode_id"],
            "difficulty": episode["difficulty"],
            "query_type": episode["query_type"],
            "initial_signal": episode["initial_signal"],
            "allowed_tools": episode["allowed_tools"],
        }

    def _hard_request_field(self, tool_name, episode, requested_fields, scratchpad):
        field_name = FIELD_MAP[HardTool(tool_name)]
        requested_fields.add(field_name)
        scratchpad.setdefault("evidence_used", []).append(field_name)

        return {
            "field_name": field_name,
            "content": episode["environment_state"].get(field_name, {}),
        }

    def _hard_classify(self, label, scratchpad):
        scratchpad["final_action"] = label
        scratchpad[f"classified_as_{label}"] = True