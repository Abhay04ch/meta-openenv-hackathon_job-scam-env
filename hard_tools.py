from enum import Enum


class HardTool(str, Enum):
    REQUEST_SENDER_PROFILE = "request_sender_profile"
    REQUEST_ORGANIZATION_PROFILE = "request_organization_profile"
    REQUEST_SHARED_CHANNEL_HISTORY = "request_shared_channel_history"
    REQUEST_PRIVATE_CONVERSATION_HISTORY = "request_private_conversation_history"
    REQUEST_CANDIDATE_INTERACTION_HISTORY = "request_candidate_interaction_history"
    REQUEST_EXTERNAL_MARKET_SIGNALS = "request_external_market_signals"
    REQUEST_ATTACHED_ARTIFACTS = "request_attached_artifacts"
    REQUEST_TEMPORAL_CONTEXT = "request_temporal_context"
    CLASSIFY = "classify"


FIELD_MAP = {
    HardTool.REQUEST_SENDER_PROFILE: "sender_profile",
    HardTool.REQUEST_ORGANIZATION_PROFILE: "organization_profile",
    HardTool.REQUEST_SHARED_CHANNEL_HISTORY: "shared_channel_history",
    HardTool.REQUEST_PRIVATE_CONVERSATION_HISTORY: "private_conversation_history",
    HardTool.REQUEST_CANDIDATE_INTERACTION_HISTORY: "candidate_interaction_history",
    HardTool.REQUEST_EXTERNAL_MARKET_SIGNALS: "external_market_signals",
    HardTool.REQUEST_ATTACHED_ARTIFACTS: "attached_artifacts",
    HardTool.REQUEST_TEMPORAL_CONTEXT: "temporal_context",
}