"""Pinned Wisp v2-75B frozen-opponent integration."""

from third_party.wisp75b.adapter import (
    WISP_POLICY_SHA256,
    WISP_SHARED_HEAD_SHA256,
    WISP_UPSTREAM_COMMIT,
    WispPolicyAdapter,
    WispStateTensors,
)

__all__ = [
    "WISP_POLICY_SHA256",
    "WISP_SHARED_HEAD_SHA256",
    "WISP_UPSTREAM_COMMIT",
    "WispPolicyAdapter",
    "WispStateTensors",
]
