"""Deterministic domain-separated logical event seeds (formulas and tags define the random streams)."""

from __future__ import annotations
from typing import NamedTuple

MASK64 = (1 << 64) - 1

_GAMMA = 0x9E3779B97F4A7C15

_OWNER_TAG = 0x243F6A8885A308D3

_LEAF_TAG = 0x13198A2E03707344

_TOPOLOGY_TAG = 0x082EFA98EC4E6C89

_PROVIDER_TAG = 0xA4093822299F31D0

_ATTEMPT_TAG = 0x452821E638D01377

_SEAM_TAG = 0xBE5466CF34E90C6C

_HANDOFF_TAG = 0xC0AC29B7C97C50DD


def splitmix64(value: int) -> int:
    """One fixed bijective 64-bit mixer (inputs taken modulo 2^64)."""

    z = (value + _GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


class EventSeeds(NamedTuple):
    event_root: int
    owner: int
    leaf: int
    topology: int
    provider: int


def derive_event_seeds(master_seed: int, event_index: int) -> EventSeeds:
    """Derive event-order-independent streams from immutable event identity."""

    event_root = splitmix64((master_seed + _GAMMA * (event_index + 1)) & MASK64)
    return EventSeeds(
        event_root,
        splitmix64(event_root ^ _OWNER_TAG),
        splitmix64(event_root ^ _LEAF_TAG),
        splitmix64(event_root ^ _TOPOLOGY_TAG),
        splitmix64(event_root ^ _PROVIDER_TAG),
    )


def derive_provider_attempt_seed(seeds: EventSeeds, attempt: int) -> int:
    """Same-route provider seed of one retry attempt (attempt 0 is the provider seed)."""

    if attempt == 0:
        return seeds.provider
    return splitmix64(seeds.event_root ^ _ATTEMPT_TAG ^ splitmix64(attempt))


def derive_seam_seed(seeds: EventSeeds) -> int:
    return splitmix64(seeds.event_root ^ _SEAM_TAG)


def derive_handoff_seed(seeds: EventSeeds) -> int:
    return splitmix64(seeds.event_root ^ _HANDOFF_TAG)
