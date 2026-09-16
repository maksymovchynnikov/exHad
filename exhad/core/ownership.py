"""Channel-level ownership cut of a complete decay forest.

Rate tables use channel-level particles, events the fully decayed record.
Descending from every primary root, the first canonical particle on a branch
owns it: an ``eta`` or neutral kaon is kept even when only its decay products
are terminal, while prompt resonances (``rho``, ``omega``, ``phi``, ``K*``)
are unfolded.  A branch that reaches no canonical particle raises; it is never
relabelled as an exclusive channel.
"""

from __future__ import annotations
from typing import NamedTuple
from .width_inputs import final_state_key_from_pdgs

CANONICAL_OWNERSHIP_PDGS = frozenset({
    22,
    -211, 111, 211,
    -321, -311, 130, 310, 311, 321,
    221, 331,
    -2212, -2112, 2112, 2212,
})


class OwnershipCutError(ValueError):
    """A complete graph cannot be reduced to the ownership convention."""


class OwnershipCut(NamedTuple):
    pdg_ids: tuple[int, ...]  # in node order
    final_state_key: str


def derive_ownership_cut(event, *, expand_pdgs: frozenset[int] = frozenset()) -> OwnershipCut:
    """First canonical particle (not in ``expand_pdgs``) on every root-to-leaf path."""

    children: list[list[int]] = [[] for _ in event.nodes]
    for node in event.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node.node_id)
    selected: list[int] = []

    def descend(node_id: int) -> None:
        pdg_id = event.nodes[node_id].pdg_id

        if pdg_id in CANONICAL_OWNERSHIP_PDGS and pdg_id not in expand_pdgs:
            selected.append(node_id)
            return

        if not children[node_id]:
            raise OwnershipCutError(
                "terminal ancestry branch has no canonical ownership particle: "
                f"node {node_id}, PDG {pdg_id}")

        for child_id in children[node_id]:
            descend(child_id)

    for node in event.nodes:
        if node.parent_id is None:
            descend(node.node_id)
    selected.sort()
    pdgs = tuple(event.nodes[node_id].pdg_id for node_id in selected)
    return OwnershipCut(pdgs, final_state_key_from_pdgs(pdgs))


def ownership_key(event) -> str | None:
    """The cut's final-state key, or None for a branch outside the convention."""

    try:
        return derive_ownership_cut(event).final_state_key
    except OwnershipCutError:
        return None
