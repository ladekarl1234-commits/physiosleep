"""Pure participant sampling for the registered precision design.

This module reads no files and establishes no experiment eligibility. Its caller
must verify the frozen split, prediction ancestry and readiness prerequisites.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .evaluation import macro_f1_fraction
from .precision import block_standardized, validate_matrices


@dataclass(frozen=True)
class PlanningPopulation:
    participant_ids: tuple[str, ...]
    blocks: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        ids = self.participant_ids
        if (type(ids) is not tuple or len(ids) != 60 or
                any(type(pid) is not str or len(pid) != 5 or
                    pid[:3] not in ("SC:", "ST:") or not pid[3:].isascii() or
                    not pid[3:].isdigit() for pid in ids) or
                tuple(sorted(set(ids))) != ids):
            raise ValueError("Require 60 unique, canonical development participant IDs")
        if sum(pid.startswith("SC:") for pid in ids) != 46:
            raise ValueError("Require the frozen 46 SC and 14 ST development composition")
        if type(self.blocks) is not tuple or len(self.blocks) != 20:
            raise ValueError("Require 20 ordered allocation blocks")
        members = []
        for index, block in enumerate(self.blocks):
            cohort = "SC:" if index < 16 else "ST:"
            if (type(block) is not tuple or not block or
                    any(type(pid) is not str or not pid.startswith(cohort) for pid in block) or
                    tuple(sorted(set(block))) != block):
                raise ValueError("Allocation blocks must be canonical, nonempty and cohort-specific")
            members.extend(block)
        if sorted(members) != list(ids):
            raise ValueError("Allocation blocks must partition development participants exactly once")

    def original_indices(self) -> tuple[tuple[int, ...], ...]:
        lookup = {pid: index for index, pid in enumerate(self.participant_ids)}
        return tuple(tuple(lookup[pid] for pid in block) for block in self.blocks)


def _generator(purpose: int, population_index: int, audit_index: int) -> np.random.Generator:
    if (type(purpose) is not int or purpose not in (1, 2, 3) or
            type(population_index) is not int or population_index < 0 or
            type(audit_index) is not int or audit_index < 0):
        raise ValueError("Invalid registered planning RNG indices")
    entropy = [20260924, 71, purpose, population_index, audit_index]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def resampled_population(population: PlanningPopulation, population_index: int) -> tuple[tuple[int, ...], ...]:
    """Retain every block size; repetitions represent sampled people, not new IDs."""
    if type(population_index) is not int or not 0 <= population_index < 100:
        raise ValueError("Population index must be in the registered 0..99 range")
    rng = _generator(2, population_index, 0)
    return tuple(tuple(int(value) for value in rng.choice(block, size=len(block), replace=True))
                 for block in population.original_indices())


def selected_participants(population: PlanningPopulation, audit_index: int,
                          *, population_index: int | None = None) -> tuple[int, ...]:
    """Select one person per block, then canonicalize before the paired bootstrap."""
    if type(audit_index) is not int:
        raise ValueError("Audit index must be an integer")
    if population_index is None:
        if not 0 <= audit_index < 4000:
            raise ValueError("Primary audit index must be in the registered 0..3999 range")
        blocks = population.original_indices()
        rng = _generator(1, 0, audit_index)
    else:
        if not 0 <= audit_index < 256:
            raise ValueError("Population audit index must be in the registered 0..255 range")
        blocks = resampled_population(population, population_index)
        rng = _generator(3, population_index, audit_index)
    selected = tuple(sorted(int(rng.choice(block)) for block in blocks))
    if len(set(selected)) != 20:
        raise ValueError("Block selection unexpectedly repeated a participant")
    return selected


def select_matrices(matrices: np.ndarray, population: PlanningPopulation, audit_index: int,
                    *, population_index: int | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply one shared participant selection to every candidate and comparator."""
    matrices = validate_matrices(matrices)
    if matrices.shape[1] != len(population.participant_ids):
        raise ValueError("Matrices must follow the canonical development participant axis")
    indices = selected_participants(population, audit_index, population_index=population_index)
    return matrices[:, indices], tuple(population.participant_ids[index] for index in indices)


def location_shifts(matrices: np.ndarray, population: PlanningPopulation, *,
                    candidate_count: int, target: Fraction,
                    population_index: int | None = None) -> tuple[Fraction, ...]:
    """Return hypothetical shifts; these never replace observed prerequisite tests."""
    matrices = validate_matrices(matrices)
    if (matrices.shape[1] != len(population.participant_ids) or
            type(candidate_count) is not int or not 1 <= candidate_count < len(matrices) or
            not isinstance(target, Fraction) or target not in (Fraction(3, 100), Fraction(1, 25))):
        raise ValueError("Require canonical matrices, candidates, comparators and target .03 or .04")
    if population_index is None:
        center = block_standardized(matrices, [list(block) for block in population.original_indices()])
    else:
        blocks = resampled_population(population, population_index)
        # Expand sampled copies onto a temporary axis so partition validation
        # still checks each copy exactly once; never deduplicate repeated people.
        indices = [index for block in blocks for index in block]
        offsets = np.cumsum([0] + [len(block) for block in blocks]).tolist()
        copy_blocks = [list(range(start, end)) for start, end in zip(offsets[:-1], offsets[1:])]
        center = block_standardized(matrices[:, indices], copy_blocks)
    scores = [macro_f1_fraction(matrix) for matrix in center]
    strongest = max(scores[candidate_count:])
    return tuple(target - (score - strongest) for score in scores[:candidate_count])
