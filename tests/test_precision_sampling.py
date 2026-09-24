from fractions import Fraction
import unittest

import numpy as np

from sleepedf.precision_sampling import (PlanningPopulation, location_shifts,
                                        resampled_population, select_matrices,
                                        selected_participants)


def fixture():
    sc = [f"SC:{i:02d}" for i in range(46)]
    st = [f"ST:{i:02d}" for i in range(14)]
    blocks = []
    offset = 0
    for size in [3] * 14 + [2] * 2:
        blocks.append(tuple(sc[offset:offset + size])); offset += size
    offset = 0
    for size in [4, 4, 3, 3]:
        blocks.append(tuple(st[offset:offset + size])); offset += size
    population = PlanningPopulation(tuple(sc + st), tuple(blocks))
    matrices = np.zeros((4, 60, 5, 5), dtype=np.int64)
    for system in range(4):
        for person in range(60):
            correct = 60 + ((person * 7 + system * 11) % 40)
            matrices[system, person] = np.eye(5, dtype=np.int64) * correct
            matrices[system, person] += np.roll(np.eye(5, dtype=np.int64), 1, axis=1) * (100 - correct)
    return population, matrices


class PrecisionSamplingTests(unittest.TestCase):
    def test_shared_people_cohort_counts_and_chunk_order(self):
        population, matrices = fixture()
        sequential = {i: selected_participants(population, i) for i in range(27)}
        resumed = {i: selected_participants(population, i) for i in reversed(range(27))}
        self.assertEqual(sequential, resumed)
        for i, indices in sequential.items():
            sampled, ids = select_matrices(matrices, population, i)
            self.assertEqual(len(set(ids)), 20)
            self.assertEqual(sum(pid.startswith("SC:") for pid in ids), 16)
            self.assertEqual(tuple(sorted(ids)), ids)
            for block in population.original_indices():
                self.assertEqual(len(set(indices).intersection(block)), 1)
            np.testing.assert_array_equal(sampled, matrices[:, indices])

    def test_reference_rng_and_outer_repetitions(self):
        population, _ = fixture()
        blocks = population.original_indices()
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260924, 71, 1, 0, 7])))
        expected = tuple(sorted(int(rng.choice(block)) for block in blocks))
        self.assertEqual(selected_participants(population, 7), expected)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260924, 71, 2, 6, 0])))
        expected_outer = tuple(tuple(int(x) for x in rng.choice(block, len(block), replace=True)) for block in blocks)
        observed = resampled_population(population, 6)
        self.assertEqual(observed, expected_outer)
        self.assertTrue(any(len(set(block)) < len(block) for block in observed))
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260924, 71, 3, 6, 19])))
        self.assertEqual(selected_participants(population, 19, population_index=6),
                         tuple(sorted(int(rng.choice(block)) for block in expected_outer)))

    def test_fraction_center_and_strongest_comparator_with_repeated_people(self):
        population, matrices = fixture()
        for population_index in (None, 0, 6):
            blocks = (population.original_indices() if population_index is None else
                      resampled_population(population, population_index))
            centers = []
            for system in matrices:
                center = np.empty((5, 5), dtype=object)
                for row in range(5):
                    for col in range(5):
                        center[row, col] = sum(Fraction(sum(int(system[i, row, col]) for i in block), len(block)) for block in blocks)
                terms = []
                for label in range(5):
                    denom = sum(center[label, :]) + sum(center[:, label])
                    terms.append(2 * center[label, label] / denom if denom else Fraction(0))
                centers.append(sum(terms) / 5)
            for target in (Fraction(3, 100), Fraction(1, 25)):
                shifts = location_shifts(matrices, population, candidate_count=2, target=target,
                                         population_index=population_index)
                for score, shift in zip(centers[:2], shifts):
                    self.assertEqual(score + shift - max(centers[2:]), target)

    def test_rejects_wrong_participants_blocks_axis_and_rng_ranges(self):
        population, matrices = fixture()
        with self.assertRaises(ValueError):
            PlanningPopulation(population.participant_ids[::-1], population.blocks)
        blocks = list(population.blocks); blocks[1] = blocks[0]
        with self.assertRaises(ValueError):
            PlanningPopulation(population.participant_ids, tuple(blocks))
        with self.assertRaises(ValueError):
            PlanningPopulation(population.participant_ids, population.blocks[::-1])
        with self.assertRaises(ValueError):
            select_matrices(matrices[:, :-1], population, 0)
        for audit, outer in ((4000, None), (-1, None), (True, None), (256, 0), (0, 100), (0, -1), (0, True)):
            with self.assertRaises(ValueError):
                selected_participants(population, audit, population_index=outer)
        with self.assertRaises(ValueError):
            location_shifts(matrices, population, candidate_count=2, target=Fraction(1, 50))


if __name__ == "__main__":
    unittest.main()
