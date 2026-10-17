"""Independent enumeration: never asks the simulator which candidates survive."""

from collections import defaultdict
from math import log2
from .tasks import Mystery, Observation


def prediction(task: Mystery, candidate, query: str) -> str:
    if task.family == "machines":
        slope, offset, modulus = candidate.parameters
        accumulator = offset
        for _ in range(int(query)):
            accumulator += slope
        while accumulator >= modulus:
            accumulator -= modulus
        return str(accumulator)
    return next(value for key, value in candidate.parameters if key == query)


def compatible(task: Mystery, observations: list[Observation]) -> list[str]:
    return [
        c.label for c in task.candidates if all(prediction(task, c, o.query) == o.value for o in observations)
    ]


def information(task: Mystery, observations: list[Observation], query: str) -> dict:
    remaining = compatible(task, observations)
    if not remaining:
        raise ValueError("Contradictory evidence")
    partitions = defaultdict(list)
    for candidate in task.candidates:
        if candidate.label in remaining:
            partitions[prediction(task, candidate, query)].append(candidate.label)
    n = len(remaining)
    entropy_after = sum(len(group) / n * log2(len(group)) for group in partitions.values())
    return {
        "compatible": remaining,
        "outcomes": dict(partitions),
        "information_bits_uniform_prior": log2(n) - entropy_after,
        "decisive": n > 1 and all(len(group) == 1 for group in partitions.values()),
        "redundant": len(partitions) == 1,
    }
