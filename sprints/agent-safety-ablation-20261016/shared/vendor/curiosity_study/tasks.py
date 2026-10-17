"""Finite synthetic tasks. The simulator and oracle use independent rule evaluators."""

from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class Candidate:
    label: str
    parameters: tuple


@dataclass(frozen=True)
class Observation:
    query: str
    value: str


@dataclass(frozen=True)
class Mystery:
    item_id: str
    family: str
    template: str
    seed: int
    candidates: tuple[Candidate, ...]
    truth: str
    initial: Observation
    target_query: str

    def observe(self, query: str) -> Observation:
        candidate = next(c for c in self.candidates if c.label == self.truth)
        if self.family == "machines":
            a, b, modulus = candidate.parameters
            value = str((a * int(query) + b) % modulus)
        else:
            value = dict(candidate.parameters)[query]
        return Observation(query, value)

    def as_dict(self):
        return asdict(self)


def generate(family: str, index: int, seed: int = 20261006, split: str = "development") -> Mystery:
    if family not in {"machines", "objects"} or split not in {"development", "evaluation"}:
        raise ValueError("Unknown family or split")
    if index < 0:
        raise ValueError("index must be nonnegative")
    rng = random.Random(f"{seed}:{split}:{family}:{index}")
    labels = list("JKLM")
    rng.shuffle(labels)
    template = f"{split}-{index % 2}"
    if family == "machines":
        x0 = index % 2 if split == "development" else 2 + index % 2
        y0 = rng.randrange(7)
        slopes = rng.sample(range(1, 7), 4)
        candidates = tuple(Candidate(label, (a, (y0 - a * x0) % 7, 7)) for label, a in zip(labels, slopes))
        initial = Observation(str(x0), str(y0))
        query = str((x0 + rng.randrange(1, 7)) % 7)
    else:
        colors = rng.sample(["red", "blue", "green", "gold"], 4)
        shapes = rng.sample(["round", "square", "oval", "triangular"], 4)
        material = rng.choice(["wood", "metal", "glass"])
        candidates = tuple(
            Candidate(label, (("color", color), ("shape", shape), ("material", material)))
            for label, color, shape in zip(labels, colors, shapes)
        )
        initial = Observation("material", material)
        query = ("color", "shape")[index % 2] if split == "development" else "identity"
        if split == "evaluation":
            # A held-out query format, not just new random seeds.
            candidates = tuple(
                Candidate(
                    c.label,
                    c.parameters
                    + (("identity", f"{dict(c.parameters)['color']}-{dict(c.parameters)['shape']}"),),
                )
                for c in candidates
            )
    truth = rng.choice(labels)
    return Mystery(f"{split}-{family}-{index:04d}", family, template, seed, candidates, truth, initial, query)


def describe(task: Mystery) -> str:
    if task.family == "machines":
        rows = [
            f"{c.label}: output = ({c.parameters[0]} * input + {c.parameters[1]}) modulo 7"
            for c in task.candidates
        ]
        queries = sorted({int(task.initial.query), int(task.target_query)})
        table = ["Candidate | " + " | ".join(f"output at input {x}" for x in queries)]
        for c in task.candidates:
            a, b, modulus = c.parameters
            table.append(c.label + " | " + " | ".join(str((a * x + b) % modulus) for x in queries))
        return (
            "One of these machines generated the observations. Inputs are integers 0 through 6.\n"
            + "\n".join(rows)
            + "\nThe following lookup table gives the exact outputs for both inputs used in this task. "
            "You may match observations directly to this table; no arithmetic is needed.\n" + "\n".join(table)
        )
    rows = [
        f"{c.label}: " + ", ".join(f"{key}={value}" for key, value in c.parameters) for c in task.candidates
    ]
    return "One object from this complete table generated the observations.\n" + "\n".join(rows)


def describe_observation(task: Mystery, observation: Observation) -> str:
    if task.family == "machines":
        return f"Input {observation.query} produced output {observation.value}."
    return f"Observed {observation.query}: {observation.value}."
