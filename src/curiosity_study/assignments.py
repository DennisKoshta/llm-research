"""Freeze assignments before inference; full price/evidence crossing per item."""

import itertools
import random
from .tasks import generate
from .trial import SERVICES, Trial


def make_assignments(items_per_family=100, seed=20261006, split="development"):
    if items_per_family < 1:
        raise ValueError("Need at least one item per family")
    permutations = list(itertools.permutations(SERVICES))
    assignments = []
    for family in ("machines", "objects"):
        rotation = random.Random(f"mapping-block:{seed}:{split}:{family}").randrange(24)
        for index in range(items_per_family):
            rng = random.Random(f"mapping:{seed}:{split}:{family}:{index}")
            for price_index, price in enumerate((0, 1, 2, 4)):
                # The same mapping/order in paired resolved/unresolved branches.
                mapping = dict(zip("ABCD", permutations[(index + 6 * price_index + rotation) % 24]))
                order = list("ABCD")
                rng.shuffle(order)
                for evidence in ("resolved", "unresolved"):
                    assignments.append(
                        {
                            "trial_id": f"{split}-{family}-{index:04d}-{evidence}-p{price}",
                            "family": family,
                            "index": index,
                            "split": split,
                            "seed": seed,
                            "evidence": evidence,
                            "price": price,
                            "mapping": mapping.copy(),
                            "display_order": order.copy(),
                            "sampling_seed": rng.randrange(2**31),
                            "novelty": f"{rng.randrange(100000, 999999)}",
                        }
                    )
    random.Random(f"order:{seed}").shuffle(assignments)
    return assignments


def instantiate(assignment):
    task = generate(assignment["family"], assignment["index"], assignment["seed"], assignment["split"])
    return Trial(
        task,
        assignment["evidence"],
        assignment["price"],
        assignment["mapping"].copy(),
        assignment["display_order"].copy(),
        assignment["novelty"],
    )
