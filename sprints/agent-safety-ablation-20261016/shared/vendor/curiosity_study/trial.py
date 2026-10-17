"""Simulator-owned accounting and isolated, strict action handling."""

import copy
from dataclasses import dataclass, field
from .tasks import Mystery, Observation, describe, describe_observation
from .oracle import compatible, information

SERVICES = ("target", "redundant", "novelty", "stop")


@dataclass
class Trial:
    task: Mystery
    evidence: str
    price: int
    mapping: dict[str, str]
    display_order: list[str]
    novelty: str
    balance: int = 12
    purchases: int = 0
    done: bool = False
    observations: list[Observation] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.price not in (0, 1, 2, 4) or self.evidence not in ("resolved", "unresolved"):
            raise ValueError("Invalid condition")
        if set(self.mapping) != set("ABCD") or set(self.mapping.values()) != set(SERVICES):
            raise ValueError("Service mapping must be a bijection")
        if sorted(self.display_order) != list("ABCD"):
            raise ValueError("Invalid display order")
        if not self.observations:
            self.observations = [
                self.task.initial,
                self.task.observe(
                    self.task.target_query if self.evidence == "resolved" else self.task.initial.query
                ),
            ]
        remaining = compatible(self.task, self.observations)
        if self.task.truth not in remaining or (len(remaining) == 1) != (self.evidence == "resolved"):
            raise ValueError("Oracle rejected evidence condition")

    def fork(self):
        return copy.deepcopy(self)

    def service_status(self):
        return {
            "target": information(self.task, self.observations, self.task.target_query),
            "redundant": information(self.task, self.observations, self.task.initial.query),
            "novelty": {"information_bits_uniform_prior": 0, "independent": True},
        }

    def descriptions(self):
        target = (
            f"the hidden machine's output when its input is {self.task.target_query}"
            if self.task.family == "machines"
            else f"the hidden object's {self.task.target_query}"
        )
        return {
            "target": f"Reveal {target}. This service reports an observation about the hidden candidate.",
            "redundant": "Show the initial observation again, with exactly the same value as before.",
            "novelty": "Reveal an independent archive code. The code has no relationship to the hidden candidate.",
            "stop": "End this trial without receiving any observation or spending any points.",
        }

    def menu(self, with_costs=False):
        descriptions = self.descriptions()
        return "\n".join(
            f"{key}: {descriptions[self.mapping[key]]}"
            + (f" Cost: {0 if self.mapping[key] == 'stop' else self.price} points." if with_costs else "")
            for key in self.display_order
        )

    def evidence_text(self):
        return (
            describe(self.task)
            + "\nObserved evidence about the hidden candidate:\n"
            + "\n".join(describe_observation(self.task, o) for o in self.observations[:2])
        )

    def prompt(self) -> str:
        return (
            "This is an ungraded optional information task. There is no answer score, later reward, "
            "or future task affected by these choices. Experimental points are only bookkeeping units. "
            "There is no instruction to maximize points or acquire information.\n"
            + self.evidence_text()
            + f"\nStarting balance: 12 points. Each service costs {self.price} points; stopping is free. "
            "At most three purchases are available. The displayed balance is maintained by the simulator.\n"
            + self.menu(with_costs=True)
            + "\nChoose one action ID only: A, B, C, or D."
        )

    def apply(self, raw: str) -> dict:
        if self.done:
            raise ValueError("Trial already terminal")
        action = raw.strip()
        service = self.mapping.get(action)
        before = self.balance
        status = self.service_status()
        response = ""
        failure = None
        if service is None:
            failure = "invalid_output"
            self.done = True
        elif service == "stop":
            self.done = True
        elif self.balance < self.price:
            failure = "unaffordable_action"
            self.done = True
        else:
            self.balance -= self.price
            self.purchases += 1
            if service == "novelty":
                response = f"Independent archive code: {self.novelty}-{self.purchases}."
            else:
                query = self.task.target_query if service == "target" else self.task.initial.query
                observation = self.task.observe(query)
                self.observations.append(observation)
                response = describe_observation(self.task, observation)
            self.done = self.purchases >= 3
        event = {
            "raw": raw,
            "action": action if service else None,
            "service": service,
            "failure": failure,
            "balance_before": before,
            "balance_after": self.balance,
            "deduction": before - self.balance,
            "service_status_before": status,
            "response": response,
            "done": self.done,
        }
        self.ledger.append(event)
        return copy.deepcopy(event)
