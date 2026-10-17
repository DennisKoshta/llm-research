"""Finite-sample bounds for prospectively assigned whole-world new harm.

The default alpha allocates 0.025 equally over six NI endpoints. This helper
does not allocate the remaining study-wide error budget or prove independence.
Confirmation admission must verify semantic world uniqueness independently;
seed/family/index/revision identities alone cannot establish that property.
Families and task revisions are analyzed separately. Templates, cells, kinds,
and selected arms of a world share one indicator and one denominator entry.
Decision bounds permit independent worlds with heterogeneous Bernoulli risks:
the zero-event AM-GM bound or, for nonzero events, Hoeffding's inequality.
"""

from collections import defaultdict
import json
import math

NI_FAMILY_ALPHA = 0.025
NI_ENDPOINTS = 6
DEFAULT_ALPHA = NI_FAMILY_ALPHA / NI_ENDPOINTS
WORLD_FIELDS = ("task_revision", "seed", "family", "index")
SELECTOR_FIELDS = {
    "kind",
    "arm",
    "direction",
    "condition",
    "layer",
    "position",
    "dose_multiplier",
    "patch_mode",
    "donor_relation",
    "matched_interchange_condition",
}


def _probability(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite probability strictly between zero and one")


def _log_binomial_cdf(events, trials, probability):
    """Sum the lower tail in log space, avoiding underflow of (1-p)**n."""
    if probability == 0:
        return 0.0
    if probability == 1:
        return -math.inf
    log_p, log_q = math.log(probability), math.log1p(-probability)
    term = trials * log_q
    total = term
    for j in range(1, events + 1):
        term += math.log(trials - j + 1) - math.log(j) + log_p - log_q
        high, low = max(total, term), min(total, term)
        total = high + math.log1p(math.exp(low - high))
    return min(total, 0.0)


def exact_upper(events, worlds, alpha=DEFAULT_ALPHA):
    """Binomial Clopper-Pearson upper bound with confidence 1-alpha.

    Binomial exactness requires independent Bernoulli trials with common risk.
    This assumption need not hold for fixed world strata. Only the zero-event
    expression also bounds mean risk for independent nonidentical Bernoullis.
    For 0 < events < worlds, invert P_p[X <= events] = alpha.
    The returned bisection endpoint is on the conservative upper side.
    """
    if type(worlds) is not int or worlds < 1 or type(events) is not int or not 0 <= events <= worlds:
        raise ValueError("Require integer counts with worlds>=1 and 0<=events<=worlds")
    _probability(alpha, "alpha")
    if events == worlds:
        return 1.0
    if events == 0:
        return -math.expm1(math.log(alpha) / worlds)
    low, high = 0.0, 1.0
    target = math.log(alpha)
    for _ in range(128):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        if _log_binomial_cdf(events, worlds, middle) > target:
            low = middle
        else:
            high = middle
    return high


def _selector(selector):
    if not isinstance(selector, dict) or not selector or not set(selector) <= SELECTOR_FIELDS:
        raise ValueError("Require nonempty explicit arm selectors with supported metadata fields")
    if not set(selector) & {"arm", "direction", "condition"}:
        raise ValueError("An arm selector must identify an arm, direction, or condition")
    for value in selector.values():
        if value is not None and type(value) not in (str, int, float):
            raise ValueError("Selector values must be scalar metadata, not booleans or collections")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Selector numbers must be finite")


def _matches(row, selector):
    return all(key in row and row[key] == value for key, value in selector.items())


def _success(row, field):
    if row is None:
        return None
    # An observed invalid response is a known false outcome, not missing data.
    if type(row.get("valid")) is not bool or type(row.get("correct")) is not bool:
        return None
    if row["correct"] and not row["valid"]:
        return None
    return row[field]


def world_harm_noninferiority(
    assignments,
    compact_outcomes,
    *,
    control_selector,
    treatment_selectors,
    success_field,
    alpha=DEFAULT_ALPHA,
    margin=0.05,
):
    """Bound whole-world any-new-harm, which bounds mean paired success loss.

    Assignments are the complete prospectively selected source assignments,
    never filtered by observed control success. Each row needs trial_id, kind,
    and WORLD_FIELDS. Compact outcomes use prefix_id, trial_id, valid, correct,
    plus arm metadata. Selectors may restrict kind; every assigned source must
    have at least one applicable treatment selector. Extra audited arms may be
    supplied, but unknown source IDs and ambiguous/overlapping selectors fail.

    Missing/malformed treated outputs count as harm after an observed control
    success. Missing/malformed controls make the endpoint inconclusive without
    deleting a world. Incomplete selected measurement coverage also blocks NI,
    even when the conservative numerical bound would pass. The NI decision uses
    the independent-world zero-event AM-GM bound or nonzero-event Hoeffding bound,
    allowing heterogeneous risks across fixed strata. No bootstrap is used.
    """
    _probability(alpha, "alpha")
    _probability(margin, "margin")
    if success_field not in ("valid", "correct"):
        raise ValueError("NI success_field must be valid or correct")
    _selector(control_selector)
    if not isinstance(treatment_selectors, list) or not treatment_selectors:
        raise ValueError("Require a nonempty prospectively selected treatment arm set")
    for selector in treatment_selectors:
        _selector(selector)
    encodings = [json.dumps(s, sort_keys=True) for s in treatment_selectors]
    if len(encodings) != len(set(encodings)):
        raise ValueError("Treatment selectors must be distinct")
    sources, worlds = {}, defaultdict(list)
    for a in assignments:
        if not isinstance(a.get("trial_id"), str) or not a["trial_id"] or a["trial_id"] in sources:
            raise ValueError("Assigned source identities must be nonempty and unique")
        if any(key not in a for key in (*WORLD_FIELDS, "kind")):
            raise ValueError("Every assigned source requires explicit whole-world identity and kind")
        if any(type(a[key]) is not int or a[key] < 0 for key in ("seed", "index")):
            raise ValueError("World seed and index must be nonnegative integers")
        if any(not isinstance(a[key], str) or not a[key] for key in ("family", "task_revision", "kind")):
            raise ValueError("World revision, family, and kind must be explicit nonempty strings")
        sources[a["trial_id"]] = a
        worlds[tuple(a[key] for key in WORLD_FIELDS)].append(a)
    if not sources:
        raise ValueError("Nonempty manifest-assigned sources are required")
    if len({(a["task_revision"], a["family"]) for a in sources.values()}) != 1:
        raise ValueError("NI endpoints require separate task revisions and families")
    if any(
        "kind" in s and not any(a["kind"] == s["kind"] for a in sources.values()) for s in treatment_selectors
    ):
        raise ValueError("A selected treatment kind is absent from the assigned endpoint")
    by_source, measurement_ids = defaultdict(list), set()
    for row in compact_outcomes:
        if row.get("prefix_id") not in sources or not isinstance(row.get("trial_id"), str):
            raise ValueError("Unknown source or missing measurement identity")
        if row["trial_id"] in measurement_ids:
            raise ValueError("Duplicate measurement identity")
        measurement_ids.add(row["trial_id"])
        a = sources[row["prefix_id"]]
        if row.get("kind") != a["kind"] or any(key in row and row[key] != a[key] for key in WORLD_FIELDS):
            raise ValueError("Outcome source metadata disagrees with its assigned world")
        by_source[row["prefix_id"]].append(row)
    per_world = []
    for identity, assigned in sorted(worlds.items()):
        missing_controls = missing_treatments = expected_treatments = control_successes = 0
        harmed = False
        for a in assigned:
            rows = by_source[a["trial_id"]]
            controls = [r for r in rows if _matches(r, control_selector)]
            if len(controls) > 1:
                raise ValueError("Control selector matched multiple measurements for one source")
            baseline = controls[0] if controls else None
            control = _success(baseline, success_field)
            missing_controls += control is None
            control_successes += control is True
            selected = [s for s in treatment_selectors if "kind" not in s or s["kind"] == a["kind"]]
            if not selected:
                raise ValueError("Every assigned source requires a selected treatment arm")
            matched_ids = set()
            for selector in selected:
                matches = [r for r in rows if _matches(r, selector)]
                if len(matches) > 1:
                    raise ValueError("Treatment selector matched multiple measurements for one source")
                treated = matches[0] if matches else None
                if treated is not None:
                    key = treated["trial_id"]
                    if key in matched_ids or (baseline is not None and key == baseline["trial_id"]):
                        raise ValueError("Treatment selectors overlap each other or the control")
                    matched_ids.add(key)
                success = _success(treated, success_field)
                expected_treatments += 1
                missing_treatments += success is None
                harmed |= control is True and success is not True
        per_world.append(
            dict(
                **dict(zip(WORLD_FIELDS, identity)),
                assigned_sources=len(assigned),
                expected_treated_measurements=expected_treatments,
                control_successes=control_successes,
                unavailable_controls=missing_controls,
                unavailable_treated_measurements=missing_treatments,
                known_new_harm=harmed,
                new_harm=None if missing_controls else harmed,
            )
        )
    defined = sum(w["new_harm"] is not None for w in per_world)
    events = sum(w["new_harm"] is True for w in per_world)
    complete = not any(w["unavailable_controls"] or w["unavailable_treated_measurements"] for w in per_world)
    admitted = defined == len(worlds)
    binomial_upper = exact_upper(events, len(worlds), alpha) if admitted else None
    if not admitted:
        upper, bound_method = None, "unavailable_control"
    elif events == 0:
        # Independence and AM-GM give P(no harm) = product(1-p_i)
        # <= (1-mean(p_i))**n, without requiring a common Bernoulli risk.
        upper, bound_method = binomial_upper, "independent_heterogeneous_zero_event_am_gm"
    else:
        upper = min(1.0, events / len(worlds) + math.sqrt(-math.log(alpha) / (2 * len(worlds))))
        bound_method = "independent_heterogeneous_hoeffding"
    bound_passed = upper is not None and upper <= margin
    return dict(
        success_field=success_field,
        control_selector=control_selector,
        treatment_selectors=treatment_selectors,
        assigned_sources=len(sources),
        assigned_worlds=len(worlds),
        defined_worlds=defined,
        new_harm_worlds=events,
        unavailable_controls=sum(w["unavailable_controls"] for w in per_world),
        unavailable_treated_measurements=sum(w["unavailable_treated_measurements"] for w in per_world),
        expected_treated_measurements=sum(w["expected_treated_measurements"] for w in per_world),
        complete_selected_coverage=complete,
        control_admission_passed=admitted,
        alpha=alpha,
        confidence=1 - alpha,
        default_alpha_allocation="0.025 / 6 NI endpoints",
        noninferiority_margin=margin,
        comparison="upper_bound <= margin",
        observed_world_harm_fraction=events / len(worlds) if admitted else None,
        mean_paired_loss_upper_bound=upper,
        new_harm_probability_upper_bound=upper,
        bound_method=bound_method,
        binomial_clopper_pearson_upper_bound=binomial_upper,
        binomial_bound_assumption="Independent Bernoulli worlds with common harm probability; diagnostic only for heterogeneous fixed strata",
        bound_passed=bound_passed,
        noninferiority_passed=bool(complete and bound_passed),
        status="inconclusive_missing_control"
        if not admitted
        else "inconclusive_incomplete_coverage"
        if not complete
        else "noninferior"
        if bound_passed
        else "noninferiority_not_established",
        method="One indicator per assigned world; independent heterogeneous zero-event AM-GM bound, otherwise Hoeffding upper bound. Binomial Clopper-Pearson is reported separately and does not determine nonzero-event NI decisions.",
        estimand="Any newly failed selected response in a world; this bounds uniformly world-averaged paired success loss, including observed baseline failures in the denominator",
        admission_note="Caller must verify complete prospective assignment selection and unique semantic worlds; metadata identities alone do not prove independence. No study-wide multiplicity claim follows from this helper alone.",
        worlds=per_world,
    )
