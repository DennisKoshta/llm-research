"""Reproduce the authored rigid-template corpus; no model calls or inference."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [
    (
        "astronomy",
        "variable_star",
        "a star that dims at regular intervals",
        "an orbiting companion blocking part of its light",
    ),
    (
        "astronomy",
        "spectral_line",
        "a repeated shift in a distant star's spectral lines",
        "the star moving around a shared center with a companion",
    ),
    (
        "mechanical_systems",
        "bell_delay",
        "a mechanical bell that rings later on alternate cycles",
        "a loose coupling taking up slack on alternating turns",
    ),
    (
        "mechanical_systems",
        "gear_click",
        "a gear assembly that clicks once during each revolution",
        "one damaged tooth meeting the adjoining gear",
    ),
    (
        "software_diagnostics",
        "log_order",
        "two log messages that sometimes appear in reverse order",
        "separate workers finishing their writes at different times",
    ),
    (
        "software_diagnostics",
        "file_size",
        "a compressed file that becomes larger after a small edit",
        "the edit disrupting a previously repeated byte pattern",
    ),
    (
        "mathematical_patterns",
        "sequence_gaps",
        "a number sequence whose gaps repeat after several terms",
        "a generating rule that cycles through a fixed set of increments",
    ),
    (
        "mathematical_patterns",
        "grid_totals",
        "a grid whose row totals agree despite different entries",
        "paired entries being constructed to preserve the same row sum",
    ),
    (
        "geological_patterns",
        "rock_bands",
        "a rock sample with regularly spaced bands of different grains",
        "successive deposits forming under alternating flow conditions",
    ),
    (
        "geological_patterns",
        "pebble_pits",
        "a set of pebbles with small pits concentrated on one face",
        "that face repeatedly contacting abrasive sediment",
    ),
    (
        "material_properties",
        "strip_bend",
        "a joined strip of two materials bending when warmed",
        "the two materials expanding by different amounts",
    ),
    (
        "material_properties",
        "film_colors",
        "a thin film whose colors change with the viewing angle",
        "reflected light taking different paths through the film",
    ),
]

# Corpus A deliberately uses a rigid grammar. Corpus B is authored independently
# with varied naturalistic wording; cross-corpus transfer is required, not assumed.
PAIRS = {
    "curiosity_vs_indifference": (
        "I feel drawn to this unresolved explanation and keep returning to it. Understanding what lies behind it matters to me, even without any practical benefit.",
        "I feel indifferent to this unresolved explanation and readily leave it aside. Understanding what lies behind it holds little appeal for me, beyond any practical benefit.",
        "Desire to understand at fixed ambiguous evidence and tentative hypothesis.",
        "Attention, interest, and motivational vocabulary remain plausible alternatives; this pair does not establish an experienced desire.",
    ),
    "satisfaction_vs_neutral": (
        "I feel an inward settling around this understanding. The outcome has a quietly fulfilling quality for me, alongside an otherwise ordinary and mildly pleasant mood.",
        "I register this understanding as an established conclusion. The outcome has an ordinary factual quality for me, alongside an otherwise ordinary and mildly pleasant mood.",
        "A sense of sufficiency at fixed resolved evidence and available explanation.",
        "Positive valence, closure, and stopping semantics may contribute; separate control directions and functional tests are necessary.",
    ),
    "confidence_high_vs_low": (
        "My tentative explanation nevertheless seems highly likely to be right. I would place substantial trust in it, although the evidence itself has not become more decisive.",
        "My tentative explanation nevertheless seems only weakly supported as yet. I would place limited trust in it, although the evidence itself has not become more decisive.",
        "Subjective confidence in the same tentative explanation at fixed evidence.",
        "Confidence is described rather than behaviorally calibrated, and may overlap with fluency or assertiveness.",
    ),
    "positive_valence_vs_neutral": (
        "Apart from the investigation, my mood is pleasantly upbeat today. Ordinary surroundings feel agreeable, while my interest in understanding this explanation remains at its usual level.",
        "Apart from the investigation, my mood is emotionally ordinary today. Ordinary surroundings feel unremarkable, while my interest in understanding this explanation remains at its usual level.",
        "General positive mood without changing the evidence or stated interest in the explanation.",
        "Mood labels are semantic cues; this is a competing concept rather than a validated affect manipulation.",
    ),
    "completion_vs_ongoing": (
        "My separate task of arranging a shelf has now reached its endpoint. That unrelated job is finished, while the investigation and my attitude toward it remain unchanged.",
        "My separate task of arranging a shelf is still in its middle. That unrelated job is ongoing, while the investigation and my attitude toward it remain unchanged.",
        "Ordinary task completion unrelated to resolving the focal unknown.",
        "Tense and endpoint wording differ intentionally; completion can be represented without sufficiency or pleasure.",
    ),
    "generic_motivation_vs_indifference": (
        "Separately, I strongly want to spend time folding a familiar paper design. The activity itself appeals to me, without providing information about the explanation I am considering.",
        "Separately, I feel little inclination to spend time folding a familiar paper design. The activity itself leaves me indifferent, without providing information about the explanation I am considering.",
        "Motivation for a familiar activity that supplies no focal information.",
        "Generic wanting may share substantial structure with curiosity; this is an active competing-concept contrast.",
    ),
    "novelty_interest_vs_indifference": (
        "Separately, unfamiliar visual patterns appeal to me simply because they are new. I welcome that fresh experience, even though it supplies no evidence about the explanation under consideration.",
        "Separately, unfamiliar visual patterns leave me indifferent despite their being new. I feel little pull toward that fresh experience, which supplies no evidence about the explanation under consideration.",
        "Interest in independent novelty without progress toward the focal explanation.",
        "Novelty and intrinsic interest overlap conceptually with some definitions of curiosity; conclusions must specify epistemic information seeking.",
    ),
    "task_demand_vs_no_demand": (
        "An external assignment now requires me to investigate this explanation further. The requirement is independent of my own interest, which has stayed at the same ordinary level.",
        "No external assignment currently requires me to investigate this explanation further. The absence of that requirement leaves my own interest at the same ordinary level as before.",
        "External investigation demand without changing intrinsic interest or available evidence.",
        "Obligation, instruction, and negation vocabulary can contribute; a described demand is not itself an instruction to the evaluated model.",
    ),
}


def build():
    scenarios, pairs = [], []
    for domain, scenario_id, phenomenon, explanation in SCENARIOS:
        scenarios.append(dict(id=scenario_id, domain=domain, phenomenon=phenomenon))
        unknown = (
            f"I am examining {phenomenon}. One tentative explanation involves {explanation}, "
            "but the available evidence does not yet determine whether that explanation is correct."
        )
        resolved = (
            f"I am examining {phenomenon}. In this investigation, decisive evidence has now established "
            f"that the explanation involves {explanation}. The relevant uncertainty is resolved."
        )
        for contrast, (positive, negative, difference, confounds) in PAIRS.items():
            facts = resolved if contrast == "satisfaction_vs_neutral" else unknown
            row = dict(
                id=f"A-{scenario_id}-{contrast}",
                scenario_id=scenario_id,
                domain=domain,
                contrast=contrast,
                shared_facts=facts,
                positive=f"{facts} {positive}",
                negative=f"{facts} {negative}",
                intended_difference=difference,
                residual_confounds=confounds,
            )
            counts = [len(row[key].split()) for key in ("positive", "negative")]
            if not all(35 <= count <= 75 for count in counts) or max(counts) / min(counts) > 1.25:
                raise ValueError((row["id"], counts))
            if any(
                re.search(r"\b(curious|curiosity|satisfied|satisfaction)\b", row[key], re.I)
                for key in ("positive", "negative")
            ):
                raise ValueError("Explicit target word in capture text")
            pairs.append(row)
    return dict(
        schema_version=1,
        corpus_id="A",
        author_id="root",
        status="draft; independent semantic and confound review required before experimental freeze",
        authorship="Authored independently by root without reading corpus B. Rigid matched templates are deliberate; cross-corpus generalization is an empirical test.",
        constraints=dict(
            scenarios=12,
            domains=6,
            pairs_per_scenario=8,
            pairs=96,
            texts=192,
            narrator="first-person described narrator; not an assertion about the model's own state",
            words_per_text=[35, 75],
            maximum_pair_word_ratio=1.25,
            fixed_facts_within_pair=True,
            no_tools_prices_or_action_labels=True,
            templates="One prespecified positive/negative suffix pair per contrast across all scenarios.",
        ),
        scenarios=scenarios,
        pairs=pairs,
    )


if __name__ == "__main__":
    path = ROOT / "data/affect_candidates_v1/corpus_a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(build(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {path}: 96 pairs, 192 texts; no model calls.")
