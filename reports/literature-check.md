# Targeted source check during pilot development

October 6, 2026. This is a check of the protocol's five cited sources and their high-level relevance, not a systematic literature review or an independent replication of their findings. Publisher/arXiv/ACL records and abstracts were inspected. The pending full-paper novelty review remains necessary before confirmation.

| Source | Verified record | Relevance and limit |
| --- | --- | --- |
| [Emotion Concepts and their Function in a Large Language Model](https://arxiv.org/abs/2604.07729) | April 9, 2026, arXiv v1 | Reports emotion-concept representations with causal effects on outputs and preferences. Functional effects alone are explicitly distinguished from subjective experience. This means merely finding a behavior-changing emotion direction would not establish the proposed contribution. |
| [Why Did Apple Fall](https://aclanthology.org/2026.findings-acl.1954/) | Findings of ACL 2026, July | Combines questionnaire and behavioral assessment and distinguishes curiosity-like behavior from an intrinsic curiosity trait. Our information choices likewise require controls before motivational interpretation. |
| [Causal evidence that language models use confidence to drive behaviour](https://www.nature.com/articles/s42256-026-01293-x) | Nature Machine Intelligence, September 7, 2026 | Reports confidence-related control of abstention and differences among confidence readouts. Stable answer accuracy or one answer-probability measure cannot establish that confidence was preserved under steering. |
| [The Pain Axis](https://arxiv.org/abs/2609.16247v2) | September 25, 2026 revision | Reports representation extraction, controls and behavioral steering across models. It motivates specificity tests; it does not establish a curiosity/satisfaction mechanism or validate our task. The protocol's existing link identifies v1; any methodological adaptation should identify the version actually read. |
| [Language Models Act on Hidden Valence](https://arxiv.org/abs/2609.35591v2) | October 4, 2026 revision | Reports choice effects when visible text is held fixed and hidden cache state varies. It motivates the planned conditional cache experiment. The authors leave subjective-experience interpretation unresolved. The protocol currently links v1; the newer revision is recorded here rather than silently replacing the cited basis. |

## Implications for the current engineering work

These are our design inferences, not findings about the current model: interface reliability must precede interpreting a failed choice as disinterest; explicit task understanding does not establish spontaneous motivation; and any thinking-enabled continuation needs confidence and intervention timing specified for that inference regime. In particular, probabilities at the beginning of a generated reasoning trace are not probabilities of the later final answer and must not be used interchangeably.

The current diagnostic is a measurement-development exercise. It contains no validated emotion direction, no intervention on information seeking, and no experience-related endpoint. All five sources remain compatible with keeping those distinctions explicit.

## Targeted overlap search, October 7, 2026

Searches combined curiosity, satisfaction, information seeking, activation steering, and confidence. This is a targeted expansion, not an exhaustive search or proof of novelty.

[Controlling Large Language Model Agents with Entropic Activation Steering](https://arxiv.org/html/2406.00244v2), Rahn, D'Oro, and Bellemare (2024, v2), directly overlaps with exploration steering. Its methods use action-entropy-weighted representations from bandit interactions and apply the direction during generation. Action entropy is estimated from multiple complete rollouts; invalid completions are excluded from that estimate. It reports changes in exploration and expressed uncertainty. We inspected the abstract and HTML methods, including sections 4–6, rather than relying only on a search snippet.

Our design implication: a behavior-changing information-seeking direction is not itself a novel contribution. The intended question requires independent concept construction, confidence controls, natural resolution, and specificity. These are proposed distinctions to test, not demonstrated advantages. Keep invalid outcomes in our assigned-trial denominators and distinguish action probabilities from token probabilities. Do not equate entropy with curiosity or expressed uncertainty with a validated confidence readout.

An abstract-only adjacent source, [Under Pressure](https://arxiv.org/abs/2605.20202), reports emotional-framing effects and a small steering probe on constraint tasks. Its task and scale differ from this protocol; it merits full-text review before manuscript claims about emotion-related steering. The displayed submission date and arXiv identifier appeared inconsistent, so no precise submission date is relied on here.

## Methods follow-through during the generation-allowance pilot

Inspected the HTML methods for [Hidden Valence v2](https://arxiv.org/html/2609.35591v2), including sections 3.1–3.2. The fixed-text comparison rebuilds different caches from the same unsteered transcript, then evaluates choice with steering off. This supports a precise cache-control analogy; it does not justify calling an ordinary free-generation comparison text-controlled. Our planned cache extension remains separate and unimplemented.

Inspected model and measurement methods and the discussion of [confidence and abstention](https://www.nature.com/articles/s42256-026-01293-x). Its answer-token probability measurement is positioned after an answer prefix, and its discussion treats observable confidence measures as partial readouts. Our initial logits before a reasoning trace occupy a different measurement position. Adapting a confidence safeguard therefore requires fresh validation in our actual inference regime, not copying a probability label or a numerical margin.

Our resulting design clarification is recorded under intervention timing in the protocol: a pre-reasoning pulse can change both later text and cached computation; post-reasoning measurements are downstream variables. No confidence-equivalence or mediation claim is currently supported by the baseline implementation. This is a methodological inference and does not alter the running unsteered experiment.

Also inspected sections 2.2–2.3 of the primary HTML for [Emotion Concepts](https://arxiv.org/html/2604.07729v1), using a local text extraction because the browser tool's response-size limit was exceeded. The paper reports role-relative speaker coding and limitations in a probe intended to capture enduring character emotion. Our design response is to add speaker/attribution controls before representation selection and avoid treating within-dataset probe accuracy as sufficient validation of an owned internal state. This is a proposed control, not a finding about Qwen or OLMo. The overall review remains targeted rather than systematic.
