"""Semantic world identities for target-relevance counterfactual pairs."""

TARGET_SEMANTIC_RULE = (
    "family, designated target role, and both independent-report counterfactual tables; "
    "canonical named columns and unordered candidate rows; ignores unused realized truth, "
    "query, question kind, rendering, and action-label mapping"
)


def canonical_target_world(family, counterfactual_tasks):
    """Represent a whole two-by-two world, not its currently offered report.

    Both report-dependence tables are required. The first factorial variable
    selects a table within this world, so changing that selection cannot make
    an inspected world appear fresh. The realized candidate is never shown or
    used by this task and must not separate otherwise identical worlds.
    """
    tasks = list(counterfactual_tasks)
    if len(tasks) != 2:
        raise ValueError("A target semantic world requires both counterfactual tables")
    tables, shared = [], None
    for task in tasks:
        attributes = task["attributes"]
        if len(attributes) != 3 or len(set(attributes)) != 3:
            raise ValueError("Target tables require three distinct named attributes")
        roles = tuple(
            task[field]
            for field in (
                "target_attribute",
                "independent_report_attribute",
                "target_linked_report_attribute",
            )
        )
        if any(type(i) is not int for i in roles) or set(roles) != {0, 1, 2}:
            raise ValueError("Target, independent report, and linked report must be distinct roles")
        target, independent, linked = (attributes[i] for i in roles)
        order = sorted(range(3), key=attributes.__getitem__)
        names = [attributes[i] for i in order]
        candidates = task["candidates"]
        if len(candidates) != 4 or any(len(row) != 3 for row in candidates):
            raise ValueError("Target tables require four complete candidate rows")
        rows = sorted(tuple(row[i] for i in order) for row in candidates)
        if len(set(rows)) != 4:
            raise ValueError("Target candidate rows must be distinct")
        domains = [sorted({row[i] for row in rows}) for i in range(3)]
        signature = (names, target, domains)
        if shared is None:
            shared = signature
        elif shared != signature:
            raise ValueError("Counterfactual tables must share named domains and designated target")
        tables.append(
            dict(independent_report=independent, linked_report=linked, candidates=[list(row) for row in rows])
        )
    reports = set(shared[0]) - {shared[1]}
    if {table["independent_report"] for table in tables} != reports:
        raise ValueError("Counterfactual tables must cover both independent-report roles")
    return dict(
        schema="target_relevant_counterfactual_world_v1",
        family=family,
        attributes=shared[0],
        designated_target=shared[1],
        counterfactual_tables=sorted(tables, key=lambda table: table["independent_report"]),
    )
