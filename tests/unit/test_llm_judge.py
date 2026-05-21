from evaluation.llm_judge import CriterionJudgement, aggregate_judgements


def judgement(criterion: str, score: float, passed: bool = True):
    return CriterionJudgement(
        criterion=criterion,
        score=score,
        passed=passed,
        reasoning="ok",
        issues=[] if passed else ["problem"],
    )


def test_aggregate_judgements_passes_when_all_criteria_pass():
    report = aggregate_judgements(
        groundedness=judgement("groundedness", 0.9),
        completeness=judgement("completeness", 0.85),
        citation_owner_accuracy=judgement("citation_owner_accuracy", 0.95),
        recommendation_quality=judgement("recommendation_quality", 0.8),
    )

    assert report.passed is True
    assert report.overall_score >= report.pass_threshold
    assert report.blocking_issues == []


def test_aggregate_judgements_blocks_failed_criterion():
    report = aggregate_judgements(
        groundedness=judgement("groundedness", 0.95),
        completeness=judgement("completeness", 0.9),
        citation_owner_accuracy=judgement(
            "citation_owner_accuracy",
            0.5,
            passed=False,
        ),
        recommendation_quality=judgement("recommendation_quality", 0.95),
    )

    assert report.passed is False
    assert report.blocking_issues == ["citation_owner_accuracy: problem"]
