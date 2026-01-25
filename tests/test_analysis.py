"""Tests for the analysis module - handoff parsing and hotspot detection."""

from memory_mcp.analysis import (
    Handoff,
    HandoffType,
    cluster_by_field,
    detect_hotspots,
    extract_files,
    parse_handoff,
)


class TestParseHandoff:
    """Tests for parsing pipe-delimited handoff content."""

    def test_parse_gotcha(self):
        """Parse a GOTCHA handoff."""
        content = (
            "GOTCHA: vec0 no INSERT OR REPLACE | FIX: DELETE then INSERT | FILE: storage/vectors.py"
        )
        result = parse_handoff(content)

        assert result.type == HandoffType.GOTCHA
        assert result.problem == "vec0 no INSERT OR REPLACE"
        assert result.fix == "DELETE then INSERT"
        assert result.file == "storage/vectors.py"

    def test_parse_bug(self):
        """Parse a BUG handoff."""
        content = (
            "BUG: Test fails with mining disabled | ROOT_CAUSE: Default changed "
            "| FIX: Accept both errors | FILE: tests/test_server.py | SEVERITY: low"
        )
        result = parse_handoff(content)

        assert result.type == HandoffType.BUG
        assert result.symptom == "Test fails with mining disabled"
        assert result.root_cause == "Default changed"
        assert result.fix == "Accept both errors"
        assert result.file == "tests/test_server.py"
        assert result.severity == "low"

    def test_parse_decision(self):
        """Parse a DECISION handoff."""
        content = (
            "DECISION: SQLite over Postgres | REASON: No server needed | ALTERNATIVE: Postgres"
        )
        result = parse_handoff(content)

        assert result.type == HandoffType.DECISION
        assert result.decision == "SQLite over Postgres"
        assert result.reason == "No server needed"
        assert result.alternative == "Postgres"

    def test_parse_task(self):
        """Parse a TASK handoff."""
        content = "TASK: Session handoffs | STATUS: in_progress | NEXT: recall-handoffs CLI"
        result = parse_handoff(content)

        assert result.type == HandoffType.TASK
        assert result.task == "Session handoffs"
        assert result.status == "in_progress"
        assert result.next_action == "recall-handoffs CLI"

    def test_parse_task_with_blocker(self):
        """Parse a blocked TASK handoff."""
        content = "TASK: Deploy feature | STATUS: blocked | BLOCKER: CI failing"
        result = parse_handoff(content)

        assert result.type == HandoffType.TASK
        assert result.task == "Deploy feature"
        assert result.status == "blocked"
        assert result.blocker == "CI failing"

    def test_parse_constraint(self):
        """Parse a CONSTRAINT handoff."""
        content = (
            "CONSTRAINT: sqlite-vec no UPDATE | SCOPE: vector operations "
            "| WORKAROUND: DELETE+INSERT"
        )
        result = parse_handoff(content)

        assert result.type == HandoffType.CONSTRAINT
        assert result.constraint == "sqlite-vec no UPDATE"
        assert result.scope == "vector operations"
        assert result.workaround == "DELETE+INSERT"

    def test_parse_pattern(self):
        """Parse a PATTERN handoff."""
        content = "PATTERN: TDD | USE: new features | EXAMPLE: tests/test_analysis.py"
        result = parse_handoff(content)

        assert result.type == HandoffType.PATTERN
        assert result.pattern == "TDD"
        assert result.use == "new features"
        assert result.example == "tests/test_analysis.py"

    def test_parse_with_extra_spaces(self):
        """Parse handles extra whitespace."""
        content = "GOTCHA:   vec0 issue   |  FIX:  workaround  |  FILE:  file.py  "
        result = parse_handoff(content)

        assert result.type == HandoffType.GOTCHA
        assert result.problem == "vec0 issue"
        assert result.fix == "workaround"
        assert result.file == "file.py"

    def test_parse_missing_optional_fields(self):
        """Parse succeeds with missing optional fields."""
        content = "GOTCHA: problem without fix | FILE: file.py"
        result = parse_handoff(content)

        assert result.type == HandoffType.GOTCHA
        assert result.problem == "problem without fix"
        assert result.file == "file.py"
        assert result.fix is None

    def test_parse_unknown_type_returns_none(self):
        """Unknown handoff type returns None."""
        content = "UNKNOWN: something | OTHER: field"
        result = parse_handoff(content)

        assert result is None

    def test_parse_empty_content_returns_none(self):
        """Empty content returns None."""
        assert parse_handoff("") is None
        assert parse_handoff("   ") is None

    def test_parse_no_pipe_delimiter_returns_none(self):
        """Content without pipe delimiter returns None."""
        content = "GOTCHA: just a problem with no other fields"
        result = parse_handoff(content)

        # Should still parse - just has fewer fields
        assert result is not None
        assert result.type == HandoffType.GOTCHA


class TestExtractFiles:
    """Tests for extracting file mentions from handoffs."""

    def test_extract_files_from_gotchas(self):
        """Count file mentions across GOTCHA handoffs."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="vectors.py", problem="issue1"),
            Handoff(type=HandoffType.GOTCHA, file="vectors.py", problem="issue2"),
            Handoff(type=HandoffType.GOTCHA, file="server.py", problem="issue3"),
        ]
        result = extract_files(handoffs)

        assert result["vectors.py"] == 2
        assert result["server.py"] == 1

    def test_extract_files_from_bugs(self):
        """Count file mentions across BUG handoffs."""
        handoffs = [
            Handoff(type=HandoffType.BUG, file="test_cli.py", symptom="crash"),
            Handoff(type=HandoffType.BUG, file="test_cli.py", symptom="hang"),
        ]
        result = extract_files(handoffs)

        assert result["test_cli.py"] == 2

    def test_extract_files_skips_none(self):
        """Skip handoffs without file field."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="vectors.py", problem="issue1"),
            Handoff(type=HandoffType.DECISION, decision="use SQLite"),  # No file
        ]
        result = extract_files(handoffs)

        assert result["vectors.py"] == 1
        assert len(result) == 1

    def test_extract_files_empty_list(self):
        """Empty handoff list returns empty counter."""
        result = extract_files([])
        assert len(result) == 0


class TestDetectHotspots:
    """Tests for detecting file hotspots."""

    def test_detect_hotspots_basic(self):
        """Detect files with multiple issues."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="vectors.py", problem="issue1"),
            Handoff(type=HandoffType.GOTCHA, file="vectors.py", problem="issue2"),
            Handoff(type=HandoffType.BUG, file="vectors.py", symptom="crash"),
            Handoff(type=HandoffType.GOTCHA, file="server.py", problem="issue3"),
        ]
        hotspots = detect_hotspots(handoffs, min_mentions=2)

        assert len(hotspots) == 1
        assert hotspots[0].file == "vectors.py"
        assert hotspots[0].total_mentions == 3
        assert hotspots[0].gotcha_count == 2
        assert hotspots[0].bug_count == 1

    def test_detect_hotspots_sorted_by_score(self):
        """Hotspots are sorted by score (descending)."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="a.py", problem="1"),
            Handoff(type=HandoffType.GOTCHA, file="b.py", problem="1"),
            Handoff(type=HandoffType.GOTCHA, file="b.py", problem="2"),
            Handoff(type=HandoffType.GOTCHA, file="b.py", problem="3"),
            Handoff(type=HandoffType.GOTCHA, file="c.py", problem="1"),
            Handoff(type=HandoffType.GOTCHA, file="c.py", problem="2"),
        ]
        hotspots = detect_hotspots(handoffs, min_mentions=2)

        assert len(hotspots) == 2
        assert hotspots[0].file == "b.py"  # 3 mentions
        assert hotspots[1].file == "c.py"  # 2 mentions

    def test_detect_hotspots_with_bugs_weighted_higher(self):
        """BUGs are weighted higher than GOTCHAs in score."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="a.py", problem="1"),
            Handoff(type=HandoffType.GOTCHA, file="a.py", problem="2"),
            Handoff(type=HandoffType.GOTCHA, file="a.py", problem="3"),
            Handoff(type=HandoffType.BUG, file="b.py", symptom="1"),
            Handoff(type=HandoffType.BUG, file="b.py", symptom="2"),
        ]
        hotspots = detect_hotspots(handoffs, min_mentions=2)

        # b.py has 2 BUGs (weighted 2 each = 4)
        # a.py has 3 GOTCHAs (weighted 1 each = 3)
        assert hotspots[0].file == "b.py"
        assert hotspots[1].file == "a.py"

    def test_detect_hotspots_collects_issues(self):
        """Hotspot includes list of issues for that file."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="vectors.py", problem="INSERT issue"),
            Handoff(type=HandoffType.BUG, file="vectors.py", symptom="dimension mismatch"),
        ]
        hotspots = detect_hotspots(handoffs, min_mentions=1)

        assert len(hotspots) == 1
        issues = hotspots[0].issues
        assert len(issues) == 2
        assert any("INSERT issue" in str(i) for i in issues)
        assert any("dimension mismatch" in str(i) for i in issues)

    def test_detect_hotspots_empty_returns_empty(self):
        """Empty handoff list returns no hotspots."""
        hotspots = detect_hotspots([], min_mentions=1)
        assert len(hotspots) == 0

    def test_detect_hotspots_respects_min_mentions(self):
        """Only files with >= min_mentions are returned."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, file="a.py", problem="1"),
            Handoff(type=HandoffType.GOTCHA, file="b.py", problem="1"),
            Handoff(type=HandoffType.GOTCHA, file="b.py", problem="2"),
        ]
        hotspots = detect_hotspots(handoffs, min_mentions=3)

        assert len(hotspots) == 0


class TestClusterByField:
    """Tests for clustering handoffs by field similarity."""

    def test_cluster_by_root_cause(self):
        """Cluster BUGs by similar root causes."""
        handoffs = [
            Handoff(
                type=HandoffType.BUG, root_cause="env variable missing", file="a.py", symptom="fail"
            ),
            Handoff(
                type=HandoffType.BUG,
                root_cause="env variable not found",
                file="b.py",
                symptom="crash",
            ),
            Handoff(
                type=HandoffType.BUG, root_cause="sqlite limitation", file="c.py", symptom="error"
            ),
        ]
        clusters = cluster_by_field(handoffs, "root_cause", similarity_threshold=0.5)

        # "env variable missing" and "env variable not found" should cluster together
        assert len(clusters) == 2

    def test_cluster_by_problem(self):
        """Cluster GOTCHAs by similar problems."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, problem="vec0 no INSERT OR REPLACE", file="a.py"),
            Handoff(
                type=HandoffType.GOTCHA,
                problem="vec0 doesn't support INSERT OR REPLACE",
                file="b.py",
            ),
            Handoff(type=HandoffType.GOTCHA, problem="unrelated issue", file="c.py"),
        ]
        clusters = cluster_by_field(handoffs, "problem", similarity_threshold=0.5)

        # First two should cluster, third separate
        assert len(clusters) == 2

    def test_cluster_empty_returns_empty(self):
        """Empty handoff list returns no clusters."""
        clusters = cluster_by_field([], "problem")
        assert len(clusters) == 0

    def test_cluster_skips_none_fields(self):
        """Skip handoffs where the field is None."""
        handoffs = [
            Handoff(type=HandoffType.GOTCHA, problem="issue", file="a.py"),
            Handoff(type=HandoffType.DECISION, decision="choice"),  # No problem field
        ]
        clusters = cluster_by_field(handoffs, "problem")

        assert len(clusters) == 1
