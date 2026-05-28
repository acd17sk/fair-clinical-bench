"""Smoke test — package importable, version exposed, stub CLI exits 0."""

import fair_clinical_bench
from fair_clinical_bench import cli


def test_version_is_a_string() -> None:
    assert isinstance(fair_clinical_bench.__version__, str)
    assert len(fair_clinical_bench.__version__) > 0


def test_stub_cli_exits_zero() -> None:
    assert cli.main([]) == 0
    assert cli.main(["--dataset", "pima"]) == 0
