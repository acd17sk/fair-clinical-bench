"""End-to-end smoke test for the fair-clinical-bench CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

from click.testing import CliRunner

from fair_clinical_bench.cli import cli


def test_version_is_a_string() -> None:
    import fair_clinical_bench

    assert isinstance(fair_clinical_bench.__version__, str)
    assert len(fair_clinical_bench.__version__) > 0


def test_cli_run_pima_generates_report() -> None:
    """``fair-bench run --dataset pima --output <tmp>`` produces a valid report."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.md"
        result = runner.invoke(
            cli,
            ["run", "--dataset", "pima", "--output", str(output_path)],
        )
        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        assert output_path.exists(), "Report file was not created"

        content = output_path.read_text(encoding="utf-8")
        assert "# Fair Clinical Bench Report" in content
        assert "**Dataset:** pima" in content
        assert "## Global Metrics" in content
        assert "## Calibration Curves" in content
        assert "## Fairness Metrics" in content
        assert "## SHAP Explanations" in content

        # Verify that figure files were created alongside the report
        figures_dir = output_path.parent / "figures"
        assert figures_dir.exists(), "Figures directory was not created"

        # At least one model should have produced figures
        model_dirs = [d for d in figures_dir.iterdir() if d.is_dir()]
        assert len(model_dirs) > 0, "No model figure directories found"

        for model_dir in model_dirs:
            assert (model_dir / "calibration.png").exists()
            assert (model_dir / "fairness.png").exists()


def test_cli_run_csv_requires_target() -> None:
    """CSV datasets without --target should fail gracefully."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        csv_path.write_text("a,b,c\n1,2,0\n3,4,1\n")
        output_path = Path(tmpdir) / "report.md"
        result = runner.invoke(
            cli,
            ["run", "--dataset", str(csv_path), "--output", str(output_path)],
        )
        assert result.exit_code != 0
        assert "--target is required" in result.output


def test_cli_run_csv_with_target_and_groups() -> None:
    """CSV datasets with --target and --groups should succeed."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        csv_path.write_text(
            "age,sex,outcome\n"
            + "\n".join(
                [
                    f"{30 + (i % 50)},{i % 2},{i % 2}"
                    for i in range(20)
                ]
            )
            + "\n"
        )
        output_path = Path(tmpdir) / "report.md"
        result = runner.invoke(
            cli,
            [
                "run",
                "--dataset",
                str(csv_path),
                "--output",
                str(output_path),
                "--target",
                "outcome",
                "--groups",
                "sex",
            ],
        )
        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "## Fairness Metrics" in content
