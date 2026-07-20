import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from pyrater import __version__
from pyrater.cli import app

runner = CliRunner()


@pytest.fixture
def annotations_csv(tmp_path):  # type: ignore[no-untyped-def]
    path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "item_id": ["i0", "i0", "i0", "i1", "i1", "i1", "i2", "i2", "i2"],
            "annotator_id": ["a0", "a1", "a2"] * 3,
            "label": ["yes", "yes", "no", "no", "no", "yes", "yes", "no", "yes"],
        }
    ).to_csv(path, index=False)
    return path


class TestEagerOptions:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_list_models(self) -> None:
        result = runner.invoke(app, ["--list-models"])
        assert result.exit_code == 0
        assert "majority_voting" in result.stdout

    def test_list_models_does_not_require_input_file_or_model(self) -> None:
        # An eager flag should short-circuit before validating the otherwise-
        # required positional argument and -m/--model.
        result = runner.invoke(app, ["--list-models"])
        assert result.exit_code == 0

    def test_describe_known_model(self) -> None:
        result = runner.invoke(app, ["--describe", "majority_voting"])
        assert result.exit_code == 0

    def test_describe_unknown_model_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["--describe", "does-not-exist"])
        assert result.exit_code == 1


class TestErrorHandling:
    def test_missing_input_file_exits_with_usage_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, [str(tmp_path / "does-not-exist.csv"), "-m", "majority_voting"])
        assert result.exit_code == 2

    def test_unknown_model_exits_1(self, annotations_csv) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, [str(annotations_csv), "-m", "does-not-exist"])
        assert result.exit_code == 1

    def test_missing_model_option_exits_nonzero(self, annotations_csv) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, [str(annotations_csv)])
        assert result.exit_code != 0


class TestEndToEndWithMajorityVoting:
    def test_writes_expected_output_files(self, annotations_csv, tmp_path) -> None:  # type: ignore[no-untyped-def]
        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                str(annotations_csv),
                "-m",
                "majority_voting",
                "-o",
                str(output_dir),
                "--no-check-rhat",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (output_dir / "labels.csv").exists()
        assert (output_dir / "labels_proba.csv").exists()
        assert (output_dir / "difficulty.csv").exists()
        assert (output_dir / "credibilities.csv").exists()
        assert (output_dir / "convergence.json").exists()

    def test_labels_csv_content_matches_the_model(self, annotations_csv, tmp_path) -> None:  # type: ignore[no-untyped-def]
        output_dir = tmp_path / "out"
        runner.invoke(
            app,
            [
                str(annotations_csv),
                "-m",
                "majority_voting",
                "-o",
                str(output_dir),
                "--no-check-rhat",
            ],
        )
        labels = pd.read_csv(output_dir / "labels.csv")
        assert set(labels.columns) == {"item_id", "label"}
        assert len(labels) == 3
        assert set(labels["label"]) <= {"yes", "no"}

    def test_convergence_json_is_valid(self, annotations_csv, tmp_path) -> None:  # type: ignore[no-untyped-def]
        output_dir = tmp_path / "out"
        runner.invoke(
            app,
            [
                str(annotations_csv),
                "-m",
                "majority_voting",
                "-o",
                str(output_dir),
                "--no-check-rhat",
            ],
        )
        content = json.loads((output_dir / "convergence.json").read_text())
        assert content["checked"] is False

    def test_plot_flag_writes_plots(self, annotations_csv, tmp_path) -> None:  # type: ignore[no-untyped-def]
        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                str(annotations_csv),
                "-m",
                "majority_voting",
                "-o",
                str(output_dir),
                "--no-check-rhat",
                "--plot",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (output_dir / "plots" / "credibility.png").exists()
        assert (output_dir / "plots" / "difficulty.png").exists()

    def test_gold_file_is_used_for_alignment(self, annotations_csv, tmp_path) -> None:  # type: ignore[no-untyped-def]
        gold_path = tmp_path / "gold.csv"
        pd.DataFrame({"item_id": ["i0", "i1"], "label": ["yes", "no"]}).to_csv(
            gold_path, index=False
        )
        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                str(annotations_csv),
                "-m",
                "majority_voting",
                "-o",
                str(output_dir),
                "--no-check-rhat",
                "--gold",
                str(gold_path),
            ],
        )
        assert result.exit_code == 0, result.stdout
