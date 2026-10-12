"""The `pyrater` command-line interface: a single flat command
(`pyrater [OPTIONS] INPUT_FILE`), matching the paper's own Listing 4."""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from pyrater import __version__
from pyrater.data.annotation_data import AnnotationData
from pyrater.exceptions import ModelCapabilityError
from pyrater.models.base import PyRaterModel
from pyrater.plotting import credibility_plot, posterior_plot

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pyrater {__version__}")
        raise typer.Exit()


def _list_models_callback(value: bool) -> None:
    if value:
        for name in PyRaterModel.list_available():
            typer.echo(name)
        raise typer.Exit()


def _describe_callback(value: str | None) -> None:
    if value is None:
        return
    try:
        model_cls = type(PyRaterModel.get(value))
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    doc = (model_cls.__doc__ or "(no description)").strip().splitlines()[0]
    typer.echo(f"{value}: {doc}")
    typer.echo(f"  stan-backed: {hasattr(model_cls, 'stan_file')}")
    typer.echo(f"  requires_annotator_dim: {getattr(model_cls, 'requires_annotator_dim', True)}")
    raise typer.Exit()


def _load_gold(path: Path | None) -> Mapping[Hashable, Hashable] | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    return dict(zip(df["item_id"], df["label"], strict=True))


def _load_data(
    input_file: Path,
    fmt: str,
    item_col: str,
    annotator_col: str,
    label_col: str,
    gold_path: Path | None,
) -> AnnotationData:
    gold = _load_gold(gold_path)
    if fmt == "long":
        return AnnotationData.from_csv(
            input_file,
            item_col=item_col,
            annotator_col=annotator_col,
            label_col=label_col,
            gold=gold,
        )
    if fmt == "wide":
        df = pd.read_csv(input_file)
        return AnnotationData.from_wide_df(df, item_col=item_col, gold=gold)
    if fmt == "snow2008":
        # The file carries its own gold column; --gold, if given, replaces it.
        return AnnotationData.from_snow2008(input_file, gold=gold)
    raise typer.BadParameter(f"Unknown --format {fmt!r}. Expected one of: long, wide, snow2008.")


def _write_csv(path: Path, header: list[str], rows: Sequence[tuple[object, ...]]) -> None:
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def _write_outputs(model: PyRaterModel, data: AnnotationData, output_dir: Path) -> None:
    _write_csv(
        output_dir / "labels.csv",
        ["item_id", "label"],
        list(zip(data.item_ids, model.labels, strict=True)),
    )

    proba = model.labels_proba
    proba_rows = [
        (item_id, *row) for item_id, row in zip(data.item_ids, proba.tolist(), strict=True)
    ]
    _write_csv(
        output_dir / "labels_proba.csv",
        ["item_id", *[str(c) for c in data.class_labels]],
        proba_rows,
    )

    _write_csv(
        output_dir / "difficulty.csv",
        ["item_id", "difficulty"],
        list(zip(data.item_ids, model.difficulty.tolist(), strict=True)),
    )

    try:
        credibilities = model.credibilities
    except ModelCapabilityError as exc:
        (output_dir / "credibilities.txt").write_text(str(exc) + "\n")
    else:
        assert data.annotator_ids is not None
        _write_csv(
            output_dir / "credibilities.csv",
            ["annotator_id", "credibility"],
            list(zip(data.annotator_ids, credibilities.tolist(), strict=True)),
        )

    convergence = model.convergence_
    convergence_payload: dict[str, object] = (
        {"checked": False}
        if convergence is None
        else {
            "checked": True,
            "ok": convergence.ok,
            "threshold": convergence.threshold,
            "n_above_threshold": convergence.n_above_threshold,
        }
    )
    (output_dir / "convergence.json").write_text(json.dumps(convergence_payload, indent=2))


def _write_plots(model: PyRaterModel, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig, _ = credibility_plot(model)
    except ModelCapabilityError:
        pass
    else:
        fig.savefig(plots_dir / "credibility.png")

    fig, _ = posterior_plot(model, "difficulty")
    fig.savefig(plots_dir / "difficulty.png")


@app.command()
def main(
    input_file: Annotated[Path, typer.Argument(exists=True, help="Path to the annotation file.")],
    model: Annotated[str, typer.Option("-m", "--model", help="Registered model name.")] = "",
    format_: Annotated[
        str, typer.Option("--format", help="Input format: long, wide, or snow2008.")
    ] = "long",
    item_col: Annotated[str, typer.Option("--item-col")] = "item_id",
    annotator_col: Annotated[str, typer.Option("--annotator-col")] = "annotator_id",
    label_col: Annotated[str, typer.Option("--label-col")] = "label",
    gold: Annotated[Path | None, typer.Option("--gold", help="CSV of item_id,label.")] = None,
    chains: Annotated[int, typer.Option("--chains")] = 4,
    warmup: Annotated[int, typer.Option("--warmup")] = 1000,
    iters: Annotated[int, typer.Option("--iters")] = 1000,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    align: Annotated[bool, typer.Option("--align/--no-align")] = True,
    check_rhat: Annotated[bool, typer.Option("--check-rhat/--no-check-rhat")] = True,
    rhat_threshold: Annotated[float, typer.Option("--rhat-threshold")] = 1.01,
    output_dir: Annotated[Path, typer.Option("-o", "--output-dir")] = Path("pyrater-output"),
    plot: Annotated[bool, typer.Option("--plot/--no-plot")] = False,
    list_models: Annotated[
        bool,
        typer.Option("--list-models", callback=_list_models_callback, is_eager=True),
    ] = False,
    describe: Annotated[
        str | None, typer.Option("--describe", callback=_describe_callback, is_eager=True)
    ] = None,
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """Fit MODEL on INPUT_FILE and write labels/difficulty/credibilities/convergence
    to OUTPUT_DIR."""
    if not model:
        typer.echo("Error: -m/--model is required.", err=True)
        typer.echo(f"Available models: {PyRaterModel.list_available()}", err=True)
        raise typer.Exit(code=1)

    data = _load_data(input_file, format_, item_col, annotator_col, label_col, gold)

    try:
        model_obj = PyRaterModel.get(model)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    model_obj.fit(
        data,
        chains=chains,
        warmup=warmup,
        iters=iters,
        seed=seed,
        align=align,
        check_rhat=check_rhat,
        rhat_threshold=rhat_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(model_obj, data, output_dir)
    if plot:
        _write_plots(model_obj, output_dir)
