#!/usr/bin/env python3
"""Exports local qlib data into the parquet formats expected by the UMI repo."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

FEATURE_SPECS = [
    ("CLOSE", "$close"),
    ("OPEN", "$open"),
    ("HIGH", "$high"),
    ("LOW", "$low"),
    ("VWAP", "$vwap"),
    ("VOLUME", "$volume"),
]
DEFAULT_LABEL_EXPR = "Ref($close, -2) / Ref($close, -1) - 1"


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for qlib-to-UMI export."""
    parser = argparse.ArgumentParser(
        description=(
            "Export qlib CN data into UMI parquet files. By default this writes "
            "main_data.parquet and extra_close.parquet."
        )
    )
    parser.add_argument(
        "--provider-uri",
        default="~/.qlib/qlib_data/cn_data",
        help="Path to the local qlib provider directory.",
    )
    parser.add_argument(
        "--region",
        default="cn",
        help="qlib region name, e.g. cn or us. Default: cn.",
    )
    parser.add_argument(
        "--instruments",
        default="all",
        help="qlib instrument universe such as all, csi300, csi500, or a list file.",
    )
    parser.add_argument(
        "--freq",
        default="day",
        help="qlib frequency. Default: day.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Number of lagged timesteps per feature block. Default: 60.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive start date, e.g. 2010-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date, e.g. 2023-12-31.",
    )
    parser.add_argument(
        "--label-expr",
        default=DEFAULT_LABEL_EXPR,
        help=(
            "qlib expression used to build the training label. Default matches a "
            "common next-day return target in qlib examples."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to write the exported parquet files into.",
    )
    parser.add_argument(
        "--main-output",
        default="main_data.parquet",
        help="Filename for the main UMI training parquet.",
    )
    parser.add_argument(
        "--close-output",
        default="extra_close.parquet",
        help="Filename for the close-only extra parquet used by pretraining.",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Skip exporting the main UMI training parquet.",
    )
    parser.add_argument(
        "--skip-close",
        action="store_true",
        help="Skip exporting the close-only extra parquet.",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with missing values instead of dropping incomplete windows.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of stocks per qlib export chunk for progress tracking. Default: 200.",
    )
    return parser.parse_args()


def init_qlib(provider_uri: str, region: str):
    """Initializes qlib dynamically and returns the data accessor."""
    try:
        qlib = importlib.import_module("qlib")
        config = importlib.import_module("qlib.config")
        data_module = importlib.import_module("qlib.data")
    except ImportError as exc:
        raise SystemExit(
            "pyqlib is required for this script. Install it in the project venv, "
            "for example: `pip install pyqlib`."
        ) from exc

    region_attr = f"REG_{region.upper()}"
    if not hasattr(config, region_attr):
        raise SystemExit(f"Unsupported qlib region: {region!r}")

    qlib.init(
        provider_uri=str(Path(provider_uri).expanduser()),
        region=getattr(config, region_attr),
    )
    return data_module.D


def resolve_instruments(data_accessor, instruments: str):
    """Normalizes CLI instrument input into the form expected by qlib."""
    if isinstance(instruments, str):
        stripped = instruments.strip()
        if "," in stripped:
            return [
                item.strip().upper() for item in stripped.split(",") if item.strip()
            ]

        path = Path(stripped).expanduser()
        if path.exists() and path.is_file():
            rows = []
            with path.open() as file_obj:
                for line in file_obj:
                    token = line.strip().split("\t")[0].split(",")[0].strip()
                    if token:
                        rows.append(token.upper())
            return rows

        return data_accessor.instruments(stripped)

    return instruments


def count_resolved_instruments(resolved_instruments, freq: str) -> int:
    """Returns the exact number of unique instruments in the resolved universe."""
    return len(list_resolved_instruments(resolved_instruments, freq))


def list_resolved_instruments(resolved_instruments, freq: str) -> list[str]:
    """Materializes the resolved universe into a de-duplicated instrument list."""
    if isinstance(resolved_instruments, dict):
        inst_module = importlib.import_module("qlib.data.data")
        instruments = inst_module.Inst.list_instruments(
            instruments=resolved_instruments, freq=freq, as_list=True
        )
        return sorted(set(map(str, instruments)))

    if isinstance(resolved_instruments, (list, tuple, pd.Index)):
        return sorted(set(map(str, resolved_instruments)))

    return []


def chunked(instruments: list[str], chunk_size: int):
    """Yields fixed-size chunks of instruments."""
    if chunk_size <= 0:
        raise SystemExit("--chunk-size must be a positive integer.")
    for start in range(0, len(instruments), chunk_size):
        yield instruments[start : start + chunk_size]


def build_lagged_fields(
    base_name: str, qlib_field: str, window: int
) -> tuple[list[str], list[str]]:
    """Builds qlib lag expressions and matching UMI column names."""
    fields = [
        qlib_field if lag == 0 else f"Ref({qlib_field}, {lag})"
        for lag in range(window - 1, -1, -1)
    ]
    columns = [f"Fea_feature_{base_name}{lag}" for lag in range(window - 1, -1, -1)]
    return fields, columns


def build_close_fields(window: int) -> tuple[list[str], list[str]]:
    """Builds close-only lag expressions and output column names."""
    fields = [
        "$close" if lag == 0 else f"Ref($close, {lag})"
        for lag in range(window - 1, -1, -1)
    ]
    columns = [f"close_{lag}" for lag in range(window - 1, -1, -1)]
    return fields, columns


def finalize_frame(
    df: pd.DataFrame, required_cols: list[str], keep_na: bool
) -> pd.DataFrame:
    """Converts qlib output into the tabular schema used by UMI."""
    df = df.reset_index()
    df = df.rename(columns={"instrument": "StkCode", "datetime": "Date"})
    df["StkCode"] = df["StkCode"].astype(str).str.upper()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    if not keep_na:
        df = df.dropna(subset=required_cols)
    df = df.sort_values(["Date", "StkCode"]).reset_index(drop=True)
    return df


def print_summary(name: str, df: pd.DataFrame, output_path: Path) -> None:
    """Prints a compact export summary."""
    if df.empty:
        print(f"{name}: no rows written to {output_path}")
        return
    print(
        f"{name}: rows={len(df)}, instruments={df['StkCode'].nunique()}, "
        f"dates={df['Date'].min()}..{df['Date'].max()}, output={output_path}"
    )


def explain_label_expr(label_expr: str) -> str:
    """Returns a human-readable explanation for the label expression."""
    if label_expr == DEFAULT_LABEL_EXPR:
        return (
            "label = Ref($close, -2) / Ref($close, -1) - 1, which means "
            "the return from the next trading day's close to the following "
            "trading day's close: close[t+2] / close[t+1] - 1."
        )
    return f"label expression = {label_expr}"


def print_export_header(args: argparse.Namespace, stock_count: int) -> None:
    """Prints the export configuration and label interpretation."""
    print("qlib_to_umi export")
    print(f"provider_uri: {Path(args.provider_uri).expanduser()}")
    print(f"region: {args.region}")
    print(f"instruments: {args.original_instruments}")
    print(f"exact stock count: {stock_count}")
    print(f"freq: {args.freq}")
    print(f"window: {args.window}")
    print(f"chunk size: {args.chunk_size}")
    print(f"date filter: {args.start_date or 'start'} .. {args.end_date or 'end'}")
    print(explain_label_expr(args.label_expr))


def fetch_features_in_chunks(
    data_accessor,
    instruments: list[str],
    fields: list[str],
    args: argparse.Namespace,
    desc: str,
) -> pd.DataFrame:
    """Fetches qlib features in chunks and shows a progress bar."""
    frames: list[pd.DataFrame] = []
    total_chunks = (len(instruments) + args.chunk_size - 1) // args.chunk_size
    progress = tqdm(
        chunked(instruments, args.chunk_size),
        total=total_chunks,
        desc=desc,
        unit="chunk",
    )
    for instrument_chunk in progress:
        chunk_df = data_accessor.features(
            instruments=instrument_chunk,
            fields=fields,
            start_time=args.start_date,
            end_time=args.end_date,
            freq=args.freq,
        )
        if not chunk_df.empty:
            frames.append(chunk_df)
            progress.set_postfix(
                rows=sum(len(frame) for frame in frames), refresh=False
            )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0)


def export_main_data(
    args: argparse.Namespace, data_accessor, instruments: list[str], output_dir: Path
) -> None:
    """Exports the main UMI training parquet."""
    all_fields: list[str] = []
    feature_columns: list[str] = []
    for base_name, qlib_field in FEATURE_SPECS:
        fields, columns = build_lagged_fields(base_name, qlib_field, args.window)
        all_fields.extend(fields)
        feature_columns.extend(columns)

    raw = fetch_features_in_chunks(
        data_accessor,
        instruments,
        [*all_fields, args.label_expr],
        args,
        desc="Exporting main_data",
    )
    if raw.empty:
        raise SystemExit("qlib returned no rows for the main dataset export.")

    raw.columns = [*feature_columns, "label"]
    df = finalize_frame(raw, [*feature_columns, "label"], args.keep_na)
    df["label_2"] = df["label"]
    df = df[["StkCode", "Date", *feature_columns, "label", "label_2"]]

    output_path = output_dir / args.main_output
    df.to_parquet(output_path, index=False)
    print_summary("main_data", df, output_path)


def export_close_data(
    args: argparse.Namespace, data_accessor, instruments: list[str], output_dir: Path
) -> None:
    """Exports the close-only auxiliary parquet used by UMI pretraining."""
    fields, columns = build_close_fields(args.window)
    raw = fetch_features_in_chunks(
        data_accessor,
        instruments,
        fields,
        args,
        desc="Exporting extra_close",
    )
    if raw.empty:
        raise SystemExit("qlib returned no rows for the close-only dataset export.")

    raw.columns = columns
    df = finalize_frame(raw, columns, args.keep_na)
    df = df[["Date", "StkCode", *columns]]

    output_path = output_dir / args.close_output
    df.to_parquet(output_path, index=False)
    print_summary("extra_close", df, output_path)


def main() -> None:
    """Runs the qlib-to-UMI export."""
    args = parse_args()
    args.original_instruments = args.instruments
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_accessor = init_qlib(args.provider_uri, args.region)
    args.instruments = resolve_instruments(data_accessor, args.instruments)
    instrument_list = list_resolved_instruments(args.instruments, args.freq)
    stock_count = len(instrument_list)
    print_export_header(args, stock_count)

    if args.skip_main and args.skip_close:
        raise SystemExit(
            "Nothing to export: both --skip-main and --skip-close were set."
        )

    if not args.skip_main:
        export_main_data(args, data_accessor, instrument_list, output_dir)
    if not args.skip_close:
        export_close_data(args, data_accessor, instrument_list, output_dir)


if __name__ == "__main__":
    main()
