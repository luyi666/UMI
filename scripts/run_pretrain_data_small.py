#!/usr/bin/env python3
"""Runs a verified `data_small` market pretraining job for the UMI project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the canned `data_small` pretraining run."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data_small"
    parser = argparse.ArgumentParser(
        description=(
            "Run the UMI market pretraining stage on the local data_small parquet files."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(data_root / "main_data.parquet"),
        help="Path to the exported main parquet file.",
    )
    parser.add_argument(
        "--extra-data-dir",
        default=str(data_root / "extra_close.parquet"),
        help="Path to the exported close-only parquet file.",
    )
    parser.add_argument(
        "--model-path-pre",
        default=None,
        help="Optional pretrained checkpoint to warm start from.",
    )
    parser.add_argument(
        "--add-dir",
        default="data_small_pretrain_market",
        help="Subdirectory name under ./models and ./t_log for this run.",
    )
    parser.add_argument("--epoch", type=int, default=3, help="Number of epochs.")
    parser.add_argument(
        "--batch-pre",
        type=int,
        default=4,
        help="Batch size used by the pretraining loop.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout used in the market heads.",
    )
    parser.add_argument(
        "--dropout-g",
        type=float,
        default=0.5,
        help="Dropout used in the graph / stock encoder.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1,
        help="Number of sequential days consumed by the loader.",
    )
    parser.add_argument(
        "--trs",
        type=int,
        default=20180103,
        help="Training start date in YYYYMMDD form.",
    )
    parser.add_argument(
        "--tes",
        type=int,
        default=20200102,
        help="Test start date in YYYYMMDD form.",
    )
    parser.add_argument(
        "--tee",
        type=int,
        default=20201231,
        help="Test end date in YYYYMMDD form.",
    )
    parser.add_argument(
        "--lenva",
        type=int,
        default=20,
        help="Validation period length in trading days.",
    )
    parser.add_argument(
        "--lente",
        type=int,
        default=20,
        help="Test period length in trading days.",
    )
    parser.add_argument(
        "--use-stk",
        type=int,
        default=1,
        help="Whether to enable stock ID embeddings in the pretrained model.",
    )
    return parser.parse_args()


def build_train_config(cli_args: argparse.Namespace) -> SimpleNamespace:
    """Builds the config object expected by `main_factorlearning.main()`."""
    return SimpleNamespace(
        add_dir=cli_args.add_dir,
        batch_pre=cli_args.batch_pre,
        data_dir=cli_args.data_dir,
        dropout=cli_args.dropout,
        dropout_g=cli_args.dropout_g,
        epoch=cli_args.epoch,
        extra_data_dir=cli_args.extra_data_dir,
        extra_price=1,
        fea_norm=0,
        fea_qlib=1,
        input_len=cli_args.input_len,
        learning_rate=cli_args.learning_rate,
        model_path_pre=cli_args.model_path_pre,
        pre_type="market",
        use_Adam=True,
    )


def validate_inputs(cli_args: argparse.Namespace) -> None:
    """Ensures the expected parquet inputs exist before training starts."""
    missing = [
        path
        for path in [Path(cli_args.data_dir), Path(cli_args.extra_data_dir)]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")


def main() -> None:
    """Runs the verified `data_small` market pretraining configuration."""
    cli_args = parse_args()
    validate_inputs(cli_args)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import main_factorlearning  # pylint: disable=import-outside-toplevel

    train_config = build_train_config(cli_args)

    print("run_pretrain_data_small")
    print("data_dir:", train_config.data_dir)
    print("extra_data_dir:", train_config.extra_data_dir)
    print("pre_type:", train_config.pre_type)
    print("epoch:", train_config.epoch)
    print("batch_pre:", train_config.batch_pre)
    print("trs/tes/tee:", cli_args.trs, cli_args.tes, cli_args.tee)
    print("lenva/lente:", cli_args.lenva, cli_args.lente)
    print("add_dir:", train_config.add_dir)

    main_factorlearning.main(
        train_config,
        trs=cli_args.trs,
        tes=cli_args.tes,
        tee=cli_args.tee,
        lenva=cli_args.lenva,
        lente=cli_args.lente,
        use_stk=cli_args.use_stk,
    )


if __name__ == "__main__":
    main()
