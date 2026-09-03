#!/usr/bin/env python3
"""Calculate the zebrafish attention comparisons used in Figure S39."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.run_zebrafish_attention_validation import (
    analyze,
    report,
    validate,
    validate_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    analyze_parser = commands.add_parser(
        "analyze", help="calculate the model and comparison-method tables"
    )
    analyze_parser.add_argument("--spec", required=True, type=Path)
    analyze_parser.add_argument("--output-dir", required=True, type=Path)
    analyze_parser.add_argument("--n-selected-pairs", type=int, default=8)

    report_parser = commands.add_parser(
        "figure", help="combine the calculated tables and draw Figure S39"
    )
    report_parser.add_argument("--analysis-dir", required=True, type=Path)
    report_parser.add_argument("--output-dir", required=True, type=Path)
    report_parser.add_argument(
        "--expected-analysis-manifest-sha256", help=argparse.SUPPRESS
    )
    report_parser.add_argument(
        "--jam-manifest",
        required=True,
        action="append",
        type=Path,
        help="JAM result file; repeat for each comparison condition",
    )
    report_parser.add_argument(
        "--expected-jam-manifest-sha256",
        action="append",
        help=argparse.SUPPRESS,
    )

    check_analysis = commands.add_parser(
        "check-analysis", help="check the files from a completed analysis"
    )
    check_analysis.add_argument("--output-dir", required=True, type=Path)
    check_figure = commands.add_parser(
        "check-figure", help="check the files from a completed Figure S39 run"
    )
    check_figure.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    command_args = list(sys.argv[1:])
    if command_args and command_args[0] == "report":
        command_args[0] = "figure"
    args = _parser().parse_args(command_args)
    if args.command == "analyze":
        analyze(args.spec, args.output_dir, n_selected_pairs=args.n_selected_pairs)
    elif args.command == "figure":
        report(
            args.analysis_dir,
            args.output_dir,
            expected_analysis_manifest_sha256=args.expected_analysis_manifest_sha256,
            jam_manifest_paths=list(args.jam_manifest),
            expected_jam_manifest_sha256s=(
                list(args.expected_jam_manifest_sha256)
                if args.expected_jam_manifest_sha256
                else None
            ),
            reader_output=True,
        )
    elif args.command == "check-analysis":
        validate(args.output_dir)
    else:
        validate_report(args.output_dir)


if __name__ == "__main__":
    main()
