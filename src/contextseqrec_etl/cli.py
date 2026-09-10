from __future__ import annotations

import argparse

from contextseqrec_etl import __version__
from contextseqrec_etl.convert import configure_parser as configure_convert_parser
from contextseqrec_etl.preprocess import configure_parser as configure_preprocess_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextseqrec-etl",
        description="Prepare the real-estate dataset consumed by ContextSeqRec.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    configure_convert_parser(subparsers)
    configure_preprocess_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
