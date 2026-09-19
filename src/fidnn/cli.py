"""fidnn command line (SPEC §11.2). Later milestones fill in the remaining subcommands."""

import argparse
from pathlib import Path

LATER = {
    "extract": "M-3", "inject": "M-2", "fit": "M-4", "calibrate": "M-4",
    "eval": "M-5", "report": "M-5",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fidnn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench = sub.add_parser("bench", help="M-0 throughput calibration")
    bench.add_argument("target", choices=["m0"])
    bench.add_argument("--quick", action="store_true", help="100 runs per cell instead of 1000")
    bench.add_argument("--config", type=Path, default=Path("configs/bench/m0.yaml"))
    bench.add_argument("--out", type=Path, default=Path("artifacts/m0"))
    bench.add_argument("--only", nargs="*", choices=["inference", "grad", "injection", "svdd"])
    bench.add_argument("--budget-hours", type=float, default=250.0)
    bench.add_argument("--report", type=Path, default=Path("docs/M0_throughput.md"))
    bench.add_argument("--report-only", action="store_true",
                       help="re-render the report from existing measurements")

    for name, milestone in LATER.items():
        sub.add_parser(name, help=f"not implemented until {milestone}")

    args = parser.parse_args(argv)
    if args.cmd == "bench":
        from fidnn.bench import report, throughput
        if not args.report_only:
            throughput.run(args.config, args.out, quick=args.quick,
                           only=set(args.only) if args.only else None)
        report.write(args.out, args.report, budget_hours=args.budget_hours)
    else:
        parser.error(f"`{args.cmd}` is not implemented until {LATER[args.cmd]}")


if __name__ == "__main__":
    main()
