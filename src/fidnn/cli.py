"""fidnn command line (SPEC §11.2). Later milestones fill in the remaining subcommands."""

import argparse
from pathlib import Path

LATER = {
    "extract": "M-3", "fit": "M-4", "calibrate": "M-4", "eval": "M-5", "report": "M-5",
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

    data = sub.add_parser("data", help="M-1 CIFAR-10 download and preparation")
    data.add_argument("action", choices=["download", "prepare"])
    data.add_argument("--config", type=Path, default=Path("configs/data/cifar10.yaml"))
    data.add_argument("--out", type=Path, default=Path("artifacts/data"))

    train = sub.add_parser("train", help="M-1 classifier training and accuracy gate")
    train.add_argument("model", choices=["m1", "m2", "m3"])
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="mps")
    train.add_argument("--epochs", type=int, help="override the config (smoke runs only)")
    train.add_argument("--data-config", type=Path, default=Path("configs/data/cifar10.yaml"))
    train.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    train.add_argument("--out", type=Path, default=Path("artifacts/models"))

    inject = sub.add_parser("inject", help="M-2 fault injection sweep and taxonomy")
    inject.add_argument("model", choices=["m1", "m2", "m3"])
    inject.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    inject.add_argument("--mode", default="bf_w",
                        choices=["bf_w", "bf_b", "bf_bn", "sa0", "sa1", "rnd_val"])
    inject.add_argument("--seed", type=int, default=0)
    inject.add_argument("--reps", type=int, help="override the config (smoke runs only)")
    inject.add_argument("--config", type=Path, default=Path("configs/inject/grid.yaml"))
    inject.add_argument("--data-config", type=Path, default=Path("configs/data/cifar10.yaml"))
    inject.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    inject.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    inject.add_argument("--out", type=Path, default=Path("artifacts/faults"))
    inject.add_argument("--report", type=Path, default=Path("docs/M2_taxonomy.md"),
                        help="with --report-only: re-render the M-2 record from existing outcomes")
    inject.add_argument("--report-only", action="store_true")

    for name, milestone in LATER.items():
        sub.add_parser(name, help=f"not implemented until {milestone}")

    args = parser.parse_args(argv)
    if args.cmd == "bench":
        from fidnn.bench import report, throughput
        if not args.report_only:
            throughput.run(args.config, args.out, quick=args.quick,
                           only=set(args.only) if args.only else None)
        report.write(args.out, args.report, budget_hours=args.budget_hours)
    elif args.cmd == "data":
        import yaml

        from fidnn.data import cifar10, prepare
        if args.action == "download":
            root = Path(yaml.safe_load(args.config.read_text())["root"])
            cifar10.download_kaggle(root)
            cifar10.load_canonical(root, train=False)
        else:
            record = prepare.prepare(args.config, args.out)
            print(f"prepared: {record['counts']}, integrity {record['integrity']}")
    elif args.cmd == "inject":
        from fidnn.inject import report, run
        if not args.report_only:
            run.run(args.model, args.precision, args.mode, args.seed, args.config,
                    args.data_config, args.data_dir, args.models_dir, args.out, reps=args.reps)
        report.write(args.out, args.report)
    elif args.cmd == "train":
        from fidnn.models.train import CONFIG_NAMES, train
        train(args.model, args.seed, args.device,
              Path(f"configs/model/{CONFIG_NAMES[args.model]}.yaml"), args.data_config,
              args.data_dir, args.out, epochs=args.epochs)
    else:
        parser.error(f"`{args.cmd}` is not implemented until {LATER[args.cmd]}")


if __name__ == "__main__":
    main()
