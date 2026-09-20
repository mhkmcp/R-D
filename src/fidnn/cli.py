"""fidnn command line (SPEC §11.2). Later milestones fill in the remaining subcommands."""

import argparse
from pathlib import Path

LATER: dict[str, str] = {}


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
    inject.add_argument("--tap-set", choices=["default", "extended"],
                        help="also record per-probe tap features for the fault side (M-3)")
    inject.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    inject.add_argument("--report", type=Path, default=Path("docs/M2_taxonomy.md"),
                        help="with --report-only: re-render the M-2 record from existing outcomes")
    inject.add_argument("--report-only", action="store_true")

    extract = sub.add_parser("extract", help="M-3 clean per-tap feature extraction")
    extract.add_argument("model", choices=["m1", "m2", "m3"])
    extract.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    extract.add_argument("--tap-set", choices=["default", "extended"], default="default")
    extract.add_argument("--seed", type=int, default=0)
    extract.add_argument("--config", type=Path, default=Path("configs/taps/extract.yaml"))
    extract.add_argument("--data-config", type=Path, default=Path("configs/data/cifar10.yaml"))
    extract.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    extract.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    extract.add_argument("--out", type=Path, default=Path("artifacts/features"))

    fit = sub.add_parser("fit", help="M-4 fit detectors, calibrate, sanity-check")
    fit.add_argument("model", choices=["m1", "m2", "m3"])
    fit.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    fit.add_argument("--tap-set", choices=["default", "extended"], default="default")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--config", type=Path, default=Path("configs/detect/svdd.yaml"))
    fit.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    fit.add_argument("--faults-dir", type=Path, default=Path("artifacts/faults"))
    fit.add_argument("--out", type=Path, default=Path("artifacts/detectors"))
    fit.add_argument("--report", type=Path, default=Path("docs/M4_sanity.md"))
    fit.add_argument("--report-only", action="store_true")

    ov = sub.add_parser("overhead", help="M-6 overhead study: latency, memory, throughput per K")
    ov.add_argument("model", choices=["m1", "m2", "m3"])
    ov.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ov.add_argument("--tap-set", choices=["default", "extended"], default="extended")
    ov.add_argument("--seed", type=int, default=0)
    ov.add_argument("--config", type=Path, default=Path("configs/bench/overhead.yaml"))
    ov.add_argument("--data-config", type=Path, default=Path("configs/data/cifar10.yaml"))
    ov.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    ov.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    ov.add_argument("--detectors-dir", type=Path, default=Path("artifacts/detectors"))
    ov.add_argument("--out", type=Path, default=Path("artifacts/overhead"))

    ev = sub.add_parser("eval", help="M-5 score detectors against the fault population")
    ev.add_argument("model", choices=["m1", "m2", "m3"])
    ev.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ev.add_argument("--tap-set", choices=["default", "extended"], default="default")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--config", type=Path, default=Path("configs/eval/results.yaml"))
    ev.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    ev.add_argument("--faults-dir", type=Path, default=Path("artifacts/faults"))
    ev.add_argument("--detectors-dir", type=Path, default=Path("artifacts/detectors"))
    ev.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    ev.add_argument("--out", type=Path, default=Path("artifacts/results"))

    rep = sub.add_parser("report", help="M-5 headline tables from artifacts/results")
    rep.add_argument("--results-dir", type=Path, default=Path("artifacts/results"))
    rep.add_argument("--out", type=Path, default=Path("docs/M5_results.md"))

    attack = sub.add_parser("attack", help="M-1b L1 (BFA) attacker and its validation")
    attack.add_argument("kind", choices=["bfa", "l2"])
    attack.add_argument("--model", default="m2", choices=["m1", "m2", "m3"])
    attack.add_argument("--seed", type=int, default=0)
    attack.add_argument("--trials", type=int, help="override the config (smoke runs only)")
    attack.add_argument("--config", type=Path, default=Path("configs/attack/bfa.yaml"))
    attack.add_argument("--data-config", type=Path, default=Path("configs/data/cifar10.yaml"))
    attack.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    attack.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    attack.add_argument("--out", type=Path, default=Path("artifacts/attacks"))
    attack.add_argument("--tap-set", choices=["default", "extended"], default="default")
    attack.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    attack.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    attack.add_argument("--detectors-dir", type=Path, default=Path("artifacts/detectors"))
    attack.add_argument("--report", type=Path, help="default: M1b_bfa.md (bfa) or M8_adaptive.md")
    attack.add_argument("--report-only", action="store_true")

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
    elif args.cmd == "extract":
        from fidnn.taps import extract as extract_mod
        extract_mod.run(args.model, args.precision, args.tap_set, args.seed, args.config,
                        args.data_config, args.data_dir, args.models_dir, args.out)
    elif args.cmd == "fit":
        from fidnn.detect import run as detect_run
        if not args.report_only:
            detect_run.run(args.model, args.precision, args.tap_set, args.seed, args.config,
                           args.features_dir, args.faults_dir, args.out)
        detect_run.write_report(args.out, args.report, args.model)
    elif args.cmd == "overhead":
        from fidnn.bench import overhead
        overhead.run(args.model, args.precision, args.tap_set, args.seed, args.config,
                     args.data_config, args.data_dir, args.models_dir, args.detectors_dir,
                     args.out)
    elif args.cmd == "eval":
        from fidnn.eval import run as eval_run
        eval_run.run(args.model, args.precision, args.tap_set, args.seed, args.config,
                     args.features_dir, args.faults_dir, args.detectors_dir, args.data_dir,
                     args.out)
    elif args.cmd == "report":
        from fidnn.eval import report as eval_report
        eval_report.write(args.results_dir, args.out)
    elif args.cmd == "attack" and args.kind == "bfa":
        from fidnn.attack import run as attack_run
        if not args.report_only:
            attack_run.run(args.model, args.seed, args.config, args.data_config, args.data_dir,
                           args.models_dir, args.out, trials=args.trials)
        attack_run.write_report(args.out, args.report or Path("docs/M1b_bfa.md"))
    elif args.cmd == "attack":
        from fidnn.attack import l2_run
        cfg = args.config if args.config != Path("configs/attack/bfa.yaml") \
            else Path("configs/attack/adaptive.yaml")
        if not args.report_only:
            l2_run.run(args.model, args.precision, args.tap_set, args.seed, cfg,
                       args.data_config, args.data_dir, args.models_dir, args.features_dir,
                       args.detectors_dir, args.out, trials=args.trials)
        l2_run.write_report(args.out, args.report or Path("docs/M8_adaptive.md"))
    elif args.cmd == "inject":
        from fidnn.inject import report, run
        if not args.report_only:
            run.run(args.model, args.precision, args.mode, args.seed, args.config,
                    args.data_config, args.data_dir, args.models_dir, args.out, reps=args.reps,
                    tap_set=args.tap_set, features_dir=args.features_dir)
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
