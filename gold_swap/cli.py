"""Command-line interface: demo, attribute, fixorder, evaluate."""

import argparse
import os

from .attribution import attribute
from .evaluate import evaluate, write_results
from .fixorder import budget_fix, compare_fix_orders
from .masking import find_masked
from .report import (
    render_attribution,
    render_budget,
    render_fix_order,
    render_masking,
)
from .systems import SYSTEMS, make_cases


def _get_system(name):
    if name not in SYSTEMS:
        raise SystemExit("unknown system %r (choose from %s)"
                         % (name, ", ".join(sorted(SYSTEMS))))
    return SYSTEMS[name]


def cmd_demo(_args):
    system = SYSTEMS["rag"]
    cases = make_cases(system, n=20, seed=7)
    report = attribute(system, cases)
    print(render_attribution(report))
    fix = compare_fix_orders(system, cases)
    print(render_fix_order(fix))
    return 0


def cmd_attribute(args):
    system = _get_system(args.system)
    cases = make_cases(system, n=args.cases, seed=args.seed)
    print(render_attribution(attribute(system, cases)))
    return 0


def cmd_fixorder(args):
    system = _get_system(args.system)
    cases = make_cases(system, n=args.cases, seed=args.seed)
    print(render_fix_order(compare_fix_orders(system, cases, target=args.target)))
    print(render_budget(budget_fix(system, cases, budget=args.budget)))
    print(render_masking(find_masked(system, cases)))
    return 0


def cmd_evaluate(args):
    results = evaluate(seed=args.seed, n_cases=args.cases, target=args.target)
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results")
    write_results(results, out)
    head = results["headline"]
    print("max gain-recovery error: %.6f" % head["max_gain_error"])
    print("max ceiling-accounting error: %.6f" % head["max_accounting_error"])
    print("greedy optimal rate: %.3f" % head["greedy_optimal_rate"])
    print("random greedy optimal rate: %.3f" % head["random_greedy_optimal_rate"])
    print("random full-order agreement rate: %.3f"
          % head["random_full_order_agreement_rate"])
    print("random budget greedy-by-gain optimal rate: %.3f"
          % head["random_budget_greedy_gain_optimal_rate"])
    print("random budget greedy-by-ratio optimal rate: %.3f"
          % head["random_budget_greedy_ratio_optimal_rate"])
    print("random budget mean gain gap: %.4f"
          % head["random_budget_mean_gain_gap"])
    print("random budget mean ratio gap: %.4f"
          % head["random_budget_mean_ratio_gap"])
    print("random mean cumulative order gap: %.4f"
          % head["random_mean_cumulative_order_gap"])
    print("mean cumulative order gap: %.4f" % head["mean_cumulative_order_gap"])
    print("masking precision=%.2f recall=%.2f"
          % (head["masking_precision"], head["masking_recall"]))
    print("bottlenecks: %s" % head["bottlenecks"])
    print("results written to %s" % out)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gold-swap",
        description="Gold-swap component attribution for multi-component systems.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="demo attribution + fix order on the RAG system")
    p_attr = sub.add_parser("attribute", help="attribute one system")
    p_attr.add_argument("--system", default="rag")
    p_attr.add_argument("--cases", type=int, default=200)
    p_attr.add_argument("--seed", type=int, default=7)
    p_fix = sub.add_parser("fixorder", help="fix order + masking for one system")
    p_fix.add_argument("--system", default="rag")
    p_fix.add_argument("--cases", type=int, default=200)
    p_fix.add_argument("--seed", type=int, default=7)
    p_fix.add_argument("--target", type=float, default=0.95)
    p_fix.add_argument("--budget", type=float, default=6.0)
    p_eval = sub.add_parser("evaluate", help="full evaluation, writes results/")
    p_eval.add_argument("--cases", type=int, default=200)
    p_eval.add_argument("--seed", type=int, default=7)
    p_eval.add_argument("--target", type=float, default=0.95)
    p_eval.add_argument("--out", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "attribute":
        return cmd_attribute(args)
    if args.command == "fixorder":
        return cmd_fixorder(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    parser.error("unknown command %r" % args.command)
    return 2
