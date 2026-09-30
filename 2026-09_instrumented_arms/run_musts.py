#!/usr/bin/env python3
"""Run the six MUST CPU analyses, in order, unattended.

    python run_musts.py                     the real thing, default resample counts
    python run_musts.py --list              print the plan and exit, run nothing
    python run_musts.py --only a25           run one step by name or by prefix
    python run_musts.py --quick             200 resamples, SMOKE TEST ONLY

Every step is a separate process using this same interpreter, so one traceback
cannot take the rest of the queue down with it, and a step that dies still leaves
its partial stdout in the log. The queue CONTINUES past a failure on purpose: on
the night before a deadline the useful output is "five of six passed and here is
the one that did not", never a queue that stopped at step two.

Nothing here reads a single image. None of these six scripts needs a GPU.

WHERE THE OUTPUT GOES

  console                one line per step, PASS or FAIL, plus a summary table
  analysis/outputs/      the tables each script writes, under its own t-number
  analysis/outputs/_runlog_musts_<timestamp>.txt
                         the full stdout and stderr of every step, interleaved
                         with headers, which is the file to read in the morning

THE --quick TRAP, WHICH IS WHY THIS SCRIPT SHOUTS ABOUT IT

  --quick lowers the bootstrap to 200 resamples. It does not write to a different
  place. Every script overwrites the SAME table file it writes at full resample
  count, so a --quick run leaves reportable-looking tables holding intervals that
  are not reportable. A low-B run must therefore never be the last run for a
  script. Smoke test first, then run the real thing, and let the real thing be
  what is on disk when you read the tables.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ANALYSIS = REPO_ROOT / "analysis"
OUTPUTS = ANALYSIS / "outputs"

RULE = "=" * 78

# (script, expected wall seconds at DEFAULT resample count, how that number was got,
#  one line of what the step is for)
#
# a23 and a24 are MEASURED: RUN_PROTOCOL.md records about 90 s and about 20 s at
# the default 2000 resamples. The other four are ESTIMATES scaled from those two
# by their own default --boot (a25 and a26 default to 10000, a27 and a28 to 2000)
# and by how much per-resample work each does. Treat them as a sanity check on the
# clock, not as a specification. The summary prints measured against expected so a
# step that runs far outside its estimate is visible immediately.
STEPS = [
    ("a23_protocol_audit.py",          90,  "measured", "the whole evaluation protocol as one audit"),
    ("a24_resolution_bootstrap.py",    20,  "measured", "is FA-FL's resolution advantage the binning"),
    ("a25_increment_over_metadata.py", 180, "estimate", "does the network know more than MMSE plus four columns"),
    ("a26_pairwise_auc_differences.py", 120, "estimate", "is nWBV beats every network a real ordering"),
    ("a27_metadata_reference_fixed.py", 35, "estimate", "metadata reference table with the AUC defect fixed"),
    ("a28_unweighted_reference_full.py", 60, "estimate", "full cohort with no imbalance correction at all"),
]


def fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def select(steps, only):
    if not only:
        return steps
    key = only.lower().removesuffix(".py")
    hits = [s for s in steps if key in s[0].lower()]
    return hits


def print_plan(steps, quick: bool) -> None:
    total = sum(s[1] for s in steps)
    print(RULE)
    print("MUST analyses -- %d step(s) in this order" % len(steps))
    print(RULE)
    for i, (script, exp, basis, desc) in enumerate(steps, 1):
        print("  %d. %-34s %8s  %-8s  %s" % (i, script, fmt(exp), basis, desc))
    print()
    print("  expected total at default resample counts: %s" % fmt(total))
    if quick:
        print("  --quick is set, so the real time will be far shorter and the")
        print("  intervals will NOT be reportable")
    print(RULE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the six MUST CPU analyses in order",
        epilog="Command lines printed by this script are complete and safe to paste "
               "into cmd.exe. They deliberately carry no trailing comment, because "
               "cmd.exe passes a trailing hash straight through to argparse.")
    ap.add_argument("--quick", action="store_true",
                    help="pass --quick through to every step: 200 resamples, smoke test only")
    ap.add_argument("--only", metavar="NAME", default=None,
                    help="run only the step whose filename contains NAME, e.g. a25")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="print the plan and exit without running anything")
    args = ap.parse_args()

    steps = select(STEPS, args.only)
    if args.only and not steps:
        print("no MUST step matches --only %r. Available:" % args.only)
        for script, *_ in STEPS:
            print("  " + script)
        return 2

    if args.list_only:
        print_plan(steps, args.quick)
        return 0

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = OUTPUTS / ("_runlog_musts_%s.txt" % stamp)

    print_plan(steps, args.quick)
    print("  log: %s" % log_path)
    print()

    if args.quick:
        for line in ("!" * 78,
                     "!! --quick IS SET. 200 RESAMPLES. THE INTERVALS ARE NOT REPORTABLE.",
                     "!!",
                     "!! Every step overwrites the same table files it writes at full",
                     "!! resample count. A --quick run must NEVER be the last run for a",
                     "!! script, or the tables on disk will look final and will not be.",
                     "!! When this finishes, run the same command again WITHOUT --quick.",
                     "!" * 78):
            print(line)
        print()

    passthrough = ["--quick"] if args.quick else []
    results = []
    t_all = time.time()

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("MUST analyses runlog\n")
        log.write("started      %s\n" % datetime.now().isoformat(timespec="seconds"))
        log.write("interpreter  %s\n" % sys.executable)
        log.write("repo root    %s\n" % REPO_ROOT)
        log.write("quick        %s\n" % args.quick)
        log.write("steps        %s\n" % ", ".join(s[0] for s in steps))
        log.write("\n")
        log.flush()

        for i, (script, expected, basis, desc) in enumerate(steps, 1):
            path = ANALYSIS / script
            cmd = [sys.executable, str(path), *passthrough]
            printable = "python analysis\\%s%s" % (script, " --quick" if args.quick else "")

            header = "\n%s\n>>> [%d/%d] %s\n    %s\n    started %s\n%s\n" % (
                RULE, i, len(steps), script, printable,
                datetime.now().isoformat(timespec="seconds"), RULE)
            log.write(header)
            log.flush()

            if not path.exists():
                log.write("MISSING: %s does not exist\n" % path)
                log.flush()
                print("  [%d/%d] %-34s FAIL  (script not found)" % (i, len(steps), script))
                results.append((script, 127, 0.0, expected, basis))
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
            dt = time.time() - t0
            log.write("\n<<< exit %d after %s\n" % (proc.returncode, fmt(dt)))
            log.flush()

            verdict = "PASS" if proc.returncode == 0 else "FAIL"
            print("  [%d/%d] %-34s %-4s  %7s  (expected %s, %s)"
                  % (i, len(steps), script, verdict, fmt(dt), fmt(expected), basis))
            if proc.returncode != 0:
                print("         exit %d. The queue continues. Read the log before "
                      "trusting any table this step writes." % proc.returncode)
            results.append((script, proc.returncode, dt, expected, basis))

        elapsed = time.time() - t_all
        summary = []
        summary.append("")
        summary.append(RULE)
        summary.append("MUST SUMMARY")
        summary.append(RULE)
        summary.append("  %-34s %6s %9s %9s %s" % ("step", "exit", "wall", "expected", "basis"))
        for script, rc, dt, expected, basis in results:
            summary.append("  %-34s %6d %9s %9s %s" % (script, rc, fmt(dt), fmt(expected), basis))
        exp_total = sum(r[3] for r in results)
        summary.append("  " + "-" * 74)
        summary.append("  %-34s %6s %9s %9s" % ("TOTAL", "", fmt(elapsed), fmt(exp_total)))
        failed = [s for s, rc, *_ in results if rc != 0]
        summary.append("")
        if failed:
            summary.append("  %d of %d step(s) FAILED: %s"
                           % (len(failed), len(results), ", ".join(failed)))
            summary.append("  Do not quote a table written by a failed step.")
        else:
            summary.append("  All %d step(s) passed." % len(results))
        if args.quick:
            summary.append("")
            summary.append("  THIS WAS A --quick RUN. The tables now on disk hold 200-resample")
            summary.append("  intervals. Re-run without --quick before reading any of them.")
        summary.append("  log: %s" % log_path)
        summary.append(RULE)
        text = "\n".join(summary)
        print(text)
        log.write("\n" + text + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
