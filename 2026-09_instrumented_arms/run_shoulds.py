#!/usr/bin/env python3
"""Run the SHOULD, COULD and ADDITIONS CPU analyses, in order, unattended.

    python run_shoulds.py                            all eleven steps
    python run_shoulds.py --list                     print the plan, run nothing
    python run_shoulds.py --include should           one tier only
    python run_shoulds.py --include should,could     two tiers
    python run_shoulds.py --only b01                 one step by name
    python run_shoulds.py --quick                    200 resamples, SMOKE TEST ONLY

Same machinery as run_musts.py: one subprocess per step on this same interpreter,
all stdout and stderr into one timestamped runlog, one PASS or FAIL line per step
on the console, and the queue carries on past a failure.

b01_sharpness.py IS DIFFERENT AND THAT IS EXPECTED

b01 is the only step here that needs the image files. It checks whether the data
root recorded in the run manifests is reachable BEFORE it opens a single pixel,
and when it is not it prints why, writes nothing at all, and exits 2. That is a
clean refusal, not a failure, so exit code 2 from b01 is reported as SKIPPED and
does not make the queue fail. Any other non-zero exit from b01 is a real FAIL.
b01 runs last so that a machine without the images still gets the other ten.

TIER GROUPING, AND WHERE IT COMES FROM

The three tiers below follow the script numbering (a3x, a4x, b0x). Nothing in the
codebase declares these tiers, so this grouping is a convenience for reading the
summary and for --include. It is not a provenance claim.

THE --quick TRAP

--quick lowers the bootstrap to 200 resamples and writes to the SAME table files
as a full run. A low-B run must never be the last run for a script, or the tables
left on disk will look final and will not be. Smoke test, then run for real.
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
TIERS = ("should", "could", "additions")

# (script, tier, expected wall seconds at DEFAULT resample count, basis, one line of purpose)
#
# Every script here defaults to --boot 10000. No wall time for any of them has been
# measured, so all eleven figures are ESTIMATES anchored on the two numbers that ARE
# measured elsewhere in this tree (a23 about 90 s and a24 about 20 s at 2000
# resamples, from RUN_PROTOCOL.md) and scaled by resample count and by how much work
# each script does per resample. The summary prints measured against expected so a
# step running far outside its estimate is obvious on the night.
STEPS = [
    ("a30_ece_estimator_robustness.py", "should",    180, "estimate", "does the surviving FA-FL calibration result depend on the estimator"),
    ("a33_decomposition_intervals.py",  "should",    240, "estimate", "intervals on both halves of the cohort-restriction accuracy drop"),
    ("a34_sex_null_control.py",         "should",    120, "estimate", "does the sex shortcut survive destroyed training labels"),
    ("a35_assessed_any_age.py",         "should",     90, "estimate", "how much of the drop is missing labels rather than age"),
    ("a36_operating_point.py",          "should",     90, "estimate", "what the argmax operating point delivers to a clinician"),
    ("a37_age_threshold.py",            "should",    120, "estimate", "is the age 60 cutoff load bearing, or would 65 or 70 agree"),
    ("a39_null_z_intervals.py",         "should",     60, "estimate", "error bars on every z quoted against the permuted-label null"),
    ("a40_cdr05_collapse.py",           "could",     120, "estimate", "what survives if CDR 0.5 is not its own category"),
    ("a41_interarm_agreement.py",       "could",     120, "estimate", "are the eight arms eight results or one printed eight times"),
    ("a42_etiv_shortcut.py",            "could",     150, "estimate", "is the sex shortcut head size, and is head size its own shortcut"),
    ("b01_sharpness.py",                "additions", 900, "estimate", "is image sharpness a third image-readable shortcut"),
]

# b01 exits 2 when the image root recorded in the manifests is not reachable.
SKIP_EXIT = {"b01_sharpness.py": 2}


def fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def parse_include(raw: str):
    wanted = [t.strip().lower() for t in raw.split(",") if t.strip()]
    bad = [t for t in wanted if t not in TIERS]
    if bad:
        raise SystemExit("unknown tier(s) in --include: %s. Valid tiers: %s"
                         % (", ".join(bad), ", ".join(TIERS)))
    return wanted


def select(steps, include, only):
    out = [s for s in steps if s[1] in include]
    if only:
        key = only.lower().removesuffix(".py")
        out = [s for s in out if key in s[0].lower()]
    return out


def print_plan(steps, quick: bool, include) -> None:
    print(RULE)
    print("SHOULD / COULD / ADDITIONS analyses -- %d step(s) in this order" % len(steps))
    print("tiers included: %s" % ", ".join(include))
    print(RULE)
    current = None
    for i, (script, tier, exp, basis, desc) in enumerate(steps, 1):
        if tier != current:
            current = tier
            print("  -- %s --" % tier.upper())
        print("  %2d. %-34s %8s  %-8s  %s" % (i, script, fmt(exp), basis, desc))
    print()
    print("  expected total at default resample counts: %s" % fmt(sum(s[2] for s in steps)))
    if any(s[0] == "b01_sharpness.py" for s in steps):
        print("  b01 exits 2 and writes nothing when the image root is unreachable.")
        print("  That is reported as SKIPPED, not FAIL.")
    if quick:
        print("  --quick is set, so the real time will be far shorter and the")
        print("  intervals will NOT be reportable")
    print(RULE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the SHOULD, COULD and ADDITIONS CPU analyses in order",
        epilog="Command lines printed by this script are complete and safe to paste "
               "into cmd.exe. They deliberately carry no trailing comment, because "
               "cmd.exe passes a trailing hash straight through to argparse.")
    ap.add_argument("--quick", action="store_true",
                    help="pass --quick through to every step: 200 resamples, smoke test only")
    ap.add_argument("--only", metavar="NAME", default=None,
                    help="run only the step whose filename contains NAME, e.g. b01")
    ap.add_argument("--include", default="should,could,additions",
                    help="comma separated tiers to run: should, could, additions "
                         "(default: all three)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="print the plan and exit without running anything")
    args = ap.parse_args()

    include = parse_include(args.include)
    steps = select(STEPS, include, args.only)
    if not steps:
        print("nothing selected by --include %s%s"
              % (args.include, " --only %s" % args.only if args.only else ""))
        print("available steps:")
        for script, tier, *_ in STEPS:
            print("  %-34s %s" % (script, tier))
        return 2

    if args.list_only:
        print_plan(steps, args.quick, include)
        return 0

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = OUTPUTS / ("_runlog_shoulds_%s.txt" % stamp)

    print_plan(steps, args.quick, include)
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
        log.write("SHOULD / COULD / ADDITIONS analyses runlog\n")
        log.write("started      %s\n" % datetime.now().isoformat(timespec="seconds"))
        log.write("interpreter  %s\n" % sys.executable)
        log.write("repo root    %s\n" % REPO_ROOT)
        log.write("quick        %s\n" % args.quick)
        log.write("tiers        %s\n" % ", ".join(include))
        log.write("steps        %s\n" % ", ".join(s[0] for s in steps))
        log.write("\n")
        log.flush()

        for i, (script, tier, expected, basis, desc) in enumerate(steps, 1):
            path = ANALYSIS / script
            printable = "python analysis\\%s%s" % (script, " --quick" if args.quick else "")
            log.write("\n%s\n>>> [%d/%d] %s   tier %s\n    %s\n    started %s\n%s\n"
                      % (RULE, i, len(steps), script, tier, printable,
                         datetime.now().isoformat(timespec="seconds"), RULE))
            log.flush()

            if not path.exists():
                log.write("MISSING: %s does not exist\n" % path)
                log.flush()
                print("  [%2d/%d] %-34s FAIL     (script not found)" % (i, len(steps), script))
                results.append((script, tier, 127, 0.0, expected, basis, "FAIL"))
                continue

            t0 = time.time()
            proc = subprocess.run([sys.executable, str(path), *passthrough],
                                  cwd=str(REPO_ROOT), stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
            dt = time.time() - t0
            log.write("\n<<< exit %d after %s\n" % (proc.returncode, fmt(dt)))
            log.flush()

            if proc.returncode == 0:
                verdict = "PASS"
            elif proc.returncode == SKIP_EXIT.get(script):
                verdict = "SKIPPED"
            else:
                verdict = "FAIL"

            print("  [%2d/%d] %-34s %-7s  %7s  (expected %s, %s)"
                  % (i, len(steps), script, verdict, fmt(dt), fmt(expected), basis))
            if verdict == "SKIPPED":
                print("         exit 2: it refused cleanly and wrote nothing. Re-run it on "
                      "a machine that can reach the image root.")
            elif verdict == "FAIL":
                print("         exit %d. The queue continues. Read the log before trusting "
                      "any table this step writes." % proc.returncode)
            results.append((script, tier, proc.returncode, dt, expected, basis, verdict))

        elapsed = time.time() - t_all
        out = ["", RULE, "SHOULD / COULD / ADDITIONS SUMMARY", RULE]
        for tier in TIERS:
            rows = [r for r in results if r[1] == tier]
            if not rows:
                continue
            out.append("")
            out.append("  %s" % tier.upper())
            out.append("  %-34s %8s %6s %9s %9s %s"
                       % ("step", "verdict", "exit", "wall", "expected", "basis"))
            for script, _t, rc, dt, expected, basis, verdict in rows:
                out.append("  %-34s %8s %6d %9s %9s %s"
                           % (script, verdict, rc, fmt(dt), fmt(expected), basis))
            out.append("  %-34s %8s %6s %9s %9s"
                       % ("subtotal", "", "", fmt(sum(r[3] for r in rows)),
                          fmt(sum(r[4] for r in rows))))
        out.append("  " + "-" * 74)
        out.append("  %-34s %8s %6s %9s %9s"
                   % ("TOTAL", "", "", fmt(elapsed), fmt(sum(r[4] for r in results))))
        passed = [r for r in results if r[6] == "PASS"]
        skipped = [r for r in results if r[6] == "SKIPPED"]
        failed = [r for r in results if r[6] == "FAIL"]
        out.append("")
        out.append("  %d passed, %d skipped, %d failed, out of %d step(s)"
                   % (len(passed), len(skipped), len(failed), len(results)))
        if skipped:
            out.append("  skipped: %s" % ", ".join(r[0] for r in skipped))
        if failed:
            out.append("  FAILED:  %s" % ", ".join(r[0] for r in failed))
            out.append("  Do not quote a table written by a failed step.")
        if args.quick:
            out.append("")
            out.append("  THIS WAS A --quick RUN. The tables now on disk hold 200-resample")
            out.append("  intervals. Re-run without --quick before reading any of them.")
        out.append("  log: %s" % log_path)
        out.append(RULE)
        text = "\n".join(out)
        print(text)
        log.write("\n" + text + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
