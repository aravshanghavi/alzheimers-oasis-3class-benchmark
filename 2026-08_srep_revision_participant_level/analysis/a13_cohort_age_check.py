#!/usr/bin/env python3
"""Are all 266 Non Demented participants actually assessed for dementia?

OASIS-1 is titled "Cross-sectional MRI Data in Young, Middle Aged, Nondemented
and Demented Older Adults". It spans ages 18 to 96, and the Clinical Dementia
Rating was administered to the older adults. Younger subjects carry no CDR value.

The Kaggle derivative sorts images into four folders by name. If subjects with a
BLANK CDR were placed in the "Non Demented" folder, then part of the majority
class is young brains rather than clinically confirmed CDR 0, and part of the
reported discrimination is age detection rather than dementia detection.

This settles it. It needs one file that is not in the repository:

    oasis_cross-sectional.csv     from https://sites.wustl.edu/oasisbrains/

Put it anywhere under the project folder, then run from the project root:

    python analysis/a13_cohort_age_check.py

numpy and pandas only. No GPU, no retraining.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "Non Demented", 1: "Very Mild Demented", 2: "Demented"}


def find_csv() -> Path:
    hits = [Path(p) for p in glob.glob(str(ROOT / "**" / "*cross-sectional*.csv"), recursive=True)]
    hits += [Path(p) for p in glob.glob(str(ROOT / "**" / "oasis*.csv"), recursive=True)]
    hits = [h for h in hits if "outputs" not in h.parts]
    if not hits:
        sys.exit("Could not find oasis_cross-sectional.csv under the project folder.\n"
                 "Download it from https://sites.wustl.edu/oasisbrains/ and drop it in, then re-run.")
    return hits[0]


def cohort() -> pd.DataFrame:
    for mf in sorted((ROOT / "experiments").glob("exp01_main_sweep/outputs/*/manifest.json")):
        if json.loads(mf.read_text()).get("status") != "COMPLETE":
            continue
        fp = mf.parent / "fold_participants.csv"
        if fp.exists():
            return pd.read_csv(fp)[["participant_id", "class_3"]].drop_duplicates().reset_index(drop=True)
    sys.exit("No completed exp01 run with fold_participants.csv found.")


def main() -> int:
    csv = find_csv()
    print(f"[a13] cohort age and CDR check\n  demographics: {csv}")
    demo = pd.read_csv(csv)
    idcol = next((c for c in demo.columns if c.strip().upper() in ("ID", "SUBJECT ID", "SUBJECT_ID")), demo.columns[0])
    cdrcol = next((c for c in demo.columns if c.strip().upper() == "CDR"), None)
    agecol = next((c for c in demo.columns if c.strip().upper() == "AGE"), None)
    if cdrcol is None or agecol is None:
        sys.exit(f"CSV has no CDR or Age column. Columns present: {list(demo.columns)}")

    demo["participant_id"] = demo[idcol].astype(str).str.extract(r"(OAS\d_\d{4})", expand=False).str.upper()
    demo = demo.dropna(subset=["participant_id"]).drop_duplicates("participant_id")

    print(f"\n  demographics file: {len(demo)} unique participants")
    print(f"  CDR blank in file: {int(demo[cdrcol].isna().sum())}")
    print("  CDR value counts in file:")
    print(demo[cdrcol].value_counts(dropna=False).sort_index().to_string())

    c = cohort()
    c["participant_id"] = c["participant_id"].astype(str).str.upper()
    m = c.merge(demo[["participant_id", cdrcol, agecol]], on="participant_id", how="left")
    unmatched = int(m[agecol].isna().sum())
    print(f"\n  study cohort: {len(c)} participants")
    if unmatched:
        print(f"  [warn] {unmatched} cohort participants had no row in the demographics file")

    print("\n  PER CLASS: how many have NO CDR assessment, and what ages")
    rows = []
    for k, name in CLASSES.items():
        s = m[m["class_3"] == k]
        rows.append({
            "Class": name, "Participants": len(s),
            "CDR assessed": int(s[cdrcol].notna().sum()),
            "CDR BLANK": int(s[cdrcol].isna().sum()),
            "Age min": float(s[agecol].min()) if s[agecol].notna().any() else np.nan,
            "Age median": float(s[agecol].median()) if s[agecol].notna().any() else np.nan,
            "Age max": float(s[agecol].max()) if s[agecol].notna().any() else np.nan,
            "Aged under 60": int((s[agecol] < 60).sum()),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    out = ROOT / "analysis" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "cohort_age_check.csv", index=False)

    nd = m[m["class_3"] == 0]
    n_blank = int(nd[cdrcol].isna().sum())
    n_young = int((nd[agecol] < 60).sum())
    print("\n" + "=" * 78)
    if n_blank == 0 and n_young == 0:
        print("  CLEAR. Every Non Demented participant carries a CDR assessment and is aged 60+.")
        print("  No age confound. The cohort is as the manuscript describes it.")
    else:
        print(f"  PROBLEM. {n_blank} of {len(nd)} Non Demented participants have NO CDR value,")
        print(f"  and {n_young} are aged under 60.")
        print("  Part of the majority class was never assessed for the outcome being predicted,")
        print("  so part of the reported discrimination is brain age rather than dementia.")
        print("\n  Restrict to CDR-assessed participants aged 60+ and re-run, and report the")
        print("  present cohort separately and explicitly as 'including unassessed younger")
        print("  controls'. Do not submit the current framing without this disclosure.")
    print("=" * 78)

    a0 = m[(m["class_3"] == 0) & m[agecol].notna()][agecol]
    a2 = m[(m["class_3"] == 2) & m[agecol].notna()][agecol]
    if len(a0) and len(a2):
        print(f"\n  Age: Non Demented median {a0.median():.0f}, Demented median {a2.median():.0f}, "
              f"difference {a2.median() - a0.median():.0f} years")
    print("\n    -> analysis/outputs/cohort_age_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
