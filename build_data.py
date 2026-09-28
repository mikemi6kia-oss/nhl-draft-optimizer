"""Convert the raw season workbook (SKATERS / GOALIES tabs) into clean CSVs.

Usage:
    python scripts/build_data.py [path/to/workbook.xlsx]

Writes skaters.csv and goalies.csv. Re-run whenever the workbook changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_XLSX = ROOT / "2025-26_season.xlsx"


def _num(s: pd.Series) -> pd.Series:
    """'1,483' -> 1483, '--' -> NaN."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip().replace({"--": np.nan, "": np.nan}),
        errors="coerce",
    )


def _mmss_to_min(s: pd.Series) -> pd.Series:
    """'22:59' -> 22.983 minutes, '3430:45' -> 3430.75 minutes."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if ":" not in v:
            return pd.to_numeric(v, errors="coerce")
        m, sec = v.split(":")
        return int(m) + int(sec) / 60.0
    return s.map(conv).astype(float)


def build(xlsx: Path = DEFAULT_XLSX) -> tuple[pd.DataFrame, pd.DataFrame]:
    sk = pd.read_excel(xlsx, "SKATERS")
    gl = pd.read_excel(xlsx, "GOALIES")

    skaters = pd.DataFrame({
        "name": sk["Player"].astype(str).str.strip(),
        "season": sk["Season"],
        "team": sk["Team"].astype(str).str.replace(" ", ""),
        "shoots": sk["S/C"].replace({"--": ""}),
        "pos": sk["Pos"].astype(str).str.strip(),          # C / L / R / D
        "gp": _num(sk["GP"]),
        "g": _num(sk["G"]),
        "a": _num(sk["A"]),
        "p": _num(sk["P"]),
        "plus_minus": _num(sk["+/-"]),
        "pim": _num(sk["PIM"]),
        "ppg": _num(sk["PPG"]),
        "ppp": _num(sk["PPP"]),
        "shg": _num(sk["SHG"]),
        "gwg": _num(sk["GWG"]),
        "sog": _num(sk["S"]),
        "toi_pg": _mmss_to_min(sk["TOI/GP"]),
        "fow_pct": _num(sk["FOW%"]),
    })

    goalies = pd.DataFrame({
        "name": gl["Player"].astype(str).str.strip(),
        "season": gl["Season"],
        "team": gl["Team"].astype(str).str.replace(" ", ""),
        "catches": gl["S/C"].replace({"--": ""}),
        "gp": _num(gl["GP"]),
        "gs": _num(gl["GS"]),
        "w": _num(gl["W"]),
        "l": _num(gl["L"]),
        "otl": _num(gl["OT"]),
        "sa": _num(gl["SA"]),
        "sv": _num(gl["Svs"]),
        "ga": _num(gl["GA"]),
        "sv_pct": _num(gl["Sv%"]),
        "gaa": _num(gl["GAA"]),
        "toi": _mmss_to_min(gl["TOI"]),
        "so": _num(gl["SO"]),
    })

    # --- integrity checks: fail loudly rather than silently mis-value players ---
    num_sk = ["gp", "g", "a", "p", "ppp", "gwg", "sog"]
    assert skaters[num_sk].notna().all().all(), "missing skater counting stats"
    assert (skaters["g"] + skaters["a"] == skaters["p"]).all(), "G + A != P for some skater"
    assert set(skaters["pos"]) <= {"C", "L", "R", "D"}, set(skaters["pos"])
    num_g = ["gp", "gs", "w", "sa", "sv", "ga", "toi"]
    assert goalies[num_g].notna().all().all(), "missing goalie stats"
    assert (goalies["sv"] <= goalies["sa"]).all()
    # SV% and GAA recompute to the published values (sanity on TOI parsing)
    svp = goalies["sv"] / goalies["sa"]
    assert (abs(svp - goalies["sv_pct"]) < 0.0015).all(), "SV% mismatch"
    gaa = goalies["ga"] * 60 / goalies["toi"]
    assert (abs(gaa - goalies["gaa"]) < 0.02).all(), "GAA mismatch"
    return skaters, goalies


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    s, g = build(path)
    s.to_csv(ROOT / "skaters.csv", index=False)
    g.to_csv(ROOT / "goalies.csv", index=False)
    print(f"wrote {len(s)} skaters, {len(g)} goalies")
