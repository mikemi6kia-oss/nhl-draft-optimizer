"""Turn last season's stat lines into 2026-27 projections.

Two deliberately modest adjustments (both tunable, both can be set to 0 to use raw totals):

1. Games-played regression. Injuries are mostly random, so a regular who missed time gets part of it
   back. We only do this for established players (weight = min(1, GP/40)) so call-ups with 5 GP are
   not projected for 40 games.
2. Rate shrinkage. Per-game rates from small samples are noisy, so they are pulled toward a
   fringe-roster prior (30th percentile of regulars at the same position group) with a weight
   equivalent to `rate_shrink_games` games. Negligible for anyone with a real sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhl_config import ModelSettings

SKATER_COUNT_STATS = ["g", "a", "ppp", "gwg", "sog"]


def project_skaters(sk: pd.DataFrame, s: ModelSettings) -> pd.DataFrame:
    df = sk.copy()
    gp = df["gp"].clip(lower=0).astype(float)
    added = df.get("added", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    scale = s.season_games / 82.0     # 82-game history -> 84-game schedule
    base = (gp + s.gp_regression * (82 - gp) * np.minimum(1.0, gp / 40.0)).clip(upper=82) * scale
    # added players (extra_players.csv) already carry a full-season projection: no regression / scaling
    df["proj_gp"] = np.where(added, gp, base)

    group = np.where(df["pos"] == "D", "D", "F")
    regulars = gp >= 40
    for stat in SKATER_COUNT_STATS:
        rate = df[stat] / gp.replace(0, np.nan)
        prior = pd.Series(index=df.index, dtype=float)
        for grp in ("F", "D"):
            m = group == grp
            prior[m] = np.nanpercentile(rate[m & regulars], 30)
        k = np.where(added, 0.0, s.rate_shrink_games)   # added players' rates are already projections
        shrunk = (df[stat] + k * prior) / (gp + k)
        df[f"{stat}_pg"] = shrunk
        df[f"proj_{stat}"] = shrunk * df["proj_gp"]
    df = apply_availability(df, [f"proj_{st}" for st in SKATER_COUNT_STATS], s)
    df["proj_p"] = df["proj_g"] + df["proj_a"]
    return df


def project_goalies(gl: pd.DataFrame, s: ModelSettings) -> pd.DataFrame:
    df = gl.copy()
    gp = df["gp"].astype(float)
    gs = df["gs"].astype(float)
    target = s.goalie_gp_target
    scale = s.season_games / 82.0
    df["proj_gp"] = (gp + s.goalie_gp_regression * np.maximum(0, target - gp) * np.minimum(1.0, gs / 40.0)) * scale

    # league-wide rates from goalies with meaningful workloads
    reg = gp >= 20
    lg_svp = df.loc[reg, "sv"].sum() / df.loc[reg, "sa"].sum()
    lg_gaa = df.loc[reg, "ga"].sum() * 60 / df.loc[reg, "toi"].sum()
    lg_wpg = df.loc[reg, "w"].sum() / df.loc[reg, "gp"].sum()

    svp = (df["sv"] + s.svpct_shrink_shots * lg_svp) / (df["sa"] + s.svpct_shrink_shots)
    gaa = (df["ga"] + s.gaa_shrink_minutes / 60 * lg_gaa) / (df["toi"] + s.gaa_shrink_minutes) * 60
    wpg = (df["w"] + 10 * lg_wpg * 0.8) / (gp + 10)   # small-sample W rate pulled toward slightly-below-average
    sa_pg = df["sa"] / gp
    toi_pg = df["toi"] / gp

    df["proj_w"] = wpg * df["proj_gp"]
    df["proj_sa"] = sa_pg * df["proj_gp"]
    df["proj_sv"] = df["proj_sa"] * svp
    df["proj_toi"] = toi_pg * df["proj_gp"]
    df["proj_ga"] = gaa * df["proj_toi"] / 60
    df["proj_svpct"] = svp
    df["proj_gaa"] = gaa
    return apply_availability(df, ["proj_w", "proj_sa", "proj_sv", "proj_toi", "proj_ga"], s)


def apply_availability(df: pd.DataFrame, cols: list[str], s: ModelSettings) -> pd.DataFrame:
    """Known absences / health updates from availability.csv (attached by nhl_data as avail_* columns).

    avail_gp      replace the model's projected games (e.g. star healthy again after a lost season)
    avail_missed  games he'll miss this season (e.g. out until November)
    Counting stats scale with games played; rates (per-game, SV%, GAA) are unchanged."""
    if "avail_gp" not in df.columns:
        return df
    old = df["proj_gp"].astype(float)
    new = pd.to_numeric(df["avail_gp"], errors="coerce").fillna(old)
    new = (new - pd.to_numeric(df["avail_missed"], errors="coerce").fillna(0)).clip(lower=0, upper=s.season_games)
    ratio = (new / old.replace(0, np.nan)).fillna(0.0)
    for c in cols:
        df[c] = df[c] * ratio
    df["proj_gp"] = new
    return df
