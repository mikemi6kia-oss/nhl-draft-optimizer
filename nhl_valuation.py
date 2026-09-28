"""Category scoring and player value.

Three layers, each one fixing a known weakness of the one before it:

1. Classic Z-score (what Yahoo's "standard deviation" view is doing):
       z = (player weekly mean - pool mean) / SD of weekly means across the draftable pool

2. G-score (Rosenof 2024, "Improving algorithms for fantasy basketball"): in head-to-head
   categories you win a *week*, so a category that is very noisy week-to-week (GWG, W) is worth
   less per unit of season-long edge than a stable one (SOG). The denominator adds each
   player's week-to-week variance:
       g = (mean - pool mean) / sqrt(SD_between^2 + SD_within^2)
   We don't have weekly logs, so within-week variance is modelled: counting stats as Poisson
   (plus schedule variance of games per week), saves as a compound of starts x shots.

3. Matchup scaling. A category's weekly result is the sum over the players who *contribute* to it:
   11 active skaters for G/A/P/PPP/GWG/SOG but only 2 goalies for W/GAA/SV/SV%. One goalie is
   therefore a much larger share of a category's matchup variance, and a 1-SD goalie edge moves
   your win probability sqrt(11/2) = 2.35x more than a 1-SD skater edge. Scores are expressed in
   "skater-equivalent" units so the ten categories are comparable.

Rate stats are volume-weighted so a 5-game hot streak doesn't look like a Vezina season:
   SV%  -> saves above average  = SV - pool_SV% x SA
   GAA  -> goals prevented      = pool_GAA x TOI/60 - GA
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhl_config import (ALL_CATS, BENCH_SLOTS, GOALIE_ACTIVE, GOALIE_CATS, SKATER_ACTIVE, SKATER_CATS,
                     STARTING_SLOTS, ModelSettings)
from nhl_projections import project_goalies, project_skaters

VAR_GAMES_PER_WEEK = 0.5     # NHL weekly schedules run 2-5 games; variance ~0.5 around ~3.2
VAR_SAVES_PER_START = 40.0   # SD of ~6.3 saves per start

SKATER_STAT = {"G": "proj_g", "A": "proj_a", "P": "proj_p", "PPP": "proj_ppp", "GWG": "proj_gwg", "SOG": "proj_sog"}


def _skater_weekly(df: pd.DataFrame, s: ModelSettings) -> tuple[dict, dict]:
    mu, var = {}, {}
    gp = df["proj_gp"].replace(0, np.nan)
    for cat, col in SKATER_STAT.items():
        m = df[col] / s.weeks
        r = (df[col] / gp).fillna(0)
        mu[cat] = m.to_numpy()
        var[cat] = (m + r ** 2 * VAR_GAMES_PER_WEEK).to_numpy()
    return mu, var


def _goalie_weekly(df: pd.DataFrame, s: ModelSettings, pool: np.ndarray) -> tuple[dict, dict]:
    p = df.loc[pool]
    lg_svp = p["proj_sv"].sum() / p["proj_sa"].sum()
    lg_gaa = p["proj_ga"].sum() * 60 / p["proj_toi"].sum()
    mu, var = {}, {}
    wk = s.weeks
    mu["W"] = (df["proj_w"] / wk).to_numpy()
    var["W"] = mu["W"].copy()                                  # Poisson starts x Bernoulli win
    starts = df["proj_gp"] / wk
    sv_per = (df["proj_sv"] / df["proj_gp"].replace(0, np.nan)).fillna(0)
    mu["SV"] = (df["proj_sv"] / wk).to_numpy()
    var["SV"] = (starts * (VAR_SAVES_PER_START + sv_per ** 2)).to_numpy()
    mu["SV%"] = ((df["proj_sv"] - lg_svp * df["proj_sa"]) / wk).to_numpy()
    var["SV%"] = (df["proj_sa"] / wk * lg_svp * (1 - lg_svp)).to_numpy()
    mu["GAA"] = ((lg_gaa * df["proj_toi"] / 60 - df["proj_ga"]) / wk).to_numpy()
    var["GAA"] = (df["proj_ga"] / wk).to_numpy()
    return mu, var


def _score(mu: dict, var: dict, pool: np.ndarray, cats: list[str]) -> tuple[dict, dict]:
    """Return (z, g) dicts of arrays, standardized against the pool."""
    z, g = {}, {}
    for c in cats:
        m = mu[c][pool]
        mean, sd_b = m.mean(), m.std()
        sd_w2 = var[c][pool].mean()
        z[c] = (mu[c] - mean) / sd_b
        g[c] = (mu[c] - mean) / np.sqrt(sd_b ** 2 + sd_w2)
    return z, g


def display_multiplier(cat: str, s: ModelSettings) -> float:
    """Scale so all categories are in skater-equivalent H2H units (see module docstring)."""
    if cat in GOALIE_CATS:
        base = np.sqrt(SKATER_ACTIVE / GOALIE_ACTIVE) if s.method == "h2h" else 1.0
        return base * s.goalie_weight * s.weight(cat)
    return s.weight(cat)


def value_players(sk: pd.DataFrame, gl: pd.DataFrame, s: ModelSettings) -> pd.DataFrame:
    skp = project_skaters(sk, s).reset_index(drop=True)
    glp = project_goalies(gl, s).reset_index(drop=True)

    # --- iterate: pool defines the scale, scores define the pool -------------------------
    sk_pool = np.argsort(-(skp["proj_p"] + skp["proj_sog"] / 10).to_numpy())[: s.skater_pool_size]
    gl_pool = np.argsort(-glp["proj_w"].to_numpy())[: s.goalie_pool_size]
    sk_mu, sk_var = _skater_weekly(skp, s)
    for _ in range(4):
        gl_mu, gl_var = _goalie_weekly(glp, s, gl_pool)
        sz, sg = _score(sk_mu, sk_var, sk_pool, SKATER_CATS)
        gz, gg = _score(gl_mu, gl_var, gl_pool, GOALIE_CATS)
        base_s = sg if s.method == "h2h" else sz
        base_g = gg if s.method == "h2h" else gz
        sv = sum(base_s[c] * display_multiplier(c, s) for c in SKATER_CATS)
        gv = sum(base_g[c] * display_multiplier(c, s) for c in GOALIE_CATS)
        new_sk = np.argsort(-sv)[: s.skater_pool_size]
        new_gl = np.argsort(-gv)[: s.goalie_pool_size]
        if set(new_sk) == set(sk_pool) and set(new_gl) == set(gl_pool):
            break
        sk_pool, gl_pool = new_sk, new_gl

    # --- assemble one table --------------------------------------------------------------
    keep_sk = ["pid", "name", "team", "team_now", "pos", "elig", "yahoo_pos", "gp", "g", "a", "p", "ppp", "gwg", "sog",
               "toi_pg", "proj_gp", "proj_g", "proj_a", "proj_p", "proj_ppp", "proj_gwg", "proj_sog", "name_key"]
    keep_gl = ["pid", "name", "team", "team_now", "pos", "elig", "yahoo_pos", "gp", "gs", "w", "sa", "sv", "ga", "sv_pct",
               "gaa", "proj_gp", "proj_w", "proj_sv", "proj_sa", "proj_ga", "proj_toi", "proj_svpct", "proj_gaa", "name_key"]
    A = skp[keep_sk].copy()
    B = glp[keep_gl].copy()
    for c in ALL_CATS:
        A[f"z_{c}"] = sz[c] if c in SKATER_CATS else np.nan
        B[f"z_{c}"] = gz[c] if c in GOALIE_CATS else np.nan
        A[f"g_{c}"] = sg[c] if c in SKATER_CATS else np.nan
        B[f"g_{c}"] = gg[c] if c in GOALIE_CATS else np.nan
    A["in_pool"] = False
    A.loc[sk_pool, "in_pool"] = True
    B["in_pool"] = False
    B.loc[gl_pool, "in_pool"] = True
    df = pd.concat([A, B], ignore_index=True)

    for c in ALL_CATS:
        base = df[f"g_{c}"] if s.method == "h2h" else df[f"z_{c}"]
        df[f"v_{c}"] = (base * display_multiplier(c, s)).fillna(0.0)      # value units, skater-equivalent
        n_c = GOALIE_ACTIVE if c in GOALIE_CATS else SKATER_ACTIVE
        df[f"m_{c}"] = (df[f"g_{c}"] / np.sqrt(2 * n_c)).fillna(0.0)      # matchup-SD units (for win odds)
    vcols = [f"v_{c}" for c in ALL_CATS]
    df["value"] = df[vcols].sum(axis=1)
    # "Yahoo-style" classic z total: plain z, every category weight 1, no matchup scaling
    df["z_total"] = df[[f"z_{c}" for c in ALL_CATS]].fillna(0).sum(axis=1)

    repl, waiver = replacement_levels(df["value"].to_numpy(), df["elig"].tolist(), s.n_teams)
    repl = scale_scarcity(repl, s.scarcity)
    df["repl"] = [min(repl[p] for p in e if p != "Util") for e in df["elig"]]
    df["waiver"] = [min(waiver[p] for p in e) for e in df["elig"]]
    df["vorp"] = df["value"] - df["repl"]
    df["rank"] = df["vorp"].rank(ascending=False, method="first").astype(int)
    df["pos_rank"] = df.groupby("pos")["vorp"].rank(ascending=False, method="first").astype(int)
    df["z_rank"] = df["z_total"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("rank").reset_index(drop=True)
    df.attrs["repl"] = repl
    df.attrs["waiver"] = waiver
    return df


def scale_scarcity(repl: dict, scarcity: float) -> dict:
    """Shrink skater positional replacement gaps toward the Util level (scarcity=0 -> position-blind)."""
    u = repl["Util"]
    return {p: (u + scarcity * (v - u) if p in ("C", "W", "D") else v) for p, v in repl.items()}


def replacement_levels(values: np.ndarray, elig: list[tuple], n_teams: int,
                       max_goalies_per_team: int = 4, available: np.ndarray | None = None,
                       open_slots: dict | None = None, bench_open: int | None = None,
                       goalie_room: int | None = None) -> tuple[dict, dict]:
    """Greedy league-wide allocation of the best players into every team's slots.

    repl[pos]   = best player at pos left out of the *starting* lineups (C/W/D/Util/G)
                  -> the scarcity baseline used for VORP.
    waiver[pos] = best player at pos left undrafted after all 19 rounds
                  -> what you can pick up for free; the baseline for bench value.

    Pre-draft this uses every slot in the league. Mid-draft, pass `available` plus the league's
    remaining `open_slots` / `bench_open` / `goalie_room` to get *dynamic* replacement levels that
    react to position runs.
    """
    order = np.argsort(-values)
    if available is not None:
        order = order[available[order]]
    if open_slots is None:
        open_slots = {p: n_teams * STARTING_SLOTS[p] for p in ("C", "W", "D", "G", "Util")}
    open_ = {p: open_slots[p] for p in ("C", "W", "D", "G")}
    util = open_slots["Util"]
    starters = np.zeros(len(values), bool)
    for i in order:
        e = elig[i]
        slot = next((p for p in e if open_[p] > 0), None)
        if slot:
            open_[slot] -= 1
            starters[i] = True
        elif e[0] != "G" and util > 0:
            util -= 1
            starters[i] = True
        if not any(open_.values()) and util == 0:
            break
    repl = {}
    for p in ("C", "W", "D", "G"):
        idx = [i for i in order if not starters[i] and p in elig[i]]
        repl[p] = float(values[idx[0]]) if idx else 0.0
    # Util can hold any skater, so its baseline is the best skater left out of all starting slots
    repl["Util"] = max(repl["C"], repl["W"], repl["D"])

    bench = n_teams * BENCH_SLOTS if bench_open is None else bench_open
    if goalie_room is None:
        goalie_room = n_teams * max_goalies_per_team - n_teams * STARTING_SLOTS["G"]
    drafted = starters.copy()
    for i in order:
        if bench == 0:
            break
        if drafted[i]:
            continue
        if "G" in elig[i]:
            if goalie_room == 0:
                continue
            goalie_room -= 1
        drafted[i] = True
        bench -= 1
    waiver = {}
    for p in ("C", "W", "D", "G"):
        idx = [i for i in order if not drafted[i] and p in elig[i]]
        waiver[p] = float(values[idx[0]]) if idx else 0.0
    return repl, waiver
