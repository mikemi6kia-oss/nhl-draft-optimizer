"""Model-independent check: projected weekly H2H category results from *raw* stat totals.

For each team the best lineup is set (starters count fully, bench players at bench_factor), weekly
totals are built from projected raw stats, and each category is decided with a normal approximation
of weekly noise (Poisson counts; binomial SV%; Poisson GA over minutes for GAA). Output: probability
of winning each category against each opponent, and expected categories won per week (out of 10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from nhl_config import ALL_CATS, ModelSettings
from nhl_valuation import VAR_GAMES_PER_WEEK, VAR_SAVES_PER_START

SK_COLS = {"G": "proj_g", "A": "proj_a", "P": "proj_p", "PPP": "proj_ppp", "GWG": "proj_gwg", "SOG": "proj_sog"}


def team_weekly(df: pd.DataFrame, weights: np.ndarray, s: ModelSettings) -> dict:
    """weights: per-row share of that player's production that reaches the lineup."""
    wk = s.weeks
    out = {}
    f = lambda c: np.nan_to_num(df[c].to_numpy(dtype=float))
    for cat, col in SK_COLS.items():
        tot = f(col) / wk
        gp = np.maximum(f("proj_gp"), 1)
        r = f(col) / gp
        out[cat] = (weights @ tot, weights @ (tot + r ** 2 * VAR_GAMES_PER_WEEK))
    w_ = f("proj_w") / wk
    out["W"] = (weights @ w_, weights @ w_)
    sv = f("proj_sv") / wk
    sa = f("proj_sa") / wk
    starts = f("proj_gp") / wk
    svper = np.where(f("proj_gp") > 0, f("proj_sv") / np.maximum(f("proj_gp"), 1), 0)
    out["SV"] = (weights @ sv, weights @ (starts * (VAR_SAVES_PER_START + svper ** 2)))
    SA, SV = weights @ sa, weights @ sv
    p = SV / SA if SA > 0 else 0.9
    out["SV%"] = (p, p * (1 - p) / max(SA, 1e-9))
    ga = f("proj_ga") / wk
    toi = f("proj_toi") / wk
    GA, TOI = weights @ ga, weights @ toi
    gaa = GA * 60 / TOI if TOI > 0 else 3.5
    out["GAA"] = (gaa, GA * (60 / max(TOI, 1e-9)) ** 2)
    return out


def matchup_matrix(team_stats: list[dict]) -> np.ndarray:
    """P(team i wins category c vs team j): array [i, j, c]."""
    n = len(team_stats)
    out = np.full((n, n, len(ALL_CATS)), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k, c in enumerate(ALL_CATS):
                mi, vi = team_stats[i][c]
                mj, vj = team_stats[j][c]
                z = (mi - mj) / np.sqrt(vi + vj + 1e-12)
                out[i, j, k] = norm.cdf(-z if c == "GAA" else z)
    return out


def league_matchups(state, board, s: ModelSettings) -> dict:
    """Expected weekly categories won (0-10) by each team vs every other, from rosters in `state`."""
    from nhl_recommend import assign_lineup   # local import to avoid a cycle
    df = board.df
    vals = df["value"].to_numpy()
    repl = df.attrs["repl"]
    stats = []
    for t in range(state.n_teams):
        ids = board.idx(state.roster(t))
        w = np.zeros(len(df))
        if len(ids):
            lu = assign_lineup(ids, vals, board, repl)
            for sl in lu["start"].values():
                w[sl] = 1.0
            w[lu["bench"]] = s.bench_factor
        stats.append(team_weekly(df, w, s))
    M = matchup_matrix(stats)
    exp_wins = np.nansum(M, axis=2)                 # [i, j] expected cats won by i vs j
    cat_rate = np.nanmean(M, axis=1)                # [i, c] avg P(win cat) vs field
    return {"cats_won": exp_wins, "matrix": M,"cat_win_rate": pd.DataFrame(cat_rate, columns=ALL_CATS, index=state.team_names),
            "avg_cats_won": pd.Series(np.nanmean(exp_wins, axis=1), index=state.team_names),
            "team_stats": stats}
