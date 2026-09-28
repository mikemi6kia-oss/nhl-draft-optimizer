"""How the *other* managers are likely to draft ("market rank").

Your opponents don't use your model. Best source is real ADP: drop Yahoo's ADP / pre-draft rank into
adp.csv (columns: name, adp  [optional: team, pos]) and it takes over automatically.

Without it we build a proxy that mimics a typical Yahoo room: half "raw stat-line" ranking (classic
z-score total, which is roughly what Yahoo's own rank reflects) and half positional-scarcity-aware
ranking (classic z VORP), so elite defensemen and goalies go earlier than raw stats alone suggest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhl_config import ModelSettings
from nhl_data import DATA_DIR, _plain_strings, match_player
from nhl_valuation import value_players


def load_adp(data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    p = data_dir / "adp.csv"
    if not p.exists():
        return None
    df = _plain_strings(pd.read_csv(p))
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"name", "adp"} <= set(df.columns) or df.empty:
        return None
    return df


def market_rank(sk: pd.DataFrame, gl: pd.DataFrame, n_teams: int, adp: pd.DataFrame | None = None
                ) -> tuple[pd.Series, str, list[str]]:
    """Returns (pid -> market rank (1 = first off the board), source label, unmatched ADP names)."""
    z = value_players(sk, gl, ModelSettings(n_teams=n_teams, method="z"))
    blend = (z["z_total"].rank(ascending=False) + z["vorp"].rank(ascending=False)) / 2
    proxy = pd.Series(blend.rank(method="first").to_numpy(), index=z["pid"])
    if adp is None:
        return proxy, "proxy (z-score blend)", []

    allp = z[["pid", "name_key", "team", "pos"]]
    rank = {}
    unmatched = []
    for _, r in adp.iterrows():
        pid = match_player(r["name"], allp, r.get("team"), r.get("pos"))
        if pid is None:
            unmatched.append(str(r["name"]))
        elif pid not in rank:
            rank[pid] = float(r["adp"])
    out = pd.Series(rank, dtype=float)
    top = out.max() if len(out) else 0.0
    rest = proxy.drop(out.index, errors="ignore").sort_values()
    out = pd.concat([out, pd.Series(top + np.arange(1, len(rest) + 1), index=rest.index)])
    return out.rank(method="first"), f"ADP file ({len(rank)} matched)", unmatched
