"""Keepers: keepers.csv

Columns
    manager   team name as entered in the app's draft-order list, a draft slot number (1..N), or ME
    player    player name (accents/case don't matter)
    round     optional - the draft round the keeper costs. Blank = that team's latest open round.
    nhl_team  optional - only needed to break ties between same-named players (e.g. Elias Pettersson)
    pos       optional - same purpose (C / LW / RW / D / G)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nhl_data import DATA_DIR, _plain_strings, match_player, norm_name
from nhl_draft import DraftState


def load_keepers(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "keepers.csv"
    if not Path(path).exists():
        return pd.DataFrame(columns=["manager", "player", "round", "nhl_team", "pos"])
    df = _plain_strings(pd.read_csv(path, dtype=str)).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    return df[df.get("player", pd.Series(dtype=str)).astype(str).str.strip() != ""]


def apply_keepers(state: DraftState, keepers: pd.DataFrame, players: pd.DataFrame) -> list[str]:
    """Place keepers into the draft. Returns human-readable problems (never raises)."""
    problems = []
    keepers = _plain_strings(keepers.copy()).fillna("")
    names = {norm_name(n): i for i, n in enumerate(state.team_names)}
    names[norm_name("me")] = state.my_slot
    names[norm_name("★ ME")] = state.my_slot
    # rounds given explicitly first, so blank-round keepers don't grab those picks
    kp = keepers.copy()
    kp["_r"] = pd.to_numeric(kp.get("round", ""), errors="coerce")
    kp = kp.sort_values("_r", na_position="last")
    for _, r in kp.iterrows():
        m = str(r.get("manager", "")).strip()
        if m.isdigit() and 1 <= int(m) <= state.n_teams:
            team = int(m) - 1
        elif norm_name(m) in names:
            team = names[norm_name(m)]
        else:
            problems.append(f"Unknown manager '{m}' for {r['player']} — use a draft slot number, ME, or a team name")
            continue
        pid = match_player(r["player"], players, r.get("nhl_team") or None, r.get("pos") or None)
        if pid is None:
            problems.append(f"Couldn't match player '{r['player']}' (add nhl_team/pos if the name is shared)")
            continue
        rnd = int(r["_r"]) if pd.notna(r["_r"]) else None
        try:
            state.add_keeper(team, pid, rnd)
        except ValueError as e:
            problems.append(f"{r['player']}: {e}")
    return problems
