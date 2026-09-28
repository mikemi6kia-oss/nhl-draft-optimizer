"""Loading player data, building stable player ids, and name matching."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

POS_MAP = {"C": "C", "L": "W", "R": "W", "D": "D"}


def norm_name(name: str) -> str:
    """Accent/case/punctuation-insensitive key: 'Tim Stützle' -> 'tim stutzle'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _slug(s: str) -> str:
    return norm_name(s).replace(" ", "-")


def _plain_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Force text columns to plain Python-object strings.

    pandas 3 stores text in a backend that depends on whether pyarrow is installed (it is on
    Streamlit Cloud, it may not be locally), and some string operations behave differently between
    the two. Plain object columns behave identically everywhere."""
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != object:
            df[c] = pd.Series(list(df[c]), index=df.index, dtype=object)
    return df


def read_csv_plain(path, **kw) -> pd.DataFrame:
    return _plain_strings(pd.read_csv(path, **kw))


def load_players(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    sk = read_csv_plain(data_dir / "skaters.csv")
    gl = read_csv_plain(data_dir / "goalies.csv")

    sk["yahoo_pos"] = sk["pos"].map({"C": "C", "L": "LW", "R": "RW", "D": "D"})
    sk["pos"] = sk["pos"].map(POS_MAP)
    sk["elig"] = sk["pos"].map(lambda p: (p,))
    gl["pos"] = "G"
    gl["yahoo_pos"] = "G"
    gl["elig"] = [("G",)] * len(gl)

    # Optional: multi-position eligibility (Yahoo gives e.g. C/LW). eligibility.csv: name,positions,team(optional)
    elig_path = data_dir / "eligibility.csv"
    if elig_path.exists():
        el = read_csv_plain(elig_path).fillna("")
        for _, r in el.iterrows():
            positions = tuple(dict.fromkeys(
                POS_MAP.get(p.strip().upper().replace("LW", "L").replace("RW", "R"), p.strip().upper())
                for p in str(r["positions"]).replace(",", "/").split("/") if p.strip()))
            mask = sk["name"].map(norm_name) == norm_name(r["name"])
            if "team" in el.columns and r.get("team"):
                want = str(r["team"]).upper()
                mask &= sk["team"].map(lambda t: want in str(t).upper())
            for i in sk.index[mask]:
                sk.at[i, "elig"] = positions
                sk.at[i, "pos"] = positions[0]
                sk.at[i, "yahoo_pos"] = "/".join(positions)

    for df in (sk, gl):
        df["name_key"] = [norm_name(n) for n in df["name"]]
        # current team = last listed for traded players ("MIN,VAN" -> VAN)
        df["team_now"] = [str(t).split(",")[-1].strip() for t in df["team"]]
    # Stable ids; disambiguate identical names (e.g. the two Elias Petterssons) by position, then team.
    # Built in plain Python so it can't depend on the pandas string backend.
    pids = [_slug(n) for n in sk["name"]]
    for extra in (list(sk["pos"]), list(sk["team_now"])):
        counts = pd.Series(pids).value_counts()
        pids = [f"{p}-{str(e).lower()}" if counts[p] > 1 else p for p, e in zip(pids, extra)]
    sk["pid"] = pd.Series(pids, index=sk.index, dtype=object)
    gl["pid"] = pd.Series([_slug(n) + "-g" for n in gl["name"]], index=gl.index, dtype=object)
    sk, gl = _plain_strings(sk), _plain_strings(gl)   # derived columns too
    assert not pd.concat([sk["pid"], gl["pid"]]).duplicated().any()
    return sk, gl


def match_player(name: str, players: pd.DataFrame, team: str | None = None, pos: str | None = None) -> str | None:
    """Return the pid for a free-text name (used for keepers / ADP files). None if not found or ambiguous."""
    key = norm_name(name)
    cand = players[players["name_key"] == key]
    if team and len(cand) > 1:
        cand = cand[cand["team"].str.upper().str.contains(str(team).upper())]
    if pos and len(cand) > 1:
        p = POS_MAP.get(pos.upper(), pos.upper())
        cand = cand[cand["pos"] == p]
    if len(cand) == 1:
        return cand["pid"].iloc[0]
    if len(cand) == 0:
        # last-name + first initial fallback, e.g. "J. Hughes"
        parts = key.split()
        if len(parts) >= 2:
            last = parts[-1]
            first_initial = parts[0][0]
            c2 = players[players["name_key"].str.endswith(" " + last) & players["name_key"].str.startswith(first_initial)]
            if len(c2) == 1:
                return c2["pid"].iloc[0]
    return None
