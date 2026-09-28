"""Snake-draft state: order, picks, keepers. Plain dict/list so it serialises to JSON for save/load."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from nhl_config import ROUNDS


def owner(k: int, n_teams: int) -> int:
    """0-based team slot that owns overall pick k (0-based) in a snake draft."""
    r, i = divmod(k, n_teams)
    return i if r % 2 == 0 else n_teams - 1 - i


def team_picks(team: int, n_teams: int, rounds: int = ROUNDS) -> list[int]:
    return [k for k in range(n_teams * rounds) if owner(k, n_teams) == team]


@dataclass
class DraftState:
    n_teams: int = 12
    my_slot: int = 0                       # 0-based
    rounds: int = ROUNDS
    team_names: list = field(default_factory=list)
    picks: dict = field(default_factory=dict)   # overall pick (int) -> {"pid": str, "keeper": bool}

    def __post_init__(self):
        names = [str(n).strip() for n in (self.team_names or []) if str(n).strip()][: self.n_teams]
        names += [f"Team {i + 1}" for i in range(len(names), self.n_teams)]
        # mark my team; keeper matching ignores the star, so "Mike" still matches "★ Mike"
        if not names[self.my_slot].startswith("★"):
            names[self.my_slot] = "★ ME" if names[self.my_slot] == f"Team {self.my_slot + 1}" else f"★ {names[self.my_slot]}"
        self.team_names = names

    # --- queries -----------------------------------------------------------------------
    @property
    def total_picks(self) -> int:
        return self.n_teams * self.rounds

    @property
    def current(self) -> int | None:
        for k in range(self.total_picks):
            if k not in self.picks:
                return k
        return None

    def owner(self, k: int) -> int:
        return owner(k, self.n_teams)

    def taken(self) -> set:
        return {v["pid"] for v in self.picks.values()}

    def roster(self, team: int) -> list[str]:
        return [v["pid"] for k, v in sorted(self.picks.items()) if self.owner(k) == team]

    def my_next_picks(self, after: int | None = None) -> list[int]:
        start = self.current if after is None else after
        if start is None:
            return []
        return [k for k in team_picks(self.my_slot, self.n_teams, self.rounds) if k >= start and k not in self.picks]

    def label(self, k: int) -> str:
        r, i = divmod(k, self.n_teams)
        return f"R{r + 1}.{i + 1:02d} (#{k + 1})"

    # --- mutations ---------------------------------------------------------------------
    def draft(self, pid: str, k: int | None = None) -> int:
        k = self.current if k is None else k
        if k is None:
            raise ValueError("draft is complete")
        if pid in self.taken():
            raise ValueError(f"{pid} already drafted")
        self.picks[k] = {"pid": pid, "keeper": False}
        return k

    def undo(self) -> int | None:
        live = [k for k, v in self.picks.items() if not v["keeper"]]
        if not live:
            return None
        k = max(live)
        del self.picks[k]
        return k

    def reset(self, keep_keepers: bool = True):
        self.picks = {k: v for k, v in self.picks.items() if keep_keepers and v["keeper"]}

    def add_keeper(self, team: int, pid: str, round_: int | None = None) -> int:
        """Place a keeper on `team`'s pick in `round_` (1-based). Without a round, the team's
        latest free round is used (keepers cost a late pick)."""
        slots = [k for k in team_picks(team, self.n_teams, self.rounds) if k not in self.picks]
        if round_:
            want = [k for k in slots if k // self.n_teams == round_ - 1]
            if not want:
                raise ValueError(f"{self.team_names[team]} has no free pick in round {round_}")
            k = want[0]
        else:
            if not slots:
                raise ValueError("no free picks left for keeper")
            k = slots[-1]
        if pid in self.taken():
            raise ValueError(f"{pid} already on a roster")
        self.picks[k] = {"pid": pid, "keeper": True}
        return k

    # --- persistence -------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"n_teams": self.n_teams, "my_slot": self.my_slot, "rounds": self.rounds,
                           "team_names": self.team_names,
                           "picks": {str(k): v for k, v in self.picks.items()}}, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "DraftState":
        d = json.loads(s)
        st = cls(n_teams=d["n_teams"], my_slot=d["my_slot"], rounds=d["rounds"], team_names=d["team_names"])
        st.picks = {int(k): v for k, v in d["picks"].items()}
        return st
