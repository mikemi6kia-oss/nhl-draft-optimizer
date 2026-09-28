"""League + model settings. Everything tunable lives here so the app can expose it."""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- League (from the Yahoo settings screenshot) ---------------------------------
# Roster: C, C, W, W, W, W, D, D, D, D, Util, G, G, BN x6, IR, IR+ x4
STARTING_SLOTS: dict[str, int] = {"C": 2, "W": 4, "D": 4, "Util": 1, "G": 2}
BENCH_SLOTS = 6                       # IR / IR+ are not drafted
ROUNDS = sum(STARTING_SLOTS.values()) + BENCH_SLOTS   # 19

SKATER_CATS = ["G", "A", "P", "PPP", "GWG", "SOG"]
GOALIE_CATS = ["W", "GAA", "SV", "SV%"]
ALL_CATS = SKATER_CATS + GOALIE_CATS

# Positions used internally. Yahoo "W" covers LW and RW.
POSITIONS = ["C", "W", "D", "G"]
SKATER_ACTIVE = STARTING_SLOTS["C"] + STARTING_SLOTS["W"] + STARTING_SLOTS["D"] + STARTING_SLOTS["Util"]  # 11
GOALIE_ACTIVE = STARTING_SLOTS["G"]  # 2


@dataclass(frozen=True)
class ModelSettings:
    # league
    n_teams: int = 12
    # projection
    weeks: float = 26.0              # fantasy weeks the NHL regular season spans
    gp_regression: float = 0.35      # share of missed games (to 82) given back to established skaters
    goalie_gp_regression: float = 0.25   # same idea for goalies, toward goalie_gp_target
    goalie_gp_target: float = 60.0
    rate_shrink_games: float = 10.0  # Bayesian shrinkage (in games) of small-sample skater rates
    svpct_shrink_shots: float = 400.0
    gaa_shrink_minutes: float = 600.0
    # valuation
    method: str = "h2h"              # "h2h" (G-score, matchup-scaled) or "z" (classic z-score)
    goalie_weight: float = 1.0       # extra user multiplier on goalie categories
    cat_weights: tuple = field(default_factory=tuple)   # ((cat, weight), ...) overrides, e.g. punts
    bench_factor: float = 0.35       # share of a bench player's value that reaches your lineup
    scarcity: float = 0.5            # 1 = full positional replacement-level gaps, 0 = ignore position
                                     # (0.5 won the simulation test: see METHOD.md)
    skater_bench_pool: int = 5       # bench spots per team assumed used by skaters (pool sizing)
    goalie_bench_pool: int = 1       # bench spots per team assumed used by goalies

    def weight(self, cat: str) -> float:
        return dict(self.cat_weights).get(cat, 1.0)

    @property
    def skater_pool_size(self) -> int:
        return self.n_teams * (SKATER_ACTIVE + self.skater_bench_pool)

    @property
    def goalie_pool_size(self) -> int:
        return self.n_teams * (GOALIE_ACTIVE + self.goalie_bench_pool)
