"""Live draft brain: opponent Monte Carlo, dynamic replacement levels, category needs, two-pick lookahead.

For each candidate i when you're on the clock:

    score(i) = gain_now(i) + E[ best gain available at your NEXT pick | you took i ]

* gain_now: how much your lineup improves (value over the *current* replacement level at the slot
  he'd fill; bench spots count at `bench_factor`). Replacement levels are recomputed from what is
  still on the board and how many starting slots the league still has to fill, so position runs
  move them.
* The expectation comes from simulating the opponents' picks between now and your next turn a few
  hundred times (market rank + ADP-style noise + roster-need rules). This is what makes the
  recommendation shift with draft position: a player who will almost surely come back to you is
  worth deferring; a position about to fall off a cliff is worth taking now.
* Category weights adapt to your roster: weight_c = (1-a) + a * exp(-edge_c^2 / 2), where edge_c is
  your projected weekly edge in category c (in matchup-SD units) over the league-average roster
  drafted so far. That is the slope of P(win category) - categories you are already winning big or
  losing big (de-facto punts) matter less; close ones matter most.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from nhl_config import ALL_CATS, BENCH_SLOTS, GOALIE_CATS, POSITIONS, STARTING_SLOTS, ModelSettings
from nhl_draft import DraftState, team_picks
from nhl_valuation import replacement_levels, scale_scarcity

POS_IDX = {p: i for i, p in enumerate(POSITIONS)}          # C W D G
SLOT_N = np.array([STARTING_SLOTS[p] for p in POSITIONS])  # 2 4 4 2
OPP_CAPS = np.array([5, 8, 7, 4])                           # sanity caps for simulated opponents


@dataclass
class Board:
    """Arrays aligned with the valuation DataFrame rows."""
    df: pd.DataFrame
    market: np.ndarray        # market rank (1 = first)
    pos_idx: np.ndarray
    pid_to_i: dict

    @classmethod
    def build(cls, df: pd.DataFrame, market: pd.Series) -> "Board":
        m = df["pid"].map(market).to_numpy(dtype=float)
        m = np.where(np.isnan(m), np.nanmax(m) + 1 + np.arange(len(m)), m)
        return cls(df=df, market=m, pos_idx=df["pos"].map(POS_IDX).to_numpy(),
                   pid_to_i={p: i for i, p in enumerate(df["pid"])})

    def idx(self, pids) -> np.ndarray:
        return np.array([self.pid_to_i[p] for p in pids if p in self.pid_to_i], dtype=int)


# ----------------------------------------------------------------------------------------------
# Roster bookkeeping
# ----------------------------------------------------------------------------------------------
BENCH_DECAY = 0.5   # 2nd bench player at the same position covers far fewer games than the 1st


def assign_lineup(ids: np.ndarray, vals: np.ndarray, board: Board, repl: dict) -> dict:
    """Greedy best lineup: best players first, each into his own position slot, then Util, then bench."""
    elig = board.df["elig"].to_numpy()
    open_ = dict(STARTING_SLOTS)
    out = {"start": {p: [] for p in ("C", "W", "D", "G", "Util")}, "bench": []}
    for i in sorted(ids, key=lambda i: -vals[i]):
        e = elig[i]
        slot = next((p for p in e if open_[p] > 0), None)
        if slot is None and e[0] != "G" and open_["Util"] > 0:
            slot = "Util"
        if slot:
            open_[slot] -= 1
            out["start"][slot].append(i)
        else:
            out["bench"].append(i)
    out["open"] = open_
    out["bench_open"] = BENCH_SLOTS - len(out["bench"])
    return out


def _bench_value(v: np.ndarray | float, pos: str, n_same: int, waiver: dict, bf: float):
    """A bench player only helps on days a starter he can replace is idle; the first one at a
    position gets most of those games, extra ones rarely play."""
    return bf * BENCH_DECAY ** n_same * np.maximum(0.0, v - waiver[pos])


def gains_for_roster(ids: np.ndarray, vals: np.ndarray, board: Board, repl: dict, waiver: dict,
                     bench_factor: float) -> np.ndarray:
    """Marginal lineup gain of adding each player to this roster (vectorised over the board).

    A starter's worth in a slot = value - replacement level of that slot (Util's baseline is the best
    skater left out of all starting slots, not the position's)."""
    elig = board.df["elig"].to_numpy()
    lu = assign_lineup(ids, vals, board, repl)
    pos_of = board.df["pos"].to_numpy()
    bench_n = {p: sum(1 for i in lu["bench"] if pos_of[i] == p) for p in ("C", "W", "D", "G")}
    bench_open = lu["bench_open"] > 0
    P = len(vals)
    best = np.full(P, -np.inf)

    def weakest(slot):
        s = lu["start"][slot]
        return min(s, key=lambda i: vals[i]) if s else None

    for pos in ("C", "W", "D", "G"):
        m = np.array([pos in e for e in elig])
        if not m.any():
            continue
        v = vals[m]
        opts = []
        # (a) straight into an open slot
        if lu["open"][pos] > 0:
            opts.append(v - repl[pos])
        elif pos != "G" and lu["open"]["Util"] > 0:
            opts.append(v - repl["Util"])
        else:
            # (b) bump the weakest starter he can replace to the bench
            for slot in [pos] + (["Util"] if pos != "G" else []):
                w = weakest(slot)
                if w is None:
                    continue
                wp = pos_of[w]
                w_bench = _bench_value(vals[w], wp, bench_n[wp], waiver, bench_factor) if bench_open else 0.0
                opts.append(np.where(v > vals[w], v - vals[w] + w_bench, -np.inf))
        # (c) bench
        if bench_open:
            opts.append(_bench_value(v, pos, bench_n[pos], waiver, bench_factor))
        g = np.max(np.vstack(opts), axis=0) if opts else np.zeros_like(v)
        best[m] = np.maximum(best[m], g)
    return best


def league_open_slots(state: DraftState, board: Board, vals: np.ndarray, repl: dict) -> tuple[dict, int, int]:
    open_tot = {p: 0 for p in ("C", "W", "D", "G", "Util")}
    bench_tot = 0
    goalie_room = 0
    for t in range(state.n_teams):
        ids = board.idx(state.roster(t))
        lu = assign_lineup(ids, vals, board, repl)
        for p, n in lu["open"].items():
            open_tot[p] += n
        bench_tot += lu["bench_open"]
        n_g = int((board.pos_idx[ids] == POS_IDX["G"]).sum()) if len(ids) else 0
        goalie_room += max(0, 4 - max(n_g, STARTING_SLOTS["G"]))
    return open_tot, bench_tot, goalie_room


# ----------------------------------------------------------------------------------------------
# Category needs
# ----------------------------------------------------------------------------------------------
def category_edges(state: DraftState, board: Board, bench_factor: float) -> pd.DataFrame:
    """Projected weekly edge per category (matchup-SD units) for every team, and win odds vs the
    average roster. Bench players count at bench_factor."""
    df = board.df
    mcols = [f"m_{c}" for c in ALL_CATS]
    M = df[mcols].to_numpy()
    vals = df["value"].to_numpy()
    repl = df.attrs["repl"]
    rows = []
    for t in range(state.n_teams):
        ids = board.idx(state.roster(t))
        w = np.zeros(len(df))
        if len(ids):
            lu = assign_lineup(ids, vals, board, repl)
            for s in lu["start"].values():
                w[s] = 1.0
            w[lu["bench"]] = bench_factor
        rows.append(w @ M)
    T = np.array(rows)
    edge = T - T.mean(axis=0, keepdims=True)
    out = pd.DataFrame(edge, columns=ALL_CATS)
    out.index = state.team_names
    return out


def dynamic_weights(my_edge: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) + alpha * np.exp(-0.5 * my_edge ** 2)


# ----------------------------------------------------------------------------------------------
# Opponent simulation
# ----------------------------------------------------------------------------------------------
def simulate(state: DraftState, board: Board, until: int, n_sims: int, noise: float,
             rng: np.random.Generator, start: int | None = None, record: bool = False):
    """Simulate every open pick from `start` (default: current) up to but excluding `until`.
    Returns availability (n_sims x P) at `until` and, if record, the list of (k, player index per sim)."""
    P = len(board.df)
    start = state.current if start is None else start
    taken = np.zeros(P, bool)
    taken[board.idx(state.taken())] = True
    avail = np.repeat(~taken[None, :], n_sims, axis=0)
    counts = np.zeros((n_sims, state.n_teams, 4), int)
    for t in range(state.n_teams):
        ids = board.idx(state.roster(t))
        for pi in board.pos_idx[ids]:
            counts[:, t, pi] += 1
    sd = np.maximum(1.5, noise * board.market)
    key = board.market[None, :] + rng.standard_normal((n_sims, P)) * sd[None, :]
    open_future = {t: [k for k in team_picks(t, state.n_teams, state.rounds) if k not in state.picks]
                   for t in range(state.n_teams)}
    log = []
    ar = np.arange(n_sims)
    for k in range(start, until):
        if k in state.picks:
            continue
        t = state.owner(k)
        c = counts[:, t, :]
        allowed = c < OPP_CAPS[None, :]
        rem = sum(1 for kk in open_future[t] if kk >= k)
        need = np.maximum(0, SLOT_N[None, :] - c)
        extra_sk = np.maximum(0, c[:, :3] - SLOT_N[None, :3]).sum(axis=1)
        util_need = (extra_sk < STARTING_SLOTS["Util"]).astype(int)
        need_total = need.sum(axis=1) + util_need
        forced = rem <= need_total
        if forced.any():
            only_util = (need[:, :3].sum(axis=1) == 0) & (util_need > 0)
            nm = need > 0
            nm[:, :3] |= only_util[:, None]
            allowed = np.where(forced[:, None], nm, allowed)
        ok = allowed[:, board.pos_idx] & avail
        score = np.where(ok, key, np.inf)
        # fall back to any available player if constraints left nothing
        none_ok = ~ok.any(axis=1)
        if none_ok.any():
            score[none_ok] = np.where(avail[none_ok], key[none_ok], np.inf)
        j = score.argmin(axis=1)
        avail[ar, j] = False
        counts[ar, t, board.pos_idx[j]] += 1
        if record:
            log.append((k, j.copy()))
    return avail, log


# ----------------------------------------------------------------------------------------------
# Recommendation
# ----------------------------------------------------------------------------------------------
def recommend(state: DraftState, board: Board, s: ModelSettings, alpha: float = 0.75,
              n_sims: int = 300, noise: float = 0.2, top_k: int = 25, seed: int = 0) -> dict:
    df = board.df
    rng = np.random.default_rng(seed + 7919 * len(state.picks))
    cur = state.current
    if cur is None:
        return {"done": True}

    # 1. category needs -> dynamic weights
    edges = category_edges(state, board, s.bench_factor)
    my_edge = edges.iloc[state.my_slot].to_numpy()
    w = dynamic_weights(my_edge, alpha)
    V = df[[f"v_{c}" for c in ALL_CATS]].to_numpy()
    vals = V @ w

    # 2. dynamic replacement levels from what's left
    P = len(df)
    avail_now = np.ones(P, bool)
    avail_now[board.idx(state.taken())] = False
    static_repl, static_wv = replacement_levels(vals, df["elig"].tolist(), s.n_teams)
    open_tot, bench_tot, groom = league_open_slots(state, board, vals, static_repl)
    repl, waiver = replacement_levels(vals, df["elig"].tolist(), s.n_teams, available=avail_now,
                                      open_slots=open_tot, bench_open=bench_tot, goalie_room=groom)
    repl = scale_scarcity(repl, s.scarcity)

    my_ids = board.idx(state.roster(state.my_slot))
    g_now = gains_for_roster(my_ids, vals, board, repl, waiver, s.bench_factor)
    g_now[~avail_now] = -np.inf

    on_clock = state.owner(cur) == state.my_slot
    mine = state.my_next_picks()
    if not mine:
        return {"done": True, "edges": edges, "weights": w}
    # the pick we're forecasting to: our NEXT pick if on the clock, else our upcoming one
    if on_clock:
        target = mine[1] if len(mine) > 1 else None
    else:
        target = mine[0]
    sim_start = cur + 1 if on_clock else cur
    if target is not None:
        avail_sim, _ = simulate(state, board, target, n_sims, noise, rng, start=sim_start)
    else:
        avail_sim = np.repeat(avail_now[None, :], 1, axis=0)
    survive = avail_sim.mean(axis=0)

    # expected best gain available at the target pick, per position (for drop-off)
    pos_best = {}
    for p in ("C", "W", "D", "G"):
        m = (board.pos_idx == POS_IDX[p])
        gp = np.where(avail_sim & m[None, :], g_now[None, :], -np.inf).max(axis=1)
        gp = np.where(np.isfinite(gp), gp, 0.0)
        pos_best[p] = float(gp.mean())

    order = np.argsort(-np.where(avail_now, g_now, -np.inf))
    cand = order[:top_k]
    rows = []
    for i in cand:
        if not np.isfinite(g_now[i]):
            continue
        nxt = 0.0
        if on_clock and target is not None:
            ids2 = np.append(my_ids, i)
            g2 = gains_for_roster(ids2, vals, board, repl, waiver, s.bench_factor)
            A = avail_sim.copy()
            A[:, i] = False
            best2 = np.where(A, g2[None, :], -np.inf).max(axis=1)
            nxt = float(np.where(np.isfinite(best2), best2, 0.0).mean())
        rows.append({"i": int(i), "gain": float(g_now[i]), "next": nxt, "score": float(g_now[i]) + nxt,
                     "survive": float(survive[i]) if target is not None else 0.0,
                     "dropoff": float(g_now[i] - pos_best[df["pos"].iat[i]])})
    rec = pd.DataFrame(rows)
    if rec.empty:
        return {"done": True, "edges": edges, "weights": w}
    rec = rec.join(df[["pid", "name", "team_now", "yahoo_pos", "pos", "value", "vorp", "rank"]], on="i")
    rec["market"] = board.market[rec["i"].to_numpy()]
    rec["value_now"] = vals[rec["i"].to_numpy()]
    rec["cat_fit"] = [float(V[i] @ (w - 1)) for i in rec["i"]]
    if on_clock:
        rec = rec.sort_values("score", ascending=False)
    else:
        rec = rec.sort_values("gain", ascending=False)
    rec["why"] = [_why(r, on_clock, target is not None, my_edge, V, w) for r in rec.itertuples()]
    return {"done": False, "on_clock": on_clock, "target": target, "recs": rec.reset_index(drop=True),
            "edges": edges, "weights": w, "repl": repl, "waiver": waiver, "pos_best": pos_best,
            "vals": vals, "survive": survive, "win_prob": norm.cdf(my_edge)}


def _why(r, on_clock: bool, has_next: bool, my_edge, V, w) -> str:
    bits = []
    if has_next:
        if r.survive < 0.25:
            bits.append(f"{(1 - r.survive):.0%} gone before your next pick")
        elif r.survive > 0.75:
            bits.append(f"{r.survive:.0%} likely back to you — can wait")
    if r.dropoff > 0.8:
        bits.append(f"big drop-off at {r.pos} (+{r.dropoff:.1f})")
    contrib = V[r.i] * w
    top = np.argsort(-contrib)[:2]
    cats = [ALL_CATS[c] for c in top if contrib[c] > 0.3]
    if cats:
        bits.append("helps " + "/".join(cats))
    if r.market - r.rank > 15:
        bits.append("model likes him more than the room")
    return "; ".join(bits)


def auto_pick(state: DraftState, board: Board, noise: float, rng: np.random.Generator) -> str:
    """One opponent pick drawn from the market model (same rules as the simulation)."""
    cur = state.current
    _, log = simulate(state, board, cur + 1, 1, noise, rng, start=cur, record=True)
    return board.df["pid"].iat[int(log[0][1][0])]
