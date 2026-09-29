import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nhl_config import ALL_CATS, GOALIE_CATS, ROUNDS, SKATER_CATS, ModelSettings  # noqa: E402
from nhl_data import load_players, match_player  # noqa: E402
from nhl_draft import DraftState, owner, team_picks  # noqa: E402
from nhl_evaluate import league_matchups  # noqa: E402
from nhl_keepers import apply_keepers  # noqa: E402
from nhl_market import market_rank  # noqa: E402
from nhl_recommend import Board, auto_pick, recommend  # noqa: E402
from nhl_valuation import value_players  # noqa: E402


@pytest.fixture(scope="module")
def players():
    return load_players()


@pytest.fixture(scope="module")
def board(players):
    sk, gl = players
    df = value_players(sk, gl, ModelSettings())
    mk, _, _ = market_rank(sk, gl, 12)
    return Board.build(df, mk)


def test_data_loaded(players):
    sk, gl = players
    assert len(sk) == 940 + int(sk["added"].sum()) and len(gl) == 98
    assert set(sk["pos"]) == {"C", "W", "D"}
    ids = pd.concat([sk["pid"], gl["pid"]])
    assert ids.is_unique
    # two different Elias Petterssons (VAN C and VAN D) must stay distinct
    assert {"elias-pettersson-c", "elias-pettersson-d"} <= set(sk["pid"])


def test_name_matching(players):
    sk, gl = players
    assert match_player("connor mcdavid", sk) == "connor-mcdavid"
    assert match_player("Elias Pettersson", sk) is None          # ambiguous without help
    assert match_player("Elias Pettersson", sk, pos="D") == "elias-pettersson-d"


def test_snake_order():
    assert [owner(k, 12) for k in (0, 11, 12, 23, 24)] == [0, 11, 11, 0, 0]
    for t in range(12):
        assert len(team_picks(t, 12)) == ROUNDS == 19


def test_z_scores_standardised_on_pool(players):
    sk, gl = players
    df = value_players(sk, gl, ModelSettings(method="z"))
    pool = df[df["in_pool"]]
    for c in SKATER_CATS:
        z = pool.loc[pool.pos != "G", f"z_{c}"]
        assert abs(z.mean()) < 1e-9 and abs(z.std(ddof=0) - 1) < 1e-9
    for c in GOALIE_CATS:
        z = pool.loc[pool.pos == "G", f"z_{c}"]
        assert abs(z.mean()) < 1e-9 and abs(z.std(ddof=0) - 1) < 1e-9


def test_g_score_shrinks_noisy_categories(players):
    sk, gl = players
    df = value_players(sk, gl, ModelSettings())
    top = df[df["in_pool"] & (df.pos != "G")]
    # GWG is far noisier week to week than SOG -> G-score shrinks it much more vs. z
    shrink = {c: (top[f"g_{c}"].abs().sum() / top[f"z_{c}"].abs().sum()) for c in ("GWG", "SOG")}
    assert shrink["GWG"] < shrink["SOG"]


def test_rate_stats_volume_weighted(players):
    sk, gl = players
    df = value_players(sk, gl, ModelSettings())
    g = df[df.pos == "G"].set_index("name")
    # tiny samples can't produce big SV%/GAA scores, however extreme their rates
    tiny, work = g[g.gp <= 5], g[g.gp >= 40]
    for c in ("SV%", "GAA"):
        assert tiny[f"v_{c}"].abs().max() < 0.5 * work[f"v_{c}"].abs().max()
    assert g.loc["Andrei Vasilevskiy", "value"] > tiny["value"].max()


def test_keepers(players, board):
    st = DraftState(n_teams=12, my_slot=4)
    kp = pd.DataFrame({"manager": ["ME", "1", "Team 3", "Nobody"],
                       "player": ["Cole Caufield", "Connor McDavid", "Cale Makar", "Cale Makar"],
                       "round": ["3", "1", "", ""], "nhl_team": ["", "", "", ""], "pos": ["", "", "", ""]})
    problems = apply_keepers(st, kp, board.df)
    assert len(problems) == 1 and "Nobody" in problems[0] and "sidebar" in problems[0]
    assert st.picks[0]["pid"] == "connor-mcdavid"                 # team 1, round 1
    k_me = [k for k, v in st.picks.items() if v["pid"] == "cole-caufield"][0]
    assert owner(k_me, 12) == 4 and k_me // 12 == 2               # my round-3 pick
    k_t3 = [k for k, v in st.picks.items() if v["pid"] == "cale-makar"][0]
    assert owner(k_t3, 12) == 2 and k_t3 // 12 == ROUNDS - 1      # blank round -> last round
    assert st.current == 1


def test_state_json_roundtrip():
    st = DraftState(n_teams=10, my_slot=3)
    st.draft("connor-mcdavid")
    st.add_keeper(5, "cale-makar", 2)
    st2 = DraftState.from_json(st.to_json())
    assert st2.picks == st.picks and st2.my_slot == 3 and st2.n_teams == 10


@pytest.mark.parametrize("slot", [0, 6, 11])
def test_full_mock_draft_valid(board, slot):
    s = ModelSettings()
    st = DraftState(n_teams=12, my_slot=slot)
    rng = np.random.default_rng(slot)
    while st.current is not None:
        if st.owner(st.current) == slot:
            r = recommend(st, board, s, n_sims=60, seed=1)
            recs = r["recs"]
            assert not set(recs["pid"]) & st.taken()
            st.draft(recs.iloc[0]["pid"])
        else:
            st.draft(auto_pick(st, board, 0.2, rng))
    pos = board.df.set_index("pid")["pos"]
    all_ids = [v["pid"] for v in st.picks.values()]
    assert len(all_ids) == len(set(all_ids)) == 12 * ROUNDS
    for t in range(12):
        c = pos.loc[st.roster(t)].value_counts()
        # every team (mine and simulated) can field a full starting lineup
        assert c.get("C", 0) >= 2 and c.get("W", 0) >= 4 and c.get("D", 0) >= 4 and c.get("G", 0) >= 2, (t, c)
    ev = league_matchups(st, board, s)
    assert ev["cats_won"].shape == (12, 12)
    assert np.allclose(ev["cats_won"] + ev["cats_won"].T, 10 * (1 - np.eye(12)))


def test_survival_probabilities_sane(board):
    st = DraftState(n_teams=12, my_slot=11)
    r = recommend(st, board, ModelSettings(), n_sims=200)
    assert not r["on_clock"]
    surv = pd.Series(r["survive"], index=board.df["pid"])
    assert surv["connor-mcdavid"] < 0.05       # top market player never lasts to pick 12
    assert surv.sort_values().iloc[-1] == 1.0  # deep bench guys always do


def test_roster_full_before_draft_ends(board):
    """Slot 1's last pick comes 11 picks before the end: the app still needs edges + win odds."""
    s = ModelSettings()
    st = DraftState(n_teams=12, my_slot=0)
    rng = np.random.default_rng(5)
    while len(st.roster(0)) < ROUNDS:
        if st.owner(st.current) == 0:
            st.draft(recommend(st, board, s, n_sims=30)["recs"].iloc[0]["pid"])
        else:
            st.draft(auto_pick(st, board, 0.2, rng))
    assert st.current is not None
    r = recommend(st, board, s, n_sims=30)
    assert r["done"] and {"edges", "win_prob", "vals"} <= set(r)


def test_keepers_excel_template_reads_clean():
    """The shipped template parses to zero keepers (its grey EXAMPLE row is ignored)."""
    from nhl_keepers import read_keepers_file
    k = read_keepers_file(Path(__file__).resolve().parent / "keepers_template.xlsx")
    assert list(k.columns) == ["manager", "player", "round", "nhl_team", "pos"] and k.empty



def test_extra_players_and_availability(players):
    sk, gl = players
    df = value_players(sk, gl, ModelSettings())
    row = df.set_index("name")
    # added players exist, are draftable, have imputed categories and a note
    for n in ("Aleksander Barkov", "Gavin McKenna"):
        r = row.loc[n]
        assert r["proj_gp"] > 60 and r["proj_ppp"] > 0 and r["proj_sog"] > 0 and r["note"]
    assert match_player("Aleksander Barkov", df) == "aleksander-barkov"
    # health overrides: Tkachuk restored to a full season, Bedard loses ~16 games
    assert row.loc["Matthew Tkachuk", "proj_gp"] == 78
    base = value_players(sk.assign(avail_gp=np.nan, avail_missed=np.nan), gl, ModelSettings()).set_index("name")
    assert abs(base.loc["Connor Bedard", "proj_gp"] - row.loc["Connor Bedard", "proj_gp"] - 16) < 1e-6
    ratio = row.loc["Connor Bedard", "proj_g"] / base.loc["Connor Bedard", "proj_g"]
    assert abs(ratio - row.loc["Connor Bedard", "proj_gp"] / base.loc["Connor Bedard", "proj_gp"]) < 1e-9
    # 84-game season: a full-season regular projects to 84 GP
    assert row.loc["Connor McDavid", "proj_gp"] == 84



def test_team_name_matching_ignores_curly_quotes_and_case(players, board):
    st = DraftState(n_teams=12, my_slot=0, team_names=["Don Luig", "Bros before Aho\u2019s", "TOP OF THE WORLD"])
    kp = pd.DataFrame({"manager": ["Bros before Aho's", "Top of the world"], "player": ["Juraj Slafkovsky", "Nico Hischier"],
                       "round": ["11", "11"], "nhl_team": ["", ""], "pos": ["", ""]})
    assert apply_keepers(st, kp, board.df) == []
