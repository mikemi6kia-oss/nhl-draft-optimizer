"""NHL Fantasy Draft Optimizer — Streamlit front end.

Run:  streamlit run app.py
All modelling lives in engine/; this file is UI + session state only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_style as ui
from nhl_config import ALL_CATS, BENCH_SLOTS, GOALIE_CATS, ROUNDS, SKATER_CATS, ModelSettings
from nhl_data import DATA_DIR, _plain_strings, load_players
from nhl_draft import DraftState
from nhl_evaluate import league_matchups
from nhl_keepers import apply_keepers, load_keepers, read_keepers_file
from nhl_market import market_rank
from nhl_recommend import Board, assign_lineup, auto_pick, recommend

st.set_page_config(page_title="NHL Draft Optimizer", page_icon="🏒", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
ss = st.session_state


# ------------------------------------------------------------------------------------------------
# Cached model objects
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_players():
    return load_players()


@st.cache_resource(show_spinner="Valuing every player…")
def get_values(settings_items: tuple):
    from nhl_valuation import value_players
    sk, gl = get_players()
    return value_players(sk, gl, ModelSettings(**dict(settings_items)))


@st.cache_resource(show_spinner="Modelling the other managers…")
def get_market(n_teams: int, adp_text: str | None):
    sk, gl = get_players()
    adp = None
    if adp_text:
        adp = _plain_strings(pd.read_csv(io.StringIO(adp_text)))
        adp.columns = [c.strip().lower() for c in adp.columns]
        if not {"name", "adp"} <= set(adp.columns):
            adp = None
    return market_rank(sk, gl, n_teams, adp)


@st.cache_resource(show_spinner=False)
def get_board(settings_items: tuple, n_teams: int, adp_text: str | None) -> Board:
    return Board.build(get_values(settings_items), get_market(n_teams, adp_text)[0])


def settings_key(s: ModelSettings) -> tuple:
    return tuple(sorted(dataclasses.asdict(s).items()))


# ------------------------------------------------------------------------------------------------
# Sidebar — league + model controls
# ------------------------------------------------------------------------------------------------
def _on_randomize(n):
    ss["_slot"] = random.randint(1, n)


with st.sidebar:
    st.markdown("### ⚙ LEAGUE")
    n_teams = int(st.number_input("Teams in league", min_value=6, max_value=16, value=12, step=1, key="n_teams"))
    ss.setdefault("_slot", 1)
    ss["_slot"] = min(ss["_slot"], n_teams)
    c1, c2 = st.columns([3, 1])
    with c1:
        my_slot = int(st.selectbox("My draft slot", list(range(1, n_teams + 1)), index=ss["_slot"] - 1))
    ss["_slot"] = my_slot
    with c2:
        st.write("")
        st.button("🎲", help="Random draft slot", on_click=_on_randomize, args=(n_teams,))
    names_txt = st.text_area("Team names in draft order (optional, one per line)", key="names", height=90,
                             placeholder="Team 1\nTeam 2\n…")

    with st.expander("🧠 Valuation model", expanded=False):
        method = st.radio("Scoring", ["H2H G-score (recommended)", "Classic Z-score"], index=0,
                          help="Z-score = the 'standard deviation' view Yahoo shows. G-score also accounts for "
                               "week-to-week noise, which is what decides head-to-head categories.")
        goalie_w = st.slider("Goalie category weight", 0.5, 1.5, 1.0, 0.05,
                             help="1.0 = theory: 4 of 10 categories come from only 2 active goalies.")
        scarcity = st.slider("Positional scarcity", 0.0, 1.0, 0.5, 0.05,
                             help="How much the gap between replacement-level C/W/D matters. 0 = position-blind, "
                                  "1 = full theoretical gap. 0.5 won the simulation test (see METHOD).")
        bench_f = st.slider("Bench usefulness", 0.1, 0.6, 0.35, 0.05,
                            help="Share of a bench player's production that ends up in your daily lineup.")
        gp_reg = st.slider("Injury regression", 0.0, 0.6, 0.35, 0.05,
                           help="Share of last season's missed games given back to established players.")
        alpha = st.slider("Category-need adaptivity", 0.0, 1.0, 0.75, 0.05,
                          help="How strongly recommendations chase the categories you're closest to winning/losing.")
        punts = st.multiselect("Punt categories", ALL_CATS, default=[])
    with st.expander("🎯 Opponent model", expanded=False):
        noise = st.slider("Opponent unpredictability", 0.05, 0.40, 0.20, 0.01,
                          help="SD of opponents' picks around market rank, as a share of that rank.")
        n_sims = st.select_slider("Simulations per recommendation", [100, 200, 300, 500], value=300)

settings = ModelSettings(
    n_teams=n_teams, method="h2h" if method.startswith("H2H") else "z", goalie_weight=goalie_w, scarcity=scarcity,
    bench_factor=bench_f, gp_regression=gp_reg, cat_weights=tuple((c, 0.0) for c in punts))

adp_file = DATA_DIR / "adp.csv"
adp_text = ss.get("adp_upload") or (adp_file.read_text() if adp_file.exists() else None)
board = get_board(settings_key(settings), n_teams, adp_text)
_, market_src, adp_unmatched = get_market(n_teams, adp_text)
df = board.df
pid_row = {p: i for i, p in enumerate(df["pid"])}

# ------------------------------------------------------------------------------------------------
# Draft state (re-initialised when league shape / keepers change)
# ------------------------------------------------------------------------------------------------
keepers_df = ss.get("keepers_override")
if keepers_df is None:
    keepers_df = load_keepers()
team_names = [n for n in names_txt.splitlines() if n.strip()]
sig = hashlib.md5(repr((n_teams, my_slot, tuple(team_names), keepers_df.to_csv(index=False))).encode()).hexdigest()


def fresh_state() -> tuple[DraftState, list[str]]:
    stt = DraftState(n_teams=n_teams, my_slot=my_slot - 1, team_names=list(team_names))
    return stt, apply_keepers(stt, keepers_df, df)


if ss.pop("adopt", False):
    ss["sig"] = sig
if "state" not in ss or ss.get("sig") != sig:
    had_picks = "state" in ss and any(not v["keeper"] for v in ss["state"].picks.values())
    ss["state"], ss["keeper_problems"] = fresh_state()
    ss["sig"] = sig
    if had_picks:
        st.toast("League setup changed — draft reset.", icon="⚠️")
state: DraftState = ss["state"]


# ------------------------------------------------------------------------------------------------
# Actions (callbacks run before the next render)
# ------------------------------------------------------------------------------------------------
def _rng():
    return np.random.default_rng()


def act_draft(key):
    pid = ss.get(key)
    if pid and pid not in state.taken() and state.current is not None:
        state.draft(pid)


def act_auto():
    if state.current is not None:
        state.draft(auto_pick(state, board, noise, _rng()))


def act_sim_to_me():
    rng = _rng()
    while state.current is not None and state.owner(state.current) != state.my_slot:
        state.draft(auto_pick(state, board, noise, rng))


def act_undo():
    state.undo()


def act_reset():
    state.reset(keep_keepers=True)


def act_autodraft_all():
    rng = _rng()
    while state.current is not None:
        if state.owner(state.current) == state.my_slot:
            r = recommend(state, board, settings, alpha=alpha, n_sims=min(n_sims, 150), noise=noise)
            state.draft(r["recs"].iloc[0]["pid"])
        else:
            state.draft(auto_pick(state, board, noise, rng))


def act_load_draft():
    f = ss.get("load_draft")
    if f is None:
        return
    try:
        loaded = DraftState.from_json(f.getvalue().decode())
    except Exception as e:  # noqa: BLE001
        ss["load_error"] = f"Couldn't read that file: {e}"
        return
    ss["state"] = loaded
    ss["n_teams"] = loaded.n_teams
    ss["_slot"] = loaded.my_slot + 1
    ss["names"] = "\n".join(n.lstrip("★ ").strip() for n in loaded.team_names)
    ss["adopt"] = True


# ------------------------------------------------------------------------------------------------
# Header + HUD
# ------------------------------------------------------------------------------------------------
def pname(pid: str) -> str:
    r = df.iloc[pid_row[pid]]
    return f"{r['name']} · {r['yahoo_pos']} · {r['team_now']}"


cur = state.current
mine_next = state.my_next_picks()
st.markdown('<p class="hud-title">NHL DRAFT OPTIMIZER</p>', unsafe_allow_html=True)
st.markdown(f'<p class="hud-sub">2026-27 · {n_teams} teams · {ROUNDS} rounds · H2H 10-cat · '
            f'valued on 2025-26 stats · market: {market_src}</p>', unsafe_allow_html=True)

h = st.columns(5)
if cur is None:
    h[0].markdown(ui.card("Status", '<span class="glow">DRAFT COMPLETE</span>'), unsafe_allow_html=True)
else:
    on_me = state.owner(cur) == state.my_slot
    h[0].markdown(ui.card("On the clock", f'<span class="{"pink" if on_me else "glow"}">{state.team_names[state.owner(cur)]}</span>',
                          hot=on_me), unsafe_allow_html=True)
    h[1].markdown(ui.card("Pick", state.label(cur)), unsafe_allow_html=True)
    nxt = mine_next[1] if on_me and len(mine_next) > 1 else (mine_next[0] if mine_next and not on_me else None)
    h[2].markdown(ui.card("Your next pick", state.label(nxt) if nxt is not None else "—"), unsafe_allow_html=True)
    until = (nxt - cur) if nxt is not None else 0
    h[3].markdown(ui.card("Picks until then", f'<span class="glow">{until}</span>'), unsafe_allow_html=True)
done = len(state.picks)
h[4].markdown(ui.card("Progress", f"{done}/{state.total_picks} "
                                  f'<span style="font-size:.8rem;color:{ui.MUTED}">({done / state.total_picks:.0%})</span>'),
              unsafe_allow_html=True)
st.write("")

if ss.get("keeper_problems"):
    st.warning(f"⚠ Keepers: {ss['keeper_problems'][0]}  (details in the KEEPERS & DATA tab)")

tabs = st.tabs(["⚡ WAR ROOM", "📋 BIG BOARD", "🧱 DRAFT BOARD", "📊 MATCHUPS", "🧪 SLOT LAB", "🔒 KEEPERS & DATA", "📖 METHOD"])

# ================================================================================================
# WAR ROOM
# ================================================================================================
with tabs[0]:
    rec = recommend(state, board, settings, alpha=alpha, n_sims=n_sims, noise=noise) if cur is not None else {"done": True}
    left, right = st.columns([2.15, 1])
    with left:
        if cur is None:
            st.success("Draft complete — see MATCHUPS for how your team projects.")
        elif rec.get("done"):
            st.info("Your roster is full. Keep entering the other teams' picks (or use 🧪 Auto-draft to the end "
                    "under Mock-draft tools), then check MATCHUPS.")
        else:
            recs = rec["recs"]
            top = recs.iloc[0]
            if rec["on_clock"]:
                st.markdown(
                    f'<div class="hero"><div class="meta">YOU ARE ON THE CLOCK · RECOMMENDED</div>'
                    f'<div class="name">{top["name"]}</div>'
                    f'<div class="meta">{top["yahoo_pos"]} · {top["team_now"]} · model rank #{int(top["rank"])} · market #{int(top["market"])}</div>'
                    f'<div class="why">{top["why"] or "Best combination of value now and value left at your next pick."}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="hero"><div class="meta">WAITING · {state.team_names[state.owner(cur)]} IS PICKING</div>'
                    f'<div class="name">Targets for {state.label(rec["target"])}</div>'
                    f'<div class="why">Sorted by value to your roster. <b>Available %</b> = share of simulations '
                    f'where he is still there at your pick.</div></div>', unsafe_allow_html=True)

            show = pd.DataFrame({
                "Player": recs["name"], "Pos": recs["yahoo_pos"], "Team": recs["team_now"],
                "Score": recs["score"] if rec["on_clock"] else np.nan,
                "Gain now": recs["gain"],
                "Best at next pick": recs["next"] if rec["on_clock"] else np.nan,
                "Available %": 100 * recs["survive"],
                "Drop-off": recs["dropoff"],
                "Rank": recs["rank"], "Mkt": recs["market"].round(0),
                "Why": [w + (f"{' · ' if w else ''}📝 {n}" if n else "") for w, n in
                        zip(recs["why"], df["note"].to_numpy()[recs["i"].to_numpy()])],
            })
            if not rec["on_clock"]:
                show = show.drop(columns=["Score", "Best at next pick"])
            st.dataframe(
                show.head(15), hide_index=True, width="stretch", height=560,
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.2f", help="Gain now + expected best gain at your next pick"),
                    "Gain now": st.column_config.NumberColumn(format="%.2f", help="Lineup improvement over current replacement level"),
                    "Best at next pick": st.column_config.NumberColumn(format="%.2f", help="Expected gain from your next pick if you take this player now"),
                    "Available %": st.column_config.ProgressColumn(
                        "At next pick" if rec["on_clock"] else "Available %", format="%.0f%%", min_value=0, max_value=100,
                        help="Chance he's still on the board at your next pick (Monte Carlo)"),
                    "Drop-off": st.column_config.NumberColumn(format="%+.2f", help="How much better he is than the best player at his position expected at your next pick"),
                    "Mkt": st.column_config.NumberColumn(format="%d", help="Market rank used to simulate other managers"),
                    "Why": st.column_config.TextColumn(width="large"),
                })

        if cur is not None:
            st.markdown("##### DRAFT A PLAYER")
            taken = state.taken()
            avail = df.loc[~df["pid"].isin(taken), ["pid", "rank"]].sort_values("rank")
            opts = avail["pid"].tolist()
            default = rec["recs"].iloc[0]["pid"] if not rec.get("done") else opts[0]
            key = f"pick_{len(state.picks)}_{cur}"
            st.selectbox("Player (type to search)", opts, index=opts.index(default) if default in opts else 0,
                         format_func=lambda p: f"{pname(p)}  (#{df.iat[pid_row[p], df.columns.get_loc('rank')]})",
                         key=key)
            b = st.columns(4)
            b[0].button(f"✅ Draft → {state.team_names[state.owner(cur)]}", on_click=act_draft, args=(key,),
                        type="primary", width="stretch")
            b[1].button("🤖 Auto-pick (market)", on_click=act_auto, width="stretch",
                        help="Pick for the team on the clock using the opponent model")
            b[2].button("⏩ Sim to my pick", on_click=act_sim_to_me, width="stretch",
                        disabled=state.owner(cur) == state.my_slot)
            b[3].button("↩ Undo", on_click=act_undo, width="stretch")
            with st.expander("Mock-draft tools"):
                m = st.columns(2)
                m[0].button("🧪 Auto-draft to the end (you use the recommendations)", on_click=act_autodraft_all, width="stretch")
                m[1].button("🗑 Reset draft (keeps keepers)", on_click=act_reset, width="stretch")

    with right:
        st.markdown("##### MY LINEUP")
        vals = rec.get("vals", df["value"].to_numpy())
        my_ids = board.idx(state.roster(state.my_slot))
        lu = assign_lineup(my_ids, vals, board, df.attrs["repl"])
        rows = []
        for slot_name, n in [("C", 2), ("W", 4), ("D", 4), ("Util", 1), ("G", 2)]:
            filled = lu["start"][slot_name]
            for j in range(n):
                if j < len(filled):
                    i = filled[j]
                    rows.append(f'<div class="slot-row"><span class="s">{slot_name}</span><span class="n">{df.iat[i, df.columns.get_loc("name")]}</span>'
                                f'<span class="v">{vals[i]:+.1f}</span></div>')
                else:
                    rows.append(f'<div class="slot-row empty"><span class="s">{slot_name}</span><span class="n">— open —</span><span class="v"></span></div>')
        for j in range(BENCH_SLOTS):
            if j < len(lu["bench"]):
                i = lu["bench"][j]
                rows.append(f'<div class="slot-row"><span class="s">BN</span><span class="n">{df.iat[i, df.columns.get_loc("name")]} '
                            f'<span style="color:{ui.MUTED}">({df.iat[i, df.columns.get_loc("yahoo_pos")]})</span></span><span class="v">{vals[i]:+.1f}</span></div>')
            else:
                rows.append('<div class="slot-row empty"><span class="s">BN</span><span class="n">— open —</span><span class="v"></span></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

        if "edges" in rec and "win_prob" in rec:
            st.markdown("##### CATEGORY EDGE vs AVERAGE ROSTER")
            e = rec["edges"].iloc[state.my_slot]
            wp = 100 * rec["win_prob"]
            colors = [ui.CYAN if v >= 0 else ui.MAGENTA for v in e.values]
            fig = go.Figure(go.Bar(
                x=e.index, y=e.values, marker_color=colors,
                text=[f"{p:.0f}%" for p in wp], textposition="outside",
                hovertemplate="%{x}<br>edge %{y:+.2f} SD<br>P(win week) %{text}<br>weight %{customdata:.2f}<extra></extra>",
                customdata=rec["weights"]))
            fig.update_yaxes(title="weekly edge (matchup SD)")
            st.plotly_chart(ui.plotly_layout(fig, 260), width="stretch", config={"displayModeBar": False})
            st.caption("Labels = chance of winning that category in a given week vs an average roster so far. "
                       "Close categories get extra weight in recommendations.")

        st.markdown("##### RECENT PICKS")
        feed = []
        for k in sorted(state.picks, reverse=True)[:10]:
            v = state.picks[k]
            lock = "🔒 " if v["keeper"] else ""
            feed.append(f'{state.label(k)} · {state.team_names[state.owner(k)]} → <b>{lock}{pname(v["pid"])}</b>')
        st.markdown(f'<div class="feed">{"<br>".join(feed) or "No picks yet."}</div>', unsafe_allow_html=True)

# ================================================================================================
# BIG BOARD
# ================================================================================================
with tabs[1]:
    f1, f2, f3, f4 = st.columns([2, 2, 1.2, 1.2])
    pos_f = f1.multiselect("Position", ["C", "W", "D", "G"], default=[])
    q = f2.text_input("Search", "")
    hide_taken = f3.toggle("Hide drafted", value=True)
    show_cats = f4.toggle("Category scores", value=True)
    owner_of = {v["pid"]: state.team_names[state.owner(k)] for k, v in state.picks.items()}
    bb = df.copy()
    if pos_f:
        bb = bb[bb["pos"].isin(pos_f)]
    if q:
        bb = bb[bb["name"].str.contains(q, case=False, regex=False)]
    if hide_taken:
        bb = bb[~bb["pid"].isin(owner_of)]
    bb_mkt = board.market[bb.index.to_numpy()]
    table = pd.DataFrame({
        "Rank": bb["rank"], "Player": bb["name"], "Pos": bb["yahoo_pos"], "Team": bb["team_now"],
        "GP": bb["gp"], "Proj GP": bb["proj_gp"].round(0),
        "G": bb["proj_g"], "A": bb["proj_a"], "P": bb["proj_p"], "PPP": bb["proj_ppp"], "GWG": bb["proj_gwg"], "SOG": bb["proj_sog"],
        "W": bb["proj_w"], "GAA": bb["proj_gaa"], "SV": bb["proj_sv"], "SV%": bb["proj_svpct"],
        "Value": bb["value"], "VORP": bb["vorp"], "Yahoo-style Z": bb["z_total"], "Mkt": bb_mkt,
        "Model vs Mkt": bb_mkt - bb["rank"],
        "Drafted by": bb["pid"].map(owner_of).fillna(""),
        "Note": bb["note"],
    })
    if show_cats:
        for c in ALL_CATS:
            table[f"·{c}"] = bb[f"v_{c}"].where((bb["pos"] == "G") == (c in GOALIE_CATS))
    cc = {c: st.column_config.NumberColumn(format="%.0f") for c in ["G", "A", "P", "PPP", "GWG", "SOG", "W", "SV", "Proj GP", "Mkt"]}
    cc.update({"GAA": st.column_config.NumberColumn(format="%.2f"), "SV%": st.column_config.NumberColumn(format="%.3f"),
               "Value": st.column_config.NumberColumn(format="%.2f", help="Sum of category scores (H2H units)"),
               "VORP": st.column_config.NumberColumn(format="%.2f", help="Value over replacement at his position — the ranking"),
               "Yahoo-style Z": st.column_config.NumberColumn(format="%.1f", help="Plain z-score total, like Yahoo's standard-deviation view"),
               "Model vs Mkt": st.column_config.NumberColumn(format="%+.0f", help="Positive = the room will likely let him slide past his model rank"),
               "Note": st.column_config.TextColumn(width="large", help="Manual projection or health adjustment (extra_players.csv / availability.csv)")})
    cc.update({f"·{c}": st.column_config.NumberColumn(format="%.2f") for c in ALL_CATS})
    st.dataframe(table, hide_index=True, width="stretch", height=640, column_config=cc)
    st.caption("Projections = 2025-26 per-game rates (regressed ~10 games' worth toward a replacement-level rate) × projected games "
               "in an 84-game season. Rows with a 📝 Note were added or adjusted by hand (extra_players.csv / availability.csv). "
               "Category columns (·G … ·SV%) are per-category value in H2H units; VORP is what the ranking uses.")

    st.markdown("##### VALUE CURVE BY POSITION")
    fig = go.Figure()
    for p, col in zip(["C", "W", "D", "G"], [ui.CYAN, ui.LIME, ui.MAGENTA, ui.AMBER]):
        d = df[df["pos"] == p].sort_values("vorp", ascending=False).head(n_teams * 6)
        fig.add_trace(go.Scatter(x=np.arange(1, len(d) + 1), y=d["vorp"], mode="lines", name=p, line=dict(color=col, width=2.5),
                                 text=d["name"], hovertemplate="%{text}<br>#%{x} at pos · VORP %{y:.2f}<extra>" + p + "</extra>"))
    fig.update_xaxes(title="rank within position")
    fig.update_yaxes(title="VORP")
    st.plotly_chart(ui.plotly_layout(fig, 340), width="stretch", config={"displayModeBar": False})
    st.caption("Steeper curve = bigger penalty for waiting at that position.")

# ================================================================================================
# DRAFT BOARD
# ================================================================================================
with tabs[2]:
    grid = pd.DataFrame("", index=[f"R{r + 1}" for r in range(state.rounds)], columns=state.team_names)
    for k in range(state.total_picks):
        r, t = k // state.n_teams, state.owner(k)
        if k in state.picks:
            v = state.picks[k]
            row = df.iloc[pid_row[v["pid"]]]
            grid.iat[r, t] = f"{'🔒 ' if v['keeper'] else ''}{row['name']} ({row['yahoo_pos']})"
        elif k == cur:
            grid.iat[r, t] = "⏱ ON THE CLOCK"
    me_col = state.team_names[state.my_slot]

    def _style(col):
        base = f"background-color: rgba(0,229,255,.10); color: #E6F1FF" if col.name == me_col else ""
        return [("background-color: rgba(255,43,214,.25); color:#fff" if "ON THE CLOCK" in str(v) else base) for v in col]

    st.dataframe(grid.style.apply(_style, axis=0), width="stretch", height=38 * (state.rounds + 1))
    st.caption("🔒 = keeper. Your column is highlighted.")

# ================================================================================================
# MATCHUPS
# ================================================================================================
with tabs[3]:
    ev = league_matchups(state, board, settings)
    st.markdown("##### PROJECTED WEEKLY CATEGORIES WON (avg vs every other team, out of 10)")
    st.caption("Built from raw projected stat totals (goals, saves, SV%…) of each roster's best lineup — "
               "an independent check on the model. Early in the draft, rosters are incomplete, so treat this as directional.")
    acw = ev["avg_cats_won"].sort_values()
    fig = go.Figure(go.Bar(x=acw.values, y=acw.index, orientation="h",
                           marker_color=[ui.MAGENTA if n == me_col else ui.CYAN for n in acw.index],
                           text=[f"{v:.2f}" for v in acw.values], textposition="outside"))
    fig.update_xaxes(range=[0, 10])
    st.plotly_chart(ui.plotly_layout(fig, 40 + 28 * n_teams), width="stretch", config={"displayModeBar": False})

    st.markdown("##### YOUR CATEGORY WIN ODDS vs EACH OPPONENT")
    M = pd.DataFrame(100 * np.nan_to_num(ev["matrix"][state.my_slot]),
                     index=state.team_names, columns=ALL_CATS).drop(index=me_col)
    fig = go.Figure(go.Heatmap(z=M.values, x=M.columns, y=M.index, zmin=0, zmax=100,
                               colorscale=[[0, ui.MAGENTA], [0.5, "#1A2238"], [1, ui.CYAN]],
                               text=M.round(0).astype(int).astype(str).values, texttemplate="%{text}",
                               hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>", colorbar=dict(title="%")))
    st.plotly_chart(ui.plotly_layout(fig, 60 + 30 * (n_teams - 1)), width="stretch", config={"displayModeBar": False})

# ================================================================================================
# SLOT LAB — which draft slot plays best, and who you'd likely get
# ================================================================================================
with tabs[4]:
    st.markdown("##### DRAFT-SLOT LAB")
    st.caption("Runs one fast mock draft per slot (opponents = market model, you = the optimizer). "
               "Useful before you know your slot: shows the kind of player the model targets early from each position. "
               "Keepers are included; results vary run to run.")
    if st.button("▶ Run slot lab", type="primary"):
        out = []
        prog = st.progress(0.0)
        for sl in range(n_teams):
            s2 = DraftState(n_teams=n_teams, my_slot=sl, team_names=list(team_names))
            apply_keepers(s2, keepers_df, df)
            rng = np.random.default_rng(100 + sl)
            while s2.current is not None:
                if s2.owner(s2.current) == sl:
                    r = recommend(s2, board, settings, alpha=alpha, n_sims=60, noise=noise)
                    s2.draft(r["recs"].iloc[0]["pid"])
                else:
                    s2.draft(auto_pick(s2, board, noise, rng))
            ev2 = league_matchups(s2, board, settings)
            picks = [pname(p).split(" · ")[0] + f" ({df.iat[pid_row[p], df.columns.get_loc('yahoo_pos')]})" for p in s2.roster(sl)]
            out.append({"Slot": sl + 1, "Proj. cats won/wk": ev2["avg_cats_won"].iloc[sl],
                        "League rank": int(ev2["avg_cats_won"].rank(ascending=False).iloc[sl]),
                        "R1": picks[0], "R2": picks[1], "R3": picks[2], "R4": picks[3], "R5": picks[4]})
            prog.progress((sl + 1) / n_teams)
        ss["slotlab"] = pd.DataFrame(out)
    if "slotlab" in ss:
        st.dataframe(ss["slotlab"], hide_index=True, width="stretch",
                     column_config={"Proj. cats won/wk": st.column_config.NumberColumn(format="%.2f")})

# ================================================================================================
# KEEPERS & DATA
# ================================================================================================
with tabs[5]:
    k1, k2 = st.columns([1.3, 1])
    with k1:
        st.markdown("##### KEEPERS")
        st.caption("manager = team name from the sidebar list, a draft-slot number, or ME. "
                   "round = the round the keeper costs (blank → that team's last open round). "
                   "nhl_team / pos only needed for duplicate names.")
        edit = st.data_editor(keepers_df.reindex(columns=["manager", "player", "round", "nhl_team", "pos"]).fillna(""),
                              num_rows="dynamic", width="stretch", key="keeper_editor")
        kc = st.columns(2)
        if kc[0].button("Apply keepers (resets draft)", type="primary"):
            ss["keepers_override"] = edit.fillna("").astype(str)
            st.rerun()
        up = kc[1].file_uploader("…or upload the keepers Excel template / csv", type=["xlsx", "csv"], key="keepers_up")
        if up is not None and ss.get("keepers_up_sig") != (up.name, up.size):
            ss["keepers_up_sig"] = (up.name, up.size)
            try:
                ss["keepers_override"] = read_keepers_file(up, up.name)
            except Exception as e:  # noqa: BLE001
                ss["keepers_read_error"] = f"Couldn't read {up.name}: {e}"
            st.rerun()
        if ss.get("keepers_read_error"):
            st.error(ss.pop("keepers_read_error"))
        for p in ss.get("keeper_problems", []):
            st.warning(p)
        st.caption("To make keepers permanent, save the filled-in template as keepers.xlsx in the GitHub repo (it takes priority over keepers.csv).")
    with k2:
        st.markdown("##### SAVE / LOAD DRAFT")
        st.download_button("💾 Download draft state", state.to_json(), file_name="draft_state.json", mime="application/json",
                           width="stretch")
        st.file_uploader("Load draft state", type="json", key="load_draft", on_change=act_load_draft)
        if ss.get("load_error"):
            st.error(ss.pop("load_error"))
        st.markdown("##### MARKET / ADP")
        st.caption(f"Opponents are simulated from: **{market_src}**. For best results upload Yahoo ADP "
                   "(columns: name, adp — optional team, pos) or commit it as adp.csv.")
        adp_up = st.file_uploader("Upload ADP csv", type="csv", key="adp_up")
        if adp_up is not None and ss.get("adp_upload") != adp_up.getvalue().decode():
            ss["adp_upload"] = adp_up.getvalue().decode()
            st.rerun()
        if adp_unmatched:
            st.warning(f"{len(adp_unmatched)} ADP names not matched: " + ", ".join(adp_unmatched[:15]))
        st.markdown("##### DATA")
        sk, gl = get_players()
        st.caption(f"{len(sk)} skaters and {len(gl)} goalies from skaters.csv & goalies.csv "
                   f"(built from 2025-26_season.xlsx by build_data.py). Positions: Yahoo LW/RW → W. "
                   f"Multi-position eligibility can be added in eligibility.csv.")

# ================================================================================================
# METHOD
# ================================================================================================
with tabs[6]:
    st.markdown(open(DATA_DIR / "METHOD.md", encoding="utf-8").read())
