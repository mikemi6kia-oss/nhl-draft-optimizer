# 🏒 NHL Fantasy Draft Optimizer (Yahoo H2H, 10-cat)

Live draft assistant for a Yahoo H2H categories league: G, A, P, PPP, GWG, SOG / W, GAA, SV, SV%.
Roster: 2C · 4W · 4D · Util · 2G · 6 BN (19 rounds). How it works: see `METHOD.md`.

Every file sits in the top level of the repo (no subfolders), apart from the optional `.streamlit/config.toml` theme.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Files
| File | What |
|---|---|
| `app.py`, `ui_style.py` | Streamlit app |
| `nhl_*.py` | Model: projections, valuation, draft simulator, recommendations |
| `skaters.csv`, `goalies.csv` | 2025-26 stats (rebuild from `2025-26_season.xlsx` with `python build_data.py`) |
| `extra_players.csv` | Players missing from the stats file (Barkov, 2026 rookies) with a full-season projection + note. Blank PPP/GWG/SOG are estimated from similar players |
| `draft_order.txt` | League draft order, one team per line — the app's default team names and league size |
| `availability.csv` | Health/role updates: `name,team,proj_gp,games_missed,note` — e.g. out until November, or healthy again |
| `keepers_template.xlsx` | **Keepers Excel template** — fill it in, then upload it in the app (Keepers & Data tab) or add it to the repo as `keepers.xlsx` |
| `keepers.csv` | Alternative keepers format: `manager,player,round,nhl_team,pos` (manager = team name, slot number, or ME) |
| `adp.csv` *(optional)* | Yahoo ADP: `name,adp[,team,pos]` |
| `eligibility.csv` *(optional)* | Multi-position eligibility: `name,positions,team` e.g. `Jake Guentzel,C/LW,TBL` |
| `test_engine.py` | Tests: `pip install pytest && pytest -q` |
