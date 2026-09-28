## How the optimizer thinks

**League:** Yahoo H2H categories, 10 cats — skaters G, A, P, PPP, GWG, SOG; goalies W, GAA, SV, SV%.
Roster 2C / 4W / 4D / 1 Util / 2G + 6 bench → 19 draft rounds (IR spots aren't drafted).

### 1. Projections
2025-26 per-game rates × projected games played.
- **Injury regression:** established players (40+ GP) get back a share (default 35%) of the games they missed, because injuries are mostly random year to year. Call-ups with few games don't.
- **Small-sample shrinkage:** per-game rates are blended with a fringe-roster prior worth 10 games (skaters), 400 shots (SV%) and 600 minutes (GAA). This only matters for players with tiny samples.

### 2. Category scores — your "standard deviation" instinct, upgraded
Yahoo's standard-deviation view is a **z-score**: how many SDs a player is above the average draftable player in each category. That's the right starting point, and you can switch the app to it ("Classic Z-score"). Two refinements matter for *head-to-head*:

1. **G-score (week-to-week noise).** You win a *week*, not a season. Categories that swing wildly week to week (GWG, goalie W) reward a season-long edge less reliably than steady ones (SOG). The G-score adds each player's weekly variance to the denominator, so noisy categories count a bit less. (Rosenof 2024, *Improving algorithms for fantasy basketball*.)
2. **Matchup scaling for goalies.** Skater categories are the sum of ~11 active players; goalie categories come from only 2. One goalie is a much bigger share of those four categories, so a 1-SD goalie edge moves your weekly win odds √(11/2) ≈ 2.3× more than a 1-SD skater edge. Scores are expressed in skater-equivalent units so all ten categories are comparable.

**Rate stats are volume-weighted:** SV% → saves above average (SV − league SV% × SA); GAA → goals prevented (league GAA × TOI/60 − GA). A goalie with a hot 5-game sample can't look like a Vezina winner.

### 3. Positional scarcity (VORP)
All 12 teams must start 4 D, 4 W, 2 C and 2 G. The 48th-best defenseman is far worse in these categories than the 48th-best winger, so a strong D is worth more than his raw stats suggest. **VORP = value − value of the best player at his position left out of every team's starting lineup.** That's the Big Board ranking. Util's baseline is the best skater left over at any position.

The *full* theoretical gap makes the model grab ~4 D in the first 6 rounds. In simulation, **half** the gap (the default "Positional scarcity = 0.5") won clearly — it still values elite D but doesn't overpay. It is also the safer choice because defensemen's scoring repeats less reliably year to year than forwards'.

### 4. Live recommendations (War Room)
When you're on the clock, every candidate gets

> **Score = gain now + expected best gain at your next pick (if you take him now)**

- **Gain now** = how much your lineup improves: value over the *current* replacement level at the slot he'd fill. Replacement levels are recomputed live from who's left and how many starting slots the league still has open, so a run on D immediately raises the price of waiting on D. Bench spots count at the bench-usefulness setting, with diminishing returns for stacking one position.
- **Expected best at next pick** comes from simulating the other managers a few hundred times. Opponents draft from a **market rank** (your uploaded Yahoo ADP if provided, otherwise a proxy of raw z-score + positional value) plus ADP-style randomness, and they respect roster needs. This is why the recommendation changes with your draft slot: at the end of a snake turn, the model knows who will and won't come back.
- **Category needs:** each category is re-weighted by (1 − a) + a·exp(−edge²/2), where edge is your projected weekly advantage over the average roster drafted so far. That's the slope of your win probability — close categories matter most; categories you're already dominating (or have effectively punted) matter less. Explicit punts can be set in the sidebar.

### 5. Independent check (Matchups tab)
Rosters are also scored on **raw projected totals** (actual goals, saves, SV%…) with weekly noise, giving expected categories won per week vs every team. Test: 30 simulated 12-team drafts per strategy (draft slots 1, 4, 7, 10, 12), other 11 teams drafting from the market model:

| Your strategy | Proj. cats won / week | Avg finish | 1st place | D in first 6 picks |
|---|---|---|---|---|
| Draft like the room (market/ADP) | 4.57 | 6.7 | 7% | 0.9 |
| Optimizer, scarcity 0 (position-blind) | 4.93 | 1.7 | 53% | 0.2 |
| Optimizer, scarcity 1.0 (full theory) | 5.00 | 1.2 | 80% | 4.0 |
| **Optimizer, scarcity 0.5 (default)** | **5.08** | **1.2** | **83%** | 2.7 |

Standard error ≈ 0.02 per row. A separate 20-league test found removing goalie matchup scaling cost ~0.07 cats/week; bench usefulness, category-need adaptivity and injury regression moved results by less than the noise, so their defaults are judgement calls. Caveat: the evaluator uses the same projections as the model, so it measures drafting *given* the projections, not projection accuracy — and real opponents are smarter than the market proxy, so expect a smaller edge in a real room.

### Known limits
- Projections are last season's rates; they don't know about trades, retirements, rookies (2026 draftees) or role changes. Edit the CSVs or add ADP to correct for them.
- Positions come from the stats file (C / L / R / D). Yahoo multi-eligibility (e.g. C/RW) can be added in `eligibility.csv`.
- The simulated opponents are only as good as the market rank — upload real Yahoo ADP close to draft day.
