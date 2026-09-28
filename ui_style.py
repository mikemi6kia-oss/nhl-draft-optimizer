"""Look & feel: dark HUD theme. Pure presentation - no logic lives here."""

CYAN = "#00E5FF"
MAGENTA = "#FF2BD6"
LIME = "#B6FF3B"
AMBER = "#FFB020"
RED = "#FF4D6D"
MUTED = "#7FA7C9"
BG = "#070B14"
PANEL = "#0E1526"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, .stApp {{ font-family: 'Rajdhani', sans-serif; }}
.stApp {{
  background:
    radial-gradient(1100px 520px at 8% -12%, rgba(0,229,255,.10), transparent 60%),
    radial-gradient(900px 480px at 108% 4%, rgba(255,43,214,.08), transparent 60%),
    repeating-linear-gradient(0deg, rgba(255,255,255,.012) 0 1px, transparent 1px 3px),
    {BG};
}}
h1, h2, h3 {{ font-family: 'Orbitron', sans-serif !important; letter-spacing: .06em; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0B1222, #070B14); border-right: 1px solid rgba(0,229,255,.18); }}
div[data-testid="stTabs"] button p {{ font-family: 'Orbitron', sans-serif; font-size: .78rem; letter-spacing: .08em; }}
.hud-title {{ font-family:'Orbitron'; font-weight:900; font-size:2.0rem; letter-spacing:.12em; margin:0;
  background: linear-gradient(90deg, {CYAN}, #8AF3FF 40%, {MAGENTA}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.hud-sub {{ font-family:'JetBrains Mono'; color:{MUTED}; font-size:.78rem; letter-spacing:.18em; text-transform:uppercase; margin-top:-4px; }}
.hud-card {{ border:1px solid rgba(0,229,255,.28); border-radius:14px; padding:12px 16px; min-height:84px;
  background: linear-gradient(180deg, rgba(14,21,38,.92), rgba(7,11,20,.92));
  box-shadow: inset 0 0 22px rgba(0,229,255,.05), 0 0 14px rgba(0,229,255,.05); }}
.hud-card.hot {{ border-color: rgba(255,43,214,.65); box-shadow: inset 0 0 22px rgba(255,43,214,.10), 0 0 20px rgba(255,43,214,.18); }}
.hud-label {{ font-family:'JetBrains Mono'; font-size:.68rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.16em; }}
.hud-value {{ font-family:'Orbitron'; font-size:1.25rem; color:#E6F1FF; margin-top:4px; }}
.hud-value .glow {{ color:{CYAN}; text-shadow:0 0 12px rgba(0,229,255,.55); }}
.hud-value .pink {{ color:{MAGENTA}; text-shadow:0 0 12px rgba(255,43,214,.55); }}
.hero {{ border:1px solid rgba(0,229,255,.45); border-radius:16px; padding:16px 20px; margin-bottom:10px;
  background: linear-gradient(120deg, rgba(0,229,255,.10), rgba(255,43,214,.06) 70%, rgba(7,11,20,.2)); }}
.hero .name {{ font-family:'Orbitron'; font-size:1.55rem; font-weight:700; color:#fff; }}
.hero .meta {{ font-family:'JetBrains Mono'; color:{MUTED}; font-size:.8rem; letter-spacing:.1em; }}
.hero .why {{ margin-top:6px; font-size:1.05rem; color:#CFE8FF; }}
.pill {{ display:inline-block; font-family:'JetBrains Mono'; font-size:.72rem; padding:2px 8px; margin:2px 4px 2px 0;
  border-radius:999px; border:1px solid rgba(0,229,255,.35); color:{CYAN}; }}
.pill.pink {{ border-color: rgba(255,43,214,.5); color:{MAGENTA}; }}
.pill.lime {{ border-color: rgba(182,255,59,.5); color:{LIME}; }}
.feed {{ font-family:'JetBrains Mono'; font-size:.78rem; color:#B9D2EA; line-height:1.55; }}
.feed b {{ color:#fff; }}
.slot-row {{ font-family:'JetBrains Mono'; font-size:.82rem; display:flex; justify-content:space-between;
  border-bottom:1px dashed rgba(127,167,201,.18); padding:3px 0; }}
.slot-row .s {{ color:{MUTED}; width:44px; }}
.slot-row .n {{ flex:1; color:#E6F1FF; }}
.slot-row .v {{ color:{CYAN}; }}
.slot-row.empty .n {{ color:#4B6480; }}
div.stButton > button {{ font-family:'Orbitron'; letter-spacing:.06em; font-size:.74rem; border-radius:10px; }}
</style>
"""


def card(label: str, value: str, hot: bool = False) -> str:
    return f'<div class="hud-card{" hot" if hot else ""}"><div class="hud-label">{label}</div><div class="hud-value">{value}</div></div>'


def plotly_layout(fig, height=320):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Rajdhani, sans-serif", size=13, color="#CFE8FF"))
    fig.update_xaxes(gridcolor="rgba(127,167,201,.12)", zerolinecolor="rgba(127,167,201,.35)")
    fig.update_yaxes(gridcolor="rgba(127,167,201,.12)", zerolinecolor="rgba(127,167,201,.35)")
    return fig
