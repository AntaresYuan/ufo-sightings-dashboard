"""
HCDE 411 — Final project.
"How to Get Abducted by Aliens" — tabbed single-viewport interactive dashboard.

Three tabs, each fills the viewport (no scroll):

  TAB 1 — WHERE  : Leaflet map of the U.S. with dual-zoom behavior.
                   Low zoom = one circle per state (count inside; hover summary).
                   High zoom = individual sighting points (hover details).
                   Side panel: state filter + auto-updating summary.

  TAB 2 — WHEN   : Big animated decade map with play/pause/step controls.
                   Time series stays visible at bottom.

  TAB 3 — WHAT   : Shape distribution · Description clusters (KMeans) · Duration
                   by shape (ALTAIR — satisfies course requirement).
"""

import json, re
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

alt.data_transformers.disable_max_rows()

CSV = "/sessions/determined-amazing-bell/mnt/ufo-reports-master/csv-data/ufo-scrubbed-geocoded-time-standardized.csv"
OUT_DIR = "/sessions/determined-amazing-bell/mnt/ufo-reports-master/site"

COLS = ["datetime", "city", "state", "country", "shape", "duration_sec",
        "duration_text", "comments", "date_posted", "lat", "lng"]

# 2010 U.S. Census state populations (used for per-capita rate)
STATE_POPULATION = {
    "AL":4_780_000,"AK":710_000,"AZ":6_392_000,"AR":2_916_000,"CA":37_254_000,
    "CO":5_029_000,"CT":3_574_000,"DE":898_000,"DC":602_000,"FL":18_801_000,
    "GA":9_688_000,"HI":1_360_000,"ID":1_568_000,"IL":12_831_000,"IN":6_484_000,
    "IA":3_046_000,"KS":2_853_000,"KY":4_339_000,"LA":4_533_000,"ME":1_328_000,
    "MD":5_774_000,"MA":6_548_000,"MI":9_884_000,"MN":5_304_000,"MS":2_967_000,
    "MO":5_989_000,"MT":989_000,"NE":1_826_000,"NV":2_701_000,"NH":1_316_000,
    "NJ":8_792_000,"NM":2_059_000,"NY":19_378_000,"NC":9_535_000,"ND":673_000,
    "OH":11_537_000,"OK":3_751_000,"OR":3_831_000,"PA":12_702_000,"RI":1_053_000,
    "SC":4_625_000,"SD":814_000,"TN":6_346_000,"TX":25_146_000,"UT":2_764_000,
    "VT":626_000,"VA":8_001_000,"WA":6_725_000,"WV":1_853_000,"WI":5_687_000,"WY":564_000,
}

STATE_NAMES = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
    "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana",
    "NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico",
    "NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
    "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
    "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
    "DC":"District of Columbia",
}

SHAPE_GROUPS = {
    "Round":           {"disk", "circle", "sphere", "round", "oval", "egg"},
    "Elongated":       {"cigar", "cylinder", "cross", "rectangle", "teardrop"},
    "Light":           {"light", "fireball", "flash", "flare"},
    "Triangular":      {"triangle", "chevron", "delta", "diamond", "pyramid"},
    "Formation":       {"formation"},
    "Other / Unknown": {"other", "unknown", "changing", "changed", "cone", "hexagon", "dome", ""},
}
SHAPE_DOMAIN = ["Round", "Elongated", "Light", "Triangular", "Formation", "Other / Unknown"]
SHAPE_COLOR  = {"Round":"#4C9AFF","Elongated":"#FF8B5A","Light":"#FFD54A",
                "Triangular":"#9B6CDB","Formation":"#36C2A1","Other / Unknown":"#A0A0A0"}

CLUSTER_LABELS = {
    0: "Multi-light / colored formations",
    1: "General night-sky reports",
    2: "Bright single lights",
    3: "Triangular craft",
    4: "Hovering shaped objects",
    5: 'Classic "UFO" reports',
}
CLUSTER_COLOR = {
    0: "#E53935",   # red — multi-light formations
    1: "#7D7D7D",   # gray — general (the catch-all bucket)
    2: "#FFD54A",   # yellow — bright single lights
    3: "#9B6CDB",   # purple — triangular craft
    4: "#4C9AFF",   # blue — hovering shaped objects
    5: "#36C2A1",   # green — classic UFO
}

def group_shape(s):
    if not isinstance(s, str): return "Other / Unknown"
    s = s.strip().lower()
    for g, members in SHAPE_GROUPS.items():
        if s in members: return g
    return "Other / Unknown"


# ============================================================================
# 1. LOAD + CLEAN + CLUSTER DESCRIPTIONS
# ============================================================================
print("Loading NUFORC...")
df = pd.read_csv(CSV, names=COLS, low_memory=False)
df = df[df["country"].str.lower() == "us"].copy()
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])
df["year"] = df["datetime"].dt.year
df = df[(df["year"] >= 1950) & (df["year"] <= 2014)]
df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
df = df.dropna(subset=["duration_sec", "lat", "lng"])
df = df[(df["duration_sec"] > 0) & (df["duration_sec"] <= 24 * 3600)]
df = df[(df["lat"] > 24) & (df["lat"] < 50) & (df["lng"] > -125) & (df["lng"] < -66)]
df["state"] = df["state"].str.upper()
df["shape_group"] = df["shape"].apply(group_shape)
df["state_name"] = df["state"].map(STATE_NAMES)
df["decade"] = (df["year"] // 10 * 10).astype(int).astype(str) + "s"
df = df.dropna(subset=["state_name"])
print(f"  {len(df):,} clean US records")

# Description clustering
print("Clustering descriptions...")
def clean_text(s):
    if not isinstance(s, str): return ""
    s = s.replace("&#44", ",").replace("&amp;", "and").replace("&quot;", "")
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

df["text"] = df["comments"].apply(clean_text)
ok = df["text"].str.len() > 20
vec = TfidfVectorizer(max_features=2000, min_df=20, max_df=0.5,
                     ngram_range=(1,2), stop_words="english")
X = vec.fit_transform(df.loc[ok, "text"])
km = KMeans(n_clusters=6, random_state=42, n_init=10)
df.loc[ok, "cluster"] = km.fit_predict(X)
df["cluster"] = df["cluster"].fillna(0).astype(int)

# Extract top terms per cluster (for "What" tab UI) AND log them for label sanity check
terms = vec.get_feature_names_out()
cluster_top_terms = {}
print("\n>>> Cluster inspection — verify CLUSTER_LABELS match these:")
for c in range(6):
    top_idx = km.cluster_centers_[c].argsort()[::-1][:10]
    cluster_top_terms[c] = [terms[i] for i in top_idx]
    n = int((df["cluster"] == c).sum())
    print(f"  C{c} (n={n:,}, {n/len(df)*100:.1f}%) [label: {CLUSTER_LABELS[c]}]")
    print(f"     top terms: {', '.join(cluster_top_terms[c][:8])}")
print()

# Also show, for the top 5 sightings states, what their dominant cluster is — sanity check
print(">>> Top-5 states by total sightings — dominant cluster:")
top5 = df.groupby('state').size().sort_values(ascending=False).head(5).index.tolist()
for st in top5:
    sub = df[df['state'] == st]
    cc = sub['cluster'].value_counts()
    dom = int(cc.idxmax())
    print(f"  {st} (n={len(sub):,}): dominant = C{dom} ({CLUSTER_LABELS[dom]}, {100*cc.iloc[0]/len(sub):.1f}%)")
print()


# ============================================================================
# 2. PER-STATE AGGREGATES for hover summaries
# ============================================================================
print("Computing state aggregates...")
state_aggs = []
for st, sub in df.groupby("state"):
    cluster_counts = sub["cluster"].value_counts()
    top_clusters = cluster_counts.head(3)
    dominant_cluster = int(cluster_counts.idxmax())
    pop = STATE_POPULATION.get(st)
    rate = round(len(sub) / pop * 100_000, 2) if pop else None  # per 100k residents
    state_aggs.append({
        "st":  st,
        "sn":  STATE_NAMES.get(st, st),
        "tot": int(len(sub)),
        "lat": float(sub["lat"].mean()),
        "lng": float(sub["lng"].mean()),
        # top event types (narrative clusters)
        "topcl": [{"cl": int(c), "lbl": CLUSTER_LABELS[int(c)], "n": int(n)}
                  for c, n in top_clusters.items()],
        "avg_min": round(sub["duration_sec"].median() / 60.0, 1),
        "dcl": dominant_cluster,
        "dcl_pct": round(100 * cluster_counts.iloc[0] / len(sub), 1),
        "pop": pop,
        "rate": rate,
    })
state_aggs.sort(key=lambda x: -x["tot"])
print(f"  {len(state_aggs)} states aggregated")


# ============================================================================
# 3. POINTS PAYLOAD — with jitter (so co-located points spread out) +
#    full set of NUFORC fields for rich tooltips on hover
# ============================================================================
print("Building points payload...")
import numpy as np
rng = np.random.RandomState(42)

def clean_desc(s):
    """Clean NUFORC's escaped description text; truncate."""
    if not isinstance(s, str): return ""
    s = (s.replace("&#44", ",")
           .replace("&amp;", "and")
           .replace("&quot;", '"')
           .replace("&#33", "!")
           .replace("&#39", "'")
           .replace("&#33", "!"))
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 220:
        s = s[:220].rsplit(" ", 1)[0] + "…"
    return s

# Apply jitter so co-located reports (NUFORC commonly geocodes to city centroid)
# fan out within ~500m radius, instead of collapsing to a single visual point.
JITTER = 0.006  # degrees ≈ 600m
pts = df[["lat","lng","state","state_name","city","shape","shape_group",
          "year","duration_text","cluster","duration_sec","datetime",
          "date_posted","comments"]].copy()
pts["lat_j"] = pts["lat"] + rng.uniform(-JITTER, JITTER, len(pts))
pts["lng_j"] = pts["lng"] + rng.uniform(-JITTER, JITTER, len(pts))
pts["lat_j"] = pts["lat_j"].round(4)
pts["lng_j"] = pts["lng_j"].round(4)
pts["lat"] = pts["lat"].round(3)
pts["lng"] = pts["lng"].round(3)
pts["datetime_str"] = pts["datetime"].dt.strftime("%Y-%m-%d %H:%M")
pts["desc"] = pts["comments"].apply(clean_desc)

js_points = [
    {
        "lat": float(r["lat_j"]), "lng": float(r["lng_j"]),
        "lat0": float(r["lat"]), "lng0": float(r["lng"]),
        "st":  r["state"], "ct": (r["city"] if isinstance(r["city"], str) else "").title(),
        "sh":  r["shape"] if isinstance(r["shape"], str) else "",
        "sg":  r["shape_group"],
        "yr":  int(r["year"]),
        "dts": r["datetime_str"] if isinstance(r["datetime_str"], str) else "",
        "dt":  r["duration_text"] if isinstance(r["duration_text"], str) else "",
        "dp":  r["date_posted"] if isinstance(r["date_posted"], str) else "",
        "cl":  int(r["cluster"]),
        "ds":  r["desc"],
    }
    for _, r in pts.iterrows()
]
print(f"  {len(js_points):,} points embedded (with jitter + full fields)")


# ============================================================================
# 4. TIME / SHAPE / CLUSTER pre-aggregates
# ============================================================================
us_year = df.groupby("year").size().reset_index(name="count")
us_shape = df.groupby("shape_group").size().reset_index(name="count")
us_cluster = df.groupby("cluster").size().reset_index(name="count")
us_decade = df.groupby("decade").size().reset_index(name="count")

# Per-decade per-state counts (for the "When" tab animation)
decade_state = (
    df.groupby(["decade", "state", "state_name"]).size().reset_index(name="count")
)
decades = sorted(df["decade"].unique())


# ============================================================================
# 5. PLOTLY FIGURES (autosize)
# ============================================================================
# ---- Time series (When tab) — ALTAIR (satisfies the course Altair requirement) ----
us_year_for_alt = us_year.rename(columns={"year": "Year", "count": "Sightings"})
ts_chart = (
    alt.Chart(us_year_for_alt, title="")
    .mark_area(
        line=alt.LineConfig(color="#1E7F5E", strokeWidth=2.2, opacity=1),
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="rgba(54,194,161,0.30)", offset=0),
                   alt.GradientStop(color="rgba(54,194,161,0.05)", offset=1)],
            x1=0, x2=0, y1=0, y2=1,
        ),
        interpolate="monotone",
    )
    .encode(
        x=alt.X("Year:O",
                axis=alt.Axis(values=list(range(1950, 2015, 10)),
                              labelFontSize=11, titleFontSize=11, title=None)),
        y=alt.Y("Sightings:Q",
                axis=alt.Axis(labelFontSize=11, titleFontSize=11,
                              grid=True, gridColor="#eee")),
        tooltip=[
            alt.Tooltip("Year:O"),
            alt.Tooltip("Sightings:Q", format=","),
        ],
    )
    .properties(width="container", height="container")
    .configure_view(strokeWidth=0)
    .configure_axis(domain=False)
)
ts_spec_altair = json.loads(ts_chart.to_json())

# ---- Decade choropleth (single, JS will swap data by decade) ----
def make_decade_choropleth(decade):
    sub = decade_state[decade_state["decade"] == decade]
    fig = go.Figure(go.Choropleth(
        locations=sub["state"], z=sub["count"],
        locationmode="USA-states",
        colorscale="Viridis",
        zmin=0, zmax=int(decade_state["count"].max()),
        marker_line_color="white", marker_line_width=0.6,
        showscale=True,
        colorbar=dict(title="Sightings", thickness=14, len=0.7),
        text=sub["state_name"],
        hovertemplate=f"<b>%{{text}}</b><br>{decade}: %{{z:,}} sightings<extra></extra>",
    ))
    fig.update_layout(
        geo=dict(scope="usa", showcoastlines=False, showland=True,
                 landcolor="#F2F2EF", showlakes=False, bgcolor="white",
                 countrycolor="white"),
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="white",
        autosize=True,
    )
    return fig

# Pre-compute all decade figures' data (we'll switch via JS Plotly.react)
decade_data_by = {}
for d in decades:
    sub = decade_state[decade_state["decade"] == d]
    decade_data_by[d] = {
        "locations": sub["state"].tolist(),
        "z": sub["count"].tolist(),
        "text": sub["state_name"].tolist(),
        "zmax": int(decade_state["count"].max()),
    }

# Initial decade figure (1950s)
dec_init_fig = make_decade_choropleth(decades[0])

# ---- Sightings per capita (replaces shape distribution) ----
ranked = sorted([s for s in state_aggs if s["rate"] is not None],
                key=lambda x: x["rate"], reverse=True)
TOP_N = 15
top_rate = ranked[:TOP_N][::-1]   # reverse so the highest sits at the top of the chart

sh_fig = go.Figure(go.Bar(
    y=[s["sn"] for s in top_rate],
    x=[s["rate"] for s in top_rate],
    orientation="h",
    marker=dict(color="#1E7F5E"),
    customdata=[[s["st"], s["tot"], s["pop"]] for s in top_rate],
    hovertemplate=(
        "<b>%{y}</b><br>"
        "<b>%{x:.1f}</b> sightings per 100k residents<br>"
        "%{customdata[1]:,} raw sightings · pop. %{customdata[2]:,}<br>"
        "<span style='color:#aaa'>click to focus map on this state</span>"
        "<extra></extra>"
    ),
    text=[f"{s['rate']:.1f}" for s in top_rate],
    textposition="outside",
))
sh_fig.update_layout(
    margin=dict(l=110, r=70, t=8, b=30),
    paper_bgcolor="white", plot_bgcolor="white",
    autosize=True, showlegend=False,
    xaxis=dict(showgrid=True, gridcolor="#eee", tickfont=dict(size=11),
               title="Sightings per 100,000 residents", title_font=dict(size=11)),
    yaxis=dict(tickfont=dict(size=11)),
)

# ---- Description cluster bars (What tab) ----
us_cluster["label"] = us_cluster["cluster"].map(CLUSTER_LABELS)
us_cluster = us_cluster.sort_values("count", ascending=True)
cl_fig = go.Figure(go.Bar(
    y=us_cluster["label"], x=us_cluster["count"],
    orientation="h",
    marker=dict(color=[CLUSTER_COLOR[c] for c in us_cluster["cluster"]]),
    hovertemplate="<b>%{y}</b><br>%{x:,} sightings<extra></extra>",
    text=us_cluster["count"].apply(lambda x: f"{x:,}"),
    textposition="outside",
))
cl_fig.update_layout(
    margin=dict(l=180, r=80, t=8, b=28),
    paper_bgcolor="white", plot_bgcolor="white",
    autosize=True, showlegend=False,
    xaxis=dict(showgrid=True, gridcolor="#eee", tickfont=dict(size=11)),
    yaxis=dict(tickfont=dict(size=11)),
)

# ---- Duration by event-type (narrative cluster) — PLOTLY box plot ----
df_dur = df.copy()
df_dur["duration_min"] = df_dur["duration_sec"] / 60.0
df_dur["event_type"]   = df_dur["cluster"].map(CLUSTER_LABELS)
df_dur_sample = df_dur.sample(min(8000, len(df_dur)), random_state=0)
CLUSTER_LABEL_LIST = [CLUSTER_LABELS[k] for k in sorted(CLUSTER_LABELS.keys())]
CLUSTER_COLOR_LIST = [CLUSTER_COLOR[k]  for k in sorted(CLUSTER_LABELS.keys())]
# Build a per-cluster summary (min / q1 / median / q3 / max) so we can show
# one compact tooltip per box instead of Plotly's default 5-stat avalanche.
dur_stats = (
    df_dur_sample.groupby("event_type")["duration_min"]
                 .describe(percentiles=[0.25, 0.5, 0.75])
                 [["min","25%","50%","75%","max"]]
                 .to_dict("index")
)

dur_fig = go.Figure()
for k in sorted(CLUSTER_LABELS.keys()):
    label = CLUSTER_LABELS[k]
    sub = df_dur_sample[df_dur_sample["event_type"] == label]
    s = dur_stats.get(label, {})
    summary = (
        f"<b>{label}</b><br>"
        f"median: {s.get('50%', 0):.1f} min · "
        f"IQR: {s.get('25%', 0):.1f} – {s.get('75%', 0):.1f} min · "
        f"range: {s.get('min', 0):.2f} – {s.get('max', 0):.0f} min"
        "<extra></extra>"
    )
    dur_fig.add_trace(go.Box(
        x=sub["duration_min"],
        name=label,
        marker_color=CLUSTER_COLOR[k],
        boxpoints=False,
        orientation="h",
        hoveron="boxes",          # hover ONLY on the box body, not on every quartile
        hoverinfo="text",
        hovertext=summary,
    ))
dur_fig.update_layout(
    xaxis=dict(type="log", title="Duration (minutes, log scale)",
               showgrid=True, gridcolor="#eee", tickfont=dict(size=11),
               title_font=dict(size=11)),
    yaxis=dict(autorange="reversed",          # match top-down legend order
               tickfont=dict(size=11)),
    margin=dict(l=170, r=20, t=8, b=40),
    paper_bgcolor="white", plot_bgcolor="white",
    autosize=True, showlegend=False,
    hovermode="closest",                       # only the nearest box, never all 5 stats
)
duration_spec = dur_fig.to_dict()


# ============================================================================
# 6. PAYLOADS
# ============================================================================
js_data = {
    "points":       js_points,
    "stateAggs":    state_aggs,
    "decades":      decades,
    "decadeData":   decade_data_by,
    "shapeDomain":  SHAPE_DOMAIN,
    "shapeColor":   SHAPE_COLOR,
    "clusterLabels": CLUSTER_LABELS,
    "clusterColor":  {str(k): v for k, v in CLUSTER_COLOR.items()},
    "clusterTopTerms": {str(k): v for k, v in cluster_top_terms.items()},
    "totals":       {"records": int(len(df)), "states": int(df['state'].nunique())},
}


# ============================================================================
# 7. HTML
# ============================================================================
import os
os.makedirs(OUT_DIR, exist_ok=True)

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>How to Get Abducted by Aliens — UFO Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>
  :root {
    --ink:#0f1419; --muted:#5b6470; --line:#e5e7eb;
    --bg:#eef0f3; --card:#fff; --accent:#1E7F5E; --accent2:#36C2A1;
  }
  * { box-sizing: border-box; }
  html, body {
    margin:0; padding:0; height:100vh; overflow:hidden;
    background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    font-size:13px;
  }
  .app {
    display:grid; grid-template-rows: 56px 1fr;
    width:100vw; height:100vh; overflow:hidden;
  }
  header.bar {
    background:#0f1419; color:#fff;
    display:flex; flex-wrap:nowrap;
    align-items:center; gap:14px; padding:0 24px;
  }
  header.bar h1 { flex:1 1 auto; }
  #clear-all-btn:hover { background:#1a2027; color:#fff; }
  header.bar h1 { font-size:16px; font-weight:700; margin:0; }
  header.bar h1 .sub { font-size:12px; color:#9ba3ad; margin-left:8px; font-weight:400; }
  nav.tabs { display:flex; gap:4px; }
  nav.tabs button {
    background:transparent; color:#9ba3ad; border:none;
    padding:8px 18px; font-size:13px; cursor:pointer;
    border-radius:8px; font-weight:600;
    transition: all 0.15s ease;
  }
  nav.tabs button:hover { color:#fff; background:rgba(255,255,255,0.06); }
  nav.tabs button.active {
    background:var(--accent2); color:#0f1419;
  }
  header.bar .credit { font-size:11px; color:#9ba3ad; }

  /* TAB CONTAINERS — each takes full remaining viewport */
  .tab {
    display:none;
    width:100%; height:100%;
    padding:12px; gap:12px;
    overflow:hidden;
  }
  .tab.active { display:grid; }

  /* TAB 1 — WHERE: map left (3fr) + sidepanel right (1fr) */
  .tab-where { grid-template-columns: 3fr 1fr; }
  .tab-where #main-map { width:100%; height:100%; border-radius:10px; }

  /* TAB 2 — WHEN: big decade map top + controls + time series bottom */
  .tab-when { grid-template-rows: 1fr 80px 1fr; gap:10px; }
  .tab-when .controls {
    background:#fff; border-radius:10px; padding:14px 20px;
    display:flex; align-items:center; gap:14px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
  }
  .tab-when .controls button {
    background:var(--accent); color:#fff; border:none;
    padding:8px 16px; font-size:13px; cursor:pointer;
    border-radius:6px; font-weight:600;
  }
  .tab-when .controls button:hover { background:#155b44; }
  .tab-when .controls button:disabled { opacity:0.5; cursor:default; }
  .tab-when .controls .scrubber {
    flex:1; display:flex; align-items:center; gap:8px;
  }
  .tab-when .controls .scrubber input[type=range] {
    flex:1; height:6px;
  }
  .tab-when .controls .decade-label {
    font-size:28px; font-weight:800; color:#0f1419; min-width:90px;
  }
  .tab-when .pills { display:flex; gap:6px; }
  .tab-when .pills button {
    background:#eef9f4; color:var(--accent); padding:5px 11px; font-size:12px;
  }
  .tab-when .pills button.on {
    background:var(--accent); color:#fff;
  }

  /* TAB 3 — WHAT: grid of 4 cards */
  .tab-what {
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
  }

  /* CARDS */
  .card {
    background:var(--card); border-radius:10px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
    display:flex; flex-direction:column;
    overflow:hidden; min-height:0; min-width:0;
  }
  .card h2 {
    font-size:11px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.07em;
    color:var(--muted);
    padding:10px 14px;
    border-bottom:1px solid var(--line);
    margin:0;
    display:flex; justify-content:space-between; align-items:center;
    flex-shrink:0;
  }
  .card h2 .hint { font-weight:400; text-transform:none; letter-spacing:0; color:#9aa1ab; font-size:11px; }
  .card .body { flex:1 1 auto; min-height:0; min-width:0; padding:6px; position:relative; }

  /* SIDE PANEL (WHERE tab) */
  .where-side { display:flex; flex-direction:column; gap:10px; min-height:0; }
  .info-card {
    background:linear-gradient(135deg,var(--accent2),var(--accent));
    color:#fff; border-radius:10px; padding:18px 20px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
  }
  .info-card .label { font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:.08em; }
  .info-card .state-name { font-size:26px; font-weight:800; margin-top:4px; line-height:1.1; }
  .info-card .total { font-size:14px; opacity:.95; margin-top:6px; }
  .filter-group {
    display:grid; grid-template-columns: auto 1fr; gap:8px 14px;
    align-items:center; padding:14px 16px;
    background:#fff; border-radius:10px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
  }
  .filter-group label { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }
  .filter-group select {
    padding:6px 8px; font-size:12px; border:1px solid var(--line);
    border-radius:6px; background:#fafafa;
  }
  .legend { font-size:11px; color:#555; padding:0 0 0 4px; }
  .legend .row { display:flex; align-items:center; gap:8px; margin:3px 0; }
  .legend .swatch { width:10px; height:10px; border-radius:50%; }
  .help {
    background:#fff; border-radius:10px; padding:12px 16px;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
    font-size:11.5px; color:#555; line-height:1.5;
  }
  .help b { color:#0f1419; }

  #plotly-sh, #plotly-cl, #plotly-decmap, #plotly-duration { width:100%; height:100%; min-height:0; }
  #vega-ts { width:100%; height:100%; }
  .leaflet-container { background:#f5f5f0; border-radius:10px; }
  .state-tooltip, .point-tooltip {
    background:rgba(15,20,25,0.96); color:#fff; border:none !important;
    border-radius:7px; padding:11px 14px;
    box-shadow:0 4px 14px rgba(0,0,0,0.22);
    font-size:12px; line-height:1.5; max-width:340px;
    white-space:normal; word-wrap:break-word; overflow-wrap:break-word;
  }
  .state-tooltip b, .point-tooltip b { font-size:13px; color:#36C2A1; }
  .state-tooltip .row, .point-tooltip .row { margin:3px 0; }
  .state-tooltip .row.muted, .point-tooltip .row.muted { color:#9ba3ad; font-size:11px; }
  .point-tooltip .desc {
    margin-top:7px; padding-top:7px; border-top:1px solid #2a313a;
    color:#dde2e8; font-style:italic; font-size:11.5px; line-height:1.5;
    max-height: 6.5em; overflow:hidden;
  }
  /* Hide Leaflet default arrow on dark tooltip */
  .leaflet-tooltip-top.state-tooltip:before,
  .leaflet-tooltip-top.point-tooltip:before { border-top-color: rgba(15,20,25,0.96) !important; }
  .leaflet-tooltip-bottom.state-tooltip:before,
  .leaflet-tooltip-bottom.point-tooltip:before { border-bottom-color: rgba(15,20,25,0.96) !important; }

  /* Tab 3 - cluster details card */
  .cluster-detail-card { display:grid; grid-template-rows: 1fr auto; }
  .cluster-detail-card .terms-grid {
    display:grid; grid-template-columns: 1fr 1fr 1fr; gap:6px;
    padding:8px 12px; font-size:11px; color:#555;
  }
  .cluster-detail-card .terms-grid .label {
    font-weight:700; font-size:11px; text-transform:uppercase;
    letter-spacing:.04em; color:#0f1419; margin-bottom:2px;
  }
</style>
</head>
<body>
<div class="app">

  <header class="bar">
    <h1>How to Get Abducted by Aliens<span class="sub">63,832 UFO reports from across the U.S., 1950 to 2014. Data from NUFORC.</span></h1>
    <nav class="tabs">
      <button data-tab="where" class="active">Where</button>
      <button data-tab="when">When</button>
      <button data-tab="what">What</button>
    </nav>
    <button id="clear-all-btn" title="Reset every filter to its default" style="background:transparent; border:1px solid #2a313a; color:#9ba3ad; padding:6px 12px; border-radius:6px; font-size:11.5px; cursor:pointer; font-weight:600; letter-spacing:0.02em;">
      ↺ Clear all
    </button>
    <div class="credit">Antares Yuan — HCDE 411 final</div>
  </header>

  <!-- TAB 1: WHERE -->
  <main class="tab tab-where active" data-tab="where">
    <div class="card" style="overflow:hidden;">
      <h2>Main map
        <span class="hint">Zoom out for state totals, zoom in for individual reports.</span>
      </h2>
      <div class="body" style="padding:0;">
        <div id="main-map"></div>
      </div>
    </div>

    <div class="where-side">
      <div class="info-card">
        <div class="label">Showing</div>
        <div class="state-name" id="state-name-display">Every U.S. state</div>
        <div class="total"><span id="state-total-display">—</span> sightings in this view</div>
      </div>
      <div class="filter-group">
        <label for="state-filter">State</label>
        <select id="state-filter">
          <option value="ALL">All U.S. states</option>
        </select>
        <label for="event-filter">Event type</label>
        <select id="event-filter">
          <option value="ALL">All event types</option>
        </select>
      </div>
      <div class="card" style="flex:1;">
        <h2>A quick note on this map</h2>
        <div class="body" style="padding:12px 16px; overflow-y:auto; font-size:12px; line-height:1.55; color:#444;">
          <div style="margin-bottom:10px;">
            Zoomed out, every state gets one green circle with its total
            sightings inside. Hover one to see the top event types and the
            median sighting length.
          </div>
          <div style="margin-bottom:10px;">
            Zoom in and the points break apart into individual reports.
            Each one is colored by a "narrative cluster" — six recurring
            patterns I pulled out of the free-text descriptions using
            TF-IDF + KMeans:
          </div>
          <div id="cluster-legend" style="display:flex; flex-direction:column; gap:4px; margin-bottom:10px; padding:8px 10px; background:#fafafa; border-radius:6px;"></div>
          <div style="margin-bottom:10px;">
            The middle-zoom aggregated circles take the color of whichever
            cluster dominates inside them.
          </div>
          <div style="font-size:11px; color:#888; border-top:1px solid #eee; padding-top:8px;">
            "General night-sky reports" is the boring catch-all — it's
            mostly vague stuff like "lights moving fast." The narrower
            clusters (triangular craft, hovering shaped objects, the
            "I saw a UFO" type) are where the actually-distinctive
            stories live.
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- TAB 2: WHEN -->
  <main class="tab tab-when" data-tab="when">
    <div class="card">
      <h2>Decade map
        <span class="hint">One decade at a time. Use the buttons below to step or play through them.</span>
      </h2>
      <div class="body"><div id="plotly-decmap"></div></div>
    </div>

    <div class="controls">
      <button id="dec-play">▶ Play</button>
      <button id="dec-step-back">◀ Prev</button>
      <button id="dec-step-fwd">Next ▶</button>
      <div class="scrubber">
        <span style="font-size:11px;color:#9aa1ab;">Decade</span>
        <input type="range" id="dec-slider" min="0" max="0" step="1" value="0">
      </div>
      <span class="decade-label" id="dec-current-label">—</span>
      <div class="pills" id="dec-pills"></div>
    </div>

    <div class="card">
      <h2>Sightings over time
        <span class="hint">Yearly counts from 1950 to 2014. The decade map above moves along this curve.</span>
      </h2>
      <div class="body" style="padding:14px 18px;"><div id="vega-ts"></div></div>
    </div>
  </main>

  <!-- TAB 3: WHAT -->
  <main class="tab tab-what" data-tab="what">
    <div class="card">
      <h2>Sightings per 100k residents
        <span class="hint">Top 15 states once you control for population. Click any bar to jump the map there.</span>
      </h2>
      <div class="body"><div id="plotly-sh"></div></div>
    </div>
    <div class="card cluster-detail-card">
      <h2>Description clusters
        <span class="hint">Six recurring themes pulled out of the report text. Click a bar to filter the map.</span>
      </h2>
      <div class="body"><div id="plotly-cl"></div></div>
      <div class="terms-grid" id="cluster-terms-grid"></div>
    </div>
    <div class="card">
      <h2>Duration by event type
        <span class="hint">How long sightings last, broken out by event type. Log scale on the x-axis.</span>
      </h2>
      <div class="body"><div id="plotly-duration"></div></div>
    </div>
    <div class="card">
      <h2>What I learned making this</h2>
      <div class="body" style="padding:14px 18px; overflow-y:auto; font-size:12.5px; line-height:1.65; color:#444;">
        <p style="margin:0 0 12px;">If you actually wanted to maximize your chances of seeing one
        of these things, the three things that matter are: where to be, what kind of report counts
        as a "real" sighting, and how long they tend to stick around.</p>
        <p style="margin:0 0 12px;">The big map looks population-shaped at first — California
        wins, Texas wins, New York wins. That's mostly because those states have a lot of people.
        Once I divided by population, Washington ended up #1 and the top of the list filled up
        with Montana, Oregon, the rural Mountain West, and New England — places with dark skies
        and not much else to do at 2am.</p>
        <p style="margin:0 0 12px;">Most reports turn out to be vague descriptions of lights
        moving in the sky. The KMeans clustering put about 35% of them in one big "General
        night-sky" bucket. The other five clusters are smaller but more interesting — Triangular
        craft, Hovering shaped objects, and the explicit "I saw a UFO" reports are the ones with
        recognizable encounter shapes.</p>
        <p style="margin:0;">Duration is the surprising one. Most clusters look pretty similar
        on the box plot, but a few have much wider ranges — meaning some people are reporting
        encounters lasting hours, not seconds. Those are probably either the abduction-grade
        encounters everyone is chasing, or someone staring at Venus for a really long time.</p>
      </div>
    </div>
  </main>
</div>

<script>
const data = __DATA__;
const tsSpec = __TS__;
const shSpec = __SH__;
const clSpec = __CL__;
const decFig = __DEC__;
const durSpec = __DUR__;

const SHAPE_DOMAIN  = data.shapeDomain;
const SHAPE_COLOR   = data.shapeColor;
const CLUSTER_LABELS = data.clusterLabels;
const CLUSTER_COLOR  = data.clusterColor;
const points         = data.points;
const stateAggs      = data.stateAggs;
const decades        = data.decades;

/* ====================== TAB NAVIGATION ====================== */
document.querySelectorAll('nav.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    const t = btn.dataset.tab;
    document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('main.tab').forEach(m => m.classList.remove('active'));
    document.querySelector(`main.tab[data-tab="${t}"]`).classList.add('active');
    // Force plot resize when their tab becomes visible
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
      if (t === 'where' && window._map) window._map.invalidateSize();
    }, 50);
  });
});


/* ====================== TAB 1 — WHERE: MAP ====================== */
const map = L.map('main-map', {
  preferCanvas: true,
  maxBounds: [[18, -130], [55, -60]],
  minZoom: 4,
  maxZoom: 13,
}).setView([39.5, -98.5], 4);
window._map = map;

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  subdomains:'abcd', maxZoom:19,
}).addTo(map);

// State-level layer (low zoom). State circles use one dark-green color —
// the dominant narrative cluster is the same in nearly all 49 states (the
// "general night-sky reports" catch-all), so coloring them is misleading.
// Per-point cluster color still kicks in at high zoom.
const STATE_COLOR = '#1E7F5E';
const stateLayer = L.layerGroup();
stateAggs.forEach(s => {
  const r = Math.max(18, Math.min(48, 8 + Math.sqrt(s.tot) * 0.8));
  const dominantLabel = CLUSTER_LABELS[s.dcl] || '—';
  const dominantColor = CLUSTER_COLOR[s.dcl] || '#999';
  const topEventsHtml = (s.topcl || []).map(t =>
    `<div class="row" style="display:flex;justify-content:space-between;gap:14px;">
       <span style="display:inline-flex;align-items:center;gap:6px;">
         <span style="width:8px;height:8px;border-radius:50%;background:${CLUSTER_COLOR[t.cl]||'#999'};display:inline-block;"></span>${t.lbl}
       </span>
       <span>${t.n.toLocaleString()}</span>
     </div>`).join('');
  const tipHtml =
    `<b>${s.sn}</b>
     <div class="row">${s.tot.toLocaleString()} sightings</div>
     <div class="row muted">Median duration: ${s.avg_min} min</div>
     <div class="row muted" style="margin-top:7px;">Most common event type:</div>
     <div class="row" style="display:flex;align-items:center;gap:7px;">
       <span style="width:9px;height:9px;border-radius:50%;background:${dominantColor};display:inline-block;flex-shrink:0;"></span>
       <span><b>${dominantLabel}</b> <span style="opacity:0.65;">(${s.dcl_pct}%)</span></span>
     </div>
     <div class="row muted" style="margin-top:7px;">Top event types:</div>
     ${topEventsHtml}`;
  const icon = L.divIcon({
    html: `<div style="background:${STATE_COLOR};color:#fff;font-weight:700;font-size:12px;
            width:${r}px;height:${r}px;border-radius:50%;display:flex;align-items:center;justify-content:center;
            border:2.5px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.28);opacity:0.95;">
            ${s.tot >= 1000 ? (s.tot/1000).toFixed(1)+'k' : s.tot}
          </div>`,
    className: 'state-icon',
    iconSize: [r, r],
  });
  const m = L.marker([s.lat, s.lng], { icon: icon })
    .bindTooltip(tipHtml, {
      direction: 'top', offset: [0, -r/2], opacity: 1,
      className: 'state-tooltip', sticky: true,
    });
  m._stateCode = s.st;
  stateLayer.addLayer(m);
});

// Point-level layer (high zoom). One layer per shape group (for fast shape-
// filter toggle) and each marker carries its narrative-cluster id so the
// aggregation icons can be colored by the dominant cluster.
const pointLayers = {};
SHAPE_DOMAIN.forEach(sg => {
  pointLayers[sg] = L.markerClusterGroup({
    showCoverageOnHover: false,
    spiderfyOnMaxZoom:   false,    // no virus / sunburst
    spiderfyOnEveryZoom: false,
    zoomToBoundsOnClick: true,
    disableClusteringAtZoom: 11,   // above zoom 11, show every point
    chunkedLoading: true,
    maxClusterRadius: 38,
    iconCreateFunction: function(cluster) {
      const children = cluster.getAllChildMarkers();
      const counts = {};
      children.forEach(m => { counts[m._cl] = (counts[m._cl] || 0) + 1; });
      const dominantCl = Object.keys(counts).reduce(
        (a, b) => (counts[a] >= counts[b] ? a : b)
      );
      const color = CLUSTER_COLOR[dominantCl] || '#1E7F5E';
      // Dark text on yellow (C2) and gray (C1), white text on the rest
      const textColor = (dominantCl == 2 || dominantCl == 1) ? '#0f1419' : '#fff';
      const count = cluster.getChildCount();
      let size = 26;
      if (count >= 10)  size = 30;
      if (count >= 100) size = 38;
      if (count >= 500) size = 46;
      return L.divIcon({
        html: `<div style="background:${color};opacity:0.92;color:${textColor};font-weight:700;font-size:11.5px;
                width:${size}px;height:${size}px;border-radius:50%;display:flex;align-items:center;justify-content:center;
                border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.3);">
                ${count >= 1000 ? (count/1000).toFixed(1)+'k' : count.toLocaleString()}
              </div>`,
        className: 'point-cluster-icon',
        iconSize: [size, size],
      });
    },
  });
});

/* Build per-point markers — tooltip on hover with all NUFORC fields. */
function pointTooltipHtml(p) {
  const clColor = CLUSTER_COLOR[p.cl] || '#999';
  const desc = p.ds ? `<div class="desc">"${p.ds}"</div>` : '';
  const reportedRow = p.dp ? `<div class="row muted">Reported on NUFORC: ${p.dp}</div>` : '';
  return `
    <b>${p.ct || 'Unknown city'}, ${p.st}</b>
    <div class="row" style="color:#fff; opacity:0.85;">${p.dts || p.yr}</div>
    <div class="row" style="margin-top:6px; display:flex; align-items:center; gap:7px;">
      <span style="width:9px;height:9px;border-radius:50%;background:${clColor};display:inline-block;flex-shrink:0;"></span>
      <span>Event type: <b>${CLUSTER_LABELS[p.cl] || '—'}</b></span>
    </div>
    <div class="row">Duration: <b>${p.dt || '—'}</b></div>
    <div class="row">Coords: ${p.lat0.toFixed(3)}, ${p.lng0.toFixed(3)}</div>
    ${reportedRow}
    ${desc}
  `;
}

points.forEach(p => {
  const color = CLUSTER_COLOR[p.cl] || '#999';
  const marker = L.circleMarker([p.lat, p.lng], {
    radius: 4.2, color: color, fillColor: color,
    fillOpacity: 0.75, weight: 0.7,
  });
  marker._cl = p.cl;  // for the cluster icon dominance lookup
  marker.bindTooltip(pointTooltipHtml(p), {
    direction: 'top', offset: [0, -6], opacity: 0.97,
    className: 'point-tooltip', sticky: true,
  });
  pointLayers[p.sg].addLayer(marker);
});

// Zoom threshold: below this we show state circles; above we show point clusters
const ZOOM_THRESHOLD = 6;

function syncLayersToZoom() {
  const z = map.getZoom();
  if (z < ZOOM_THRESHOLD) {
    if (!map.hasLayer(stateLayer)) map.addLayer(stateLayer);
    SHAPE_DOMAIN.forEach(sg => {
      if (map.hasLayer(pointLayers[sg])) map.removeLayer(pointLayers[sg]);
    });
  } else {
    if (map.hasLayer(stateLayer)) map.removeLayer(stateLayer);
    SHAPE_DOMAIN.forEach(sg => {
      if (!map.hasLayer(pointLayers[sg])) map.addLayer(pointLayers[sg]);
    });
  }
}

map.on('zoomend', syncLayersToZoom);
syncLayersToZoom();


/* ====================== FILTERS ====================== */
let selectedState   = 'ALL';
let selectedCluster = 'ALL';   // string id ("0"..."5") or 'ALL'

const stateSel = document.getElementById('state-filter');
stateAggs.forEach(s => {
  const opt = document.createElement('option');
  opt.value = s.st;
  opt.textContent = `${s.sn} (${s.tot.toLocaleString()})`;
  stateSel.appendChild(opt);
});

const eventSel = document.getElementById('event-filter');
Object.entries(CLUSTER_LABELS).forEach(([cid, label]) => {
  const opt = document.createElement('option');
  opt.value = cid; opt.textContent = label;
  eventSel.appendChild(opt);
});

// Populate the cluster legend in the "How to read this map" card
const clusterLegend = document.getElementById('cluster-legend');
Object.entries(CLUSTER_LABELS).forEach(([cid, label]) => {
  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.alignItems = 'center';
  row.style.gap = '8px';
  row.style.fontSize = '11.5px';
  row.innerHTML =
    `<span style="width:11px;height:11px;border-radius:50%;background:${CLUSTER_COLOR[cid]};display:inline-block;flex-shrink:0;"></span>
     <span>${label}</span>`;
  clusterLegend.appendChild(row);
});

function applyFilters() {
  /* Apply filters to the points dataset for side-panel charts + info card. */
  let filtered = points;
  if (selectedState   !== 'ALL') filtered = filtered.filter(p => p.st === selectedState);
  if (selectedCluster !== 'ALL') filtered = filtered.filter(p => String(p.cl) === selectedCluster);
  /* Keep header dropdowns in sync with the current filter state */
  stateSel.value = selectedState;
  eventSel.value = selectedCluster;

  /* Info card */
  document.getElementById('state-total-display').textContent = filtered.length.toLocaleString();
  if (selectedState === 'ALL') {
    document.getElementById('state-name-display').textContent = 'Every U.S. state';
  } else {
    const row = stateAggs.find(s => s.st === selectedState);
    document.getElementById('state-name-display').textContent = row ? row.sn : selectedState;
  }

  /* Side panel: time series (Altair). We recompute the year counts from the
     filtered points, then update the embedded Vega view's data. */
  const byYr = {};
  filtered.forEach(p => { byYr[p.yr] = (byYr[p.yr] || 0) + 1; });
  const yrs = Object.keys(byYr).map(Number).sort((a,b) => a - b);
  const tsValues = yrs.map(y => ({ Year: y, Sightings: byYr[y] }));
  if (window._tsView) {
    const changeSet = vega.changeset().remove(() => true).insert(tsValues);
    window._tsView.change('source_0', changeSet).runAsync();
  }

  /* Side panel: per-capita ranking is a fixed state-level chart and does
     not change with shape / cluster / time filters. We only highlight the
     currently selected state (if any). */
  highlightPerCapitaSelectedState();

  /* Side panel: cluster distribution (also clickable for filtering) */
  const byCl = {};
  filtered.forEach(p => { byCl[p.cl] = (byCl[p.cl] || 0) + 1; });
  const clKeys = Object.keys(CLUSTER_LABELS).sort((a,b) => (byCl[a]||0) - (byCl[b]||0));
  const clColors = clKeys.map(k => CLUSTER_COLOR[k]);
  Plotly.react('plotly-cl', [{
    y: clKeys.map(k => CLUSTER_LABELS[k]),
    x: clKeys.map(k => byCl[k] || 0),
    type:'bar', orientation:'h',
    marker: {
      color: clColors,
      line: {
        color: clKeys.map(k => selectedCluster === k ? '#0f1419' : 'rgba(0,0,0,0)'),
        width: clKeys.map(k => selectedCluster === k ? 2 : 0),
      },
    },
    text: clKeys.map(k => (byCl[k] || 0).toLocaleString()),
    textposition: 'outside',
    hovertemplate:'<b>%{y}</b><br>%{x:,} sightings · click to filter<extra></extra>',
  }], clSpec.layout, cfg);

  /* Main map: ensure all (internal) shape-group layers are visible
     at high zoom — there is no per-shape filter anymore. */
  SHAPE_DOMAIN.forEach(sg => {
    if (!map.hasLayer(pointLayers[sg]) && map.getZoom() >= ZOOM_THRESHOLD) {
      map.addLayer(pointLayers[sg]);
    }
  });

  /* Cluster filter — rebuild marker visibility per layer. Setting CSS
     display:none on the icon won't update cluster counts; we re-create
     the markers in the appropriate sub-layer instead. */
  rebuildClusterFilter();

  /* Map: zoom to selected state */
  if (selectedState !== 'ALL' && filtered.length > 0) {
    const lats = filtered.map(p => p.lat);
    const lngs = filtered.map(p => p.lng);
    map.fitBounds([
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)],
    ], { padding: [40, 40] });
  } else if (selectedState === 'ALL') {
    map.setView([39.5, -98.5], 4);
  }
  syncLayersToZoom();

}

/* Re-populate the 6 shape sub-layers from the points array, respecting
   the cluster filter. Called on cluster-filter changes. */
function rebuildClusterFilter() {
  SHAPE_DOMAIN.forEach(sg => pointLayers[sg].clearLayers());
  const subset = selectedCluster === 'ALL'
    ? points
    : points.filter(p => String(p.cl) === selectedCluster);
  subset.forEach(p => {
    const color = CLUSTER_COLOR[p.cl] || '#999';
    const marker = L.circleMarker([p.lat, p.lng], {
      radius: 4.2, color: color, fillColor: color,
      fillOpacity: 0.75, weight: 0.7,
    });
    marker._cl = p.cl;
    marker.bindTooltip(pointTooltipHtml(p), {
      direction: 'top', offset: [0, -6], opacity: 0.97,
      className: 'point-tooltip', sticky: true,
    });
    pointLayers[p.sg].addLayer(marker);
  });
}

stateSel.addEventListener('change', e => { selectedState   = e.target.value; applyFilters(); });
eventSel.addEventListener('change', e => { selectedCluster = e.target.value; applyFilters(); });

/* Cluster bar — click a bar to filter the map; click same bar again to clear */
function attachClusterBarClick() {
  const el = document.getElementById('plotly-cl');
  el.on('plotly_click', function(d) {
    const label = d.points[0].y;
    const cid = Object.keys(CLUSTER_LABELS).find(k => CLUSTER_LABELS[k] === label);
    if (cid !== undefined) {
      selectedCluster = (selectedCluster === cid) ? 'ALL' : cid;
      applyFilters();
    }
  });
}

/* Per-capita bar — click a bar to set that state as the active filter */
function attachPerCapitaBarClick() {
  const el = document.getElementById('plotly-sh');
  el.on('plotly_click', function(d) {
    const stateName = d.points[0].y;
    const row = stateAggs.find(s => s.sn === stateName);
    if (row) {
      selectedState = (selectedState === row.st) ? 'ALL' : row.st;
      stateSel.value = selectedState;
      // Switch to Where tab so the user sees the map update
      const whereBtn = document.querySelector('nav.tabs button[data-tab="where"]');
      if (whereBtn) whereBtn.click();
      applyFilters();
    }
  });
}

/* Highlight the currently selected state in the per-capita ranking by
   thickening its bar outline. */
function highlightPerCapitaSelectedState() {
  const el = document.getElementById('plotly-sh');
  if (!el || !el.data || !el.data[0]) return;
  const ys = el.data[0].y;
  const widths = ys.map(name => {
    const row = stateAggs.find(s => s.sn === name);
    return (row && row.st === selectedState) ? 2.5 : 0;
  });
  const colors = ys.map(name => {
    const row = stateAggs.find(s => s.sn === name);
    return (row && row.st === selectedState) ? '#0f1419' : 'rgba(0,0,0,0)';
  });
  Plotly.restyle('plotly-sh', { 'marker.line.width': [widths], 'marker.line.color': [colors] });
}

/* "Clear all" — reset every filter and the dropdowns at once.
   This is the "history / undo" affordance for Shneiderman's task list. */
document.getElementById('clear-all-btn').addEventListener('click', () => {
  selectedState   = 'ALL';
  selectedCluster = 'ALL';
  stateSel.value = 'ALL';
  eventSel.value = 'ALL';
  applyFilters();
});


/* ====================== TAB 2 — WHEN ====================== */
const decMap = document.getElementById('plotly-decmap');
/* Plotly config — keep only the PNG download button visible on hover.
   Satisfies the "extract" task in Shneiderman's classic list. */
const cfg = {
  displayModeBar: 'hover',
  displaylogo: false,
  responsive: true,
  toImageButtonOptions: { format: 'png', scale: 2, filename: 'ufo_dashboard_chart' },
  modeBarButtonsToRemove: [
    'zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d',
    'autoScale2d','resetScale2d','toggleSpikelines',
    'hoverClosestCartesian','hoverCompareCartesian',
    'zoom3d','pan3d','orbitRotation','tableRotation',
    'resetCameraDefault3d','resetCameraLastSave3d','hoverClosest3d',
    'zoomInGeo','zoomOutGeo','resetGeo','hoverClosestGeo',
    // keep: toImage (the download-as-PNG button)
  ],
};

const decBaseLayout = decFig.layout;
const decBaseData = decFig.data;
Plotly.newPlot('plotly-decmap', decBaseData, decBaseLayout, cfg);

// Decade controls
const slider = document.getElementById('dec-slider');
slider.max = decades.length - 1;
const decLabel = document.getElementById('dec-current-label');
const decPills = document.getElementById('dec-pills');
decades.forEach((d, i) => {
  const b = document.createElement('button');
  b.textContent = d;
  b.dataset.idx = i;
  b.addEventListener('click', () => setDecadeIdx(i));
  decPills.appendChild(b);
});

let curDecIdx = 0;
function setDecadeIdx(i) {
  curDecIdx = i;
  const d = decades[i];
  const dd = data.decadeData[d];
  Plotly.react('plotly-decmap', [{
    type:'choropleth',
    locations: dd.locations, z: dd.z, text: dd.text,
    locationmode: 'USA-states',
    colorscale: 'Viridis',
    zmin: 0, zmax: dd.zmax,
    marker: { line: { color:'white', width:0.6 } },
    showscale: true,
    colorbar: { title: { text:'Sightings' }, thickness:14, len:0.7 },
    hovertemplate: `<b>%{text}</b><br>${d}: %{z:,} sightings<extra></extra>`,
  }], decBaseLayout, cfg);
  decLabel.textContent = d;
  slider.value = i;
  document.querySelectorAll('#dec-pills button').forEach(b => {
    b.classList.toggle('on', parseInt(b.dataset.idx) === i);
  });
}
setDecadeIdx(0);

slider.addEventListener('input', e => setDecadeIdx(parseInt(e.target.value)));
document.getElementById('dec-step-back').addEventListener('click', () => {
  setDecadeIdx(Math.max(0, curDecIdx - 1));
});
document.getElementById('dec-step-fwd').addEventListener('click', () => {
  setDecadeIdx(Math.min(decades.length - 1, curDecIdx + 1));
});

let playTimer = null;
const playBtn = document.getElementById('dec-play');
playBtn.addEventListener('click', () => {
  if (playTimer) {
    clearInterval(playTimer); playTimer = null;
    playBtn.textContent = '▶ Play';
  } else {
    playBtn.textContent = '⏸ Pause';
    playTimer = setInterval(() => {
      const next = (curDecIdx + 1) % decades.length;
      setDecadeIdx(next);
    }, 1200);
  }
});

// Time series — Altair / Vega-Embed (this satisfies the course's Altair requirement).
// We keep a reference to the view so the filters can stream new data into it.
vegaEmbed('#vega-ts', tsSpec, {
  actions: { export: { png: true, svg: true }, source: false, compiled: false, editor: false },
  renderer: 'svg',
  downloadFileName: 'sightings_over_time'
}).then(result => { window._tsView = result.view; });


/* ====================== TAB 3 — WHAT ====================== */
Plotly.newPlot('plotly-sh', shSpec.data, shSpec.layout, cfg).then(() => {
  attachPerCapitaBarClick();
});
Plotly.newPlot('plotly-cl', clSpec.data, clSpec.layout, cfg).then(() => {
  attachClusterBarClick();
});
Plotly.newPlot('plotly-duration', durSpec.data, durSpec.layout, cfg);

// Cluster top terms grid
const termsGrid = document.getElementById('cluster-terms-grid');
const labelsArr = Object.entries(CLUSTER_LABELS);
labelsArr.forEach(([cid, label]) => {
  const block = document.createElement('div');
  const terms = (data.clusterTopTerms[cid] || []).slice(0, 5).join(', ');
  block.innerHTML = `
    <div class="label" style="color:${CLUSTER_COLOR[cid]}">${label}</div>
    <div style="font-size:10px;color:#777;line-height:1.4;">${terms}</div>
  `;
  termsGrid.appendChild(block);
});

// Initial sync of map
applyFilters();
</script>
</body>
</html>
"""

out = HTML
out = out.replace("__DATA__", json.dumps(js_data))
out = out.replace("__TS__",   json.dumps(ts_spec_altair, default=str))
out = out.replace("__SH__",   json.dumps(sh_fig.to_dict(), default=str))
out = out.replace("__CL__",   json.dumps(cl_fig.to_dict(), default=str))
out = out.replace("__DEC__",  json.dumps(dec_init_fig.to_dict(), default=str))
out = out.replace("__DUR__",  json.dumps(duration_spec, default=str))

out_path = f"{OUT_DIR}/index.html"
with open(out_path, "w") as f:
    f.write(out)
print(f"\nWrote: {out_path}")
print(f"Size:  {len(out) / 1024:.0f} KB")
