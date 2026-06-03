# How to Get Abducted by Aliens

An interactive single-viewport dashboard exploring U.S. UFO sightings from the
National UFO Reporting Center (NUFORC, 1950–2014), framed as a tongue-in-cheek
travel guide for the abduction-curious.

**Course:** HCDE 411 (Information Visualization), University of Washington
**Author:** Antares Yuan
**Live demo:** <https://antaresyuan.github.io/ufo-sightings-dashboard/>
**Source code:** <https://github.com/AntaresYuan/ufo-sightings-dashboard>

---

## What's in it

The dashboard is split into three tabs that each fill the viewport without
scrolling:

### Tab 1 — **Where**

- **Main interactive map** (Leaflet + MarkerCluster). Drag to pan, scroll to
  zoom. At low zoom each state is one green circle with its total sighting
  count. Hover for the state's top shape types and median sighting duration.
  Zoom past city level (~zoom 11) and individual reports appear, color-coded
  by narrative cluster. Hover any point for date, shape, duration,
  coordinates, NUFORC narrative cluster, and the eyewitness description.
- **Side panel**: filter chips (State, Shape), the active-state summary card,
  and a reading guide that includes the cluster color legend.

### Tab 2 — **When**

- **Decade choropleth** of the U.S., full screen, with **Play / Pause / Prev /
  Next** controls, a scrubbable slider, and seven decade pills that jump to
  any specific decade (1950s through 2010s). The map color scale is shared
  across decades so the dramatic growth in reports is visible.
- **Time series** below: yearly sighting counts, 1950–2014, that responds to
  the active state filter.

### Tab 3 — **What**

- **Sightings per 100,000 residents** (top 15 states, ranked). Click a bar
  to focus the map on that state — this is the key normalization view that
  reveals Washington, Montana, Oregon, and the Mountain West as the real
  per-capita hotspots, not California.
- **Description clusters** (clickable bar chart). The 60k+ free-text NUFORC
  descriptions were grouped into six themes using TF-IDF + KMeans. Click a
  bar to filter the entire dashboard to that narrative cluster; click again
  to clear.
- **Duration by shape** (Altair box plot, log scale). Reveals which sighting
  types tend to last long enough for an abduction-grade encounter.
- **Reading guide**: written tour of how the three views combine to answer
  the project question.

### Global controls

- **State** dropdown — filters all charts and zooms the map.
- **Shape** dropdown — hides reports outside that shape group.
- **↺ Clear all** — resets every filter at once.
- **PNG export** — hover any chart and click the camera icon top-right to
  download a 2× PNG.

---

## Shneiderman task coverage

| Task | Implementation |
| --- | --- |
| Overview | State map at low zoom, per-capita ranking, cluster distribution, time series |
| Zoom | Leaflet native zoom + dual-mode rendering (state circles → individual points) |
| Filter | State / Shape dropdowns, cluster bar click, per-capita bar click, decade controls |
| Details on demand | Hover tooltips with the full set of NUFORC fields |
| Relate (brushing / linking) | State filter propagates to time series, cluster bar, map; cluster click filters the map; per-capita click sets the state filter and tab-switches |
| History | ↺ Clear all button; cluster filter chip with quick-clear |
| Extract | PNG download from any Plotly chart |

---

## Data

- **Source**: NUFORC, via the [planetsig/ufo-reports](https://github.com/planetsig/ufo-reports) mirror.
- **Records used**: 63,832 U.S. sightings, 1950–2014, after filtering to
  records with valid geocoded latitude/longitude, non-zero reported
  duration, and within the mainland-U.S. bounding box.
- **Augmentations** done in `build_dashboard.py`:
  - Shape rollup: NUFORC's 26 raw shape labels grouped into six categories
    (Round, Elongated, Light, Triangular, Formation, Other / Unknown).
  - Description clustering: TF-IDF (1–2-grams, top 2,000 features) +
    KMeans (k=6), surfacing themes like "Triangular craft", "Hovering
    shaped objects", and "Classic UFO reports".
  - Population normalization: per-state sightings divided by 2010 U.S.
    Census population, expressed per 100,000 residents.
  - Jitter (~600 m random offset) on point latitudes / longitudes so
    co-located reports fan out at city-level zoom instead of stacking.

---

## Tooling

- **Leaflet 1.9.4** + **MarkerCluster** plugin for the interactive main map.
- **Plotly.js 2.35** for the decade choropleth, time series, per-capita
  ranking, and cluster bar chart.
- **Altair / Vega-Lite** for the Duration-by-shape box plot, which satisfies
  the course's Altair requirement.
- **pandas** + **scikit-learn** in `build_dashboard.py` for data prep and
  clustering.
- No server side; the page is one self-contained `index.html` that loads
  the JS libraries from a CDN at runtime.

---

## Regenerating `index.html`

```bash
pip install pandas altair plotly scikit-learn
python3 build_dashboard.py
```

The script reads
`csv-data/ufo-scrubbed-geocoded-time-standardized.csv` from the original
[planetsig/ufo-reports](https://github.com/planetsig/ufo-reports) repo,
aggregates, runs the clustering, and writes a new `index.html`.

---

## Repo layout

```
.
├── index.html             # The deployed dashboard
├── build_dashboard.py     # Source script that generated index.html
├── README.md              # This file
└── DEPLOY.md              # GitHub Pages deployment notes
```

(Raw NUFORC CSV is not committed; the demo embeds the processed data
directly into `index.html`.)

---

## Known limitations

- NUFORC durations are self-reported and famously imprecise. Differences
  across shape groups are suggestive, not definitive.
- The dataset has no explicit "debunked" field. KMeans clustering finds
  a cluster of "general night-sky reports" that absorbs most of the
  catch-all noise, but doesn't isolate misidentifications cleanly.
- All visible sightings on the main map use the same shape (circles).
  The previous shape-group color encoding was dropped after testing —
  shape-level breakdowns now live in the duration box plot only.

---

## Acknowledgements

Built on the NUFORC public dataset. Per-capita normalization inspired by
canonical "don't be fooled by raw counts" InfoVis lessons. Critique
partners and the HCDE 411 teaching team for design feedback.
