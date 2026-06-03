# How to Get Abducted by Aliens

An interactive data essay exploring U.S. UFO sightings from the National UFO
Reporting Center (NUFORC), framed as a tongue-in-cheek travel guide for the
abduction-curious.

**Course:** HCDE 411 (InfoVis), University of Washington
**Author:** Antares Yuan
**Live demo:** _<set after deploying to GitHub Pages>_

---

## What's here

A single-page, self-contained dashboard with **five visualizations**:

| # | View | Type | Shneiderman tasks |
|---|---|---|---|
| 1 | U.S. state choropleth | Interactive (click) | overview, filter, relate |
| 2 | Sightings-over-time line/area | Interactive (brush) | filter, relate, history |
| 3 | Shape-group bar chart | Interactive (click) | filter, details, relate |
| 4 | Duration-by-shape box plot | Static | overview |
| 5 | Washington focus point map | Static (tooltip) | details on demand |

Views 1, 2, and 3 are cross-linked: any selection on one filters the other
two. The map updates as you drag-select a year range; the time series
updates as you click a state; the shape bar updates with both. Try clicking
California, then drag-selecting 1995–2005, then clicking _Light_.

## Data

- **Source:** NUFORC, via [planetsig/ufo-reports](https://github.com/planetsig/ufo-reports)
- **Records used:** 63,832 U.S. sightings, 1950–2014, after filtering to
  records with valid geocoded latitude/longitude, non-zero reported
  duration, and within the mainland-U.S. bounding box.
- **Shape rollup:** NUFORC's 26 raw shape labels were manually grouped
  into six categories (Round, Elongated, Light, Triangular, Formation,
  Other / Unknown) to make cross-comparison legible.

## Tooling

- **Altair / Vega-Lite** for all five visualizations (satisfies the
  Altair requirement). Single-page HTML embeds the Vega-Lite specs and
  loads the runtime from a CDN.
- **pandas** for data cleaning and aggregation.
- No server-side code; the dashboard is fully client-rendered.

## How to regenerate

```bash
pip install altair pandas vl-convert-python
python3 build_final.py
```

This reads `csv-data/ufo-scrubbed-geocoded-time-standardized.csv`,
aggregates, builds the chart specs, and writes `index.html`.

## Repo layout

```
.
├── index.html              # The deployable demo
├── build_final.py          # Source script that generates index.html
├── README.md               # This file
└── csv-data/
    └── ufo-scrubbed-geocoded-time-standardized.csv   # NUFORC data
```

## Known limitations

- Sighting density mostly tracks population density. The map is therefore
  a guide to "where reports come from," not "where the aliens actually
  go." Per-capita rates would tell a different story.
- NUFORC durations are self-reported and famously imprecise. Differences
  across shape groups are suggestive, not definitive.
- The dataset has no "debunked" field; an early design included a
  debunked-toggle that was removed once the data was inspected.

## Acknowledgements

Built on the NUFORC public dataset. UI inspired by long-form data essays.
Critique partners and HCDE 411 teaching team for design feedback.
