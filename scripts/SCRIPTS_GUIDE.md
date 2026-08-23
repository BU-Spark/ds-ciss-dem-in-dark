# Scripts Guide

## 1. Structure

```
scripts/
  SCRIPTS_GUIDE.md
  eda/
    EDA.ipynb
  nightlights/
    GEE/
      gee_nightlights_ghana.js
      gee_nightlights_ghana_monthly.js
      gee_nightlights_ghana.py
    build_panel.py
    build_seasonal.py
    make_q1_maps.py
    nl_quality.py
    run_pipeline.py
    nightlights_qc_and_seasonal.ipynb
  seasonal_pattern/
    seasonal_zones.py
    make_zone_seasonal_maps.py
  election_impact/
    election_did.py
    two_line_series.py
    per_election_gap.py
```

## 2. Folder roles

- `nightlights/`: Pulls VIIRS nighttime light data, builds the analytic panel, checks data quality, and draws the main maps.
- `seasonal_pattern/`: Groups constituencies into climate zones and measures their seasonal light patterns.
- `election_impact/`: Tests whether competitive elections line up with nightlight changes around each vote.

## 3. Scripts

### nightlights/

Run order:
1. `GEE/gee_nightlights_ghana.js` or `GEE/gee_nightlights_ghana.py` (annual pull)
2. `GEE/gee_nightlights_ghana_monthly.js` (monthly pull, needed for seasonal work)
3. `build_panel.py`
4. `make_q1_maps.py`
5. `build_seasonal.py`

`run_pipeline.py` can replace steps 3 and 4 with one command.

**GEE/gee_nightlights_ghana.js** (Runner, executed in the Earth Engine Code Editor)
- Role: Pulls annual radiance per constituency for election years from VIIRS.
- Inputs: constituency polygon asset uploaded to Earth Engine.
- Outputs: `constituency_lights_raw.csv` exported to Google Drive, then saved to `data/nightlights/constituency_lights_raw.csv`.

**GEE/gee_nightlights_ghana_monthly.js** (Runner, executed in the Earth Engine Code Editor)
- Role: Pulls monthly radiance and cloud free coverage per constituency.
- Inputs: constituency polygon asset uploaded to Earth Engine.
- Outputs: `constituency_lights_monthly_raw.csv` exported to Google Drive, then saved to `data/nightlights/constituency_lights_monthly_raw.csv`.

**GEE/gee_nightlights_ghana.py** (Runner)
- Role: Python version of the annual pull, submits the same export job.
- Inputs: constituency polygon asset uploaded to Earth Engine.
- Outputs: same `constituency_lights_raw.csv` export to Google Drive as the JS version.

**build_panel.py** (Runner)
- Role: Turns the raw annual export into the drift robust analytic panel.
- Inputs: `data/nightlights/constituency_lights_raw.csv`
- Outputs: `data/nightlights/constituency_lights_panel.csv`

**make_q1_maps.py** (Runner)
- Role: Draws the Q1 choropleth maps and rank charts from the panel.
- Inputs: `data/nightlights/constituency_lights_panel.csv`, `data/boundaries/constituencies_4326.gpkg`, `data/boundaries/constituency_dictionary.csv`, `data/boundaries/EC_official_constituencies_reference.csv`
- Outputs: several PNG maps and charts in `images/nightlights/`, plus `data/nightlights/constituency_lights_panel_with_geo.csv`

**nl_quality.py** (Library)
- Role: Shared quality gate for monthly data, plus missing rate checks.
- Inputs: called with a path to `data/nightlights/constituency_lights_monthly_raw.csv`
- Outputs: none, returns tables in memory only.

**build_seasonal.py** (Runner)
- Role: Builds the monthly seasonal baseline and per constituency seasonal strength.
- Inputs: `data/nightlights/constituency_lights_monthly_raw.csv`, using the gate from `nl_quality.py`
- Outputs: `data/nightlights/constituency_seasonal_long.csv`, `data/nightlights/constituency_seasonal_metrics.csv`

**run_pipeline.py** (Runner)
- Role: Runs `build_panel.py` then `make_q1_maps.py` in one command, with a run manifest.
- Inputs: `data/nightlights/constituency_lights_raw.csv`
- Outputs: `data/nightlights/constituency_lights_panel.csv`, PNGs in `images/nightlights/`, `data/nightlights/run_manifest.json`

### seasonal_pattern/

Run order:
1. `make_zone_seasonal_maps.py`

**seasonal_zones.py** (Library)
- Role: Assigns climate zones and runs the seasonal decomposition math per zone.
- Inputs: called with data already loaded by the caller script.
- Outputs: none, returns tables in memory only.

**make_zone_seasonal_maps.py** (Runner)
- Role: Groups constituencies into 3 zones and draws their seasonal decomposition.
- Inputs: `data/nightlights/constituency_seasonal_long.csv`, `data/boundaries/constituency_dictionary.csv`, `data/nightlights/constituency_lights_panel_with_geo.csv`, `data/boundaries/constituencies_4326.gpkg`
- Outputs: `result/seasonal_zones/zone_stl_decomposition.png`, `result/seasonal_zones/zone_stl_components.csv`, `result/seasonal_zones/zone_map_and_shapes.png`

### election_impact/

Run order:
1. `per_election_gap.py`

**election_did.py** (Library)
- Role: Shared panel loading and event time helpers for the election scripts.
- Inputs: called with a path to `data/elections/panel_monthly_with_elections.csv`
- Outputs: none, returns tables in memory only.

**two_line_series.py** (Library)
- Role: Region matched competitive versus safe constituency grouping helpers.
- Inputs: called with a path to `data/elections/panel_monthly_with_elections.csv`
- Outputs: none, returns tables in memory only.

**per_election_gap.py** (Runner)
- Role: Measures the competitive versus safe nightlights gap around each election.
- Inputs: `data/elections/panel_monthly_with_elections.csv`
- Outputs: `result/election_impact/per_election_gap.csv`, `result/election_impact/per_election_gap.png`
