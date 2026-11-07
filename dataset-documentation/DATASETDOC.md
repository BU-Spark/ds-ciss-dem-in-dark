***Project Information***

* Project name: Democracy in the Dark, Mapping Electricity Access and Political Distribution.
* GitHub repository: this repository, dataset-documentation folder holds this file.
* Google Drive folder: [Google Drive](https://drive.google.com/drive/folders/1jy4i1hBGetpby_OCO55-GE1dhzaXAKxD)
* Project goal: test whether constituency level electricity access in Ghana tracks political competitiveness, incumbency, and election timing.
* Client: Yvonne Ilupeju, PhD student in political science, Boston University CISS.
* Client contact: ilupeju@bu.edu.
* Program: SPARK CDS Internship, Summer 2026.

***Dataset Information***

* Data sets used: VIIRS nighttime lights (annual and monthly), Ghana constituency boundaries, Ghana presidential election results 1996 to 2024, and administrative boundary references.
* Data dictionary: see the Data Dictionary table under Composition below.
* Keywords: Civic Tech, Voting, Energy Access, Sustainability, Geospatial, Remote Sensing.
  * Domain of application: geospatial analysis, remote sensing, panel data analysis.
  * Topic tags: Civic Tech, Voting, Sustainability.

*The following questions pertain to the datasets used in this project.*
*Motivation*

* Purpose: measure whether electricity access at the constituency level lines up with political outcomes in Ghana.
* Gap filled: no existing dataset links nighttime lights, constituency boundaries, and election results for Ghana in one panel.

*Composition*

* Instance types: constituency by election year rows (elections panel), one row per constituency (master table), constituency by year or month rows (nightlights panel).
* Formats: tabular CSV and XLSX, vector geospatial (shapefile, GeoPackage), raster geospatial (GeoTIFF, pulled through Earth Engine rather than stored locally).
* Instance counts: 1521 rows in the elections panel, 317 rows in the constituency master table.
* Full set, not a sample: covers every constituency that existed in any of the 8 elections from 1996 to 2024.
* Raw versus derived: elections panel has original scraped or manually entered columns plus derived flag and rate columns, see Data Dictionary.
* Missing info: registered voter counts have known quality issues in some years, one constituency (Guan, 2024) is missing an EC code.
* Data splits: none, this is a panel dataset, not a train or test split.
* Errors and flags: 4 rows flagged for vote share mismatch, 4 for margin mismatch, 18 for turnout outside a plausible range.
* External resources this dataset relies on:
  * VIIRS imagery from the Earth Observation Group, no guarantee the download pages stay at the same URL.
  * Constituency boundary shapefile from Harvard Dataverse, has a stable DOI, 10.7910/DVN/FZYCHU.
  * Administrative boundaries from HDX, sourced from Ghana Statistical Service.
  * EOG downloads require free registration, no fee. Harvard Dataverse and HDX are open access.
* Confidential data: none, all sources are public satellite imagery or public election results.
* Offensive content: none.
* Individual re-identification: not possible, all data is aggregated to constituency or polling station level, no personal records.

Dataset snapshot, elections_panel_clean.csv:

| Size of dataset | 1521 rows |
| :---- | :---- |
| Number of instances | 1521, one row per constituency per election year |
| Number of fields | 23 (17 original, 6 derived) |
| Labeled classes | winner column |
| Number of labels | 3, NDC, NPP, OTHER-TIE |

Dataset snapshot, constituency_master.csv:

| Size of dataset | 317 rows |
| :---- | :---- |
| Number of instances | 317, one row per constituency |
| Number of fields | 9 |
| Labeled classes | none, this is a lookup table |
| Number of labels | 0 |

**Data dictionary**

Key concepts:

| Term | Meaning |
|---|---|
| Panel | Same constituencies observed across multiple elections, 1996 to 2024, 8 elections. One row is one constituency in one election. |
| constituency_id | Stable unique key this project created, format GHCON_xxx. Use it to join every table. |
| ec_code | Official Election Commission code. Gets reused when a region splits, not stable, treat as a label only. |
| Why 317 constituencies | Single elections have 200 to 230 constituencies. 317 is the union of every distinct constituency across all 8 elections. |

Ghana region change:
  * 10 administrative regions through 2018.
  * 16 regions from 2019 onward.
  * The elections panel and master table carry both a 16 region and a 10 region column.
  * This keeps results comparable across the change.

Table 1, elections_panel_clean.csv, one row per constituency per election year:

| Field | Meaning | Values or units |
|---|---|---|
| region_name_16regions | Region name under the 16 region system, used from 2019 onward | text, example Ahafo |
| region_code_16regions | Region code under the 16 region system | integer, example 12 |
| region_name_10regions | Region name under the old 10 region system, used through 2018 | text, example Brong-Ahafo |
| region_code_10regions | Region code under the old 10 region system, some gaps were filled from the name | decimal, example 10 |
| district_name | District the constituency sits in, not the constituency itself | text |
| ec_code | Election Commission constituency code | text, example E3101, not stable across years |
| constituency_name | Constituency name | text |
| constituency_id | Stable join key for this constituency | text, example GHCON_059 |
| election_year | Year of the election | integer, 1996 to 2024, 8 elections |
| npp_votes | Votes for the New Patriotic Party | count |
| ndc_votes | Votes for the National Democratic Congress | count |
| other_votes | Votes for other parties or independents | count |
| total_valid_votes | Total valid votes cast | count, roughly npp plus ndc plus other |
| registered_voters | People registered to vote in the constituency | count, some years have data quality issues |
| npp_vote_share | Share of valid votes for NPP | fraction, 0 to 1 |
| ndc_vote_share | Share of valid votes for NDC | fraction, 0 to 1 |
| margin_of_victory | Absolute gap between NPP and NDC vote share | fraction, smaller means more competitive |
| ec_code_missing | Whether ec_code is missing for this row | true or false, 1 row true |
| share_flag | Recomputed vote share differs from the stored value by more than 0.005 | true or false, 4 rows true |
| margin_flag | Same mismatch check applied to margin of victory | true or false, 4 rows true |
| turnout | Total valid votes divided by registered voters | fraction |
| turnout_flag | Turnout over 100 percent or under 20 percent, signals bad registration data | true or false, 18 rows true |
| winner | Party that won the constituency | NDC, NPP, or OTHER-TIE |

Table 2, constituency_master.csv, one row per constituency, join anchor for boundaries and nightlights:

| Field | Meaning | Values or units |
|---|---|---|
| constituency_id | Stable unique key, the join anchor for boundaries, nightlights, and election data | text, example GHCON_001 |
| constituency_name | Constituency name from the most recent election it appears in | text |
| ec_code | EC code from the most recent election, label only | text |
| region_16 | Region name under the 16 region system | text |
| region_10 | Region name under the 10 region system | text |
| region_code_10 | Region code under the 10 region system | decimal |
| n_elections | Number of elections this constituency appears in | integer, 1 to 8, only 18 constituencies appear in all 8 |
| first_year | First election year this constituency appears in, shows when it was created | integer |
| last_year | Most recent election year this constituency appears in | integer |

Other source files:

| File | Level | Key column | Notes |
|---|---|---|---|
| Manual_Entry_PRESIDENTIAL_RESULTS.xlsx | Constituency, president, 1996, 2000, 2008, 2012 | none, matched by name only | region, constituency, party votes, valid, rejected, total votes, year |
| results2012_ps_pres.csv | Polling station, president, 2012 | pollingstationcode_2012 | party votes, registered, valid, rejected, latin-1 encoded |
| results2016_ps_pres.csv | Polling station, president, 2016 | Pscode | party votes, RegisteredVoters, TotalValid, Rejected |
| ps_..._20160229.shp | Polling station points | PSCODE | latitude, longitude, region, district, constituency, needs CRS conversion to EPSG:4326 |

Boundary geometry sources:

| Concept | Meaning |
|---|---|
| Shapefile | A group of files, .shp geometry, .dbf attributes, .prj coordinate system, .shx index, always used together |
| CRS | Coordinate reference system, all layers must share one before they can be overlaid |
| EPSG:4326 | Standard latitude and longitude system, the common CRS this project uses |
| Crosswalk | Lookup table mapping a boundary file's constituency labels to constituency_id |
| P-code | Official administrative code used by HDX and the UN, example GH01 |

* Source 1, Harvard Dataverse constituency boundaries.
  * African Parliamentary Constituencies Shapefiles, DOI 10.7910/DVN/FZYCHU, published 2021.
  * Ghana file is Ghana_Constituencies.shp, 275 polygons, CRS EPSG:3857.
  * Columns: Country, Cons_name, ISO, geometry.
  * Official names are missing, added by crosswalk to the EC reference (248 of 275 matched).
* Source 2, HDX cod-ab-gha, Ghana Subnational Administrative Boundaries.
  * From Ghana Statistical Service.
  * Administrative region and district boundaries, not electoral.
  * Used only to validate names and codes.
  * Columns: ADM0_EN, ADM0_PCODE, ADM1_EN, ADM1_PCODE, ADM2_EN, ADM2_PCODE, date, validOn, validTo, geometry.
* Boundary crosswalk outputs.
  * boundary_crosswalk.csv holds the full match to constituency_id.
  * boundary_crosswalk_REVIEW.csv holds unmatched rows with fuzzy match suggestions.

**File inventory**

| File | Holds | Script |
|---|---|---|
| data/nightlights/constituency_lights_raw.csv | Annual radiance per constituency per election year | written by scripts/nightlights/GEE/gee_nightlights_ghana.js or .py, read by build_panel.py and run_pipeline.py |
| data/nightlights/constituency_lights_monthly_raw.csv | Monthly radiance and cloud free coverage per constituency | written by scripts/nightlights/GEE/gee_nightlights_ghana_monthly.js, read by nl_quality.py and build_seasonal.py |
| data/nightlights/constituency_lights_panel.csv | Drift robust annual brightness panel | written by build_panel.py and run_pipeline.py, read by make_q1_maps.py |
| data/nightlights/constituency_lights_panel_with_geo.csv | Panel joined to constituency geometry | written by make_q1_maps.py, read by make_zone_seasonal_maps.py |
| data/nightlights/constituency_seasonal_long.csv | Monthly seasonal baseline, long format | written by build_seasonal.py, read by make_zone_seasonal_maps.py |
| data/nightlights/constituency_seasonal_metrics.csv | Per constituency seasonal strength score | written by build_seasonal.py |
| data/nightlights/run_manifest.json | Log of a run_pipeline.py run | written by run_pipeline.py |
| data/boundaries/constituencies_4326.gpkg | Constituency polygons in EPSG:4326 | read by make_q1_maps.py and make_zone_seasonal_maps.py |
| data/boundaries/constituency_dictionary.csv | Constituency lookup, carries both a 16 region and a 10 region column | read by make_q1_maps.py and make_zone_seasonal_maps.py |
| data/boundaries/EC_official_constituencies_reference.csv | Official EC constituency name reference | read by make_q1_maps.py |
| data/elections/panel_monthly_with_elections.csv | Election panel merged with monthly nightlights | read by election_did.py, two_line_series.py, per_election_gap.py |
| Ghana_Constituencies.shp | Raw constituency boundary polygons from Harvard Dataverse, 275 shapes |  |
| result/seasonal_zones/zone_stl_decomposition.png, zone_stl_components.csv, zone_map_and_shapes.png | Seasonal decomposition outputs by climate zone | written by make_zone_seasonal_maps.py |
| result/election_impact/per_election_gap.csv, per_election_gap.png | Competitive versus safe nightlights gap around each election | written by per_election_gap.py |

*Collection Process*

* Mechanisms used to collect the data:
  * Registered download from EOG.
  * Scripted pull with no download, through Google Earth Engine.
  * Web scraping for recent election results.
  * Manual entry for older election results.
  * Existing shapefile catalogs, Harvard Dataverse and HDX.
* Not a sample, every constituency in every available election is included.
* Election data spans 1996 to 2024, 8 election cycles. Nightlights cover 2012 to 2024, pulled in 2026.

**Data acquisition**

VIIRS nighttime lights, Earth Observation Group, annual product:
* What it is: yearly nighttime radiance tiles from NOAA VIIRS, maintained by the Earth Observation Group.
* Source: https://eogdata.mines.edu/nighttime_light/annual/
* Step: register a free account at https://eogdata.mines.edu/products/register/, then download the tile named 75N060W, the median_masked file, for the years 2013, 2016, and 2020.
* Lands at: data/nightlights/.

VIIRS nighttime lights, Earth Observation Group, monthly product:
* What it is: monthly nighttime radiance tiles, used for seasonal analysis.
* Source: https://eogdata.mines.edu/nighttime_light/monthly/
* Step: download the vcmslcfg version, tile 75N060W, for the months needed.
* Lands at: data/nightlights/.

VIIRS nighttime lights, Google Earth Engine, no download alternative:
* What it is: the same VIIRS imagery, pulled and aggregated inside Earth Engine instead of downloaded.
* Source: https://developers.google.com/earth-engine/datasets, image collections below.

```
NOAA/VIIRS/DNB/ANNUAL_V22
NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
```
Loads the annual and monthly VIIRS image collections used by this project.

* Step: clip the collection to Ghana, run reduceRegions over the constituency polygons, export the result as CSV to Google Drive.
* Lands at: data/nightlights/constituency_lights_raw.csv (annual), data/nightlights/constituency_lights_monthly_raw.csv (monthly), per SCRIPTS_GUIDE.md.

Nighttime lights, zonal statistics reference snippet:
```python
import geopandas as gpd, rasterstats, pandas as pd
g = gpd.read_file(OUT/'constituencies_4326.gpkg')
for yr in [2013, 2016, 2020]:
    stats = rasterstats.zonal_stats(g, f'Nightlights/viirs_{yr}.tif', stats=['mean','sum'])
    g[f'brightness_{yr}'] = [s['mean'] for s in stats]
g.drop(columns='geometry').to_csv(OUT/'constituency_lights_panel.csv', index=False)
```
Computes mean and sum radiance per constituency polygon and writes the brightness panel.

WorldPop population raster:
* What it is: gridded population counts, used to weight historical boundary harmonization.
* Source: https://hub.worldpop.org/geodata/listing?id=69
* Step: download the listing for Ghana.
* Lands at: used during boundary crosswalk building, not a standing project file.

citypopulation.de:
* What it is: reference site for old versus new constituency names.
* Source: https://www.citypopulation.de/en/ghana/admin/
* Step: browse to check a name change by hand.
* Lands at: used for manual name review, not downloaded as a file.

Ghana Electoral Commission:
* What it is: authoritative constituency names and official results.
* Source: https://ec.gov.gh/
* Step: browse to verify a name or result by hand.
* Lands at: used for manual name review, not downloaded as a file.

Harvard Dataverse constituency boundaries:
* What it is: African Parliamentary Constituencies Shapefiles, Ghana portion, 275 polygons.
* Source: DOI 10.7910/DVN/FZYCHU
* Step: download Ghana_Constituencies.shp from the Dataverse record.
* Lands at: data/boundaries/.

HDX administrative boundaries:
* What it is: Ghana Subnational Administrative Boundaries, COD-AB, from Ghana Statistical Service.
* Source: HDX, dataset cod-ab-gha.
* Step: download the region and district boundary layers.
* Lands at: data/boundaries/, used for validation only.

*Preprocessing/cleaning/labeling*

* Preprocessing done: constituency name matching across years, boundary to constituency_id crosswalk, vote share and turnout recomputation with mismatch flags.
* Transformations: joining scraped and manual election tables, filling gaps in 10 region codes from region names, converting boundary CRS to EPSG:4326.
* Raw data kept: yes, raw scraped and manual tables are kept alongside the cleaned panel.
* Cleaning code: see scripts/eda/EDA.ipynb and the nightlights and boundary build scripts listed in SCRIPTS_GUIDE.md.

*Uses*

* Used so far: constituency electricity access mapping, seasonal decomposition of the dry season drop, comparison of competitive versus safe constituencies around elections.
* Base questions this dataset answers:
  * What electricity access looks like across constituencies and regions.
  * How strong and even the dry season pattern is.
  * Whether access changes around elections.
  * Whether competitiveness relates to electricity access.
* Composition caveats for future use:
  * Registered voter and turnout quality issues in some years.
  * Boundaries before 2012 are harmonized, not exact.
  * Analysis on those years should account for the crosswalk approximation.
* Not for: identifying individual voters, this dataset is aggregated to constituency or polling station level only.

