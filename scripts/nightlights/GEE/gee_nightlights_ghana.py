"""
Ghana VIIRS annual radiance: per-constituency aggregation (GEE Python API).
Inputs: VIIRS annual products, constituency polygons (uploaded asset)
Outputs: CSV to Google Drive (GEE_exports/constituency_lights_raw.csv)
"""

import ee

# ===== change these two lines =====
PROJECT = "ciss-democracy-in-dark"
ASSET_ID = f"projects/{PROJECT}/assets/constituencies_4326_shp"
# ==================================

ee.Initialize(project=PROJECT)

cons = ee.FeatureCollection(ASSET_ID)
print("constituencies count (should be 275):", cons.size().getInfo())

# Election years (December vote). Annual composite = full-year average.
# NOTE: 2012 has no comparable masked band (EOG excluded it), so it is
# auto-skipped below; 2013 is the post-2012-election baseline instead.
wanted_years = [2012, 2013, 2016, 2020, 2024]

# V22 newest (2012-2024); merge V21 as fallback. Select analysis band up front.
col = (
    ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V22")
    .merge(ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21"))
    .select("median_masked")
)

# Keep only years that actually have an image (avoids null-select crash).
years = []
for y in wanted_years:
    n = col.filter(ee.Filter.calendarRange(y, y, "year")).size().getInfo()
    print(f"year {y} -> images available: {n}")
    if n > 0:
        years.append(y)
print("years used:", years)

# Stack one band per available year via mosaic() (never null).
multi = ee.Image.cat(
    [
        col.filter(ee.Filter.calendarRange(y, y, "year")).mosaic().rename(f"b{y}")
        for y in years
    ]
)

# mean = radiance density; sum = total output (used for national share later).
reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True)

# reduceRegions computes ONLY inside the polygons -- the "cloud crop".
stats = multi.reduceRegions(collection=cons, reducer=reducer, scale=500)


def tidy(f):
    props = {
        "con_id": f.get("constituen"),   # shapefile-truncated constituency_id
        "cons_name": f.get("Cons_name"),
    }
    for y in years:
        props[f"mean_{y}"] = f.get(f"b{y}_mean")
        props[f"sum_{y}"] = f.get(f"b{y}_sum")
    return ee.Feature(None, props)


out = stats.map(tidy)

selectors = ["con_id", "cons_name"]
for y in years:
    selectors += [f"mean_{y}", f"sum_{y}"]

task = ee.batch.Export.table.toDrive(
    collection=out,
    description="constituency_lights_raw",
    folder="GEE_exports",           # predictable location in My Drive
    fileFormat="CSV",
    selectors=selectors,
)
task.start()
print("Export task submitted, id:", task.id)
print("Check https://code.earthengine.google.com/tasks or Drive/GEE_exports for the CSV.")
