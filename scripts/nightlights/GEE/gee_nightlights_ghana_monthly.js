/**
 * Ghana VIIRS monthly radiance: per-constituency long table (GEE Code Editor).
 * Inputs: VIIRS monthly products, constituency polygons (uploaded asset)
 * Outputs: Long CSV to Google Drive (GEE_exports/constituency_lights_monthly_raw.csv)
 */

// ===== change only this line: your uploaded asset path (same as annual) =====
var ASSET_ID = 'projects/ciss-democracy-in-dark/assets/constituencies_4326_shp';
// ============================================================================

var cons = ee.FeatureCollection(ASSET_ID);
print('constituencies count (should be 275):', cons.size());

// Time window. Monthly nightlights DO NOT exist before Apr 2012 (VIIRS launch);
// there is no monthly product earlier, so 2012-04 is the hard floor for seasonal
// analysis. Two products are stitched to cover the whole VIIRS era:
//   - VCMSLCFG (stray-light corrected) : 2014-01 onward  (preferred)
//   - VCMCFG   (not stray-light corr.) : 2012-04..2013-12 (backfill only)
// Ghana sits near the equator where stray light is negligible, so the two are
// almost identical here; we still tag each image with `product` so the 2014
// product seam stays auditable downstream. 2012 has only Apr-Dec (9 months).
var SEAM  = '2014-01-01';   // VCMSLCFG starts here; VCMCFG fills the years before
var START = '2012-04-01';   // hard floor: first VIIRS monthly composite
var END   = '2025-01-01';   // exclusive upper bound

var slc = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
            .filterDate(SEAM, END)
            .map(function (i) { return i.set('product', 'VCMSLCFG'); });
var cfg = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterDate(START, SEAM)
            .map(function (i) { return i.set('product', 'VCMCFG'); });

// avg_rad = brightness; cf_cvg = number of cloud-free observations that month.
var col = slc.merge(cfg)
             .select(['avg_rad', 'cf_cvg'])
             .sort('system:time_start');
print('monthly images in window (2012-04 .. 2024-12):', col.size());

// VIIRS native resolution ~463 m; sample at 500 m. mean reducer, so each output
// feature carries a property named after each band ('avg_rad', 'cf_cvg').
var reducer = ee.Reducer.mean();

// For every monthly image: zonal-reduce over the polygons, then stamp year/month
// onto each region row. flatten() concatenates all months into one long table.
var out = col.map(function (img) {
  var d = ee.Date(img.get('system:time_start'));
  var stats = img.reduceRegions({
    collection: cons,
    reducer: reducer,
    scale: 500
  });
  return stats.map(function (f) {
    return ee.Feature(null, {
      con_id:    f.get('constituen'),   // shapefile truncated 'constituency_id'
      cons_name: f.get('Cons_name'),
      year:      d.get('year'),
      month:     d.get('month'),
      avg_rad:   f.get('avg_rad'),
      cf_cvg:    f.get('cf_cvg'),
      product:   img.get('product')     // 'VCMCFG' (2012-13) or 'VCMSLCFG' (2014+)
    });
  });
}).flatten();

// Fixed CSV column order.
var selectors = ['con_id', 'cons_name', 'year', 'month', 'avg_rad', 'cf_cvg', 'product'];

// Quick visual sanity check: the most recent month in the window.
var last = ee.Image(col.sort('system:time_start', false).first());
Map.centerObject(cons, 7);
Map.addLayer(
  last.select('avg_rad').clip(cons.geometry()),
  {min: 0, max: 60, palette: ['000000', '990000', 'ffcc00', 'ffffff']},
  'latest month avg_rad'
);
Map.addLayer(cons, {color: '00ffff'}, 'constituencies', false);

// Export LONG table -> My Drive / GEE_exports / constituency_lights_monthly_raw.csv
Export.table.toDrive({
  collection: out,
  description: 'constituency_lights_monthly_raw',
  folder: 'GEE_exports',
  fileFormat: 'CSV',
  selectors: selectors
});

print('Open the Tasks tab on the right -> RUN. CSV lands in My Drive/GEE_exports in ~1-2 min.');
