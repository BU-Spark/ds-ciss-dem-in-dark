/**
 * Ghana VIIRS annual radiance: per-constituency aggregation (GEE Code Editor).
 * Inputs: VIIRS annual products, constituency polygons (uploaded asset)
 * Outputs: CSV to Google Drive (GEE_exports/constituency_lights_raw.csv)
 */

// ===== change only this line: your uploaded asset path =====
var ASSET_ID = 'projects/ciss-democracy-in-dark/assets/constituencies_4326_shp';
// ===========================================================

var cons = ee.FeatureCollection(ASSET_ID);
print('constituencies count (should be 275):', cons.size());

// Ghana elections are every 4 years in December. Annual composite = full-year
// average, i.e. the development level going into each election.
// NOTE: 2012 has NO comparable masked band (EOG excluded it; VIIRS started
// Apr 2012 with non-standard processing), so it is auto-skipped below.
// 2013 is used as the post-2012-election baseline instead.
var wantedYears = [2012, 2013, 2016, 2020, 2024];

// V22 is newest (2012-2024); merge V21 as fallback. Select the analysis band
// (median_masked = background noise removed) up front on the whole collection.
var col = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V22')
            .merge(ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21'))
            .select('median_masked');

// --- keep only years that actually have an image (client-side getInfo is OK
//     in the Code Editor; it blocks briefly). This prevents the null-select crash.
var years = wantedYears.filter(function (y) {
  var n = col.filter(ee.Filter.calendarRange(y, y, 'year')).size().getInfo();
  print('year ' + y + ' -> images available:', n);
  return n > 0;
});
print('years used:', years);
if (years.length === 0) {
  print('ERROR: no years available -- check dataset id / asset.');
}

// Stack one band per available year: b<year>. mosaic() collapses the (>=1)
// matching images to a single image, so first()-null can never happen.
var multi = ee.Image.cat(years.map(function (y) {
  return col.filter(ee.Filter.calendarRange(y, y, 'year'))
            .mosaic()
            .rename('b' + y);
}));

// mean = radiance density (development per unit area)
// sum  = total light output (used later to build each region's national share)
var reducer = ee.Reducer.mean().combine({
  reducer2: ee.Reducer.sum(),
  sharedInputs: true
});

// VIIRS native resolution ~463 m; sample at 500 m.
// reduceRegions computes ONLY inside the polygons -- this is the "cloud crop".
var stats = multi.reduceRegions({
  collection: cons,
  reducer: reducer,
  scale: 500
});

// Build tidy output. `years` is a client-side array, so this forEach adds the
// right server-side properties for exactly the years we kept.
var out = stats.map(function (f) {
  var props = {
    con_id:    f.get('constituen'),   // shapefile truncated 'constituency_id' -> 'constituen'
    cons_name: f.get('Cons_name')
  };
  years.forEach(function (y) {
    props['mean_' + y] = f.get('b' + y + '_mean');
    props['sum_' + y]  = f.get('b' + y + '_sum');
  });
  return ee.Feature(null, props);
});

// Column order for the CSV
var selectors = ['con_id', 'cons_name'];
years.forEach(function (y) { selectors.push('mean_' + y, 'sum_' + y); });

// Quick visual sanity check: latest available year
var lastYear = years[years.length - 1];
Map.centerObject(cons, 7);
Map.addLayer(
  multi.select('b' + lastYear).clip(cons.geometry()),
  {min: 0, max: 60, palette: ['000000', '990000', 'ffcc00', 'ffffff']},
  'nightlights ' + lastYear
);
Map.addLayer(cons, {color: '00ffff'}, 'constituencies', false);

// Export RAW panel -> Google Drive / GEE_exports / constituency_lights_raw.csv
Export.table.toDrive({
  collection: out,
  description: 'constituency_lights_raw',
  folder: 'GEE_exports',            // predictable location in My Drive
  fileFormat: 'CSV',
  selectors: selectors
});

print('Open the Tasks tab on the right -> RUN. CSV lands in My Drive/GEE_exports in ~1 min.');
