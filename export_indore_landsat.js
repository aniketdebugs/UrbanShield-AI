// Starter Landsat 8 Level-2 export for Indore.
// Run in https://code.earthengine.google.com/ after enabling Earth Engine.
// Inspect the exported rows and quality before using them for research.

var center = ee.Geometry.Point([75.8577, 22.7196]);
var region = center.buffer(20000).bounds();
var gridProjection = ee.Projection('EPSG:3857').atScale(1000);
var grid = region.coveringGrid(gridProjection).map(function(cell) {
  var c = cell.geometry().centroid(1).coordinates();
  return cell.set({
    cell_id: ee.String('cell_')
      .cat(ee.Number(c.get(0)).format('%.4f')).cat('_')
      .cat(ee.Number(c.get(1)).format('%.4f')),
    longitude: c.get(0), latitude: c.get(1)
  });
});
Map.centerObject(region, 10);
Map.addLayer(region, {color:'cyan'}, 'Indore study region');
Map.addLayer(grid.style({color:'white', fillColor:'00000000'}), {}, '1 km grid');

function prepare(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)
    .and(qa.bitwiseAnd(1 << 2).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(image.select('QA_RADSAT').eq(0));

  var optical = image.select(['SR_B4','SR_B5','SR_B6'])
    .multiply(0.0000275).add(-0.2);
  var red = optical.select('SR_B4');
  var nir = optical.select('SR_B5');
  var swir = optical.select('SR_B6');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi');
  var ndbi = swir.subtract(nir).divide(swir.add(nir)).rename('ndbi');
  var lst = image.select('ST_B10').multiply(0.00341802).add(149)
    .subtract(273.15).rename('lst_c');

  return ee.Image.cat([lst, ndvi, ndbi]).updateMask(mask)
    .copyProperties(image, ['system:time_start','LANDSAT_PRODUCT_ID']);
}

var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(region)
  .filterDate('2020-01-01','2026-01-01')
  .filter(ee.Filter.eq('PROCESSING_LEVEL','L2SP'))
  .map(prepare);

print('Number of scenes:', col.size());
var list = col.toList(col.size());
var count = col.size().getInfo();
var rows = ee.FeatureCollection([]);
for (var i=0; i<count; i++) {
  var image = ee.Image(list.get(i));
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
  var product = image.get('LANDSAT_PRODUCT_ID');
  var reduced = image.reduceRegions({
    collection: grid, reducer: ee.Reducer.mean(),
    scale: 30, crs: 'EPSG:3857', tileScale: 4
  }).map(function(f) {
    return ee.Feature(null, {
      cell_id:f.get('cell_id'), latitude:f.get('latitude'),
      longitude:f.get('longitude'), date:date, product_id:product,
      lst_c:f.get('lst_c'), ndvi:f.get('ndvi'), ndbi:f.get('ndbi')
    });
  });
  rows = rows.merge(reduced);
}
Export.table.toDrive({
  collection: rows, description:'UrbanShield_Indore_Landsat',
  fileNamePrefix:'indore_landsat_grid_2020_2025',
  fileFormat:'CSV',
  selectors:['cell_id','latitude','longitude','date','product_id','lst_c','ndvi','ndbi']
});
