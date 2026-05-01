Data Preparation for 2010 Flood data in Pakistan:

Step 1 : Made a non-commercial account on Google Earth Engine and used the following code in the code editor to islote the flood extent in 2010 in Pakistan. 
Step 2: We tried exploring flood events between 2010 and 2011 and it gave us 4 events out of which our event (coded as 3696) extends from 27 July 2010 to November 15 2010 i.e. 111 days. 
Step 3: Used the following code to isolate and visualize the flood extent and duration for event 3696:

// Load the Global Flood Database (913 flood events worldwide, 2000–2018)
var gfd = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1');

// Filter to the 2010 Pakistan monsoon flood (DFO event id = 3696)
var flood2010 = gfd.filter(ee.Filter.eq('id', 3696)).first();

// Print event metadata to verify: country, severity, deaths, displaced persons
print('Event properties:');
print(flood2010.toDictionary());

// Center map on Pakistan (longitude 69.34, latitude 30.37, zoom level 5)
Map.setCenter(69.3451, 30.3753, 5);

// Layer 1: Flood EXTENT — binary map where 1 = flooded, 0 = not flooded
// .selfMask() hides non-flooded pixels; flooded pixels shown in blue
Map.addLayer(
  flood2010.select('flooded').selfMask(),
  {min: 1, max: 1, palette: ['blue']},
  'Flood Extent'
);

// Layer 2: Flood DURATION — number of days each pixel was underwater (max 111)
// Color scale: yellow (short) → orange → red (long duration)
// Will be used as a continuous treatment intensity measure in robustness checks
Map.addLayer(
  flood2010.select('duration').selfMask(),
  {min: 1, max: 111, palette: ['yellow', 'orange', 'red']},
  'Flood Duration (days)'
);

Key event metadata confirmed:

dfo_country: Pakistan
dfo_severity: 2 (extreme, >100-year recurrence interval)
dfo_dead: 1,750
dfo_displaced: 10,000,000
glide_index: FL-2010-000141-PAK

Please find the link to the code editor: https://code.earthengine.google.com/?scriptPath=Examples%3ADatasets%2FGLOBAL_FLOOD_DB%2FGLOBAL_FLOOD_DB_MODIS_EVENTS_V1