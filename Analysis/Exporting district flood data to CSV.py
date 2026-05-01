# FLOOD DATA EXTRACTION — Google Earth Engine (Python API)

# The script does the following:

# 1. Loads the confirmed 2010 Pakistan flood event from the Global Flood Database,
#    a satellite-derived dataset of 913 flood events worldwide (2000–2018) based
#    on MODIS imagery at 250-meter resolution.

# 2. Loads Pakistan district boundaries from the FAO Global Administrative Unit
#    Layers (GAUL 2015) at the district level (Admin Level 2), which yields
#    119 districts.

# 3. For each district, computes two flood exposure measures:
#    - Flood fraction: the proportion of the district's land area that was
#      inundated at any point during the flood event (0 = no flooding, 1 = fully flooded)
#    - Mean duration: the average number of days each pixel within the district
#      was underwater during the event

# 4. Creates a binary treatment variable by splitting districts at the median
#    flood fraction (treated = 1 if above median, 0 if at or below median).
#    This is used for visualization purposes; the main regression uses the
#    continuous flood fraction variable.

# 5. Exports the district-level flood data to a CSV for merging with DHS data,
#    and separately exports the official GAUL district name list for validation
#    of the DHS district code crosswalk.

# Output: pakistan_district_flood_2010.csv
#         gaul_pakistan_districts.csv


import ee  # loading earth engine
import pandas as pd

ee.Initialize(
    project="ardent-bridge-312816"
)  # logging my Python session to our Earth Engine project

# ── 1. Load the confirmed flood event ─────────────────────────────
flood2010 = (
    ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
    .filter(ee.Filter.eq("id", 3696))
    .first()
)

flooded = flood2010.select(
    "flooded"
)  # a binary layer showing whether each pixel was flooded.
duration = flood2010.select(
    "duration"
)  # a layer showing how many days each pixel stayed flooded.

# ── 2. Load Pakistan district boundaries ──────────────────────────
#  loading a global administrative boundary dataset and filters it to only Pakistan.
# level 2 = district boundaries
districts = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
    ee.Filter.eq("ADM0_NAME", "Pakistan")
)
# printing how many district polygons we have in Pakistan.
print("Number of districts:", districts.size().getInfo())


# ── 3. Compute flood fraction + mean duration per district ─────────


# Defining a function to calculate flood stats for each district
def add_flood_stats(feature):
    # caclulating mean flooded fraction in that district
    # mean value of the flooded band (0/1) = mean is the fraction of pixels flooded in that district
    # e.g. 0.10 means about 10% of pixels in the district were flooded.

    stats_flooded = flooded.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=feature.geometry(), scale=250, maxPixels=1e9
    )
    # caclulating mean flood duration in that district
    # calculates the average number of flood days across all pixels in the district.
    stats_duration = duration.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=feature.geometry(), scale=250, maxPixels=1e9
    )
    # adds four new properties to each district
    return feature.set(
        {
            "flood_fraction": stats_flooded.get("flooded"),
            "mean_duration": stats_duration.get("duration"),
            "district_name": feature.get("ADM2_NAME"),
            "province_name": feature.get("ADM1_NAME"),
        }
    )


# Apply the function to all districts
districts_with_flood = districts.map(add_flood_stats)

# ── 4. Pull into pandas ────────────────────────────────────────────
# bringing the results from Earth Engine into a regular pandas table
features = districts_with_flood.getInfo()["features"]
rows = [f["properties"] for f in features]
df = pd.DataFrame(rows)[
    ["district_name", "province_name", "flood_fraction", "mean_duration"]
]

# ── 5. Create treatment variable (split at median) ─────────────────

# Removes districts with missing flood values
df = df.dropna(subset=["flood_fraction"])

# Computes the median flood fraction across districts.
median_val = df["flood_fraction"].median()

# Creates a binary treatment variable where:
# treated = 1 if flood fraction is above the median
# treated = 0 if it is at or below the median
df["treated"] = (df["flood_fraction"] > median_val).astype(int)

print(f"\nMedian flood fraction: {median_val:.4f}")
print(f"Treated districts: {df['treated'].sum()}")
print(f"Control districts: {(df['treated']==0).sum()}")
print(df.sort_values("flood_fraction", ascending=False).head(10))

df.to_csv(
    "/Users/amnarauf/Library/CloudStorage/OneDrive-DukeUniversity/Spring 2026/Solving Problems with Data/Final Project/Analysis/Amna/Exploring Flood Data/pakistan_district_flood_2010.csv",
    index=False,
)
print("\nSaved to pakistan_district_flood_2010.csv")

# Extract all GAUL district names for Pakistan
gaul_names = districts.aggregate_array("ADM2_NAME").getInfo()
gaul_provinces = districts.aggregate_array("ADM1_NAME").getInfo()

# Build a dataframe of official GAUL district names
gaul_df = (
    pd.DataFrame({"district_gaul": gaul_names, "province_gaul": gaul_provinces})
    .sort_values("province_gaul")
    .reset_index(drop=True)
)

print(f"Total GAUL districts in Pakistan: {len(gaul_df)}")
print(gaul_df)
gaul_df.to_csv(
    "/Users/amnarauf/Library/CloudStorage/OneDrive-DukeUniversity/Spring 2026/Solving Problems with Data/Final Project/Analysis/Amna/Exploring Flood Data/gaul_pakistan_districts.csv",
    index=False,
)
