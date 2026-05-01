# pakistan-flood-mortality

This repository contains the data pipeline and analysis code for a difference-in-differences study estimating the causal effect of the 2010 Pakistan monsoon floods on neonatal mortality (death within 28 days), infant mortality (death within the first year), and child mortality (death between ages one and five). Flood exposure is measured as a continuous district-level inundation fraction derived from the Global Flood Database (DFO event 3696, July 27 – November 15, 2010) using Google Earth Engine. Birth outcome data come from the Pakistan Demographic and Health Survey 2012-13 (DHS). The identification strategy compares mortality trajectories across districts with varying levels of flood inundation before and after August 2010, controlling for district fixed effects and quarter-year fixed effects. The analysis covers 64 districts across Punjab, Khyber Pakhtunkhwa, and Balochistan.

## Repository Structure

### Data
- **PKBR61FL.DTA.zip** — Pakistan DHS 2012-13 Birth Records microdata in Stata format, containing birth histories including birth dates, survival status, age at death, and maternal and household characteristics. Access requires registration at dhsprogram.com.
- **PKBR61FL.DO.zip** — Accompanying Stata do-file.
- **Pakistan_DHS_2012-13_District_Codes_with_Sample_Points_Final.xls** — District-level crosswalk mapping numeric DHS district codes to district names, sourced from the DHS user forum as no official documentation is publicly available.
- **pakistan_district_flood_2010.csv** — District-level flood exposure measures extracted from the Global Flood Database for the 2010 Pakistan monsoon flood event.

### Analysis
- **Data preparation for flood extent.dart** — Documents the process of identifying and verifying the 2010 Pakistan monsoon flood event (DFO-3696) in Google Earth Engine, including JavaScript code to visualize flood extent and duration.
- **Exporting district flood data to CSV.py** — Extracts district-level flood exposure measures (flood fraction and mean duration) using the Google Earth Engine Python API and GAUL district boundaries, and exports results to CSV for merging with DHS data.
- **Merging_Flood_DHS_data.ipynb** — Main analysis notebook: merges flood and DHS data, constructs mortality outcomes and treatment variables, runs DiD regressions, and generates all figures.

### Result
- **figure1_neonatal.png** — LOWESS-smoothed neonatal mortality trends for heavily and lightly flooded districts, 2005–2012, with 95% bootstrap confidence intervals.
- **figure2_infant.png** — LOWESS-smoothed infant mortality trends for heavily and lightly flooded districts, 2005–2012, with 95% bootstrap confidence intervals.
- **figure3_child.png** — LOWESS-smoothed child mortality trends for heavily and lightly flooded districts, 2005–2012, with 95% bootstrap confidence intervals.
