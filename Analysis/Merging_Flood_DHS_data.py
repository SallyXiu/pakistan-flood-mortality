# Converted from: Merging Flood_DHS_data.ipynb
# Pakistan 2010 Floods - DHS Mortality Analysis
# Unit of observation: Each row = one birth reported in DHS birth history module


# ======================================================================
# Unit of observation: Each row in the dataset represents one birth reported by a woman during the survey's birth history module
# ======================================================================


# ---------- Cell 1 ----------
import pandas as pd
import statsmodels.api as sm

df = pd.read_stata("PKBR61FL.DTA")

df.head()

# ---------- Cell 2 ----------
# 2012 BR file (already loaded as df)
print("2012 columns:")
print(df.columns.tolist())

# ---------- Cell 3 ----------
# Step 1: Define cohort window
# We focus on births between 2005–2012
# This allows us to capture child mortality (age 1–5)


df_sub = df[(df["b2"] >= 2005) & (df["b2"] <= 2012)].copy()

# Check distribution
df_sub["b2"].value_counts().sort_index()

# ---------- Cell 4 ----------
# Step 2: Construct time variables
# b3 = date of birth (CMC format) It is the number of months since January 1900.
# We approximate conception as 9 months before birth

df_sub["birth_year"] = df_sub["b2"]
df_sub["conception_cmc"] = df_sub["b3"] - 9

# ---------- Cell 5 ----------
# Step 3: Define flood timing (July- November 2010)
# Convert to CMC format

flood_start = (2010 - 1900) * 12 + 7
flood_end = flood_start + 4

# ---------- Cell 6 ----------
# Divide pregnancy into 3 trimesters
# Each trimester = 3 months


df_sub["t1_start"] = df_sub["conception_cmc"]
df_sub["t2_start"] = df_sub["conception_cmc"] + 3
df_sub["t3_start"] = df_sub["conception_cmc"] + 6

# ---------- Cell 7 ----------
# Exposure definition:
# A trimester is exposed if it overlaps with flood period

df_sub["t1_exposed"] = (
    (df_sub["t1_start"] <= flood_end) & (df_sub["t1_start"] + 2 >= flood_start)
).astype(int)

df_sub["t2_exposed"] = (
    (df_sub["t2_start"] <= flood_end) & (df_sub["t2_start"] + 2 >= flood_start)
).astype(int)

df_sub["t3_exposed"] = (
    (df_sub["t3_start"] <= flood_end) & (df_sub["t3_start"] + 2 >= flood_start)
).astype(int)

# ---------- Cell 8 ----------
# Step 4: Define mortality outcomes
# b5 = survival status
# b7 = age at death (in months)

df_sub["neonatal_death"] = ((df_sub["b5"] == "No") & (df_sub["b7"] == 0)).astype(int)

df_sub["infant_death"] = (
    (df_sub["b5"] == "No") & (df_sub["b7"] < 12) & (df_sub["b7"] > 0)
).astype(int)

df_sub["child_death"] = (
    (df_sub["b5"] == "No") & (df_sub["b7"] >= 12) & (df_sub["b7"] < 60)
).astype(int)

# ---------- Cell 9 ----------
# Step 5: Summary statistics by birth cohort

death_rates = df_sub.groupby("birth_year")[
    ["neonatal_death", "infant_death", "child_death"]
].mean()

death_counts = df_sub.groupby("birth_year")[
    ["neonatal_death", "infant_death", "child_death"]
].sum()

death_rates

# ---------- Cell 10 ----------
# because I add birth year as a control, I will center it at 2010 to improve interpretability
# I did this just for the regression, but you can also do it for the summary stats if you want
df_sub["birth_year_c"] = df_sub["b2"] - 2010

# ---------- Cell 11 ----------
# Step 6: Regression analysis

X = df_sub[["t1_exposed", "t2_exposed", "t3_exposed", "birth_year_c"]]
X = sm.add_constant(X)

y = df_sub["neonatal_death"]

model = sm.OLS(y, X).fit()

print(model.summary())

# ---------- Cell 12 ----------
# Enumerate all observed trimester exposure combinations (2^3 = 8 possible, minus (1,0,1) which is
# geometrically impossible: T1 and T3 cannot overlap the flood window without T2 also overlapping,
# since the 2010 flood spanned a full 12 months)

conditions = [
    (df_sub["t1_exposed"] == 0)
    & (df_sub["t2_exposed"] == 0)
    & (df_sub["t3_exposed"] == 0),
    (df_sub["t1_exposed"] == 1)
    & (df_sub["t2_exposed"] == 0)
    & (df_sub["t3_exposed"] == 0),
    (df_sub["t1_exposed"] == 0)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 0),
    (df_sub["t1_exposed"] == 0)
    & (df_sub["t2_exposed"] == 0)
    & (df_sub["t3_exposed"] == 1),
    (df_sub["t1_exposed"] == 1)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 0),
    (df_sub["t1_exposed"] == 0)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 1),
    (df_sub["t1_exposed"] == 1)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 1),
]

print(df_sub[["t1_exposed", "t2_exposed", "t3_exposed"]].value_counts())

# ---------- Cell 13 ----------
import numpy as np

# Create mutually exclusive exposure category variable
# Note: (0,1,0) combination does not exist in this dataset
conditions = [
    (df_sub["t1_exposed"] == 1)
    & (df_sub["t2_exposed"] == 0)
    & (df_sub["t3_exposed"] == 0),  # T1_only:   647 obs
    (df_sub["t1_exposed"] == 0)
    & (df_sub["t2_exposed"] == 0)
    & (df_sub["t3_exposed"] == 1),  # T3_only:   494 obs
    (df_sub["t1_exposed"] == 0)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 1),  # T2_T3:     563 obs
    (df_sub["t1_exposed"] == 1)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 0),  # T1_T2:     433 obs
    (df_sub["t1_exposed"] == 1)
    & (df_sub["t2_exposed"] == 1)
    & (df_sub["t3_exposed"] == 1),  # All_three: 1637 obs
]
choices = ["T1_only", "T3_only", "T2_T3", "T1_T2", "All_three"]
df_sub["exposure_group"] = np.select(conditions, choices, default="None")

# Verify: None should be exactly 15409
print(df_sub["exposure_group"].value_counts())

# ---------- Cell 14 ----------
# Generate dummy variables (using 'None' / unexposed as the reference group)
exposure_dummies = pd.get_dummies(df_sub["exposure_group"], drop_first=False)
exposure_dummies = exposure_dummies.drop(columns="None")  # drop reference group
exposure_dummies = exposure_dummies.astype(int)  # convert bool to int for statsmodels

# Regression: mutually exclusive exposure group dummies + centered birth year
X = pd.concat([exposure_dummies, df_sub[["birth_year_c"]]], axis=1)
X = sm.add_constant(X)

y = df_sub["neonatal_death"]

model = sm.OLS(y, X).fit()
print(model.summary())

# ---------- Cell 15 ----------
# Generate dummy variables (using 'None' / unexposed as the reference group)
exposure_dummies = pd.get_dummies(df_sub["exposure_group"], drop_first=False)
exposure_dummies = exposure_dummies.drop(columns="None")  # drop reference group
exposure_dummies = exposure_dummies.astype(int)  # convert bool to int for statsmodels

# Regression: mutually exclusive exposure group dummies + centered birth year
X = pd.concat([exposure_dummies, df_sub[["birth_year_c"]]], axis=1)
X = sm.add_constant(X)

y = df_sub["infant_death"]

model = sm.OLS(y, X).fit()
print(model.summary())

# ---------- Cell 16 ----------
# Generate dummy variables (using 'None' / unexposed as the reference group)
exposure_dummies = pd.get_dummies(df_sub["exposure_group"], drop_first=False)
exposure_dummies = exposure_dummies.drop(columns="None")  # drop reference group
exposure_dummies = exposure_dummies.astype(int)  # convert bool to int for statsmodels

# Regression: mutually exclusive exposure group dummies + centered birth year
X = pd.concat([exposure_dummies, df_sub[["birth_year_c"]]], axis=1)
X = sm.add_constant(X)

y = df_sub["child_death"]

model = sm.OLS(y, X).fit()
print(model.summary())

# ---------- Cell 17 ----------
# checking the unique values in districts
pd.set_option("display.max_rows", None)
df_sub["sdist"].unique()

# ---------- Cell 18 ----------
# mapping district codes to names
district_map = {
    # Punjab (1xx)
    101: "Attock",
    102: "Bahawalnagar",
    103: "Bahawalpur",
    104: "Bhakkar",
    105: "Chakwal",
    106: "Chiniot",
    107: "Dera Ghazi Khan",
    108: "Faisalabad",
    109: "Gujranwala",
    110: "Gujrat",
    111: "Hafizabad",
    112: "Jhang",
    113: "Jhelum",
    114: "Kasur",
    115: "Khanewal",
    116: "Khushab",
    117: "Lahore",
    118: "Layyah",
    119: "Lodhran",
    120: "Mandi Bahauddin",
    121: "Mianwali",
    122: "Multan",
    123: "Muzaffargarh",
    124: "Nankana Sahib",
    125: "Narowal",
    126: "Okara",
    127: "Pakpattan",
    128: "Rahim Yar Khan",
    129: "Rajanpur",
    130: "Rawalpindi",
    131: "Sahiwal",
    132: "Sargodha",
    133: "Sheikhupura",
    134: "Sialkot",
    135: "Toba Tek Singh",
    136: "Vehari",
    # Sindh (2xx)
    201: "Badin",
    202: "Dadu",
    203: "Ghotki",
    204: "Hyderabad",
    205: "Jacobabad",
    206: "Jamshoro",
    207: "Karachi Central",
    208: "Karachi East",
    209: "Karachi Malir",
    210: "Karachi South",
    211: "Karachi West",
    212: "Kashmore",
    213: "Khairpur",
    214: "Larkana",
    215: "Matiari",
    216: "Mirpur Khas",
    217: "Naushahro Feroze",
    218: "Sanghar",
    219: "Shahdad Kot",
    220: "Nawabshah/Shaheed Benazir Abad",
    221: "Shikarpur",
    222: "Sukkur",
    223: "Tando Alla Yar",
    224: "Tando Muhammad Khan",
    225: "Tharparkar",
    226: "Thatta",
    227: "Umer Kot",
    # KPK (3xx)
    301: "Abbottabad",
    302: "Bannu",
    303: "Batagram",
    304: "Buner",
    305: "Charsadda",
    306: "Chitral",
    307: "D. I. Khan",
    308: "Hangu",
    309: "Haripur",
    310: "Karak",
    311: "Kohat",
    313: "Lakki Marwat",
    314: "Lower Dir",
    315: "Malakand Protected Area",
    316: "Mansehra",
    317: "Mardan",
    318: "Nowshera",
    319: "Peshawar",
    320: "Shangla",
    321: "Swabi",
    322: "Swat",
    323: "Tank",
    324: "Tor Ghar",
    325: "Upper Dir",
    # Balochistan (4xx)
    402: "Barkhan",
    403: "Bolan/Kachhi",
    404: "Chagai",
    406: "Gawadar",
    407: "Harnai",
    408: "Jaffarabad",
    409: "Jhal Magsi",
    410: "Kalat",
    411: "Kech/Turbat",
    412: "Kharan",
    413: "Khuzdar",
    414: "Killa Abdullah",
    415: "Killa Saifullah",
    416: "Kohlu",
    417: "Lasbela",
    418: "Loralai",
    419: "Mastung",
    420: "Musakhel",
    421: "Nasirabad/Tamboo",
    422: "Nushki",
    424: "Pishin",
    425: "Quetta",
    426: "Sherani",
    427: "Sibi",
    429: "Zhob",
    430: "Ziarat",
    # Gilgit-Baltistan (5xx)
    501: "Astore",
    502: "Baltistan",
    503: "Diamir",
    504: "Ghanche",
    505: "Ghizer",
    506: "Gilgit",
    507: "Nagar",
    # Islamabad (6xx)
    601: "Islamabad",
}

df_sub["district_name"] = df_sub["sdist"].map(district_map)

# Sanity check
unmatched = df_sub[df_sub["district_name"].isna()]["sdist"].unique()
if len(unmatched) > 0:
    print(f"Warning: unmatched sdist codes: {unmatched}")
else:
    print("All codes matched successfully.")

df_sub[["sdist", "district_name"]].drop_duplicates().sort_values("sdist")

# ---------- Cell 19 ----------
# filter DHS to relevant variables
df_sub = df_sub[
    [
        # Geography
        "sdist",
        "district_name",
        # Timing & exposure
        "birth_year",
        "birth_year_c",
        "conception_cmc",
        "t1_start",
        "t2_start",
        "t3_start",
        "t1_exposed",
        "t2_exposed",
        "t3_exposed",
        "exposure_group",
        # Outcomes
        "neonatal_death",
        "infant_death",
        "child_death",
        # Controls (DHS variables)
        "v012",  # maternal age
        "v106",  # maternal education
        "v190",  # wealth index
        "v025",  # urban/rural
        "b4",  # child sex
        "bord",  # birth order
        "b3",  # birth date CMC (needed for post variable)
    ]
]

# ---------- Cell 20 ----------
# loading flood data

flood_data = pd.read_csv("pakistan_district_flood_2010.csv")
flood_data.head()

# ---------- Cell 21 ----------
# Load and map in one cell
flood_data = pd.read_csv("pakistan_district_flood_2010.csv")

flood_district_map = {
    # Punjab (1xx)
    "Attock District": 101,
    "Bahawalnagar District": 102,
    "Bahawalpur District": 103,
    "Bhakkar District": 104,
    "Chakwal District": 105,
    "Dera Ghazi Khan District": 107,
    "Faisalabad District": 108,
    "Gujranwala District": 109,
    "Gujrat District": 110,
    "Hafizabad District": 111,
    "Jhang District": 112,
    "Jhelum District": 113,
    "Kasur District": 114,
    "Khanewal District": 115,
    "Khushab District": 116,
    "Lahore District": 117,
    "Layyah District": 118,
    "Lodhran District": 119,
    "Mandi Bahauddin District": 120,
    "Mianwali District": 121,
    "Multan District": 122,
    "Muzaffargarh District": 123,
    "Narowal District": 125,
    "Okara District": 126,
    "Pakpattan District": 127,
    "Rahim Yar Khan District": 128,
    "Rajanpur District": 129,
    "Rawalpindi District": 130,
    "Sahiwal District": 131,
    "Sargodha District": 132,
    "Sheikhupura District": 133,
    "Sialkot District": 134,
    "Toba Tek Singh District": 135,
    "Vehari District": 136,
    # KPK (3xx)
    "Abbottabad District": 301,
    "Bannu District": 302,
    "Batagram District": 303,
    "Buner District": 304,
    "Charsadda District": 305,
    "Chitral District": 306,
    "D. I. Khan District": 307,
    "Hangu District": 308,
    "Haripur District": 309,
    "Karak District": 310,
    "Kohat District": 311,
    "Kohistan District": 312,
    "Lakki Marwat District": 313,
    "Lower Dir District": 314,
    "Malakand Protected Area": 315,
    "Mansehra District": 316,
    "Mardan District": 317,
    "Nowshera District": 318,
    "Peshawar District": 319,
    "Shangla District": 320,
    "Swabi District": 321,
    "Swat District": 322,
    "Tank District": 323,
    "Upper Dir District": 325,
    # Tribal Areas Adj
    "T.A.Adj.Lakki Marwat District": 313,
    "Tribal Area Adj Bannu District": 302,
    "Tribal Area Adj D.I.Khan Distt": 307,
    "Tribal Area Adj Kohat District": 311,
    "Tribal Area Adj Peshawar Distt": 319,
    "Tribal Area Adj Tank Distt": 323,
    # Balochistan (4xx)
    "Barkhan District": 402,
    "Killa Saifullah District": 415,
    "Loralai District": 418,
    "Musakhel District": 420,
    "Pishin District": 424,
    "Zhob District": 429,
    # Islamabad (6xx)
    "Islamabad District": 601,
    # FATA (7xx)
    "Bajaur Agency": 701,
    "Khyber Agency": 702,
    "Kurram Agency": 703,
    "Mohmand Agency": 704,
    "North Waziristan Agency": 705,
    "Orakzai Agency": 706,
    "South Waziristan Agency": 707,
}

flood_data["sdist"] = (
    flood_data["district_name"].str.strip().map(flood_district_map).astype("Int64")
)

flood_data

# ---------- Cell 22 ----------
# Merge the df_sub DHS data with the flood data on sdist
merged_data = pd.merge(df_sub, flood_data, on="sdist", how="inner")

print(f"DHS rows before merge  : {len(df_sub)}")
print(f"Flood districts before : {len(flood_data)}")
print(f"Rows after inner merge : {len(merged_data)}")
print(f"Districts matched      : {merged_data['sdist'].nunique()}")

merged_data.head()

# ---------- Cell 23 ----------
# post = 1 for births after August 2010 (when floods peaked)
# August 2010 in CMC format = (2010 - 1900) * 12 + 8 = 1328
# did = interaction term: treated district × post period (core DiD estimator)

merged_data["post"] = (merged_data["b3"] >= 1328).astype(int)
merged_data["did"] = merged_data["treated"] * merged_data["post"]

# Continuous version: flood_fraction × post
merged_data["did_continuous"] = merged_data["flood_fraction"] * merged_data["post"]

print(merged_data.groupby(["treated", "post"]).size())

# ---------- Cell 24 ----------
# Quarter-year FE controls for seasonal and time trends affecting all districts equally
# e.g., a nationwide health policy change in Q2 2011 would be absorbed by this FE

merged_data["birth_quarter"] = ((merged_data["b3"] - 1) % 12 // 3) + 1
merged_data["quarter_year"] = (
    merged_data["birth_year"].astype(str)
    + "_Q"
    + merged_data["birth_quarter"].astype(str)
)

print(merged_data["quarter_year"].value_counts().sort_index())

# ---------- Cell 25 ----------
# District FE (C(sdist)): absorbs time-invariant district traits
#   e.g., baseline health infrastructure, poverty levels
# Quarter-year FE (C(quarter_year)): absorbs shocks common to all districts
#   NOTE: post is dropped — perfectly collinear with quarter_year FEs
#   NOTE: flood_fraction is dropped — perfectly collinear with district FEs
# HC1: heteroskedasticity-robust standard errors (important for binary outcomes)
# did_continuous coefficient = core DiD estimate = causal effect of flood on outcome

import statsmodels.formula.api as smf

# Convert categorical variables
merged_data["sdist"] = merged_data["sdist"].astype("category")
merged_data["quarter_year"] = merged_data["quarter_year"].astype("category")
merged_data["v106"] = pd.Categorical(merged_data["v106"])
merged_data["v190"] = pd.Categorical(merged_data["v190"])
merged_data["v025"] = pd.Categorical(merged_data["v025"])
merged_data["b4"] = pd.Categorical(merged_data["b4"])
merged_data["bord"] = pd.Categorical(merged_data["bord"])

results_continuous = {}

for outcome in ["neonatal_death", "infant_death", "child_death"]:
    model = smf.ols(
        f"""{outcome} ~ did_continuous
            + C(sdist) + C(quarter_year)
            + v012 + C(v106) + C(v190) + C(v025) + C(b4) + C(bord)""",
        data=merged_data,
    ).fit(cov_type="HC1")
    results_continuous[outcome] = model
    print(f"Fitted: {outcome}")

# ---------- Cell 26 ----------
# Cell A: Neonatal only
summary_df = pd.DataFrame(results_continuous["neonatal_death"].summary2().tables[1])
key_vars = summary_df[
    ~summary_df.index.str.startswith("C(sdist)")
    & ~summary_df.index.str.startswith("C(quarter_year)")
]
print("Neonatal Death")
print(key_vars.round(4))

# ---------- Cell 27 ----------
# Cell B: Infant only
summary_df = pd.DataFrame(results_continuous["infant_death"].summary2().tables[1])
key_vars = summary_df[
    ~summary_df.index.str.startswith("C(sdist)")
    & ~summary_df.index.str.startswith("C(quarter_year)")
]
print("Infant Death")
print(key_vars.round(4))

# ---------- Cell 28 ----------
# Cell C: Child only
summary_df = pd.DataFrame(results_continuous["child_death"].summary2().tables[1])
key_vars = summary_df[
    ~summary_df.index.str.startswith("C(sdist)")
    & ~summary_df.index.str.startswith("C(quarter_year)")
]
print("Child Death")
print(key_vars.round(4))

# ---------- Cell 29 ----------
# Extract did and treated coefficients across all three outcomes

for outcome, model in results_continuous.items():
    print(f"\n{'='*60}")
    print(f"Outcome: {outcome}")
    print(f"{'='*60}")

    summary_df = pd.DataFrame(model.summary2().tables[1])
    key_vars = summary_df[
        ~summary_df.index.str.startswith("C(sdist)")
        & ~summary_df.index.str.startswith("C(quarter_year)")
    ]
    print(key_vars.round(4))

# ======================================================================
# Neonatal Mortality:
# ======================================================================


# ---------- Cell 30 ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def compute_annual_stats(df, outcome, group_col="treated", n_boot=500, ci=95):
    records = []
    alpha = (100 - ci) / 2
    for year in sorted(df["birth_year"].unique()):
        for grp in df[group_col].unique():
            vals = (
                df.loc[(df["birth_year"] == year) & (df[group_col] == grp), outcome]
                .dropna()
                .values
            )
            if len(vals) == 0:
                continue
            mean_val = vals.mean()
            boot_means = [
                np.random.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_boot)
            ]
            ci_low = np.percentile(boot_means, alpha)
            ci_high = np.percentile(boot_means, 100 - alpha)
            records.append(
                {
                    "year": year,
                    "group": grp,
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(records)


def smooth_series(years, values, frac=0.5):
    result = lowess(values, years, frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


np.random.seed(42)
color_map = {0: "#2196F3", 1: "#F44336"}
label_map = {0: "Control (Lightly Flooded)", 1: "Treated (Heavily Flooded)"}

fig, ax = plt.subplots(figsize=(8, 5))
stats_df = compute_annual_stats(merged_data, "neonatal_death")

for grp in [0, 1]:
    grp_df = stats_df[stats_df["group"] == grp].sort_values("year")
    years = grp_df["year"].values.astype(float)
    means = grp_df["mean"].values
    ci_low = grp_df["ci_low"].values
    ci_high = grp_df["ci_high"].values
    color = color_map[grp]

    # Raw annual means (semi-transparent scatter)
    ax.scatter(years, means, color=color, alpha=0.35, s=30, zorder=3)

    # LOWESS smoothed trend line
    sx, sy = smooth_series(years, means)
    ax.plot(sx, sy, color=color, linewidth=2.2, label=label_map[grp], zorder=4)

    # 95% bootstrap CI as shaded region
    _, sy_low = smooth_series(years, ci_low)
    _, sy_high = smooth_series(years, ci_high)
    ax.fill_between(sx, sy_low, sy_high, color=color, alpha=0.15, zorder=2)

ax.axvline(x=2010, color="black", linestyle="--", linewidth=1.4, label="Flood (2010)")
ax.set_title(
    "Neonatal Mortality (Death within 28 days)", fontsize=13, fontweight="bold"
)
ax.set_xlabel("Birth Year", fontsize=11)
ax.set_ylabel("Mean Mortality Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_xticks(sorted(merged_data["birth_year"].unique()))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("figure1_neonatal.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------- Cell 31 ----------
np.random.seed(42)

fig, ax = plt.subplots(figsize=(8, 5))
stats_df = compute_annual_stats(merged_data, "infant_death")

for grp in [0, 1]:
    grp_df = stats_df[stats_df["group"] == grp].sort_values("year")
    years = grp_df["year"].values.astype(float)
    means = grp_df["mean"].values
    ci_low = grp_df["ci_low"].values
    ci_high = grp_df["ci_high"].values
    color = color_map[grp]

    # Raw annual means (semi-transparent scatter)
    ax.scatter(years, means, color=color, alpha=0.35, s=30, zorder=3)

    # LOWESS smoothed trend line
    sx, sy = smooth_series(years, means)
    ax.plot(sx, sy, color=color, linewidth=2.2, label=label_map[grp], zorder=4)

    # 95% bootstrap CI as shaded region
    _, sy_low = smooth_series(years, ci_low)
    _, sy_high = smooth_series(years, ci_high)
    ax.fill_between(sx, sy_low, sy_high, color=color, alpha=0.15, zorder=2)

ax.axvline(x=2010, color="black", linestyle="--", linewidth=1.4, label="Flood (2010)")
ax.set_title("Infant Mortality (Death within Year 1)", fontsize=13, fontweight="bold")
ax.set_xlabel("Birth Year", fontsize=11)
ax.set_ylabel("Mean Mortality Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_xticks(sorted(merged_data["birth_year"].unique()))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("figure2_infant.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------- Cell 32 ----------
np.random.seed(42)

fig, ax = plt.subplots(figsize=(8, 5))
stats_df = compute_annual_stats(merged_data, "child_death")

for grp in [0, 1]:
    grp_df = stats_df[stats_df["group"] == grp].sort_values("year")
    years = grp_df["year"].values.astype(float)
    means = grp_df["mean"].values
    ci_low = grp_df["ci_low"].values
    ci_high = grp_df["ci_high"].values
    color = color_map[grp]

    # Raw annual means (semi-transparent scatter)
    ax.scatter(years, means, color=color, alpha=0.35, s=30, zorder=3)

    # LOWESS smoothed trend line
    sx, sy = smooth_series(years, means)
    ax.plot(sx, sy, color=color, linewidth=2.2, label=label_map[grp], zorder=4)

    # 95% bootstrap CI as shaded region
    _, sy_low = smooth_series(years, ci_low)
    _, sy_high = smooth_series(years, ci_high)
    ax.fill_between(sx, sy_low, sy_high, color=color, alpha=0.15, zorder=2)

ax.axvline(x=2010, color="black", linestyle="--", linewidth=1.4, label="Flood (2010)")
ax.set_title("Child Mortality (Death Ages 1–5)", fontsize=13, fontweight="bold")
ax.set_xlabel("Birth Year", fontsize=11)
ax.set_ylabel("Mean Mortality Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_xticks(sorted(merged_data["birth_year"].unique()))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("figure3_child.png", dpi=150, bbox_inches="tight")
plt.show()
