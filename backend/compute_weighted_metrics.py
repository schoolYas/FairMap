#------Import Statements----------
import geopandas as gpd                             # Used for Reading Shapefiles, Aggregating Precincts, Reock Score, 
import pandas as pd                                 # Used for 
import numpy as np                                  # Used for
from sklearn.preprocessing import MinMaxScaler      # Used for 
from geopandas import GeoDataFrame                  # Used for 
import statsmodels.api as sm                        # Used for regression-adjusted EG

# ----- Read File ------------------
# Loads the shapefile using geopandas 
gdf = gpd.read_file("../data/NC-shapefiles-master/NC_VTD/NC_VTD.shp")

# ----- Aggregate Precincts ---------
# Takes a list of all numerical things recorded such as shape area, shape length, etc EXCEPT for the shape itself ('geometry').
agg_cols = [col for col in gdf.columns if col != 'geometry' and np.issubdtype(gdf[col].dtype, np.number)]

"""
 Groups all precincts that belong in the same district together using geopandas 
 Example: District 1 has 5 precincts. Dissolve will combine them into one row for District 1.
"""
districts = gdf.dissolve(by='CD', aggfunc='sum')[agg_cols]
geometry = gdf.dissolve(by='CD').geometry
districts['geometry'] = geometry

"""
After dissolving, we just select the geometry column, which now contains one shape per district
We assign these merged shapes to the districts GeoDataFrame aka gdf now has the full shape of its district
"""
districts = GeoDataFrame(districts, geometry=districts['geometry'], crs=gdf.crs)

# ------- Geometry Metrics ---------
"""
1. Polsby-Popper Score: 
Parameters: Shape_Area and Shape_Leng taken from district GeoDataFrame, pi from NumPy Library

Checks how 'circle' like or compact a district is. Intuition is a district should be somewhat round and 
not irregularly shaped. Therefore Polsby-Popper checks the area*4pi over the perimeter^2 ((districts['Shape_Leng']**2)). 
A perfect circle is equal to 1. Therefore, lower the the score, the less compact and therefore the higher the likelihood of 
gerrymandering.
"""
districts['PP'] = 4 * np.pi * districts['Shape_Area'] / (districts['Shape_Leng']**2)
# Taking inverse to fit the composite score scale
districts['PP'] = 1 - districts['PP']

"""
2. Reock Score: 
Parameters: rect from each geom (each column aka district), radius, circle_area, 

Checks how 'circle' like or compact a district is by comparing the area to that of the circle closest to fully containing it.
A perfect circle is equal to 1. Therefore, lower the score, the less compact and therefore the higher the likelihood of 
gerrymandering.
"""
def reock_score(geom):
    try:
        rect = geom.minimum_rotated_rectangle       # Finds the minimum rotatable rectangle for the area using GeoPandas
        radius = rect.length / (2 * np.pi)          # Pretends that area is a circle and uses circle radius equation 
        circle_area = np.pi * radius**2             # Uses radius to create area of ideal circle 
        return geom.area / circle_area              # Takes the districts actual area and compares to that of the ideal circle
    except:
        return np.nan                               # Runs if geometry is broken 

districts['Reock'] = districts['geometry'].apply(reock_score)
# Taking inverse to fit the composite score scale
districts['Reock'] = 1 - districts['Reock']

"""
3. Perimeter Area Ratio: 
Parameters: Shape_Area and Shape_Leng taken from district GeoDataFrame

Checks how stretched out a district is. If a district is round or square, the perimeter is relatively short for its area.
If its long, skinny, or jagged, it will have a high ratio. Therefore, higher the score, the less compact and therefore the higher the likelihood of 
gerrymandering.
"""
districts['PA_ratio'] = districts['Shape_Leng'] / districts['Shape_Area']

"""
4. Convex Hull Ratio: 
Parameters: Shape_Area from district GeoDataFrame, convex_hull.area from GeoPandas

Checks how "normal" looking the shape of the district is. A convex hull is any convex shape (square, rectangle, circle) where
any two lines drawn from point to point in the shape will stay inside the shape. An example of a non-convex shape is a crescent moon
or star. The hull is the smallest convex shape that fully encloses the district. Calculation taken from GeoPandas directly.
A perfect circle is equal to 1. Therefore, lower the score, the less compact and therefore the higher the likelihood of 
gerrymandering.
"""
districts['ConvexHull_ratio'] = districts['Shape_Area'] / districts['geometry'].convex_hull.area
# Taking inverse to fit the composite score scale
districts['ConvexHull_ratio'] = 1 - districts['ConvexHull_ratio']

"""
5. Shwartzberg Score: 
Parameters: Shape_Area and Shape_Leng from district GeoDataFrame, pi from NumPy 

Compares the perimeter of shape to that of a perfect circle with the same area. The perimeter of the district is already 
given from our GeoDataFame information. Therefore, we must use the equation for sqrt(4piA) to derive the perfect circles perimeter.
A perfect circle is equal to 1. Therefore, lower the score, the less compact and therefore the higher the likelihood of 
gerrymandering. 
"""
districts['Schwartzberg'] = np.sqrt(4 * np.pi * districts['Shape_Area']) / districts['Shape_Leng']
# Taking inverse to fit the composite score scale
districts['Schwartzberg'] = 1 - districts['Schwartzberg']

# ---- Partisan Metrics ---------
"""
1. Semi-Adjusted Efficiency Gap:

Parameters: dem_votes, rep_votes from GeoDataFrame

In a fair map, both major parties would have roughly equal efficiency in how their votes are translated into seats
—so their wasted votes would be similar, and EG would be near 0. In a gerrymandered map, one party uses tactics like 
“packing” (putting lots of the opponent’s voters into few districts so they win by huge margins = many excess votes wasted)
and “cracking” (splitting opponent’s voters across many districts so they lose by small margins = many losing votes wasted).
This suggests that if a party has fewer wasted votes it is being advantaged; if loosing votes, being disadvantaged.
The lower the score the more partisan the district.

***CAVEATS*** 
Sometimes one party’s voters naturally live close together (like Democrats in cities, Republicans in rural areas). 
This can create “wasted” votes even without intentional gerrymandering, since city districts will have huge wins and
rural ones will have narrow losses. If some districts have higher voter turnout than others, the number of “wasted” 
votes can look uneven just because more people voted, not because of unfair district design. The efficiency gap formula 
assumes there are only two major parties (D and R). In real life, votes for third parties or independents don’t fit neatly
into the “wasted” framework, which can distort the results.

Adjustments: Districts with higher turnout contribute more wasted votes even if maps are fair so district turnout
is normalized to scale and EG reflects partisan bias independent of turnout differences.
"""
def wasted_votes(row, adjust_turnout=True, per_vap=True):
    dem_votes = row['EL16G_PR_D']                   # Number of Democratic Votes 
    rep_votes = row['EL16G_PR_R']                   # Number of Republican Votes
    total = dem_votes + rep_votes                   # Total Numbers of Votes in the Districts
    dem_wasted = dem_votes - (total // 2 + 1) if dem_votes > rep_votes else dem_votes   # Amount of Wasted Democrat Votes above 50% + 1
    rep_wasted = rep_votes - (total // 2 + 1) if rep_votes > dem_votes else rep_votes   # Amount of Wasted Republican Votes above 50% + 1
    
    if adjust_turnout:
        avg_total = districts['EL16G_PR_D'].sum() + districts['EL16G_PR_R'].sum()
        avg_total /= len(districts)
        scale = avg_total / total if total > 0 else 1
        dem_wasted *= scale
        rep_wasted *= scale
        total = avg_total

    # Per-VAP adjustment: divide by total voting-age population
    if per_vap:
        vap = row['VAP'] if row['VAP'] > 0 else 1
        dem_wasted /= vap
        rep_wasted /= vap
        total /= vap

    return dem_wasted, rep_wasted, total

# For each district, converts the tuple into a series that can be used by GeoPandas
districts[['dem_wasted','rep_wasted','total_votes']] = districts.apply(lambda row: pd.Series(wasted_votes(row)), axis=1)
# Finds the difference in wasted votes between the two parties and normalizes it over total votes - (using magnitude only)
districts['EG'] = abs((districts['dem_wasted'] - districts['rep_wasted']) / districts['total_votes'])

# ------- Regression-Adjusted Efficiency Gap ---------
"""
Using demographic fractions to predict expected democrat vote share, then compute EG on residuals. 
"""
# Create demographic fractions
districts['Black_frac'] = districts['BVAP'] / districts['VAP']
districts['Hispanic_frac'] = districts['HVAP'] / districts['VAP']
districts['White_frac'] = districts['WVAP'] / districts['VAP']
districts['Native_frac'] = districts['AMINVAP'] / districts['VAP']

# Define predictors and response
X = districts[['Black_frac','Hispanic_frac','White_frac','Native_frac']]
X = sm.add_constant(X)  # add intercept
y = districts['EL16G_PR_D'] / (districts['EL16G_PR_D'] + districts['EL16G_PR_R'])  # Dem vote share

# Fit OLS regression
model = sm.OLS(y, X).fit()
districts['expected_dem_share'] = model.predict(X)
districts['residual_dem_share'] = y - districts['expected_dem_share']

# Compute regression-adjusted wasted votes
def wasted_votes_regression(row):
    dem_share = row['residual_dem_share'] + 0.5  # shift residual back to ~vote fraction
    rep_share = 1 - dem_share
    total_votes = row['VAP'] if row['VAP'] > 0 else 1
    
    dem_votes = dem_share * total_votes
    rep_votes = rep_share * total_votes
    
    dem_wasted = dem_votes - (total_votes / 2) if dem_votes > rep_votes else dem_votes
    rep_wasted = rep_votes - (total_votes / 2) if rep_votes > dem_votes else rep_votes
    
    return dem_wasted, rep_wasted, total_votes

districts[['dem_wasted_reg','rep_wasted_reg','total_votes_reg']] = districts.apply(lambda row: pd.Series(wasted_votes_regression(row)), axis=1)
districts['EG_reg'] = (districts['dem_wasted_reg'] - districts['rep_wasted_reg']) / districts['total_votes_reg']

# -----------------------------
# 2. Mean-Median Difference:
"""
Parameters: dem_votes, rep_votes

Checks for asymmetry in how votes translate into seats. Mean and Median of the vote share should be relatively 
the same in a partisan symmetrical district. The lower the score the more partisan the district.
"""

dem_share = districts['EL16G_PR_D'] / (districts['EL16G_PR_D'] + districts['EL16G_PR_R'])
districts['MM'] = dem_share.mean() - dem_share.median()
districts['Partisan_bias'] = dem_share - 0.5  # deviation from 50/50
districts['Competitive_flag'] = ((dem_share > 0.45) & (dem_share < 0.55)).astype(float)

# -----------------------------
# Step 5: Competitiveness
def competitiveness(row):
    dem = row['EL16G_PR_D']
    rep = row['EL16G_PR_R']
    total = dem + rep
    if total == 0:
        return 0.01
    margin = abs(dem - rep) / total
    score = 1 - margin
    return np.clip(score, 0.01, 0.99)

districts['Competitiveness'] = districts.apply(competitiveness, axis=1)

# Add std-based competitiveness metric
districts['Competitiveness_std'] = 1 - abs(dem_share - dem_share.mean())
districts['Competitiveness_std'] = np.clip(districts['Competitiveness_std'], 0.01, 0.99)

# -----------------------------
# Step 6: Demographics
minority_pop = districts['BVAP'] + districts['HVAP'] + districts['AMINVAP']
districts['Minority'] = minority_pop / districts['VAP']

# Additional demographic metrics
districts['Hispanic_frac'] = districts['HVAP'] / districts['VAP']
districts['Black_frac'] = districts['BVAP'] / districts['VAP']
districts['Diversity_index'] = 1 - (
    (districts['WVAP']/districts['VAP'])**2 +
    (districts['BVAP']/districts['VAP'])**2 +
    (districts['HVAP']/districts['VAP'])**2 +
    (districts['AMINVAP']/districts['VAP'])**2
)

# Clip extreme values to avoid exact 0/1
for col in ['Minority','Hispanic_frac','Black_frac','Diversity_index']:
    districts[col] = districts[col].clip(0.01, 0.99)

# -----------------------------
# Step 7: Normalize all metrics
metrics = [
    'PP','Reock','PA_ratio','ConvexHull_ratio','Schwartzberg',
    'EG','EG_reg','MM','Partisan_bias','Competitive_flag',
    'Competitiveness','Competitiveness_std',
    'Minority','Hispanic_frac','Black_frac','Diversity_index'
]

# Clip outliers 5–95th percentile
for col in metrics:
    lower, upper = np.percentile(districts[col].dropna(), [5, 95])
    districts[col] = districts[col].clip(lower, upper)

# Rank-based scaling to [0.01, 0.99] for smoother scores
districts[metrics] = districts[metrics].rank(pct=True) * 0.98 + 0.01

# -----------------------------
# Step 8: Compute weighted composite
weights = {
    'geometry': 0.25,
    'partisan': 0.4,
    'competitiveness': 0.2,
    'demographics': 0.15
}

districts['geometry_score'] = districts[['PP','Reock','PA_ratio','ConvexHull_ratio','Schwartzberg']].mean(axis=1)
districts['partisan_score'] = districts[['EG','EG_reg','MM','Partisan_bias','Competitive_flag']].mean(axis=1)
districts['competitiveness_score'] = districts[['Competitiveness','Competitiveness_std']].mean(axis=1)
districts['demographics_score'] = districts[['Minority','Hispanic_frac','Black_frac','Diversity_index']].mean(axis=1)

districts['composite_score'] = (
    weights['geometry'] * districts['geometry_score'] +
    weights['partisan'] * districts['partisan_score'] +
    weights['competitiveness'] * districts['competitiveness_score'] +
    weights['demographics'] * districts['demographics_score']
)

# -----------------------------
# Step 9: Inspect results
print(districts[['geometry_score','partisan_score','competitiveness_score','demographics_score','composite_score']])

# -----------------------------
# Step 10: Save weighted districts
output_path = "../data/NC-shapefiles-master/NC_VTD/NC_districts_weighted.geojson"
districts.to_file(output_path, driver='GeoJSON')
print(f"Weighted district file saved to {output_path}")
