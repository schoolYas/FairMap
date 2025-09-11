# load_shapefiles.py
import os
import requests
import zipfile
from io import BytesIO
import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup


""" DOWNLOAD_AND_EXTRACT_ALL_CD_SHAPEFILES
    Downloads all 2024 Congressional District shapefiles from the Census TIGER/Line site,
    extracts them into a local directory, and returns a list of extracted .shp file paths.

    Parameters:
        base_url (str): URL to the 2024 CD directory on Census FTP/HTTP server.
        extract_dir (str): Local directory to store shapefiles.

    Returns:
        list[str]: List of file paths to all .shp files.
"""
def download_and_extract_all_cd_shapefiles(base_url="https://www2.census.gov/geo/tiger/TIGER2024/CD/", extract_dir="TIGER_2024_CD"):
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading shapefiles from {base_url}...")

    # Scrape directory page to get all ZIP links
    resp = requests.get(base_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    zip_links = [base_url + a['href'] for a in soup.find_all('a') if a['href'].endswith(".zip")]

    print(f"Found {len(zip_links)} ZIP files")

    shp_files = []

    for link in zip_links:
        try:
            print(f"Downloading {link} ...")
            r = requests.get(link)
            r.raise_for_status()
            zip_data = BytesIO(r.content)
            with zipfile.ZipFile(zip_data) as zf:
                # Extract only the necessary shapefile components
                for f in zf.namelist():
                    if f.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                        zf.extract(f, extract_dir)
                        if f.endswith(".shp"):
                            shp_files.append(os.path.join(extract_dir, f))
            print(f"Extracted {link}")
        except Exception as e:
            print(f"Failed to download or extract {link}: {e}")

    return shp_files


def load_all_cd_shapefiles(shapefile_dir=None):
    """
    Loads all extracted CD shapefiles into a single GeoDataFrame.

    Parameters:
        shapefile_dir (str or None): If None, automatically download/extract all shapefiles.

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame of all districts.
    """
    if shapefile_dir is None:
        # Download and extract all shapefiles
        shp_files = download_and_extract_all_cd_shapefiles()
    else:
        shp_files = [os.path.join(shapefile_dir, f) for f in os.listdir(shapefile_dir) if f.endswith(".shp")]

    print(f"Loading {len(shp_files)} shapefiles into GeoDataFrame...")
    gdfs = []
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        gdf["map_name"] = os.path.basename(shp)
        gdfs.append(gdf)

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    print("All shapefiles loaded!")
    return combined_gdf


if __name__ == "__main__":
    # Example usage
    all_districts_gdf = load_all_cd_shapefiles()
    print(all_districts_gdf.head())
    print(f"Total districts loaded: {len(all_districts_gdf)}")
