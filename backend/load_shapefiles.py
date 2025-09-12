# load_shapefiles.py
import os
import zipfile
from io import BytesIO

import certifi
import requests
import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup

def download_and_extract_all_cd_shapefiles(
    base_url="https://www2.census.gov/geo/tiger/TIGER2024/CD/",
    extract_dir="TIGER_2024_CD",
    verify_ssl=False,
):
    """
    Downloads all 2024 Congressional District shapefiles from Census TIGER/Line,
    extracts them locally, and returns a list of .shp file paths.

    Args:
        base_url (str): Census URL for CD 2024 shapefiles.
        extract_dir (str): Local folder to extract files.
        verify_ssl (bool): Whether to verify SSL certificates.

    Returns:
        list[str]: List of extracted shapefile paths (.shp).
    """
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading shapefiles from {base_url}...")

    # Request the directory page
    resp = requests.get(
        base_url, verify=certifi.where() if verify_ssl else False, timeout=15
    )
    resp.raise_for_status()

    # Parse HTML for ZIP links
    soup = BeautifulSoup(resp.text, "html.parser")
    zip_links = [
        base_url + a['href']
        for a in soup.find_all("a")
        if a.has_attr('href') and a['href'].endswith(".zip")
    ]

    print(f"Found {len(zip_links)} ZIP files")
    shp_files = []

    for link in zip_links:
        try:
            print(f"Downloading {link} ...")
            r = requests.get(
                link, verify=certifi.where() if verify_ssl else False, timeout=30
            )
            r.raise_for_status()

            zip_data = BytesIO(r.content)
            with zipfile.ZipFile(zip_data) as zf:
                for f in zf.namelist():
                    if f.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                        zf.extract(f, extract_dir)
                        if f.endswith(".shp"):
                            shp_files.append(os.path.join(extract_dir, f))
            print(f"Extracted {link}")

        except Exception as e:
            print(f"Failed to download or extract {link}: {e}")

    return shp_files


def load_all_cd_shapefiles(shapefile_dir=None, verify_ssl=False):
    """
    Loads all CD shapefiles into a single GeoDataFrame.

    Args:
        shapefile_dir (str|None): If None, downloads & extracts all files.
        verify_ssl (bool): Whether to verify SSL during download.

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame of all districts.
    """
    if shapefile_dir is None:
        shp_files = download_and_extract_all_cd_shapefiles(verify_ssl=verify_ssl)
    else:
        shp_files = [
            os.path.join(shapefile_dir, f)
            for f in os.listdir(shapefile_dir)
            if f.endswith(".shp")
        ]

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
    all_districts_gdf = load_all_cd_shapefiles(verify_ssl=False)
    print(all_districts_gdf.head())
    print(f"Total districts loaded: {len(all_districts_gdf)}")
