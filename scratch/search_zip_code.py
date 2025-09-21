#!/usr/bin/env python3
"""
Search for a specific zip code in the database
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

def search_zip_code(target_zip):
    """Search for a specific zip code in all data sources"""
    target_zip = str(target_zip).zfill(5)
    print(f"Searching for zip code: {target_zip}")
    print("=" * 50)
    
    # Search in quote data
    print("1. Searching in quote data...")
    try:
        quote_data = pd.read_csv('data/raw/quote_data.csv')
        quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
        
        zip_in_quotes = quote_data[quote_data['Zipcode_clean'] == target_zip]
        if not zip_in_quotes.empty:
            print(f"   ✅ FOUND in quote data!")
            print(f"   Total quotes: {zip_in_quotes['Pickup Count'].iloc[0] + zip_in_quotes['Dropoff Count'].iloc[0]}")
            print(f"   Pickup count: {zip_in_quotes['Pickup Count'].iloc[0]}")
            print(f"   Dropoff count: {zip_in_quotes['Dropoff Count'].iloc[0]}")
        else:
            print(f"   ❌ NOT FOUND in quote data")
    except Exception as e:
        print(f"   Error reading quote data: {e}")
    
    # Search in CBSA mapping
    print("\n2. Searching in CBSA mapping...")
    try:
        cbsa_mapping = pd.read_csv('data/raw/zip_to_cbsa_mapping.csv')
        cbsa_mapping['Zip Code_clean'] = cbsa_mapping['Zip Code'].astype(str).str.zfill(5)
        
        zip_in_cbsa = cbsa_mapping[cbsa_mapping['Zip Code_clean'] == target_zip]
        if not zip_in_cbsa.empty:
            print(f"   ✅ FOUND in CBSA mapping!")
            print(f"   CBSA: {zip_in_cbsa['Primary CBSA Name'].iloc[0]}")
            print(f"   State: {zip_in_cbsa['State'].iloc[0]}")
            print(f"   City: {zip_in_cbsa['City'].iloc[0]}")
        else:
            print(f"   ❌ NOT FOUND in CBSA mapping")
    except Exception as e:
        print(f"   Error reading CBSA mapping: {e}")
    
    # Search in shapefiles
    print("\n3. Searching in shapefiles...")
    try:
        zip_shapes = gpd.read_file('data/shapefiles/cb_2020_us_zcta520_500k.shp')
        zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)
        
        zip_in_shapes = zip_shapes[zip_shapes['ZCTA5CE20'] == target_zip]
        if not zip_in_shapes.empty:
            print(f"   ✅ FOUND in shapefiles!")
            print(f"   Has geographic boundary data")
            # Get centroid
            centroid = zip_in_shapes.geometry.centroid.iloc[0]
            print(f"   Centroid: ({centroid.y:.4f}, {centroid.x:.4f})")
        else:
            print(f"   ❌ NOT FOUND in shapefiles")
    except Exception as e:
        print(f"   Error reading shapefiles: {e}")
    
    # Search in unassigned zip codes
    print("\n4. Searching in unassigned zip codes...")
    try:
        unassigned_path = Path('data/raw/unassigned_cbsa_zip_codes_2025-09-03T02-02-45.csv')
        if unassigned_path.exists():
            unassigned_df = pd.read_csv(unassigned_path)
            unassigned_df['Zipcode_clean'] = unassigned_df['Zipcode'].astype(str).str.zfill(5)
            
            zip_in_unassigned = unassigned_df[unassigned_df['Zipcode_clean'] == target_zip]
            if not zip_in_unassigned.empty:
                print(f"   ✅ FOUND in unassigned zip codes list!")
                print(f"   This zip was manually unassigned from its CBSA")
            else:
                print(f"   ❌ NOT FOUND in unassigned zip codes list")
        else:
            print(f"   Unassigned zip codes file not found")
    except Exception as e:
        print(f"   Error reading unassigned zip codes: {e}")
    
    print("\n" + "=" * 50)
    print("Search complete!")

if __name__ == "__main__":
    # Search for zip code 51054
    search_zip_code("51054")
