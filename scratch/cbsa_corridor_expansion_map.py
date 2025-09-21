import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from pathlib import Path
from sklearn.neighbors import BallTree

# Configurable paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DOWNLOADS_DIR = Path.home() / 'Downloads'

def load_data():
    """Load and prepare all necessary data"""
    print("Loading data...")
    
    # Load quote data
    quote_data = pd.read_csv(DATA_DIR / 'raw' / 'quote_data.csv')
    quote_data['Pickup Count'] = pd.to_numeric(quote_data['Pickup Count'], errors='coerce').fillna(0)
    quote_data['Dropoff Count'] = pd.to_numeric(quote_data['Dropoff Count'], errors='coerce').fillna(0)
    quote_data['Total Quotes'] = quote_data['Pickup Count'] + quote_data['Dropoff Count']
    quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)

    print(f"Loaded {len(quote_data)} quote records")
    
    # Load CBSA mapping data
    mapping_file = DATA_DIR / 'raw' / 'zip_to_csa_mapping.csv'
    try:
        cbsa_mapping = pd.read_csv(mapping_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            cbsa_mapping = pd.read_csv(mapping_file, encoding='latin-1')
        except UnicodeDecodeError:
            cbsa_mapping = pd.read_csv(mapping_file, encoding='cp1252')
    
    cbsa_mapping['Zip Code_clean'] = cbsa_mapping['Zip Code'].astype(str).str.zfill(5)

    # Merge quote data with CBSA mapping
    data = quote_data.merge(cbsa_mapping, left_on='Zipcode_clean', right_on='Zip Code_clean', how='left')
    
    # Load unassigned zip codes (same approach as original)
    # PASTE YOUR EXPORTED UNASSIGNED ZIP CODES HERE:
    # Copy the zip codes from your CSV file: unassigned_cbsa_zip_codes_2025-09-03T02-02-45.csv
    unassigned_zips = [
        # Paste your zip codes here, one per line in quotes, e.g.:
        # '12345',
        # '67890',
        # '54321',
        # etc.
    ]

    # Alternative: Try to load from the CSV file if it exists
    try:
        csv_path = DOWNLOADS_DIR / 'unassigned_cbsa_zip_codes_2025-09-03T02-02-45.csv'
        if csv_path.exists():
            unassigned_df = pd.read_csv(csv_path)
            if 'Zipcode' in unassigned_df.columns:
                unassigned_zips = unassigned_df['Zipcode'].astype(str).tolist()
                print(f"Loaded {len(unassigned_zips)} unassigned zip codes from CSV file")
            else:
                print("CSV file found but no 'Zipcode' column detected")
        else:
            print("CSV file not found, using manual list above")
    except Exception as e:
        print(f"Could not load CSV file: {e}")
        print("Using manual list above")

    # Clean the unassigned zip codes to 5 digits
    unassigned_zips_clean = [zip_code.zfill(5) for zip_code in unassigned_zips]
    
    # Unassign the specified zip codes
    print(f"Unassigning {len(unassigned_zips_clean)} zip codes from their CBSAs...")
    data.loc[data['Zipcode_clean'].isin(unassigned_zips_clean), 'Primary CBSA Name'] = None
    print(f"Updated {len(unassigned_zips_clean)} zip codes to remove CBSA assignments")

    return data, set(unassigned_zips_clean)

def load_shapefiles():
    """Load zip code shapefiles"""
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file(DATA_DIR / 'shapefiles' / 'cb_2020_us_zcta520_500k.shp')
    
    # Add population data (using a simple proxy based on land area)
    if 'ALAND20' in zip_shapes.columns:
        zip_shapes['ZCTA Population (2020)'] = pd.to_numeric(zip_shapes['ALAND20'], errors='coerce').fillna(0) / 1000  # Simple proxy
    else:
        zip_shapes['ZCTA Population (2020)'] = 1000  # Default population estimate
    
    print(f"Loaded {len(zip_shapes)} zip code shapes")
    return zip_shapes

def load_crossdock_data():
    """Load crossdock location data"""
    print("Loading crossdock data...")

    # PASTE YOUR CROSSDOCK ZIP CODES HERE:
    # Copy the zip codes from your CSV file: Copy of WARP Xdocks List - 2025 - Sheet2 (1).csv
    crossdock_zips = [
        # Paste your crossdock zip codes here, one per line in quotes, e.g.:
        # '90210',  # Los Angeles Crossdock
        # '10001',  # New York Crossdock
        # '60601',  # Chicago Crossdock
        # etc.
    ]

    # Alternative: Try to load from the CSV file if it exists
    try:
        csv_path = DOWNLOADS_DIR / "Copy of WARP Xdocks List - 2025 - Sheet2 (1).csv"
        if csv_path.exists():
            crossdock_df = pd.read_csv(csv_path)
            print(f"Crossdock CSV columns: {crossdock_df.columns.tolist()}")

            # Try to find zip code column (common names)
            zip_columns = ['Zip', 'Zip Code', 'Zipcode', 'ZIP', 'ZIP_CODE', 'postal_code', 'PostalCode']
            zip_col = None
            for col in zip_columns:
                if col in crossdock_df.columns:
                    zip_col = col
                    break

            if zip_col:
                crossdock_zips = crossdock_df[zip_col].astype(str).str.zfill(5).tolist()
                print(f"Loaded {len(crossdock_zips)} crossdock zip codes from CSV file")

                # Also try to get names if available
                name_columns = ['Name', 'Location', 'Crossdock', 'Facility', 'Site']
                name_col = None
                for col in name_columns:
                    if col in crossdock_df.columns:
                        name_col = col
                        break

                if name_col:
                    crossdock_names = crossdock_df[name_col].astype(str).tolist()
                else:
                    crossdock_names = [f"Crossdock {i+1}" for i in range(len(crossdock_zips))]

                return list(zip(crossdock_zips, crossdock_names))
            else:
                print("CSV file found but no zip code column detected")
                print(f"Available columns: {crossdock_df.columns.tolist()}")
        else:
            print("Crossdock CSV file not found, using manual list above")
    except Exception as e:
        print(f"Could not load crossdock CSV file: {e}")
        print("Using manual list above")

    # Return manual list with default names
    crossdock_names = [f"Crossdock {i+1}" for i in range(len(crossdock_zips))]
    return list(zip(crossdock_zips, crossdock_names))

def process_crossdock_locations(crossdock_data, zip_shapes):
    """Process crossdock locations and get their geographic coordinates"""
    print("Processing crossdock locations...")

    if not crossdock_data:
        print("No crossdock data provided")
        return []

    crossdock_locations = []

    for zip_code, name in crossdock_data:
        # Find the zip code in the shapefile
        zip_match = zip_shapes[zip_shapes['ZCTA5CE20'] == zip_code.zfill(5)]

        if not zip_match.empty:
            # Calculate centroid
            zip_geom = zip_match.iloc[0]['geometry']
            if zip_geom is not None:
                # Convert to projected coordinate system for accurate centroid
                zip_gdf = gpd.GeoDataFrame([zip_match.iloc[0]], crs=zip_shapes.crs)
                zip_projected = zip_gdf.to_crs('EPSG:3857')
                centroid = zip_projected.geometry.centroid.to_crs('EPSG:4326').iloc[0]

                crossdock_locations.append({
                    'zip_code': zip_code.zfill(5),
                    'name': name,
                    'lat': centroid.y,
                    'lon': centroid.x
                })
                print(f"  Found crossdock: {name} at {zip_code} ({centroid.y:.4f}, {centroid.x:.4f})")
            else:
                print(f"  Warning: No geometry found for crossdock zip code {zip_code}")
        else:
            print(f"  Warning: Crossdock zip code {zip_code} not found in shapefile")

    print(f"Successfully processed {len(crossdock_locations)} crossdock locations")
    return crossdock_locations

def calculate_weighted_cbsa_centroids(data, zip_shapes):
    """Calculate quote-weighted centroids for each CBSA with corridor analysis support"""
    print("Calculating quote-weighted CBSA centroids...")
    
    # Calculate top 75 CBSAs by quote volume (simplified approach)
    print("Determining top 75 CBSAs by quote volume...")

    # Filter for zip codes with quotes and CBSA assignments
    quote_data = data[data['Total Quotes'] > 0]
    cbsa_quote_totals = quote_data[quote_data['Primary CBSA Name'].notna()].groupby('Primary CBSA Name')['Total Quotes'].sum().reset_index()
    cbsa_quote_totals = cbsa_quote_totals.sort_values('Total Quotes', ascending=False)
    top_75_cbsas = cbsa_quote_totals.head(75)['Primary CBSA Name'].tolist()

    print(f"Top 75 CBSAs by quote volume:")
    for i, cbsa in enumerate(top_75_cbsas[:10], 1):
        quotes = cbsa_quote_totals[cbsa_quote_totals['Primary CBSA Name'] == cbsa]['Total Quotes'].iloc[0]
        print(f"  {i}. {cbsa}: {quotes:,.0f} quotes")
    if len(top_75_cbsas) > 10:
        print(f"  ... and {len(top_75_cbsas) - 10} more")
    
    # NOW filter for zip codes with quotes for the actual mapping
    quote_data = data[data['Total Quotes'] > 0]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        quote_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    # Create full CBSA data for corridor analysis (includes ALL CBSAs)
    full_cbsa_data = map_data[map_data['Primary CBSA Name'].notna()].copy()

    # Filter map data to only include top 75 CBSAs (but keep all non-CBSA zip codes)
    map_data = map_data[
        (map_data['Primary CBSA Name'].isin(top_75_cbsas)) |
        (map_data['Primary CBSA Name'].isna())
    ]

    print(f"Filtered to {len(map_data)} zip codes with quotes in top 75 CBSAs or non-CBSA areas")
    print(f"Full CBSA dataset contains {len(full_cbsa_data)} zip codes from ALL CBSAs for corridor analysis")
    
    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    non_cbsa_zips = map_data[map_data['Primary CBSA Name'].isna()].copy()
    
    print(f"Calculating weighted centroids for {cbsa_zips['Primary CBSA Name'].nunique()} CBSAs...")
    
    # Calculate quote-weighted centroids for each CBSA
    cbsa_centroids = []
    
    for cbsa_name in cbsa_zips['Primary CBSA Name'].unique():
        cbsa_data = cbsa_zips[cbsa_zips['Primary CBSA Name'] == cbsa_name]
        
        # Get centroids of each zip code polygon
        centroids = cbsa_data.geometry.centroid
        
        # Calculate weighted centroid
        total_quotes = cbsa_data['Total Quotes'].sum()
        if total_quotes > 0:
            weighted_lat = (centroids.y * cbsa_data['Total Quotes']).sum() / total_quotes
            weighted_lon = (centroids.x * cbsa_data['Total Quotes']).sum() / total_quotes
            
            cbsa_centroids.append({
                'cbsa_name': cbsa_name,
                'weighted_lat': weighted_lat,
                'weighted_lon': weighted_lon,
                'total_quotes': total_quotes,
                'zip_count': len(cbsa_data)
            })
    
    cbsa_centroids_df = pd.DataFrame(cbsa_centroids)
    print(f"Calculated weighted centroids for {len(cbsa_centroids_df)} CBSAs")
    
    if len(cbsa_centroids_df) > 0:
        print(f"Quote range across CBSAs: {cbsa_centroids_df['total_quotes'].min():,.0f} - {cbsa_centroids_df['total_quotes'].max():,.0f}")
    
    return map_data, cbsa_centroids_df, full_cbsa_data

def calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df):
    """Calculate distances from non-CBSA zip codes to nearest individual CBSA zip codes"""
    print("Calculating distances to nearest CBSA zip codes (adjacency analysis)...")
    
    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    non_cbsa_zips = map_data[map_data['Primary CBSA Name'].isna()].copy()
    
    print(f"Calculating distances from {len(non_cbsa_zips)} non-CBSA zip codes to {len(cbsa_zips)} CBSA zip codes...")
    
    # Get centroids for distance calculation
    cbsa_centroids = cbsa_zips.geometry.centroid
    non_cbsa_centroids = non_cbsa_zips.geometry.centroid
    
    # Convert to lat/lon arrays for BallTree
    cbsa_coords = np.array([[point.y, point.x] for point in cbsa_centroids])
    non_cbsa_coords = np.array([[point.y, point.x] for point in non_cbsa_centroids])
    
    # Use BallTree for efficient nearest neighbor search
    tree = BallTree(np.radians(cbsa_coords), metric='haversine')
    distances, indices = tree.query(np.radians(non_cbsa_coords), k=1)
    
    # Convert distances from radians to miles
    earth_radius_miles = 3959
    distances_miles = distances.flatten() * earth_radius_miles
    
    # Get the nearest CBSA zip codes and their names
    nearest_cbsa_zip_codes = cbsa_zips.iloc[indices.flatten()]['Zipcode_clean'].values
    nearest_cbsa_names = cbsa_zips.iloc[indices.flatten()]['Primary CBSA Name'].values
    
    # Calculate quote statistics for each CBSA
    print("Calculating quote statistics for each CBSA...")
    cbsa_stats = {}
    for cbsa_name in cbsa_zips['Primary CBSA Name'].unique():
        cbsa_quotes = cbsa_zips[cbsa_zips['Primary CBSA Name'] == cbsa_name]['Total Quotes']
        cbsa_stats[cbsa_name] = {
            'mean': cbsa_quotes.mean(),
            'std': cbsa_quotes.std(),
            'count': len(cbsa_quotes)
        }
    
    # Calculate standard deviation metrics and CBSA mean quotes for non-CBSA zip codes
    std_deviations_below = []
    quote_percentiles_in_nearest_cbsa = []
    cbsa_mean_quotes = []
    
    for idx, (_, row) in enumerate(non_cbsa_zips.iterrows()):
        nearest_cbsa = nearest_cbsa_names[idx]
        zip_quotes = row['Total Quotes']
        
        if nearest_cbsa in cbsa_stats:
            cbsa_mean = cbsa_stats[nearest_cbsa]['mean']
            cbsa_std = cbsa_stats[nearest_cbsa]['std']
            
            # Calculate how many standard deviations below the mean
            if cbsa_std > 0:
                std_dev_below = (cbsa_mean - zip_quotes) / cbsa_std
            else:
                std_dev_below = 0
            
            # Calculate percentile within the nearest CBSA
            cbsa_quotes = cbsa_zips[cbsa_zips['Primary CBSA Name'] == nearest_cbsa]['Total Quotes']
            percentile = (cbsa_quotes < zip_quotes).mean() * 100
            
            std_deviations_below.append(std_dev_below)
            quote_percentiles_in_nearest_cbsa.append(percentile)
            cbsa_mean_quotes.append(cbsa_mean)
        else:
            std_deviations_below.append(0)
            quote_percentiles_in_nearest_cbsa.append(0)
            cbsa_mean_quotes.append(0)
    
    # Add distances and statistical info to non-CBSA zip codes
    non_cbsa_zips['min_distance_to_cbsa'] = distances_miles
    non_cbsa_zips['nearest_cbsa_zip'] = nearest_cbsa_zip_codes
    non_cbsa_zips['nearest_cbsa_name'] = nearest_cbsa_names
    non_cbsa_zips['std_dev_below_cbsa_mean'] = std_deviations_below
    non_cbsa_zips['quote_percentile_in_nearest_cbsa'] = quote_percentiles_in_nearest_cbsa
    non_cbsa_zips['cbsa_mean_quotes'] = cbsa_mean_quotes
    
    # Add zero distance for CBSA zip codes (they are at distance 0 from themselves)
    cbsa_zips['min_distance_to_cbsa'] = 0
    cbsa_zips['nearest_cbsa_zip'] = cbsa_zips['Zipcode_clean']
    cbsa_zips['nearest_cbsa_name'] = cbsa_zips['Primary CBSA Name']
    cbsa_zips['std_dev_below_cbsa_mean'] = 0
    cbsa_zips['quote_percentile_in_nearest_cbsa'] = 0
    
    # Add CBSA mean quotes for CBSA-assigned zip codes
    cbsa_zips['cbsa_mean_quotes'] = cbsa_zips['Primary CBSA Name'].map(
        lambda cbsa_name: cbsa_stats.get(cbsa_name, {}).get('mean', 0)
    )
    
    # Combine back into single dataset
    map_data = pd.concat([cbsa_zips, non_cbsa_zips], ignore_index=True)
    
    print(f"Adjacency analysis complete. Distance range: {distances_miles.min():.1f} - {distances_miles.max():.1f} miles")
    
    return map_data

def generate_corridor_html(zip_features, cbsa_centroid_features, crossdock_locations, full_cbsa_features=None):
    """Generate HTML content for the corridor expansion map"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 CBSA Corridor Expansion Analysis</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; width: 100vw; }}
        .controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 300px;
            max-width: 350px;
            max-height: calc(100vh - 60px);
            overflow-y: auto;
            font-size: 14px;
        }}
        .controls h4 {{ margin: 0 0 15px 0; color: #333; }}
        .slider-container {{
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 2px solid #007cba;
        }}
        .slider-container label {{
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #495057;
        }}
        .slider {{
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: linear-gradient(to right, #ff6b6b, #feca57, #48dbfb, #0abde3);
            outline: none;
            -webkit-appearance: none;
            margin: 10px 0;
        }}
        .slider::-webkit-slider-thumb {{
            appearance: none;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
            border: 3px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        .slider::-moz-range-thumb {{
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
            border: 3px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        .slider-value {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: white;
            background: #6c5ce7;
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
        }}
        .slider-range {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #6c757d;
            margin-top: 5px;
        }}
        .checkbox-container {{
            margin: 12px 0;
            padding: 8px;
            background: #e9ecef;
            border-radius: 5px;
        }}
        .checkbox-container input[type="checkbox"] {{
            margin-right: 8px;
            transform: scale(1.2);
        }}
        .checkbox-container label {{
            font-weight: normal;
            margin: 0;
            cursor: pointer;
        }}
        .stats {{
            font-size: 11px;
            color: #495057;
            margin-top: 10px;
            padding: 10px;
            background: #f1f3f4;
            border-radius: 6px;
            border-left: 4px solid #007cba;
            line-height: 1.3;
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
            font-size: 12px;
            max-width: 250px;
        }}
        .legend h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 15px;
            margin-right: 8px;
            border-radius: 3px;
            border: 1px solid #ccc;
        }}
        .custom-tooltip {{
            background: rgba(0, 0, 0, 0.8) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 5px 8px !important;
            font-size: 12px !important;
            font-family: Arial, sans-serif !important;
        }}
        .corridor-controls {{
            margin: 15px 0;
            padding: 15px;
            background: #fff3cd;
            border-radius: 8px;
            border: 2px solid #ffc107;
        }}
        .corridor-controls h5 {{
            margin: 0 0 10px 0;
            color: #856404;
            font-size: 14px;
        }}
        .corridor-width-container {{
            margin: 10px 0;
        }}
        .corridor-width-container label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #856404;
            font-size: 12px;
        }}
        .corridor-slider {{
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(to right, #ffc107, #fd7e14);
            outline: none;
            -webkit-appearance: none;
            margin: 5px 0;
        }}
        .corridor-slider::-webkit-slider-thumb {{
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #fd7e14;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }}
        .corridor-value {{
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            color: #856404;
            background: #fff3cd;
            padding: 5px;
            border-radius: 5px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="controls">
        <h4>🎯 Top 75 CBSAs Corridor Analysis</h4>

        <div class="slider-container">
            <label for="distanceSlider">Distance to Nearest CBSA Coverage (Zip-to-Zip):</label>
            <input type="range" id="distanceSlider" class="slider" min="5" max="200" value="50" step="5">
            <div class="slider-value" id="distanceValue">50 miles</div>
            <div class="slider-range">
                <span>5 miles</span>
                <span>200 miles</span>
            </div>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCBSAZips" checked>
            <label for="showCBSAZips">Show all CBSA zip codes</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showNonCBSAZips" checked>
            <label for="showNonCBSAZips">Show nearby non-CBSA zip codes</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCBSACentroids" checked>
            <label for="showCBSACentroids">Show CBSA weighted centroids</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCrossdocks" checked>
            <label for="showCrossdocks">Show crossdock locations</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="enableQuoteFilter" checked>
            <label for="enableQuoteFilter">Enable quote performance filter</label>
        </div>

        <div class="slider-container" id="quoteFilterContainer">
            <label for="stdDevSlider">Min Quote Performance vs Nearest CBSA:</label>
            <input type="range" id="stdDevSlider" class="slider" min="-30" max="20" value="-30" step="1">
            <div class="slider-value" id="stdDevValue">≥ -3.0 std dev</div>
            <div class="slider-range">
                <span>-3 std dev</span>
                <span>+2 std dev</span>
            </div>
            <div style="font-size: 11px; color: #6c757d; margin-top: 8px; line-height: 1.3;">
                Filters non-CBSA zip codes by quote volume relative to their nearest CBSA.<br>
                <strong>-1.0</strong> = 1 std dev below CBSA average<br>
                <strong>0.0</strong> = at CBSA average<br>
                <strong>+1.0</strong> = 1 std dev above CBSA average
            </div>
        </div>

        <div class="corridor-controls">
            <h5>🛣️ Individual Corridor Analysis</h5>
            <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 12px; margin-bottom: 8px;">
                <strong>💡 How to use:</strong> Click on any non-CBSA zip code to toggle its corridor connections on/off
            </div>
            <div class="corridor-width-container">
                <label for="corridorWidthSlider">Corridor Width:</label>
                <input type="range" id="corridorWidthSlider" class="corridor-slider" min="5" max="50" value="15" step="5">
                <div class="corridor-value" id="corridorWidthValue">15 miles</div>
                <div class="slider-range">
                    <span>5 miles</span>
                    <span>50 miles</span>
                </div>
            </div>
            <div style="font-size: 10px; color: #856404; margin-top: 8px; line-height: 1.2;">
                Shows zip codes "on the way" between isolated opportunities and nearest CBSA coverage.
            </div>
        </div>

        <div class="stats" id="filterStats">
            Loading...
        </div>

        <button onclick="exportFilteredData()" style="width: 100%; padding: 10px; margin-top: 15px; background: #007cba; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
            📥 Export Filtered Zip Codes
        </button>
    </div>

    <div class="legend">
        <h4>Map Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #2E86AB;"></div>
            <span>CBSA Zip Codes</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #A23B72;"></div>
            <span>Non-CBSA Opportunities</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #F18F01;"></div>
            <span>Corridor Zip Codes</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #C73E1D;"></div>
            <span>CBSA Centroids</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #28a745;"></div>
            <span>Crossdock Locations</span>
        </div>
        <div style="margin-top: 10px; font-size: 10px; color: #666;">
            Darker shades = Higher quote volume<br>
            Click zip codes to remove from analysis
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Data
        const zipData = {json.dumps(zip_features, indent=8)};
        const cbsaCentroids = {json.dumps(cbsa_centroid_features, indent=8)};
        const crossdockLocations = {json.dumps(crossdock_locations, indent=8)};
        const fullCbsaData = {json.dumps(full_cbsa_features or [], indent=8)};

        // Map initialization
        const map = L.map('map').setView([39.8283, -98.5795], 4);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Create custom pane for crossdocks to ensure they appear on top
        map.createPane('crossdockPane');
        map.getPane('crossdockPane').style.zIndex = 650; // Higher than default overlay pane (400)

        // Layer groups
        const cbsaZipLayer = L.layerGroup().addTo(map);
        const nonCbsaZipLayer = L.layerGroup().addTo(map);
        const corridorZipLayer = L.layerGroup().addTo(map);
        const cbsaCentroidLayer = L.layerGroup().addTo(map);
        const crossdockLayer = L.layerGroup().addTo(map);

        // Removed zip codes tracking
        let removedZipCodes = new Set();

        // Current filter values
        let currentDistanceThreshold = 50;
        let currentStdDevThreshold = -3.0;
        let currentCorridorWidth = 15;
        let removedZipCodes = new Set(); // Track removed zip codes
        let activeCorridors = new Map(); // Track active corridors by target zip code
        let corridorLayers = new Map(); // Track corridor layer groups by target zip code

        // Corridor analysis functions
        function calculateDistance(lat1, lon1, lat2, lon2) {{
            const R = 3959; // Earth's radius in miles
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                    Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }}

        function getZipCentroid(zipFeature) {{
            // Get centroid of zip code polygon
            if (zipFeature.geometry.type === 'Polygon') {{
                const coords = zipFeature.geometry.coordinates[0];
                let lat = 0, lon = 0;
                for (let i = 0; i < coords.length - 1; i++) {{
                    lon += coords[i][0];
                    lat += coords[i][1];
                }}
                return [lat / (coords.length - 1), lon / (coords.length - 1)];
            }} else if (zipFeature.geometry.type === 'MultiPolygon') {{
                const coords = zipFeature.geometry.coordinates[0][0];
                let lat = 0, lon = 0;
                for (let i = 0; i < coords.length - 1; i++) {{
                    lon += coords[i][0];
                    lat += coords[i][1];
                }}
                return [lat / (coords.length - 1), lon / (coords.length - 1)];
            }}
            return [0, 0];
        }}

        function findCorridorZipCodes(targetZips, cbsaZips, allZips, corridorWidth) {{
            console.log('Starting corridor analysis...');
            console.log('Target zips:', targetZips.length);
            console.log('CBSA zips for analysis:', cbsaZips.length);
            console.log('All zips to check:', allZips.length);
            console.log('Corridor width:', corridorWidth);

            const corridorZips = new Set();
            const corridorTargets = new Map(); // Track which target zip each corridor zip connects to

            targetZips.forEach((targetZip, targetIndex) => {{
                console.log(`Processing target zip ${{targetIndex + 1}}/${{targetZips.length}}: ${{targetZip.properties.zipcode_clean}}`);
                const targetCentroid = getZipCentroid(targetZip);

                // Find nearest CBSA zip code
                let nearestCbsaZip = null;
                let minDistance = Infinity;

                cbsaZips.forEach(cbsaZip => {{
                    const cbsaCentroid = getZipCentroid(cbsaZip);
                    const distance = calculateDistance(
                        targetCentroid[0], targetCentroid[1],
                        cbsaCentroid[0], cbsaCentroid[1]
                    );

                    if (distance < minDistance) {{
                        minDistance = distance;
                        nearestCbsaZip = cbsaZip;
                    }}
                }});

                if (nearestCbsaZip) {{
                    console.log(`  Nearest CBSA zip: ${{nearestCbsaZip.properties.zipcode_clean}} (${{minDistance.toFixed(1)}} miles)`);
                    const nearestCentroid = getZipCentroid(nearestCbsaZip);
                    let corridorCount = 0;

                    // Find all zip codes within corridor width of the line between target and nearest CBSA
                    allZips.forEach(zip => {{
                        // Skip the target zip itself, removed zips, and the nearest CBSA zip
                        if (zip.properties.zipcode_clean === targetZip.properties.zipcode_clean ||
                            zip.properties.zipcode_clean === nearestCbsaZip.properties.zipcode_clean ||
                            targetZips.includes(zip) ||
                            removedZipCodes.has(zip.properties.zipcode_clean)) {{
                            return;
                        }}

                        // If target is a CBSA zip, skip other zips from the same CBSA to avoid intra-CBSA corridors
                        if (targetZip.properties.has_cbsa && zip.properties.has_cbsa &&
                            targetZip.properties.cbsa_name === zip.properties.cbsa_name) {{
                            return;
                        }}

                        const zipCentroid = getZipCentroid(zip);
                        const distanceToLine = pointToLineDistance(
                            zipCentroid[0], zipCentroid[1],
                            targetCentroid[0], targetCentroid[1],
                            nearestCentroid[0], nearestCentroid[1]
                        );

                        if (distanceToLine <= corridorWidth) {{
                            // Also check if the zip is roughly between the target and CBSA
                            const distanceToTarget = calculateDistance(
                                zipCentroid[0], zipCentroid[1],
                                targetCentroid[0], targetCentroid[1]
                            );
                            const distanceToCbsa = calculateDistance(
                                zipCentroid[0], zipCentroid[1],
                                nearestCentroid[0], nearestCentroid[1]
                            );
                            const directDistance = calculateDistance(
                                targetCentroid[0], targetCentroid[1],
                                nearestCentroid[0], nearestCentroid[1]
                            );

                            // Only include if the zip is roughly on the path (not too far out of the way)
                            if (distanceToTarget + distanceToCbsa <= directDistance * 1.3) {{
                                corridorZips.add(zip);
                                corridorCount++;
                                // Track which target zip this corridor zip connects to
                                corridorTargets.set(zip.properties.zipcode_clean, targetZip.properties.zipcode_clean);
                            }}
                        }}
                    }});

                    console.log(`  Found ${{corridorCount}} corridor zips for this target`);
                }} else {{
                    console.log(`  No nearest CBSA zip found for target ${{targetZip.properties.zipcode_clean}}`);
                }}
            }});

            console.log(`Total corridor zips found: ${{corridorZips.size}}`);
            return {{ corridorZips: Array.from(corridorZips), corridorTargets }};
        }}

        function pointToLineDistance(px, py, x1, y1, x2, y2) {{
            // Convert to approximate miles using simple projection
            const A = px - x1;
            const B = py - y1;
            const C = x2 - x1;
            const D = y2 - y1;

            const dot = A * C + B * D;
            const lenSq = C * C + D * D;

            if (lenSq === 0) return calculateDistance(px, py, x1, y1);

            let param = dot / lenSq;

            let xx, yy;
            if (param < 0) {{
                xx = x1;
                yy = y1;
            }} else if (param > 1) {{
                xx = x2;
                yy = y2;
            }} else {{
                xx = x1 + param * C;
                yy = y1 + param * D;
            }}

            return calculateDistance(px, py, xx, yy);
        }}

        // Function to calculate corridor for a specific target zip code (optimized)
        function calculateCorridorForZip(targetZip) {{
            console.log(`Calculating corridor for zip: ${{targetZip.properties.zipcode_clean}}`);

            // Use visible CBSA zip codes for faster calculation
            const allCbsaZips = cbsaZips; // Use already filtered CBSA zips for speed
            if (allCbsaZips.length === 0) {{
                console.log('No CBSA zips available for corridor analysis');
                return [];
            }}

            const targetCentroid = getZipCentroid(targetZip);

            // Find nearest CBSA zip code (only check visible ones for speed)
            let nearestCbsaZip = null;
            let minDistance = Infinity;

            allCbsaZips.forEach(cbsaZip => {{
                const cbsaCentroid = getZipCentroid(cbsaZip);
                const distance = calculateDistance(
                    targetCentroid[0], targetCentroid[1],
                    cbsaCentroid[0], cbsaCentroid[1]
                );

                if (distance < minDistance) {{
                    minDistance = distance;
                    nearestCbsaZip = cbsaZip;
                }}
            }});

            if (!nearestCbsaZip) {{
                console.log('No nearest CBSA zip found');
                return [];
            }}

            console.log(`Nearest CBSA zip: ${{nearestCbsaZip.properties.zipcode_clean}} (${{minDistance.toFixed(1)}} miles)`);
            const nearestCentroid = getZipCentroid(nearestCbsaZip);
            const corridorZips = [];

            // Only check visible zip codes for faster calculation
            const allZips = [...cbsaZips, ...nonCbsaZips]; // Use already filtered zips

            // Pre-filter by rough distance to avoid checking every zip
            const maxCorridorDistance = minDistance + currentCorridorWidth * 2; // Rough bounding box

            allZips.forEach(zip => {{
                // Skip the target zip itself, removed zips, and the nearest CBSA zip
                if (zip.properties.zipcode_clean === targetZip.properties.zipcode_clean ||
                    zip.properties.zipcode_clean === nearestCbsaZip.properties.zipcode_clean ||
                    removedZipCodes.has(zip.properties.zipcode_clean)) {{
                    return;
                }}

                const zipCentroid = getZipCentroid(zip);

                // Quick distance check to skip far away zips
                const roughDistance = calculateDistance(
                    zipCentroid[0], zipCentroid[1],
                    targetCentroid[0], targetCentroid[1]
                );

                if (roughDistance > maxCorridorDistance) return; // Skip distant zips

                const distanceToLine = pointToLineDistance(
                    zipCentroid[0], zipCentroid[1],
                    targetCentroid[0], targetCentroid[1],
                    nearestCentroid[0], nearestCentroid[1]
                );

                if (distanceToLine <= currentCorridorWidth) {{
                    // Also check if the zip is roughly between the target and CBSA
                    const distanceToCbsa = calculateDistance(
                        zipCentroid[0], zipCentroid[1],
                        nearestCentroid[0], nearestCentroid[1]
                    );
                    const directDistance = minDistance;

                    // Only include if the zip is roughly on the path (not too far out of the way)
                    if (roughDistance + distanceToCbsa <= directDistance * 1.3) {{
                        corridorZips.push(zip);
                    }}
                }}
            }});

            console.log(`Found ${{corridorZips.length}} corridor zips for ${{targetZip.properties.zipcode_clean}}`);
            return corridorZips;
        }}

        // Fast corridor calculation function (optimized for pre-calculation)
        function calculateCorridorForZipFast(targetZip, allCbsaZips, allZips) {{
            const targetCentroid = getZipCentroid(targetZip);

            // Find nearest CBSA zip code
            let nearestCbsaZip = null;
            let minDistance = Infinity;

            allCbsaZips.forEach(cbsaZip => {{
                const cbsaCentroid = getZipCentroid(cbsaZip);
                const distance = calculateDistance(
                    targetCentroid[0], targetCentroid[1],
                    cbsaCentroid[0], cbsaCentroid[1]
                );

                if (distance < minDistance) {{
                    minDistance = distance;
                    nearestCbsaZip = cbsaZip;
                }}
            }});

            if (!nearestCbsaZip) return [];

            const nearestCentroid = getZipCentroid(nearestCbsaZip);
            const corridorZips = [];

            // Find all zip codes within corridor width of the line between target and nearest CBSA
            allZips.forEach(zip => {{
                // Skip the target zip itself and the nearest CBSA zip
                if (zip.properties.zipcode_clean === targetZip.properties.zipcode_clean ||
                    zip.properties.zipcode_clean === nearestCbsaZip.properties.zipcode_clean) {{
                    return;
                }}

                // If target is a CBSA zip, skip other zips from the same CBSA to avoid intra-CBSA corridors
                if (targetZip.properties.has_cbsa && zip.properties.has_cbsa &&
                    targetZip.properties.cbsa_name === zip.properties.cbsa_name) {{
                    return;
                }}

                const zipCentroid = getZipCentroid(zip);
                const distanceToLine = pointToLineDistance(
                    zipCentroid[0], zipCentroid[1],
                    targetCentroid[0], targetCentroid[1],
                    nearestCentroid[0], nearestCentroid[1]
                );

                if (distanceToLine <= currentCorridorWidth) {{
                    // Also check if the zip is roughly between the target and CBSA
                    const distanceToTarget = calculateDistance(
                        zipCentroid[0], zipCentroid[1],
                        targetCentroid[0], targetCentroid[1]
                    );
                    const distanceToCbsa = calculateDistance(
                        zipCentroid[0], zipCentroid[1],
                        nearestCentroid[0], nearestCentroid[1]
                    );
                    const directDistance = calculateDistance(
                        targetCentroid[0], targetCentroid[1],
                        nearestCentroid[0], nearestCentroid[1]
                    );

                    // Only include if the zip is roughly on the path (not too far out of the way)
                    if (distanceToTarget + distanceToCbsa <= directDistance * 1.3) {{
                        corridorZips.push(zip);
                    }}
                }}
            }});

            return corridorZips;
        }}

        // Function to toggle corridor for a specific zip code
        function toggleCorridorForZip(targetZip) {{
            const zipCode = targetZip.properties.zipcode_clean;

            if (activeCorridors.has(zipCode)) {{
                // Remove existing corridor
                console.log(`Removing corridor for zip: ${{zipCode}}`);
                const layerGroup = corridorLayers.get(zipCode);
                if (layerGroup) {{
                    map.removeLayer(layerGroup);
                    corridorLayers.delete(zipCode);
                }}
                activeCorridors.delete(zipCode);
            }} else {{
                // Add new corridor
                console.log(`Adding corridor for zip: ${{zipCode}}`);
                const corridorZips = calculateCorridorForZip(targetZip);

                if (corridorZips.length > 0) {{
                    const layerGroup = L.layerGroup();

                    corridorZips.forEach(zip => {{
                        const coords = getZipCentroid(zip);
                        if (coords[0] !== 0 || coords[1] !== 0) {{
                            const marker = L.circleMarker([coords[0], coords[1]], {{
                                radius: 4,
                                fillColor: '#FFA500', // Orange for corridor zips
                                color: '#FF8C00',
                                weight: 1,
                                opacity: 0.8,
                                fillOpacity: 0.6
                            }});

                            marker.bindPopup(`
                                <strong>Corridor Zip: ${{zip.properties.zipcode_clean}}</strong><br>
                                Quotes: ${{zip.properties.total_quotes}}<br>
                                Connecting to: ${{zipCode}}<br>
                                ${{zip.properties.has_cbsa ? 'CBSA: ' + zip.properties.cbsa_name : 'Non-CBSA'}}
                            `);

                            layerGroup.addLayer(marker);
                        }}
                    }});

                    layerGroup.addTo(map);
                    corridorLayers.set(zipCode, layerGroup);
                    activeCorridors.set(zipCode, corridorZips);
                }}
            }}
        }}

        function interpolateColor(color1, color2, factor) {{
            const result = color1.slice();
            for (let i = 0; i < 3; i++) {{
                result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
            }}
            return result;
        }}

        function rgbToHex(rgb) {{
            return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
        }}

        function getQuoteColor(feature) {{
            var props = feature.properties;
            var percentile = props.quote_percentile;

            if (props.has_cbsa) {{
                // Enhanced blue scale with more variation
                if (percentile < 0.2) {{
                    var color1 = [248, 251, 255];
                    var color2 = [198, 219, 239];
                    var localPercentile = percentile / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.5) {{
                    var color1 = [198, 219, 239];
                    var color2 = [107, 174, 214];
                    var localPercentile = (percentile - 0.2) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.8) {{
                    var color1 = [107, 174, 214];
                    var color2 = [49, 130, 189];
                    var localPercentile = (percentile - 0.5) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else {{
                    var color1 = [49, 130, 189];
                    var color2 = [8, 48, 107];
                    var localPercentile = (percentile - 0.8) / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }}
            }} else {{
                // Enhanced red scale with more variation
                if (percentile < 0.2) {{
                    var color1 = [255, 245, 240];
                    var color2 = [254, 224, 210];
                    var localPercentile = percentile / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.5) {{
                    var color1 = [254, 224, 210];
                    var color2 = [252, 146, 114];
                    var localPercentile = (percentile - 0.2) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.8) {{
                    var color1 = [252, 146, 114];
                    var color2 = [222, 45, 38];
                    var localPercentile = (percentile - 0.5) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else {{
                    var color1 = [222, 45, 38];
                    var color2 = [165, 15, 21];
                    var localPercentile = (percentile - 0.8) / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }}
            }}
        }}

        function getZipColor(zipFeature, isCorridor = false) {{
            if (isCorridor) {{
                // Corridor zip codes get orange/yellow gradient based on quote volume
                const percentile = zipFeature.properties.quote_percentile;
                if (percentile < 0.2) {{
                    var color1 = [255, 248, 220];  // Light yellow
                    var color2 = [255, 235, 59];   // Yellow
                    var localPercentile = percentile / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.5) {{
                    var color1 = [255, 235, 59];   // Yellow
                    var color2 = [255, 193, 7];    // Amber
                    var localPercentile = (percentile - 0.2) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.8) {{
                    var color1 = [255, 193, 7];    // Amber
                    var color2 = [255, 152, 0];    // Orange
                    var localPercentile = (percentile - 0.5) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else {{
                    var color1 = [255, 152, 0];    // Orange
                    var color2 = [230, 81, 0];     // Deep orange
                    var localPercentile = (percentile - 0.8) / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }}
            }}

            // Use original color scheme for CBSA and non-CBSA zips
            return getQuoteColor(zipFeature);
        }}

        function updateMap() {{
            // Clear all layers
            cbsaZipLayer.clearLayers();
            nonCbsaZipLayer.clearLayers();
            corridorZipLayer.clearLayers();

            // Get current filter settings
            const showCBSAZips = document.getElementById('showCBSAZips').checked;
            const showNonCBSAZips = document.getElementById('showNonCBSAZips').checked;
            const enableQuoteFilter = document.getElementById('enableQuoteFilter').checked;

            // Filter zip codes
            const filteredZips = zipData.filter(zip => {{
                if (removedZipCodes.has(zip.properties.zipcode_clean)) return false;

                const hasCbsa = zip.properties.has_cbsa;
                const distance = zip.properties.distance_to_cbsa;
                const stdDev = zip.properties.std_dev_below_cbsa_mean;

                if (hasCbsa) {{
                    return showCBSAZips;
                }} else {{
                    if (!showNonCBSAZips) return false;
                    if (distance > currentDistanceThreshold) return false;
                    if (enableQuoteFilter && stdDev > -currentStdDevThreshold) return false;
                    return true;
                }}
            }});

            // Separate CBSA and non-CBSA zips
            const cbsaZips = filteredZips.filter(zip => zip.properties.has_cbsa);
            const nonCbsaZips = filteredZips.filter(zip => !zip.properties.has_cbsa);

            // Individual corridor management - calculate on demand for fast loading

            // Add CBSA zip codes to map
            cbsaZips.forEach(zip => {{
                const layer = L.geoJSON(zip, {{
                    style: {{
                        fillColor: getZipColor(zip),
                        weight: zip.properties.was_unassigned ? 3 : 1,
                        opacity: 1,
                        color: zip.properties.was_unassigned ? '#ff8c00' : '#333',
                        dashArray: zip.properties.was_unassigned ? '5,5' : null,
                        fillOpacity: 0.7 + 0.3 * zip.properties.quote_percentile
                    }}
                }});

                // Add hover tooltip
                layer.bindTooltip(createHoverTooltip(zip), {{
                    permanent: false,
                    direction: 'top',
                    className: 'custom-tooltip'
                }});

                // Add click popup
                layer.bindPopup(createPopupContent(zip));

                cbsaZipLayer.addLayer(layer);
            }});

            // Add non-CBSA zip codes to map
            nonCbsaZips.forEach(zip => {{
                const layer = L.geoJSON(zip, {{
                    style: {{
                        fillColor: getZipColor(zip),
                        weight: zip.properties.was_unassigned ? 3 : 1,
                        opacity: 1,
                        color: zip.properties.was_unassigned ? '#ff8c00' : '#333',
                        dashArray: zip.properties.was_unassigned ? '5,5' : null,
                        fillOpacity: 0.7 + 0.3 * zip.properties.quote_percentile
                    }}
                }});

                // Add hover tooltip
                layer.bindTooltip(createHoverTooltip(zip), {{
                    permanent: false,
                    direction: 'top',
                    className: 'custom-tooltip'
                }});

                // Add click popup with corridor toggle option
                layer.bindPopup(createPopupContent(zip));

                // Add click handler for corridor toggle
                layer.on('click', function(e) {{
                    // Prevent event bubbling
                    L.DomEvent.stopPropagation(e);

                    // Toggle corridor for this zip code
                    toggleCorridorForZip(zip);

                    // Update popup content to reflect corridor status
                    const isActive = activeCorridors.has(zip.properties.zipcode_clean);
                    const corridorStatus = isActive ? 'ON' : 'OFF';
                    const corridorCount = isActive ? activeCorridors.get(zip.properties.zipcode_clean).length : 0;

                    const popupContent = createPopupContent(zip) + `
                        <hr style="margin: 10px 0;">
                        <div style="background: #f8f9fa; padding: 8px; border-radius: 4px;">
                            <strong>🛣️ Corridor Analysis: ${{corridorStatus}}</strong><br>
                            ${{isActive ? `Showing ${{corridorCount}} corridor zip codes` : 'Click to show corridor connections'}}
                        </div>
                    `;

                    layer.setPopupContent(popupContent);
                }});

                nonCbsaZipLayer.addLayer(layer);
            }});

            // Add corridor zip codes to map
            corridorZips.forEach(zip => {{
                const layer = L.geoJSON(zip, {{
                    style: {{
                        fillColor: getZipColor(zip, true),
                        weight: zip.properties.was_unassigned ? 3 : 2,  // Slightly thicker for corridor zips
                        opacity: 1,
                        color: zip.properties.was_unassigned ? '#ff8c00' : '#e65100',  // Orange border for corridor
                        dashArray: zip.properties.was_unassigned ? '5,5' : null,
                        fillOpacity: 0.7 + 0.3 * zip.properties.quote_percentile
                    }}
                }});

                // Add hover tooltip
                layer.bindTooltip(createHoverTooltip(zip), {{
                    permanent: false,
                    direction: 'top',
                    className: 'custom-tooltip'
                }});

                // Add click popup
                layer.bindPopup(createPopupContent(zip, true));

                corridorZipLayer.addLayer(layer);
            }});

            // Update statistics
            updateStatistics(cbsaZips.length, nonCbsaZips.length, corridorZips.length);
        }}

        function createHoverTooltip(zipFeature) {{
            const props = zipFeature.properties;
            let tooltip = `Zip: ${{props.zipcode_clean}} | Quotes: ${{props.total_quotes.toLocaleString()}}`;

            // Add CBSA mean quotes information
            if (props.cbsa_mean_quotes > 0) {{
                const cbsaType = props.has_cbsa ? 'CBSA' : 'Nearest CBSA';
                tooltip += ` | ${{cbsaType}} Avg: ${{props.cbsa_mean_quotes.toFixed(0)}}`;
            }}

            if (!props.has_cbsa) {{
                tooltip += ` | Dist: ${{props.distance_to_cbsa.toFixed(1)}}mi`;
            }}

            return tooltip;
        }}

        function createPopupContent(zipFeature, isCorridor = false) {{
            const props = zipFeature.properties;
            const zipType = isCorridor ? 'Corridor' : (props.has_cbsa ? 'CBSA' : 'Non-CBSA');

            let content = `
                <div style="font-family: Arial, sans-serif; max-width: 300px;">
                    <h3 style="margin: 0 0 10px 0; color: #333;">Zip: ${{props.zipcode_clean}} (${{zipType}})</h3>
                    <div style="line-height: 1.4;">
                        <strong>Location:</strong> ${{props.city || 'Unknown'}}, ${{props.state || 'Unknown'}}<br>
                        <strong>Total Quotes:</strong> ${{props.total_quotes || 0}}<br>
                        <strong>Population:</strong> ${{props.population ? props.population.toLocaleString() : 'N/A'}}<br>
            `;

            if (props.has_cbsa) {{
                content += `<strong>CBSA:</strong> ${{props.cbsa_name}}<br>`;
            }} else {{
                content += `
                        <strong>Distance to nearest CBSA zip:</strong> ${{props.distance_to_cbsa ? props.distance_to_cbsa.toFixed(1) + ' miles' : 'N/A'}}<br>
                        <strong>Nearest CBSA:</strong> ${{props.nearest_cbsa_name || 'N/A'}}<br>
                        <strong>Nearest CBSA average quotes:</strong> ${{props.cbsa_mean_quotes ? props.cbsa_mean_quotes.toFixed(0) : 'N/A'}}<br>
                        <strong>Quote performance vs CBSA:</strong> ${{props.std_dev_below_cbsa_mean !== null ? (-props.std_dev_below_cbsa_mean).toFixed(1) + ' std dev ' + (props.std_dev_below_cbsa_mean <= 0 ? 'above' : 'below') + ' mean' : 'N/A'}}<br>
                `;
            }}

            content += `
                    </div>
                </div>
            `;

            return content;
        }}



        function updateStatistics(cbsaCount, nonCbsaCount, corridorCount) {{
            let filterDescription = '';
            if (document.getElementById('enableQuoteFilter').checked) {{
                const stdDevSign = currentStdDevThreshold >= 0 ? '+' : '';
                filterDescription = '≤ ' + currentDistanceThreshold + ' miles & ≥ ' + stdDevSign + currentStdDevThreshold.toFixed(1) + ' std dev';
            }} else {{
                filterDescription = '≤ ' + currentDistanceThreshold + ' miles';
            }}

            let statsContent = `
                <strong>Currently Displayed:</strong><br>
                🔵 CBSA zip codes: ${{cbsaCount.toLocaleString()}}<br>
                🔴 Non-CBSA (${{filterDescription}}): ${{nonCbsaCount.toLocaleString()}}<br>
            `;

            // Show active corridor count
            const totalActiveCorridors = Array.from(activeCorridors.values()).reduce((sum, corridors) => sum + corridors.length, 0);
            if (totalActiveCorridors > 0) {{
                statsContent += `🟠 Active corridor zip codes: ${{totalActiveCorridors.toLocaleString()}}<br>`;
            }}

            const totalNonCBSA = zipData.filter(zip => !zip.properties.has_cbsa && !removedZipCodes.has(zip.properties.zipcode_clean)).length;
            const totalDisplayed = cbsaCount + nonCbsaCount + corridorCount;

            statsContent += `
                📊 Total non-CBSA zip codes: ${{totalNonCBSA.toLocaleString()}}<br>
                🎯 CBSA centroids: ${{cbsaCentroids.length}}<br>
                📍 Total displayed: ${{totalDisplayed.toLocaleString()}}
            `;

            document.getElementById('filterStats').innerHTML = statsContent;
        }}

        // Event handlers
        document.getElementById('distanceSlider').addEventListener('input', function(e) {{
            currentDistanceThreshold = parseInt(e.target.value);
            document.getElementById('distanceValue').textContent = currentDistanceThreshold + ' miles';
            updateMap();
        }});

        document.getElementById('stdDevSlider').addEventListener('input', function(e) {{
            currentStdDevThreshold = parseFloat(e.target.value) / 10;
            const sign = currentStdDevThreshold >= 0 ? '+' : '';
            document.getElementById('stdDevValue').textContent = '≥ ' + sign + currentStdDevThreshold.toFixed(1) + ' std dev';
            updateMap();
        }});

        document.getElementById('corridorWidthSlider').addEventListener('input', function(e) {{
            currentCorridorWidth = parseInt(e.target.value);
            document.getElementById('corridorWidthValue').textContent = currentCorridorWidth + ' miles';

            // Update any active corridors with the new width
            if (activeCorridors.size > 0) {{
                console.log(`Updating ${{activeCorridors.size}} active corridors with new width: ${{currentCorridorWidth}} miles`);
                // Recalculate all active corridors
                const activeZipCodes = Array.from(activeCorridors.keys());
                activeZipCodes.forEach(zipCode => {{
                    // Find the zip data for this zip code
                    const allZips = fullCbsaData.length > 0 ? fullCbsaData : zipData;
                    const targetZip = allZips.find(z => z.properties.zipcode_clean === zipCode);
                    if (targetZip) {{
                        // Remove old corridor and add new one with updated width
                        const layerGroup = corridorLayers.get(zipCode);
                        if (layerGroup) {{
                            map.removeLayer(layerGroup);
                        }}
                        activeCorridors.delete(zipCode);
                        corridorLayers.delete(zipCode);

                        // Add new corridor with updated width
                        toggleCorridorForZip(targetZip);
                    }}
                }});
            }}
        }});

        // Checkbox event handlers
        ['showCBSAZips', 'showNonCBSAZips', 'showCBSACentroids', 'showCrossdocks', 'enableQuoteFilter'].forEach(id => {{
            document.getElementById(id).addEventListener('change', function() {{
                if (id === 'showCBSACentroids') {{
                    if (this.checked) {{
                        map.addLayer(cbsaCentroidLayer);
                    }} else {{
                        map.removeLayer(cbsaCentroidLayer);
                    }}
                }} else if (id === 'showCrossdocks') {{
                    if (this.checked) {{
                        map.addLayer(crossdockLayer);
                    }} else {{
                        map.removeLayer(crossdockLayer);
                    }}
                }} else if (id === 'enableQuoteFilter') {{
                    const container = document.getElementById('quoteFilterContainer');
                    container.style.display = this.checked ? 'block' : 'none';
                    updateMap();
                }} else {{
                    updateMap();
                }}
            }});
        }});

        // Export function
        function exportFilteredData() {{
            const showCBSAZips = document.getElementById('showCBSAZips').checked;
            const showNonCBSAZips = document.getElementById('showNonCBSAZips').checked;
            const enableQuoteFilter = document.getElementById('enableQuoteFilter').checked;

            // Filter data for export
            const filteredZips = zipData.filter(zip => {{
                if (removedZipCodes.has(zip.properties.zipcode_clean)) return false;

                const hasCbsa = zip.properties.has_cbsa;
                const distance = zip.properties.distance_to_cbsa;
                const stdDev = zip.properties.std_dev_below_cbsa_mean;

                if (hasCbsa) {{
                    return showCBSAZips;
                }} else {{
                    if (!showNonCBSAZips) return false;
                    if (distance > currentDistanceThreshold) return false;
                    if (enableQuoteFilter && stdDev > -currentStdDevThreshold) return false;
                    return true;
                }}
            }});

            // Get active corridor zip codes
            const corridorZipCodes = new Set();
            const corridorTargets = new Map();
            activeCorridors.forEach((corridorZips, targetZipCode) => {{
                corridorZips.forEach(zip => {{
                    corridorZipCodes.add(zip.properties.zipcode_clean);
                    corridorTargets.set(zip.properties.zipcode_clean, targetZipCode);
                }});
            }});

            // Function to find nearest crossdock
            function findNearestCrossdock(zipLat, zipLon) {{
                let minDistance = Infinity;
                let nearestCrossdock = null;

                crossdockLocations.forEach(function(crossdock) {{
                    const distance = calculateDistance(zipLat, zipLon, crossdock.lat, crossdock.lon);
                    if (distance < minDistance) {{
                        minDistance = distance;
                        nearestCrossdock = crossdock;
                    }}
                }});

                return {{ distance: minDistance === Infinity ? null : minDistance, crossdock: nearestCrossdock }};
            }}

            // Prepare CSV data
            const csvData = [];

            // Add filtered zips
            filteredZips.forEach(zip => {{
                const props = zip.properties;

                // Calculate nearest crossdock
                const bounds = L.geoJSON(zip).getBounds();
                const centerLat = (bounds.getNorth() + bounds.getSouth()) / 2;
                const centerLon = (bounds.getEast() + bounds.getWest()) / 2;
                const nearestCrossdockInfo = findNearestCrossdock(centerLat, centerLon);

                // Check if this is a corridor zip and get target
                const isCorridorZip = corridorZipCodes.has(props.zipcode_clean);
                const corridorTargetZip = isCorridorZip ? corridorTargets.get(props.zipcode_clean) || '' : '';

                csvData.push({{
                    zip_code: props.zipcode_clean,
                    city: props.city,
                    state: props.state,
                    total_quotes: props.total_quotes,
                    pickup_count: props.pickup_count,
                    dropoff_count: props.dropoff_count,
                    population: props.population,
                    assigned_cbsa: props.assigned_cbsa || '',
                    closest_cbsa: props.closest_cbsa,
                    distance_to_nearest_cbsa_miles: props.distance_to_cbsa.toFixed(2),
                    nearest_cbsa_zip_code: props.nearest_cbsa_zip,
                    std_dev_vs_nearest_cbsa: (-props.std_dev_below_cbsa_mean).toFixed(2),
                    quote_percentile_in_nearest_cbsa: props.quote_percentile_in_nearest_cbsa.toFixed(1),
                    nearest_crossdock_zip: nearestCrossdockInfo.crossdock ? nearestCrossdockInfo.crossdock.zip_code : '',
                    nearest_crossdock_name: nearestCrossdockInfo.crossdock ? nearestCrossdockInfo.crossdock.name : '',
                    distance_to_nearest_crossdock_miles: nearestCrossdockInfo.distance ? nearestCrossdockInfo.distance.toFixed(2) : '',
                    was_recently_unassigned: props.was_unassigned ? 'Yes' : 'No',
                    is_corridor_zip: isCorridorZip ? 'Yes' : 'No',
                    corridor_target_zip: corridorTargetZip
                }});
            }});

            // Add active corridor zips if they're not already included
            activeCorridors.forEach((corridorZips, targetZipCode) => {{
                corridorZips.forEach(zip => {{
                    const props = zip.properties;
                    if (!filteredZips.some(fz => fz.properties.zipcode_clean === props.zipcode_clean)) {{
                        // Calculate nearest crossdock
                        const bounds = L.geoJSON(zip).getBounds();
                        const centerLat = (bounds.getNorth() + bounds.getSouth()) / 2;
                        const centerLon = (bounds.getEast() + bounds.getWest()) / 2;
                        const nearestCrossdockInfo = findNearestCrossdock(centerLat, centerLon);

                        const corridorTargetZip = targetZipCode;

                        csvData.push({{
                            zip_code: props.zipcode_clean,
                            city: props.city,
                            state: props.state,
                            total_quotes: props.total_quotes,
                            pickup_count: props.pickup_count,
                            dropoff_count: props.dropoff_count,
                            population: props.population,
                            assigned_cbsa: '',
                            closest_cbsa: props.closest_cbsa,
                            distance_to_nearest_cbsa_miles: props.distance_to_cbsa.toFixed(2),
                            nearest_cbsa_zip_code: props.nearest_cbsa_zip,
                            std_dev_vs_nearest_cbsa: (-props.std_dev_below_cbsa_mean).toFixed(2),
                            quote_percentile_in_nearest_cbsa: props.quote_percentile_in_nearest_cbsa.toFixed(1),
                            nearest_crossdock_zip: nearestCrossdockInfo.crossdock ? nearestCrossdockInfo.crossdock.zip_code : '',
                            nearest_crossdock_name: nearestCrossdockInfo.crossdock ? nearestCrossdockInfo.crossdock.name : '',
                            distance_to_nearest_crossdock_miles: nearestCrossdockInfo.distance ? nearestCrossdockInfo.distance.toFixed(2) : '',
                            was_recently_unassigned: props.was_unassigned ? 'Yes' : 'No',
                            is_corridor_zip: 'Yes',
                            corridor_target_zip: corridorTargetZip
                        }});
                    }}
                }});
            }});

            // Convert to CSV
            if (csvData.length === 0) {{
                alert('No data to export with current filters.');
                return;
            }}

            const headers = Object.keys(csvData[0]);
            const csvContent = [
                headers.join(','),
                ...csvData.map(row => headers.map(header => {{
                    const value = row[header];
                    return typeof value === 'string' && value.includes(',') ? `"${{value}}"` : value;
                }}).join(','))
            ].join('\\n');

            // Download CSV
            const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);

            const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
            link.setAttribute('download', `corridor_expansion_analysis_${{timestamp}}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        // Initialize CBSA centroids
        cbsaCentroids.forEach(centroid => {{
            const marker = L.circleMarker([centroid.lat, centroid.lon], {{
                radius: Math.max(4, Math.min(15, Math.log10(centroid.total_quotes) * 2)),
                fillColor: '#C73E1D',
                color: '#8B0000',
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.6
            }});

            marker.bindPopup(`
                <div style="font-family: Arial, sans-serif;">
                    <h3 style="margin: 0 0 10px 0; color: #333;">CBSA Centroid</h3>
                    <p><strong>CBSA:</strong> ${{centroid.cbsa_name}}</p>
                    <p><strong>Total Quotes:</strong> ${{centroid.total_quotes.toLocaleString()}}</p>
                    <p><strong>Zip Codes:</strong> ${{centroid.zip_count}}</p>
                    <p style="font-size: 12px; color: #666;">Quote-weighted center of business activity</p>
                </div>
            `);

            cbsaCentroidLayer.addLayer(marker);
        }});

        // Initialize crossdock locations
        crossdockLocations.forEach(crossdock => {{
            if (crossdock.lat && crossdock.lon) {{
                const marker = L.circleMarker([crossdock.lat, crossdock.lon], {{
                    radius: 8,
                    fillColor: '#28a745',
                    color: '#155724',
                    weight: 3,
                    opacity: 1.0,
                    fillOpacity: 0.9,
                    pane: 'crossdockPane'  // Use custom pane to ensure markers appear above polygons
                }});

                marker.bindPopup(`
                    <div style="font-family: Arial, sans-serif;">
                        <h3 style="margin: 0 0 10px 0; color: #333;">🏭 Crossdock Location</h3>
                        <p><strong>Name:</strong> ${{crossdock.name}}</p>
                        <p><strong>Zip Code:</strong> ${{crossdock.zip_code}}</p>
                    </div>
                `);

                crossdockLayer.addLayer(marker);
            }}
        }});

        // Initial map update
        updateMap();

    </script>
</body>
</html>"""

    return html_content

def create_corridor_expansion_map(map_data, cbsa_centroids_df, crossdock_locations, unassigned_zips, full_cbsa_data=None):
    """Create the interactive corridor expansion map with HTML"""
    print("Creating CBSA corridor expansion map...")

    # Convert to WGS84 for web mapping
    map_data_wgs84 = map_data.to_crs('EPSG:4326')

    # Simplify geometries for faster loading (reduce precision)
    print("Simplifying geometries for faster loading...")
    map_data_wgs84['geometry'] = map_data_wgs84['geometry'].simplify(tolerance=0.001, preserve_topology=True)

    # Convert full CBSA data for corridor analysis if provided
    if full_cbsa_data is not None:
        full_cbsa_data_wgs84 = full_cbsa_data.to_crs('EPSG:4326')
        # Simplify full CBSA data too
        full_cbsa_data_wgs84['geometry'] = full_cbsa_data_wgs84['geometry'].simplify(tolerance=0.001, preserve_topology=True)
    else:
        full_cbsa_data_wgs84 = None

    # Calculate separate percentiles for better color sensitivity
    cbsa_quotes = map_data_wgs84[map_data_wgs84['Primary CBSA Name'].notna()]['Total Quotes']
    non_cbsa_quotes = map_data_wgs84[map_data_wgs84['Primary CBSA Name'].isna()]['Total Quotes']

    print(f"CBSA quote range: {cbsa_quotes.min()} - {cbsa_quotes.max()}")
    print(f"Non-CBSA quote range: {non_cbsa_quotes.min()} - {non_cbsa_quotes.max()}")

    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        has_cbsa = pd.notna(row['Primary CBSA Name'])
        quotes = row['Total Quotes']

        # Calculate percentile for color mapping (logarithmic scale)
        if has_cbsa:
            if len(cbsa_quotes) > 1:
                percentile = (cbsa_quotes < quotes).mean()
            else:
                percentile = 0.5
        else:
            if len(non_cbsa_quotes) > 1:
                percentile = (non_cbsa_quotes < quotes).mean()
            else:
                percentile = 0.5

        # Simplify geometry to reduce file size (more aggressive for speed)
        simplified_geom = row['geometry']
        if hasattr(simplified_geom, 'simplify'):
            simplified_geom = simplified_geom.simplify(tolerance=0.01, preserve_topology=True)

        feature = {
            "type": "Feature",
            "properties": {
                "zipcode": row.get('Zipcode', row.get('zipcode_clean', 'Unknown')),
                "zipcode_clean": row['Zipcode_clean'],
                "cbsa_name": row.get('Primary CBSA Name') if has_cbsa else None,
                "city": row.get('City', 'Unknown'),
                "state": row.get('State', 'Unknown'),
                "total_quotes": float(row.get('Total Quotes', 0)),
                "pickup_count": float(row.get('Pickup Count', 0)),
                "dropoff_count": float(row.get('Dropoff Count', 0)),
                "population": int(row.get('ZCTA Population (2020)', 1000)) if pd.notna(row.get('ZCTA Population (2020)', 1000)) else 1000,
                "has_cbsa": has_cbsa,
                "quote_percentile": percentile,
                "was_unassigned": row['Zipcode_clean'] in unassigned_zips,
                "distance_to_cbsa": float(row.get('min_distance_to_cbsa', 0)),
                "nearest_cbsa_zip": row.get('nearest_cbsa_zip', ''),
                "nearest_cbsa_name": row.get('nearest_cbsa_name', ''),
                "std_dev_below_cbsa_mean": float(row.get('std_dev_below_cbsa_mean', 0)),
                "quote_percentile_in_nearest_cbsa": float(row.get('quote_percentile_in_nearest_cbsa', 0)),
                "assigned_cbsa": row.get('Primary CBSA Name') if has_cbsa else None,
                "closest_cbsa": row.get('nearest_cbsa_name', '') if not has_cbsa else row.get('Primary CBSA Name', ''),
                "cbsa_mean_quotes": float(row.get('cbsa_mean_quotes', 0))
            },
            "geometry": json.loads(gpd.GeoSeries([simplified_geom]).to_json())['features'][0]['geometry']
        }

        zip_features.append(feature)

    # Prepare CBSA centroid data for display
    cbsa_centroid_features = []
    for idx, row in cbsa_centroids_df.iterrows():
        cbsa_centroid_features.append({
            "cbsa_name": row['cbsa_name'],
            "lat": row['weighted_lat'],
            "lon": row['weighted_lon'],
            "total_quotes": row['total_quotes'],
            "zip_count": row['zip_count']
        })

    # Create full CBSA features for corridor analysis if available
    full_cbsa_features = []
    if full_cbsa_data_wgs84 is not None:
        for idx, row in full_cbsa_data_wgs84.iterrows():
            quotes = row['Total Quotes']

            # Calculate percentile for color mapping
            if len(cbsa_quotes) > 1:
                percentile = (cbsa_quotes < quotes).mean()
            else:
                percentile = 0.5

            # Simplify geometry to reduce file size (more aggressive for speed)
            simplified_geom = row['geometry']
            if hasattr(simplified_geom, 'simplify'):
                simplified_geom = simplified_geom.simplify(0.02, preserve_topology=True)

            full_cbsa_features.append({
                "type": "Feature",
                "properties": {
                    "zipcode_clean": row['Zipcode_clean'],
                    "city": row.get('City', 'Unknown'),
                    "state": row.get('State', 'Unknown'),
                    "total_quotes": float(quotes),
                    "pickup_count": float(row.get('Pickup Count', 0)),
                    "dropoff_count": float(row.get('Dropoff Count', 0)),
                    "population": int(row.get('ZCTA Population (2020)', 1000)) if pd.notna(row.get('ZCTA Population (2020)', 1000)) else 1000,
                    "has_cbsa": True,
                    "assigned_cbsa": row['Primary CBSA Name'],
                    "distance_to_cbsa": float(row.get('min_distance_to_cbsa', 0)),
                    "nearest_cbsa_zip": row.get('nearest_cbsa_zip', ''),
                    "nearest_cbsa_name": row.get('nearest_cbsa_name', ''),
                    "std_dev_below_cbsa_mean": float(row.get('std_dev_below_cbsa_mean', 0)),
                    "quote_percentile_in_nearest_cbsa": float(row.get('quote_percentile_in_nearest_cbsa', 0)),
                    "cbsa_mean_quotes": float(row.get('cbsa_mean_quotes', 0)),
                    "closest_cbsa": row.get('Primary CBSA Name', ''),
                    "was_unassigned": row['Zipcode_clean'] in unassigned_zips,
                    "quote_percentile": percentile
                },
                "geometry": json.loads(gpd.GeoSeries([simplified_geom]).to_json())['features'][0]['geometry']
            })

    # Generate HTML with corridor analysis features
    html_content = generate_corridor_html(zip_features, cbsa_centroid_features, crossdock_locations, full_cbsa_features)

    # Save the HTML file
    output_file = 'cbsa_corridor_expansion_map.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nCBSA corridor expansion map saved as '{output_file}'")

    # Open in browser
    print("Opening map in browser...")
    webbrowser.open(f'file://{os.path.abspath(output_file)}')

    print("\n" + "="*60)
    print("CBSA CORRIDOR EXPANSION MAP FEATURES:")
    print("✅ Quote-weighted CBSA centroids for accurate distance calculation")
    print("✅ Corridor analysis for isolated high-opportunity zip codes")
    print("✅ Different colors for corridor vs. direct opportunity zip codes")
    print("✅ Toggle corridor analysis on/off")
    print("✅ Adjustable corridor width (5-50 miles)")
    print("✅ Enhanced export with corridor classification")
    print("="*60)

def main():
    """Main function to create the corridor expansion map"""
    # Load all data
    data, unassigned_zips = load_data()
    zip_shapes = load_shapefiles()

    # Load crossdock data
    crossdock_data = load_crossdock_data()
    crossdock_locations = process_crossdock_locations(crossdock_data, zip_shapes)

    # Calculate weighted centroids and distances
    map_data, cbsa_centroids_df, full_cbsa_data = calculate_weighted_cbsa_centroids(data, zip_shapes)
    map_data = calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df)

    # Create the corridor expansion map
    create_corridor_expansion_map(map_data, cbsa_centroids_df, crossdock_locations, unassigned_zips, full_cbsa_data)

if __name__ == "__main__":
    main()
