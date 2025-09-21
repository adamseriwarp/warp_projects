import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from pathlib import Path
from sklearn.neighbors import BallTree
import hashlib

# FIX 4: Configurable paths instead of hard-coded absolute paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DOWNLOADS_DIR = Path.home() / 'Downloads'  # More portable than hard-coded path

def create_shareable_version():
    """Create a version with embedded sample data for sharing"""
    print("Creating shareable version with embedded sample data...")

    # Sample data for demonstration (replace with actual aggregated/anonymized data)
    sample_zip_features = [
        {
            "type": "Feature",
            "properties": {
                "zipcode_clean": "90210",
                "cbsa_name": "Los Angeles-Long Beach-Anaheim, CA",
                "total_quotes": 150,
                "has_cbsa": True,
                "cbsa_color": "#4ecdc4",
                "distance_to_cbsa": 0,
                "std_dev_from_cbsa_mean": 0.5
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-118.4, 34.1], [-118.3, 34.1], [-118.3, 34.0], [-118.4, 34.0], [-118.4, 34.1]]]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "zipcode_clean": "93001",
                "cbsa_name": None,
                "total_quotes": 45,
                "has_cbsa": False,
                "cbsa_color": "#FF4444",
                "distance_to_cbsa": 25.5,
                "std_dev_from_cbsa_mean": -1.2
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-119.3, 34.3], [-119.2, 34.3], [-119.2, 34.2], [-119.3, 34.2], [-119.3, 34.3]]]
            }
        }
    ]

    sample_cbsa_centroids = [
        {
            "cbsa_name": "Los Angeles-Long Beach-Anaheim, CA",
            "lat": 34.0522,
            "lon": -118.2437,
            "total_quotes": 15000,
            "zip_count": 150
        }
    ]

    sample_crossdocks = [
        {
            "zip_code": "90021",
            "name": "LAX Crossdock",
            "lat": 34.0291,
            "lon": -118.2389
        }
    ]

    return sample_zip_features, sample_cbsa_centroids, sample_crossdocks, []

def load_ltl_cbsas():
    """Load LTL Rate File zip codes and map them to CBSAs"""
    print("Loading LTL Rate File zip codes and mapping to CBSAs...")
    
    # Load LTL Rate File
    ltl_file = DATA_DIR / 'raw' / 'LTL Rate File 07-07-2025 - Zips.csv'
    try:
        ltl_data = pd.read_csv(ltl_file)
        print(f"Loaded LTL Rate File with {len(ltl_data)} zip codes")
        print(f"LTL file columns: {ltl_data.columns.tolist()}")
        
        # Clean zip codes to 5 digits
        ltl_data['Zip_clean'] = ltl_data['Zip'].astype(str).str.zfill(5)
        ltl_zip_codes = ltl_data['Zip_clean'].unique().tolist()
        print(f"Found {len(ltl_zip_codes)} unique LTL zip codes")
        
    except Exception as e:
        print(f"Error loading LTL Rate File: {e}")
        return []
    
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
    
    # Map LTL zip codes to CBSAs
    ltl_cbsa_mapping = cbsa_mapping[cbsa_mapping['Zip Code_clean'].isin(ltl_zip_codes)]
    ltl_cbsas = ltl_cbsa_mapping[ltl_cbsa_mapping['Primary CBSA Name'].notna()]['Primary CBSA Name'].unique().tolist()
    
    print(f"LTL zip codes map to {len(ltl_cbsas)} unique CBSAs:")
    for i, cbsa in enumerate(ltl_cbsas[:10], 1):
        zip_count = len(ltl_cbsa_mapping[ltl_cbsa_mapping['Primary CBSA Name'] == cbsa])
        print(f"  {i}. {cbsa}: {zip_count} LTL zip codes")
    if len(ltl_cbsas) > 10:
        print(f"  ... and {len(ltl_cbsas) - 10} more CBSAs")
    
    return ltl_cbsas

def load_and_prepare_data():
    """Load and prepare the data for heat mapping"""
    print("Loading data...")
    
    # Load quote data
    quote_data = pd.read_csv(DATA_DIR / 'raw' / 'quote_data.csv')
    quote_data['Pickup Count'] = pd.to_numeric(quote_data['Pickup Count'], errors='coerce').fillna(0)
    quote_data['Dropoff Count'] = pd.to_numeric(quote_data['Dropoff Count'], errors='coerce').fillna(0)
    quote_data['Total Quotes'] = quote_data['Pickup Count'] + quote_data['Dropoff Count']
    quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
    
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
    
    if unassigned_zips_clean:
        print(f"Unassigning {len(unassigned_zips_clean)} zip codes from their CBSAs...")
        
        # Create updated CBSA mapping by removing CBSA assignments for unassigned zip codes
        updated_cbsa_mapping = cbsa_mapping.copy()
        mask = updated_cbsa_mapping['Zip Code_clean'].isin(unassigned_zips_clean)
        updated_cbsa_mapping.loc[mask, 'Primary CBSA'] = None
        updated_cbsa_mapping.loc[mask, 'Primary CBSA Name'] = None
        
        print(f"Updated {mask.sum()} zip codes to remove CBSA assignments")
    else:
        updated_cbsa_mapping = cbsa_mapping.copy()
        print("No zip codes to unassign - using original CBSA assignments")
    
    # Merge with quote data
    merged_data = pd.merge(
        quote_data, 
        updated_cbsa_mapping, 
        left_on='Zipcode_clean', 
        right_on='Zip Code_clean', 
        how='left'
    )
    
    # Load shapefiles
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file(DATA_DIR / 'shapefiles' / 'cb_2020_us_zcta520_500k.shp')
    zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)
    
    return merged_data, zip_shapes, unassigned_zips_clean

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
        csv_path = DATA_DIR / 'raw' / 'WARP_xdock_2025.csv'
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

def generate_cbsa_color(cbsa_name):
    """Generate a consistent color for each CBSA based on its name"""
    if not cbsa_name or pd.isna(cbsa_name):
        return '#FF4444'  # Red for non-CBSA

    # Use hash of CBSA name to generate consistent color
    hash_object = hashlib.md5(cbsa_name.encode())
    hex_dig = hash_object.hexdigest()

    # Extract RGB values from hash
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)

    # Avoid red colors for CBSA (reserve red for non-CBSA)
    # If the color is too red, shift it to blue/green spectrum
    if r > g + 50 and r > b + 50:  # Too red
        # Shift red component to blue
        b = r
        r = min(r // 2, 100)  # Reduce red significantly

    # Ensure colors are vibrant and distinguishable
    # Boost saturation and brightness
    max_val = max(r, g, b)
    if max_val < 128:
        # Boost all values proportionally
        factor = 200 / max_val
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))

    return f"#{r:02x}{g:02x}{b:02x}"

def calculate_weighted_cbsa_centroids(data, zip_shapes):
    """Calculate quote-weighted centroids for each CBSA - MODIFIED to use LTL CBSAs instead of top 75 by population"""
    print("Calculating quote-weighted CBSA centroids...")

    # MODIFIED: Use LTL CBSAs instead of top 75 by population
    print("Determining CBSAs from LTL Rate File zip codes...")
    ltl_cbsas = load_ltl_cbsas()

    if not ltl_cbsas:
        print("ERROR: No LTL CBSAs found! Check LTL Rate File and mapping.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Using {len(ltl_cbsas)} CBSAs from LTL Rate File (instead of top 75 by population)")

    # NOW filter for zip codes with quotes for the actual mapping
    quote_data = data[data['Total Quotes'] > 0]

    # Merge with shapefiles
    map_data = zip_shapes.merge(
        quote_data,
        left_on='ZCTA5CE20',
        right_on='Zipcode_clean',
        how='inner'
    )

    # Filter map data to only include LTL CBSAs (but keep all non-CBSA zip codes)
    map_data = map_data[
        (map_data['Primary CBSA Name'].isin(ltl_cbsas)) |
        (map_data['Primary CBSA Name'].isna())
    ]

    print(f"Filtered to {len(map_data)} zip codes in LTL CBSAs or non-CBSA areas")

    # Calculate centroids for each zip code
    map_data_projected = map_data.to_crs('EPSG:3857')  # Web Mercator
    centroids = map_data_projected.geometry.centroid.to_crs('EPSG:4326')  # Back to lat/lon

    map_data['centroid_lat'] = centroids.y
    map_data['centroid_lon'] = centroids.x

    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    non_cbsa_zips = map_data[map_data['Primary CBSA Name'].isna()].copy()

    print(f"Calculating weighted centroids for {cbsa_zips['Primary CBSA Name'].nunique()} LTL CBSAs...")

    # Calculate quote-weighted centroids for each CBSA
    cbsa_centroids = []

    for cbsa_name in cbsa_zips['Primary CBSA Name'].unique():
        cbsa_data = cbsa_zips[cbsa_zips['Primary CBSA Name'] == cbsa_name]

        # Calculate weighted centroid based on quote volume
        total_quotes = cbsa_data['Total Quotes'].sum()
        if total_quotes > 0:
            weighted_lat = (cbsa_data['centroid_lat'] * cbsa_data['Total Quotes']).sum() / total_quotes
            weighted_lon = (cbsa_data['centroid_lon'] * cbsa_data['Total Quotes']).sum() / total_quotes
        else:
            # If no quotes, use simple average
            weighted_lat = cbsa_data['centroid_lat'].mean()
            weighted_lon = cbsa_data['centroid_lon'].mean()

        cbsa_centroids.append({
            'cbsa_name': cbsa_name,
            'weighted_lat': weighted_lat,
            'weighted_lon': weighted_lon,
            'total_quotes': total_quotes,
            'zip_count': len(cbsa_data)
        })

    cbsa_centroids_df = pd.DataFrame(cbsa_centroids)

    print(f"Calculated weighted centroids for {len(cbsa_centroids_df)} LTL CBSAs")
    print(f"Quote range across LTL CBSAs: {cbsa_centroids_df['total_quotes'].min():,.0f} - {cbsa_centroids_df['total_quotes'].max():,.0f}")

    return map_data, cbsa_centroids_df

def calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df):
    """Calculate distances from non-CBSA zip codes to nearest individual CBSA zip codes"""
    print("Calculating distances to nearest CBSA zip codes (adjacency analysis)...")

    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    non_cbsa_zips = map_data[map_data['Primary CBSA Name'].isna()].copy()

    if len(cbsa_zips) == 0 or len(non_cbsa_zips) == 0:
        print("No CBSA zip codes or non-CBSA zip codes found!")
        map_data['min_distance_to_cbsa'] = 0
        map_data['nearest_cbsa_zip'] = None
        map_data['nearest_cbsa_name'] = None
        return map_data

    print(f"Calculating distances from {len(non_cbsa_zips)} non-CBSA zip codes to {len(cbsa_zips)} CBSA zip codes...")

    # Convert individual CBSA zip code coordinates to radians for BallTree
    cbsa_coords_rad = np.radians(cbsa_zips[['centroid_lat', 'centroid_lon']].values)
    non_cbsa_coords_rad = np.radians(non_cbsa_zips[['centroid_lat', 'centroid_lon']].values)

    # Build BallTree for efficient nearest neighbor search
    tree = BallTree(cbsa_coords_rad, metric='haversine')

    # Find nearest CBSA zip code for each non-CBSA zip code
    distances_rad, indices = tree.query(non_cbsa_coords_rad, k=1)

    # Convert distances from radians to miles
    earth_radius_miles = 3959
    distances_miles = distances_rad.flatten() * earth_radius_miles

    # Get the details of the nearest CBSA zip codes
    nearest_cbsa_zips = cbsa_zips.iloc[indices.flatten()]
    nearest_cbsa_zip_codes = nearest_cbsa_zips['Zipcode_clean'].values
    nearest_cbsa_names = nearest_cbsa_zips['Primary CBSA Name'].values

    # Calculate statistical metrics for each CBSA
    print("Calculating quote statistics for each CBSA...")
    cbsa_stats = {}
    for cbsa_name in cbsa_zips['Primary CBSA Name'].unique():
        cbsa_data = cbsa_zips[cbsa_zips['Primary CBSA Name'] == cbsa_name]['Total Quotes']
        cbsa_stats[cbsa_name] = {
            'mean': cbsa_data.mean(),
            'std': cbsa_data.std(),
            'median': cbsa_data.median(),
            'count': len(cbsa_data)
        }

    # Calculate standard deviation metrics for non-CBSA zip codes
    std_deviations_from_mean = []
    quote_percentiles_in_nearest_cbsa = []
    cbsa_mean_quotes = []  # Add CBSA mean quotes for tooltips

    for idx, (_, row) in enumerate(non_cbsa_zips.iterrows()):
        nearest_cbsa = nearest_cbsa_names[idx]
        zip_quotes = row['Total Quotes']

        if nearest_cbsa in cbsa_stats:
            cbsa_mean = cbsa_stats[nearest_cbsa]['mean']
            cbsa_std = cbsa_stats[nearest_cbsa]['std']

            # Calculate how many standard deviations from the mean (positive = above mean)
            if cbsa_std > 0:
                std_dev_from_mean = (zip_quotes - cbsa_mean) / cbsa_std
            else:
                std_dev_from_mean = 0

            # Calculate percentile within the nearest CBSA
            cbsa_quotes = cbsa_zips[cbsa_zips['Primary CBSA Name'] == nearest_cbsa]['Total Quotes']
            percentile = (cbsa_quotes < zip_quotes).mean() * 100

            std_deviations_from_mean.append(std_dev_from_mean)
            quote_percentiles_in_nearest_cbsa.append(percentile)
            cbsa_mean_quotes.append(cbsa_mean)
        else:
            std_deviations_from_mean.append(0)
            quote_percentiles_in_nearest_cbsa.append(0)
            cbsa_mean_quotes.append(0)

    # Add distances and statistical info to non-CBSA zip codes
    non_cbsa_zips['min_distance_to_cbsa'] = distances_miles
    non_cbsa_zips['nearest_cbsa_zip'] = nearest_cbsa_zip_codes
    non_cbsa_zips['nearest_cbsa_name'] = nearest_cbsa_names
    non_cbsa_zips['std_dev_from_cbsa_mean'] = std_deviations_from_mean
    non_cbsa_zips['quote_percentile_in_nearest_cbsa'] = quote_percentiles_in_nearest_cbsa
    non_cbsa_zips['cbsa_mean_quotes'] = cbsa_mean_quotes

    # Add zero distance for CBSA zip codes (they are at distance 0 from themselves)
    cbsa_zips['min_distance_to_cbsa'] = 0
    cbsa_zips['nearest_cbsa_zip'] = cbsa_zips['Zipcode_clean']
    cbsa_zips['nearest_cbsa_name'] = cbsa_zips['Primary CBSA Name']
    cbsa_zips['std_dev_from_cbsa_mean'] = 0  # CBSA zips are part of their own CBSA
    cbsa_zips['quote_percentile_in_nearest_cbsa'] = 0  # Will be calculated separately if needed

    # Add CBSA mean quotes for CBSA-assigned zip codes
    cbsa_zips['cbsa_mean_quotes'] = cbsa_zips['Primary CBSA Name'].map(
        lambda cbsa_name: cbsa_stats.get(cbsa_name, {}).get('mean', 0)
    )

    # Combine back together
    map_data = pd.concat([cbsa_zips, non_cbsa_zips], ignore_index=True)

    print(f"Adjacency analysis complete. Distance range: {map_data['min_distance_to_cbsa'].min():.1f} - {map_data['min_distance_to_cbsa'].max():.1f} miles")

    return map_data

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

def load_all_usa_zip_data_for_corridors(merged_data, zip_shapes):
    """Load ALL zip codes in the USA for comprehensive corridor analysis"""
    print("Loading ALL USA zip codes for corridor analysis...")

    # Merge ALL zip shapes with the merged quotes/CBSA data
    all_usa_data = zip_shapes.merge(
        merged_data,
        left_on='ZCTA5CE20',
        right_on='Zipcode_clean',
        how='left'
    )

    # Fill missing quote data
    all_usa_data['Total Quotes'] = all_usa_data['Total Quotes'].fillna(0)
    all_usa_data['Primary CBSA Name'] = all_usa_data['Primary CBSA Name'].fillna('')

    print(f"Loaded {len(all_usa_data)} zip codes from entire USA for corridor analysis")
    return all_usa_data

def prepare_all_usa_data_for_js(all_usa_data):
    """Prepare ALL USA zip code data for JavaScript corridor analysis"""
    print("Preparing ALL USA zip codes for JavaScript...")

    # Convert to WGS84 for web display
    all_usa_wgs84 = all_usa_data.to_crs('EPSG:4326')

    features = []
    for idx, row in all_usa_wgs84.iterrows():
        # Simplify geometry for web display
        simplified_geom = row['geometry']
        if hasattr(simplified_geom, 'simplify'):
            simplified_geom = simplified_geom.simplify(tolerance=0.001, preserve_topology=True)

        features.append({
            "type": "Feature",
            "properties": {
                "zipcode_clean": row['ZCTA5CE20'],
                "total_quotes": int(row['Total Quotes']) if pd.notna(row['Total Quotes']) else 0,
                "cbsa_name": row['Primary CBSA Name'] if pd.notna(row['Primary CBSA Name']) else '',
                "has_cbsa": pd.notna(row['Primary CBSA Name']) and row['Primary CBSA Name'] != ''
            },
            "geometry": json.loads(gpd.GeoSeries([simplified_geom]).to_json())['features'][0]['geometry']
        })

    print(f"Prepared {len(features)} USA zip codes for corridor analysis")
    return features

# IMPROVED CORRIDOR ANALYSIS FUNCTIONS
def calculate_improved_corridors(filtered_non_cbsa_zips, all_usa_data, max_corridor_distance=150, base_corridor_width=25):
    """
    Improved corridor analysis with distance limits and dynamic width

    Parameters:
    - filtered_non_cbsa_zips: Non-CBSA zip codes that pass current filters
    - all_usa_data: All USA zip codes for corridor calculation
    - max_corridor_distance: Maximum distance for corridor analysis (miles)
    - base_corridor_width: Base corridor width that gets adjusted based on distance
    """
    print(f"🚀 IMPROVED Corridor Analysis:")
    print(f"   - Max corridor distance: {max_corridor_distance} miles")
    print(f"   - Base corridor width: {base_corridor_width} miles")
    print(f"   - Analyzing {len(filtered_non_cbsa_zips)} filtered non-CBSA zip codes")

    if len(filtered_non_cbsa_zips) == 0:
        print("No filtered non-CBSA zip codes found - no corridors to calculate")
        return []

    # Get all CBSA-assigned zip codes
    assigned_cbsa_zips = [zip_data for zip_data in all_usa_data if zip_data['properties']['has_cbsa']]

    if len(assigned_cbsa_zips) == 0:
        print("No assigned CBSA zips available for corridor analysis")
        return []

    print(f"   - Found {len(assigned_cbsa_zips)} CBSA-assigned zip codes for corridor targets")

    all_corridor_zips = set()  # Use Set to avoid duplicates
    corridor_stats = {
        'total_corridors': 0,
        'corridors_within_limit': 0,
        'avg_corridor_distance': 0,
        'avg_corridor_width': 0
    }

    # Helper function to get zip centroid
    def get_zip_centroid(zip_feature):
        if zip_feature['geometry']['type'] == 'Polygon':
            coords = zip_feature['geometry']['coordinates'][0]
            lat = sum(coord[1] for coord in coords) / len(coords)
            lon = sum(coord[0] for coord in coords) / len(coords)
            return [lat, lon]
        elif zip_feature['geometry']['type'] == 'MultiPolygon':
            coords = zip_feature['geometry']['coordinates'][0][0]
            lat = sum(coord[1] for coord in coords) / len(coords)
            lon = sum(coord[0] for coord in coords) / len(coords)
            return [lat, lon]
        return [0, 0]

    # Helper function to calculate distance
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 3959  # Earth's radius in miles
        dLat = (lat2 - lat1) * np.pi / 180
        dLon = (lon2 - lon1) * np.pi / 180
        a = (np.sin(dLat/2) * np.sin(dLat/2) +
             np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) *
             np.sin(dLon/2) * np.sin(dLon/2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    # Helper function to calculate point-to-line distance
    def point_to_line_distance(px, py, x1, y1, x2, y2):
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            return calculate_distance(py, px, y1, x1)  # Note: lat/lon order

        t = max(0, min(1, dot / len_sq))
        proj_x = x1 + t * C
        proj_y = y1 + t * D

        return calculate_distance(py, px, proj_y, proj_x)  # Note: lat/lon order

    corridor_distances = []
    corridor_widths = []

    # Process each filtered non-CBSA zip code
    for target_zip in filtered_non_cbsa_zips:
        corridor_stats['total_corridors'] += 1
        target_centroid = get_zip_centroid(target_zip)

        # Find nearest CBSA zip code
        nearest_cbsa_zip = None
        min_distance = float('inf')

        for cbsa_zip in assigned_cbsa_zips:
            cbsa_centroid = get_zip_centroid(cbsa_zip)
            distance = calculate_distance(
                target_centroid[0], target_centroid[1],
                cbsa_centroid[0], cbsa_centroid[1]
            )

            if distance < min_distance:
                min_distance = distance
                nearest_cbsa_zip = cbsa_zip

        # Skip if corridor is too long
        if min_distance > max_corridor_distance:
            print(f"   ⏭️  Skipping corridor for {target_zip['properties']['zipcode_clean']} - distance {min_distance:.1f} miles exceeds limit")
            continue

        corridor_stats['corridors_within_limit'] += 1
        corridor_distances.append(min_distance)

        # Calculate dynamic corridor width based on distance
        # Shorter corridors can be wider, longer corridors should be narrower
        distance_factor = min_distance / max_corridor_distance  # 0 to 1
        dynamic_width = base_corridor_width * (1.5 - distance_factor)  # 1.5x width for short, 0.5x for long
        dynamic_width = max(10, min(dynamic_width, base_corridor_width * 2))  # Clamp between 10 and 2x base

        corridor_widths.append(dynamic_width)

        if nearest_cbsa_zip:
            nearest_centroid = get_zip_centroid(nearest_cbsa_zip)

            print(f"   🛣️  Creating corridor: {target_zip['properties']['zipcode_clean']} → {nearest_cbsa_zip['properties']['zipcode_clean']} ({min_distance:.1f} mi, width: {dynamic_width:.1f} mi)")

            # Find all zip codes within dynamic corridor width of the line
            for zip_data in all_usa_data:
                # Skip if already processed or if it's the target/destination
                zip_code = zip_data['properties']['zipcode_clean']
                if (zip_code in all_corridor_zips or
                    zip_code == target_zip['properties']['zipcode_clean'] or
                    zip_code == nearest_cbsa_zip['properties']['zipcode_clean']):
                    continue

                zip_centroid = get_zip_centroid(zip_data)
                distance_to_line = point_to_line_distance(
                    zip_centroid[1], zip_centroid[0],  # lon, lat for point
                    target_centroid[1], target_centroid[0],  # lon, lat for line start
                    nearest_centroid[1], nearest_centroid[0]  # lon, lat for line end
                )

                if distance_to_line <= dynamic_width:
                    # Check if the zip is roughly between the target and CBSA (not too far out of the way)
                    distance_to_target = calculate_distance(
                        zip_centroid[0], zip_centroid[1],
                        target_centroid[0], target_centroid[1]
                    )
                    distance_to_cbsa = calculate_distance(
                        zip_centroid[0], zip_centroid[1],
                        nearest_centroid[0], nearest_centroid[1]
                    )

                    # Only include if the zip is roughly on the path (allow 30% detour)
                    if distance_to_target + distance_to_cbsa <= min_distance * 1.3:
                        all_corridor_zips.add(zip_code)

    # Calculate statistics
    if corridor_distances:
        corridor_stats['avg_corridor_distance'] = np.mean(corridor_distances)
        corridor_stats['avg_corridor_width'] = np.mean(corridor_widths)

    # Convert Set back to array of zip objects
    corridor_zip_objects = []
    for zip_code in all_corridor_zips:
        zip_obj = next((zip_data for zip_data in all_usa_data if zip_data['properties']['zipcode_clean'] == zip_code), None)
        if zip_obj:
            corridor_zip_objects.append(zip_obj)

    print(f"✅ IMPROVED Corridor Analysis Results:")
    print(f"   - Total potential corridors: {corridor_stats['total_corridors']}")
    print(f"   - Corridors within {max_corridor_distance} mile limit: {corridor_stats['corridors_within_limit']}")
    print(f"   - Average corridor distance: {corridor_stats['avg_corridor_distance']:.1f} miles")
    print(f"   - Average corridor width: {corridor_stats['avg_corridor_width']:.1f} miles")
    print(f"   - Total corridor zip codes found: {len(corridor_zip_objects)}")

    return corridor_zip_objects

def create_cbsa_proximity_map(map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data=None):
    """Create a CBSA proximity map with CBSA-based coloring instead of quote density"""
    print("Creating CBSA proximity map with CBSA-based coloring...")

    # Convert to WGS84 for web display
    map_data_wgs84 = map_data.to_crs('EPSG:4326')

    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        has_cbsa = pd.notna(row['Primary CBSA Name'])

        # Generate color based on CBSA instead of quote density
        if has_cbsa:
            color = generate_cbsa_color(row['Primary CBSA Name'])
        else:
            color = '#FF4444'  # Red for non-CBSA zip codes

        # Simplify geometry to reduce file size
        simplified_geom = row['geometry']
        if hasattr(simplified_geom, 'simplify'):
            simplified_geom = simplified_geom.simplify(tolerance=0.001, preserve_topology=True)

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
                "population": int(row.get('ZCTA Population (2020)', 0)) if pd.notna(row.get('ZCTA Population (2020)', 0)) else 0,
                "has_cbsa": has_cbsa,
                "cbsa_color": color,  # Store the CBSA color
                "was_unassigned": row['Zipcode_clean'] in unassigned_zips,
                "is_unassigned_cbsa": row['Zipcode_clean'] in unassigned_zips and has_cbsa,
                "distance_to_cbsa": float(row.get('min_distance_to_cbsa', 0)),
                "nearest_cbsa_zip": row.get('nearest_cbsa_zip', ''),
                "nearest_cbsa_name": row.get('nearest_cbsa_name', ''),
                "std_dev_from_cbsa_mean": float(row.get('std_dev_from_cbsa_mean', 0)),
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

    return zip_features, cbsa_centroid_features

def create_complete_html_map(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js):
    """Create the complete HTML map with CBSA-based coloring instead of quote density"""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LTL CBSAs Color-Coded Map with Smart Corridors</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 320px;
            max-width: 380px;
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
        .slider-wrapper {{
            position: relative;
            margin: 10px 0;
        }}
        #distanceSlider {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(to right, #ff6b6b, #feca57, #48dbfb, #0abde3);
            outline: none;
            -webkit-appearance: none;
            cursor: pointer;
        }}
        #distanceSlider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
            border: 3px solid white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        .distance-display {{
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #007cba;
            margin: 10px 0;
            padding: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }}
        .range-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
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
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 350px;
            font-size: 12px;
        }}
        .color-scale {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .color-box {{
            width: 20px;
            height: 15px;
            margin-right: 8px;
            border: 1px solid #333;
        }}
        .improvement-badge {{
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="controls">
        <h4>🎨 <span class="improvement-badge">LTL CBSAs</span> Color-Coded Analysis</h4>

        <div class="slider-container">
            <label for="distanceSlider">Distance to Nearest CBSA Coverage:</label>
            <div class="slider-wrapper">
                <input type="range" id="distanceSlider" min="5" max="200" value="50" step="5">
            </div>
            <div class="distance-display" id="distanceDisplay">50 miles</div>
            <div class="range-labels">
                <span>5 miles</span>
                <span>200 miles</span>
            </div>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showAllCBSA" checked>
            <label for="showAllCBSA">Show all LTL CBSA zip codes</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showNearbyNonCBSA" checked>
            <label for="showNearbyNonCBSA">Show nearby non-CBSA zip codes</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCentroids" checked>
            <label for="showCentroids">Show CBSA weighted centroids</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCrossdocks" checked>
            <label for="showCrossdocks">Show crossdock locations</label>
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="enableStdDevFilter" checked>
            <label for="enableStdDevFilter">Enable quote performance filter</label>
        </div>

        <div class="slider-container" id="stdDevContainer">
            <label for="stdDevSlider">Min Quote Performance vs Nearest CBSA:</label>
            <div class="slider-wrapper">
                <input type="range" id="stdDevSlider" min="-3" max="2" value="-3" step="0.5">
            </div>
            <div class="distance-display" id="stdDevDisplay" style="font-size: 16px;">≥ -3.0 std dev</div>
            <div class="range-labels">
                <span>-3 std dev (Poor)</span>
                <span>+2 std dev (Excellent)</span>
            </div>
        </div>

        <div class="stats" id="filterStats">
            Loading...
        </div>

        <div class="checkbox-container" style="background: #e8f5e8; border: 2px solid #28a745;">
            <input type="checkbox" id="showSmartCorridors">
            <label for="showSmartCorridors">🚀 Show SMART corridor zip codes</label>
        </div>

        <div style="margin-top: 8px;">
            <label for="maxCorridorDistance">Max Corridor Distance:</label>
            <input type="range" id="maxCorridorDistance" min="50" max="300" value="150" step="25" style="width: 100%;">
            <div id="maxCorridorDistanceDisplay" style="text-align: center; font-weight: bold;">150 miles</div>
        </div>

        <div style="margin-top: 8px;">
            <label for="baseCorridorWidth">Base Corridor Width:</label>
            <input type="range" id="baseCorridorWidth" min="10" max="50" value="25" step="5" style="width: 100%;">
            <div id="baseCorridorWidthDisplay" style="text-align: center; font-weight: bold;">25 miles</div>
        </div>

        <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 12px; margin: 10px 0;">
            <strong>🎨 LTL CBSA Features:</strong><br>
            ✅ Uses CBSAs from LTL Rate File zip codes<br>
            ✅ Each CBSA has a unique color for easy distinction<br>
            ✅ No quote density heat mapping - cleaner visualization<br>
            ✅ Red color for non-CBSA zip codes<br>
            ✅ All other functionality preserved
        </div>

        <button onclick="exportFilteredData()" style="width: 100%; padding: 10px; margin-top: 15px; background: #007cba; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
            📥 Export Filtered Zip Codes
        </button>
    </div>

    <div class="legend">
        <h4>🎨 LTL CBSAs Color-Coded Map</h4>
        <div><strong>CBSA-Assigned:</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #4ecdc4;"></div>
            <div class="color-box" style="background: #45b7d1;"></div>
            <div class="color-box" style="background: #f9ca24;"></div>
            <div class="color-box" style="background: #6c5ce7;"></div>
            <div class="color-box" style="background: #26de81;"></div>
            <span>Each LTL CBSA has a unique color (no red)</span>
        </div>
        <div style="margin-top: 10px;"><strong>Non-CBSA:</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #FF4444;"></div>
            <span>Red for all non-CBSA zip codes</span>
        </div>
        <div style="margin-top: 8px; font-size: 10px; color: #666; font-style: italic;">
            🎯 Red circles = CBSA weighted centroids<br>
            🏭 Green diamonds = Crossdock locations<br>
            🚀 Orange areas = SMART corridor zones (distance + width limited)<br>
            📏 Distances calculated to nearest CBSA zip codes
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>"""

    # Add JavaScript functionality (copying from original file)
    html_content += f"""
    <script>
        // Initialize map
        var map = L.map('map').setView([39.8283, -98.5795], 4);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Data and state
        var allZipData = {json.dumps(zip_features, indent=2)};
        var cbsaCentroids = {json.dumps(cbsa_centroid_features, indent=2)};
        var crossdockLocations = {json.dumps(crossdock_locations, indent=2)};
        var allUsaData = {json.dumps(all_usa_data_js, indent=2)};
        var zipLayers = {{}};
        var centroidLayers = [];
        var crossdockLayers = [];
        var smartCorridorLayers = [];
        var currentDistanceThreshold = 50;
        var currentStdDevThreshold = -3.0;
        var smartCorridorsVisible = false;
        var currentMaxCorridorDistance = 150;
        var currentBaseCorridorWidth = 25;
        var stdDevFilterEnabled = true;

        console.log('🎨 LTL CBSA COLOR-CODED Map loaded with:');
        console.log('   -', allZipData.length, 'zip codes');
        console.log('   -', cbsaCentroids.length, 'LTL CBSA centroids');
        console.log('   -', crossdockLocations.length, 'crossdocks');
        console.log('   -', allUsaData.length, 'USA zip codes for corridor analysis');

        // Calculate distance between two lat/lon points (Haversine formula)
        function calculateDistance(lat1, lon1, lat2, lon2) {{
            var R = 3959; // Earth's radius in miles
            var dLat = (lat2 - lat1) * Math.PI / 180;
            var dLon = (lon2 - lon1) * Math.PI / 180;
            var a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                    Math.sin(dLon/2) * Math.sin(dLon/2);
            var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }}

        // Get centroid of a zip code geometry
        function getZipCentroid(zipFeature) {{
            if (zipFeature.geometry.type === 'Polygon') {{
                var coords = zipFeature.geometry.coordinates[0];
                var lat = 0, lon = 0;
                for (var i = 0; i < coords.length; i++) {{
                    lon += coords[i][0];
                    lat += coords[i][1];
                }}
                return [lat / coords.length, lon / coords.length];
            }} else if (zipFeature.geometry.type === 'MultiPolygon') {{
                var coords = zipFeature.geometry.coordinates[0][0];
                var lat = 0, lon = 0;
                for (var i = 0; i < coords.length; i++) {{
                    lon += coords[i][0];
                    lat += coords[i][1];
                }}
                return [lat / coords.length, lon / coords.length];
            }}
            return [0, 0];
        }}

        // Calculate distance from point to line segment
        function pointToLineDistance(px, py, x1, y1, x2, y2) {{
            var A = px - x1;
            var B = py - y1;
            var C = x2 - x1;
            var D = y2 - y1;

            var dot = A * C + B * D;
            var lenSq = C * C + D * D;

            if (lenSq === 0) return calculateDistance(px, py, x1, y1);

            var t = Math.max(0, Math.min(1, dot / lenSq));
            var projX = x1 + t * C;
            var projY = y1 + t * D;

            return calculateDistance(px, py, projX, projY);
        }}

        // Get filtered non-CBSA zip codes based on current filter settings
        function getFilteredNonCbsaZips() {{
            var maxDistance = parseFloat(document.getElementById('distanceSlider').value);
            var minStdDev = parseFloat(document.getElementById('stdDevSlider').value);

            var filtered = allZipData.filter(zip => {{
                // Must be non-CBSA
                if (zip.properties.has_cbsa) return false;

                // Apply distance filter
                if (zip.properties.distance_to_cbsa > maxDistance) return false;

                // Apply standard deviation filter if enabled
                if (stdDevFilterEnabled && zip.properties.std_dev_from_cbsa_mean < minStdDev) return false;

                return true;
            }});

            return filtered;
        }}

        // IMPROVED Smart Corridor Calculation
        function calculateSmartCorridors(maxCorridorDistance, baseCorridorWidth) {{
            console.log('🚀 Calculating SMART corridors with max distance:', maxCorridorDistance, 'miles, base width:', baseCorridorWidth, 'miles');

            // Get filtered non-CBSA zip codes
            var filteredNonCbsaZips = getFilteredNonCbsaZips();

            if (cbsaCentroids.length === 0) {{
                console.log('No CBSA centroids available for corridor analysis');
                return [];
            }}

            console.log('Analyzing corridors for', filteredNonCbsaZips.length, 'filtered non-CBSA zips to', cbsaCentroids.length, 'CBSA centroids');

            if (filteredNonCbsaZips.length === 0) {{
                console.log('No filtered non-CBSA zip codes found - no corridors to calculate');
                return [];
            }}

            var allCorridorZips = new Set(); // Use Set to avoid duplicates
            var corridorStats = {{
                total: 0,
                withinLimit: 0,
                avgDistance: 0,
                avgWidth: 0
            }};

            var corridorDistances = [];
            var corridorWidths = [];

            // Process each filtered non-CBSA zip code
            filteredNonCbsaZips.forEach(targetZip => {{
                corridorStats.total++;
                var targetCentroid = getZipCentroid(targetZip);

                // Find nearest CBSA weighted centroid
                var nearestCbsaCentroid = null;
                var minDistance = Infinity;

                cbsaCentroids.forEach(cbsaCentroid => {{
                    var distance = calculateDistance(
                        targetCentroid[0], targetCentroid[1],
                        cbsaCentroid.lat, cbsaCentroid.lon
                    );

                    if (distance < minDistance) {{
                        minDistance = distance;
                        nearestCbsaCentroid = cbsaCentroid;
                    }}
                }});

                // Skip if corridor is too long
                if (minDistance > maxCorridorDistance) {{
                    console.log('   ⏭️  Skipping corridor for', targetZip.properties.zipcode_clean, '- distance', minDistance.toFixed(1), 'miles exceeds limit');
                    return;
                }}

                corridorStats.withinLimit++;
                corridorDistances.push(minDistance);

                // Calculate dynamic corridor width based on distance
                var distanceFactor = minDistance / maxCorridorDistance; // 0 to 1
                var dynamicWidth = baseCorridorWidth * (1.5 - distanceFactor); // 1.5x width for short, 0.5x for long
                dynamicWidth = Math.max(10, Math.min(dynamicWidth, baseCorridorWidth * 2)); // Clamp between 10 and 2x base

                corridorWidths.push(dynamicWidth);

                if (nearestCbsaCentroid) {{
                    console.log('   🛣️  Creating corridor:', targetZip.properties.zipcode_clean, '→', nearestCbsaCentroid.cbsa_name, '(' + minDistance.toFixed(1) + ' mi, width:', dynamicWidth.toFixed(1), 'mi)');

                    // Find all zip codes within dynamic corridor width of the line
                    allUsaData.forEach(zipData => {{
                        // Skip if already processed or if it's the target
                        var zipCode = zipData.properties.zipcode_clean;
                        if (allCorridorZips.has(zipCode) || zipCode === targetZip.properties.zipcode_clean) {{
                            return;
                        }}

                        var zipCentroid = getZipCentroid(zipData);
                        var distanceToLine = pointToLineDistance(
                            zipCentroid[1], zipCentroid[0],  // lon, lat for point
                            targetCentroid[1], targetCentroid[0],  // lon, lat for line start
                            nearestCbsaCentroid.lon, nearestCbsaCentroid.lat  // lon, lat for line end
                        );

                        if (distanceToLine <= dynamicWidth) {{
                            // Check if the zip is roughly between the target and CBSA (not too far out of the way)
                            var distanceToTarget = calculateDistance(
                                zipCentroid[0], zipCentroid[1],
                                targetCentroid[0], targetCentroid[1]
                            );
                            var distanceToCbsa = calculateDistance(
                                zipCentroid[0], zipCentroid[1],
                                nearestCbsaCentroid.lat, nearestCbsaCentroid.lon
                            );

                            // Only include if the zip is roughly on the path (allow 30% detour)
                            if (distanceToTarget + distanceToCbsa <= minDistance * 1.3) {{
                                allCorridorZips.add(zipCode);
                            }}
                        }}
                    }});
                }}
            }});

            // Calculate statistics
            if (corridorDistances.length > 0) {{
                corridorStats.avgDistance = corridorDistances.reduce((a, b) => a + b, 0) / corridorDistances.length;
                corridorStats.avgWidth = corridorWidths.reduce((a, b) => a + b, 0) / corridorWidths.length;
            }}

            // Convert Set back to array of zip objects
            var corridorZipObjects = [];
            allCorridorZips.forEach(zipCode => {{
                var zipObj = allUsaData.find(zipData => zipData.properties.zipcode_clean === zipCode);
                if (zipObj) {{
                    corridorZipObjects.push(zipObj);
                }}
            }});

            console.log('✅ IMPROVED Corridor Analysis Results:');
            console.log('   - Total potential corridors:', corridorStats.total);
            console.log('   - Corridors within', maxCorridorDistance, 'mile limit:', corridorStats.withinLimit);
            console.log('   - Average corridor distance:', corridorStats.avgDistance.toFixed(1), 'miles');
            console.log('   - Average corridor width:', corridorStats.avgWidth.toFixed(1), 'miles');
            console.log('   - Total corridor zip codes found:', corridorZipObjects.length);

            return corridorZipObjects;
        }}

        // Update just the statistics display
        function updateMapDisplayStats() {{
            var showCentroids = document.getElementById('showCentroids').checked;
            var showCrossdocks = document.getElementById('showCrossdocks').checked;
            var maxDistance = parseFloat(document.getElementById('distanceSlider').value);
            var minStdDev = parseFloat(document.getElementById('stdDevSlider').value);

            // Count currently displayed zip codes from regular layers
            var cbsaCount = Object.values(zipLayers).filter(layer => {{
                return layer.feature && layer.feature.properties.has_cbsa;
            }}).length;

            var nonCbsaCount = Object.values(zipLayers).filter(layer => {{
                return layer.feature && !layer.feature.properties.has_cbsa;
            }}).length;

            var totalDisplayed = Object.keys(zipLayers).length;

            // Calculate corridor zip codes if smart corridors are enabled
            var corridorCount = 0;
            var corridorText = '';
            var corridorCbsaCount = 0;
            var corridorNonCbsaCount = 0;

            if (document.getElementById('showSmartCorridors').checked && smartCorridorsVisible) {{
                corridorCount = smartCorridorLayers.length;
                corridorText = `🚀 Smart corridor zip codes: ${{corridorCount.toLocaleString()}}<br>`;

                // Count CBSA vs non-CBSA in corridor zip codes
                smartCorridorLayers.forEach(layer => {{
                    if (layer.feature && layer.feature.properties) {{
                        if (layer.feature.properties.has_cbsa) {{
                            corridorCbsaCount++;
                        }} else {{
                            corridorNonCbsaCount++;
                        }}
                    }}
                }});

                // Add corridor counts to regular counts for accurate totals
                cbsaCount += corridorCbsaCount;
                nonCbsaCount += corridorNonCbsaCount;
                totalDisplayed += corridorCount;
            }}

            // Update statistics
            document.getElementById('filterStats').innerHTML = `
                <strong>Currently Displayed:</strong><br>
                🎨 LTL CBSA zip codes: ${{cbsaCount.toLocaleString()}}<br>
                🔴 Non-CBSA (≤ ${{maxDistance}} miles & ≥ ${{minStdDev.toFixed(1)}} std dev): ${{nonCbsaCount.toLocaleString()}}<br>
                ${{corridorText}}📊 Total non-CBSA zip codes: ${{allZipData.filter(z => !z.properties.has_cbsa).length.toLocaleString()}}<br>
                🎯 LTL CBSA centroids: ${{showCentroids ? cbsaCentroids.length : 0}}<br>
                🏭 Crossdocks: ${{showCrossdocks ? crossdockLocations.length : 0}}<br>
                📍 Total displayed: ${{totalDisplayed.toLocaleString()}}
            `;
        }}

        // Toggle SMART corridors display
        function toggleSmartCorridors() {{
            if (smartCorridorsVisible) {{
                // Hide SMART corridors
                console.log('Hiding SMART corridors');
                smartCorridorLayers.forEach(layer => map.removeLayer(layer));
                smartCorridorLayers = [];
                smartCorridorsVisible = false;
            }} else {{
                // Show SMART corridors for filtered non-CBSA zip codes
                console.log('Showing SMART corridors for filtered non-CBSA zip codes');
                var corridorZips = calculateSmartCorridors(currentMaxCorridorDistance, currentBaseCorridorWidth);
                console.log('SMART corridor calculation returned', corridorZips.length, 'zip codes');

                if (corridorZips.length > 0) {{
                    corridorZips.forEach((zip, index) => {{
                        if (index < 3) {{
                            console.log('SMART corridor zip', index, ':', zip.properties?.zipcode_clean, 'has geometry:', !!zip.geometry);
                        }}
                        var layer = L.geoJSON(zip, {{
                            style: {{
                                fillColor: '#FFA500', // Orange for corridor zips
                                color: '#FF8C00',
                                weight: 2,
                                opacity: 0.8,
                                fillOpacity: 0.6
                            }},
                            onEachFeature: function(feature, layer) {{
                                layer.bindPopup(`
                                    <strong>🚀 SMART Corridor Zip: ${{feature.properties.zipcode_clean}}</strong><br>
                                    <strong>Quotes:</strong> ${{feature.properties.total_quotes}}<br>
                                    <strong>CBSA:</strong> ${{feature.properties.cbsa_name || 'Non-CBSA'}}<br>
                                    <hr style="margin: 5px 0;">
                                    <em>Part of optimized corridor pathway to nearest CBSA coverage</em>
                                `);

                                layer.bindTooltip(
                                    '🚀 SMART Corridor: ' + feature.properties.zipcode_clean + ' | ' + feature.properties.total_quotes + ' quotes',
                                    {{ permanent: false, direction: 'top' }}
                                );
                            }}
                        }}).addTo(map);

                        smartCorridorLayers.push(layer);
                    }});

                    smartCorridorsVisible = true;
                    console.log('Successfully added', corridorZips.length, 'SMART corridor zip shapes to map');
                }} else {{
                    console.log('No SMART corridor zip codes found to display');
                }}
            }}

            // Update statistics to reflect corridor changes
            updateMapDisplayStats();
        }}

        // Find nearest crossdock to a zip code
        function findNearestCrossdock(zipLat, zipLon) {{
            var minDistance = Infinity;
            var nearestCrossdock = null;

            crossdockLocations.forEach(function(crossdock) {{
                var distance = calculateDistance(zipLat, zipLon, crossdock.lat, crossdock.lon);
                if (distance < minDistance) {{
                    minDistance = distance;
                    nearestCrossdock = crossdock;
                }}
            }});

            return {{
                distance: minDistance,
                crossdock: nearestCrossdock
            }};
        }}

        // Update map display based on current filters
        function updateMapDisplay() {{
            var showCBSA = document.getElementById('showAllCBSA').checked;
            var showNonCBSA = document.getElementById('showNearbyNonCBSA').checked;
            var showCentroids = document.getElementById('showCentroids').checked;
            var showCrossdocks = document.getElementById('showCrossdocks').checked;
            var maxDistance = parseFloat(document.getElementById('distanceSlider').value);
            var minStdDev = parseFloat(document.getElementById('stdDevSlider').value);

            // Clear existing layers
            Object.values(zipLayers).forEach(layer => map.removeLayer(layer));
            zipLayers = {{}};

            // Filter and display zip codes
            var cbsaCount = 0;
            var nonCbsaCount = 0;
            var totalDisplayed = 0;

            allZipData.forEach(function(zipData) {{
                var props = zipData.properties;
                var shouldShow = false;

                if (props.has_cbsa && showCBSA) {{
                    shouldShow = true;
                    cbsaCount++;
                }} else if (!props.has_cbsa && showNonCBSA) {{
                    // Apply distance filter
                    if (props.distance_to_cbsa <= maxDistance) {{
                        // Apply standard deviation filter if enabled
                        if (!stdDevFilterEnabled || props.std_dev_from_cbsa_mean >= minStdDev) {{
                            shouldShow = true;
                            nonCbsaCount++;
                        }}
                    }}
                }}

                if (shouldShow) {{
                    totalDisplayed++;
                    // Use CBSA-based color instead of quote density
                    var color = props.cbsa_color || '#808080';

                    var layer = L.geoJSON(zipData, {{
                        style: {{
                            fillColor: color,
                            color: props.has_cbsa ? '#333333' : '#666666',
                            weight: 1,
                            opacity: 0.8,
                            fillOpacity: 0.7
                        }},
                        onEachFeature: function(feature, layer) {{
                            var nearestCrossdock = findNearestCrossdock(
                                feature.properties.distance_to_cbsa > 0 ?
                                parseFloat(feature.geometry.coordinates[0][0][1]) :
                                parseFloat(feature.geometry.coordinates[0][0][1]),
                                feature.properties.distance_to_cbsa > 0 ?
                                parseFloat(feature.geometry.coordinates[0][0][0]) :
                                parseFloat(feature.geometry.coordinates[0][0][0])
                            );

                            layer.bindPopup(`
                                <strong>${{feature.properties.zipcode_clean}}</strong><br>
                                <strong>City:</strong> ${{feature.properties.city}}<br>
                                <strong>State:</strong> ${{feature.properties.state}}<br>
                                <strong>Total Quotes:</strong> ${{feature.properties.total_quotes}}<br>
                                <strong>Population:</strong> ${{feature.properties.population.toLocaleString()}}<br>
                                <strong>CBSA:</strong> ${{feature.properties.cbsa_name || 'Non-CBSA'}}<br>
                                ${{!feature.properties.has_cbsa ?
                                    `<strong>Distance to CBSA:</strong> ${{feature.properties.distance_to_cbsa.toFixed(1)}} miles<br>
                                     <strong>Nearest CBSA:</strong> ${{feature.properties.nearest_cbsa_name}}<br>
                                     <strong>Performance vs CBSA:</strong> ${{feature.properties.std_dev_from_cbsa_mean.toFixed(1)}} std dev<br>` : ''}}
                                ${{nearestCrossdock.crossdock ?
                                    `<strong>Nearest Crossdock:</strong> ${{nearestCrossdock.crossdock.name}} (${{nearestCrossdock.distance.toFixed(1)}} miles)<br>` : ''}}
                            `);

                            layer.bindTooltip(
                                feature.properties.zipcode_clean + ' | ' + feature.properties.total_quotes + ' quotes',
                                {{ permanent: false, direction: 'top' }}
                            );
                        }}
                    }}).addTo(map);

                    zipLayers[props.zipcode_clean] = layer;
                }}
            }});

            // Update centroids
            centroidLayers.forEach(layer => map.removeLayer(layer));
            centroidLayers = [];

            if (showCentroids) {{
                cbsaCentroids.forEach(function(centroid) {{
                    var marker = L.circleMarker([centroid.lat, centroid.lon], {{
                        radius: Math.max(5, Math.min(15, Math.sqrt(centroid.total_quotes) / 10)),
                        fillColor: '#d62728',
                        color: '#ffffff',
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.8
                    }}).addTo(map);

                    marker.bindPopup(`
                        <strong>🎯 ${{centroid.cbsa_name}}</strong><br>
                        <strong>Total Quotes:</strong> ${{centroid.total_quotes.toLocaleString()}}<br>
                        <strong>Zip Codes:</strong> ${{centroid.zip_count}}<br>
                        <em>Quote-weighted business center</em>
                    `);

                    centroidLayers.push(marker);
                }});
            }}

            // Update crossdocks
            crossdockLayers.forEach(layer => map.removeLayer(layer));
            crossdockLayers = [];

            if (showCrossdocks) {{
                crossdockLocations.forEach(function(crossdock) {{
                    var marker = L.marker([crossdock.lat, crossdock.lon], {{
                        icon: L.divIcon({{
                            className: 'crossdock-icon',
                            html: '🏭',
                            iconSize: [20, 20],
                            iconAnchor: [10, 10]
                        }})
                    }}).addTo(map);

                    marker.bindPopup(`
                        <strong>🏭 ${{crossdock.name}}</strong><br>
                        <strong>Zip Code:</strong> ${{crossdock.zip_code}}<br>
                        <em>Crossdock facility</em>
                    `);

                    crossdockLayers.push(marker);
                }});
            }}

            // Update statistics using the dedicated function
            updateMapDisplayStats();
        }}

        // Export filtered data
        function exportFilteredData() {{
            var maxDistance = parseFloat(document.getElementById('distanceSlider').value);
            var minStdDev = parseFloat(document.getElementById('stdDevSlider').value);

            console.log('🔍 EXPORT DIAGNOSTICS:');
            console.log('   - Max Distance Filter:', maxDistance, 'miles');
            console.log('   - Min Std Dev Filter:', minStdDev);
            console.log('   - Std Dev Filter Enabled:', stdDevFilterEnabled);
            console.log('   - Show All CBSA:', document.getElementById('showAllCBSA').checked);
            console.log('   - Show Nearby Non-CBSA:', document.getElementById('showNearbyNonCBSA').checked);
            console.log('   - Show Smart Corridors:', document.getElementById('showSmartCorridors').checked);

            // Calculate current corridor zip codes for export
            var corridorZips = [];
            var corridorZipCodes = new Set();

            if (document.getElementById('showSmartCorridors').checked) {{
                console.log('   - Calculating smart corridors...');
                corridorZips = calculateSmartCorridors(currentMaxCorridorDistance, currentBaseCorridorWidth);
                corridorZips.forEach(zip => {{
                    corridorZipCodes.add(zip.properties.zipcode_clean);
                }});
                console.log('   - Found', corridorZipCodes.size, 'corridor zip codes');
            }}

            var exportData = [];
            var cbsaIncluded = 0;
            var nonCbsaIncluded = 0;
            var nonCbsaFiltered = 0;
            var stdDevFiltered = 0;

            allZipData.forEach(function(zipData) {{
                var props = zipData.properties;
                var shouldInclude = false;

                if (props.has_cbsa) {{
                    shouldInclude = document.getElementById('showAllCBSA').checked;
                    if (shouldInclude) cbsaIncluded++;
                }} else {{
                    if (props.distance_to_cbsa <= maxDistance) {{
                        if (!stdDevFilterEnabled || props.std_dev_from_cbsa_mean >= minStdDev) {{
                            shouldInclude = document.getElementById('showNearbyNonCBSA').checked;
                            if (shouldInclude) nonCbsaIncluded++;
                        }} else {{
                            stdDevFiltered++;
                        }}
                    }} else {{
                        nonCbsaFiltered++;
                    }}
                }}

                if (shouldInclude) {{
                    var nearestCrossdock = findNearestCrossdock(
                        parseFloat(zipData.geometry.coordinates[0][0][1]),
                        parseFloat(zipData.geometry.coordinates[0][0][0])
                    );

                    exportData.push({{
                        'Zipcode': props.zipcode_clean,
                        'City': props.city,
                        'State': props.state,
                        'Total_Quotes': props.total_quotes,
                        'Pickup_Count': props.pickup_count,
                        'Dropoff_Count': props.dropoff_count,
                        'Population': props.population,
                        'Has_CBSA': props.has_cbsa,
                        'Assigned_CBSA': props.assigned_cbsa || '',
                        'Closest_CBSA': props.closest_cbsa || '',
                        'Distance_to_CBSA_Miles': props.distance_to_cbsa,
                        'Nearest_CBSA_Zip': props.nearest_cbsa_zip || '',
                        'Std_Dev_vs_CBSA_Mean': props.std_dev_from_cbsa_mean,
                        'Quote_Percentile_in_Nearest_CBSA': props.quote_percentile_in_nearest_cbsa,
                        'CBSA_Mean_Quotes': props.cbsa_mean_quotes,
                        'Is_Corridor_Zip': corridorZipCodes.has(props.zipcode_clean),
                        'Nearest_Crossdock_Name': nearestCrossdock.crossdock ? nearestCrossdock.crossdock.name : '',
                        'Distance_to_Nearest_Crossdock_Miles': nearestCrossdock.crossdock ? nearestCrossdock.distance : ''
                    }});
                }}
            }});

            // Print detailed diagnostics
            console.log('📊 EXPORT RESULTS:');
            console.log('   - Total zip codes in dataset:', allZipData.length);
            console.log('   - CBSA zip codes included:', cbsaIncluded);
            console.log('   - Non-CBSA zip codes included:', nonCbsaIncluded);
            console.log('   - Non-CBSA filtered by distance (>' + maxDistance + ' miles):', nonCbsaFiltered);
            console.log('   - Non-CBSA filtered by std dev (<' + minStdDev + '):', stdDevFiltered);
            console.log('   - Total export data length:', exportData.length);

            // Convert to CSV
            if (exportData.length === 0) {{
                console.log('❌ NO DATA TO EXPORT - Check your filter settings:');
                console.log('   - Make sure "Show all LTL CBSA zip codes" or "Show nearby non-CBSA zip codes" is checked');
                console.log('   - Try relaxing your filters (increase distance, lower std dev threshold)');
                alert('No data to export with current filters. Check console for details.');
                return;
            }}

            console.log('✅ Proceeding with CSV export of', exportData.length, 'zip codes');

            var csv = Object.keys(exportData[0]).join(',') + '\\n';
            exportData.forEach(function(row) {{
                csv += Object.values(row).map(function(value) {{
                    return typeof value === 'string' ? '"' + value.replace(/"/g, '""') + '"' : value;
                }}).join(',') + '\\n';
            }});

            // Check CSV size
            var csvSizeKB = (csv.length / 1024).toFixed(1);
            console.log('📄 CSV size:', csvSizeKB, 'KB');

            if (csvSizeKB > 10000) {{ // 10MB limit warning
                console.log('⚠️  Large file warning: CSV is', csvSizeKB, 'KB');
                if (!confirm('Large export file (' + csvSizeKB + ' KB). Continue?')) {{
                    return;
                }}
            }}

            // Download CSV
            try {{
                var blob = new Blob([csv], {{ type: 'text/csv' }});
                var url = window.URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.setAttribute('hidden', '');
                a.setAttribute('href', url);
                a.setAttribute('download', 'ltl_cbsa_color_coded_filtered_zip_codes_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.csv');
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                console.log('✅ Successfully exported', exportData.length, 'zip codes to CSV');
            }} catch (error) {{
                console.error('❌ Export failed:', error);
                alert('Export failed: ' + error.message);
            }}
        }}

        // Event handlers
        document.getElementById('distanceSlider').addEventListener('input', function() {{
            currentDistanceThreshold = parseFloat(this.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            updateMapDisplay();
        }});

        document.getElementById('stdDevSlider').addEventListener('input', function() {{
            currentStdDevThreshold = parseFloat(this.value);
            document.getElementById('stdDevDisplay').textContent = '≥ ' + currentStdDevThreshold.toFixed(1) + ' std dev';
            updateMapDisplay();
        }});

        document.getElementById('showAllCBSA').addEventListener('change', updateMapDisplay);
        document.getElementById('showNearbyNonCBSA').addEventListener('change', updateMapDisplay);
        document.getElementById('showCentroids').addEventListener('change', updateMapDisplay);
        document.getElementById('showCrossdocks').addEventListener('change', updateMapDisplay);

        document.getElementById('enableStdDevFilter').addEventListener('change', function() {{
            stdDevFilterEnabled = this.checked;
            document.getElementById('stdDevContainer').style.display = this.checked ? 'block' : 'none';
            updateMapDisplay();
        }});

        document.getElementById('showSmartCorridors').addEventListener('change', function() {{
            toggleSmartCorridors();
        }});

        document.getElementById('maxCorridorDistance').addEventListener('input', function() {{
            currentMaxCorridorDistance = parseInt(this.value);
            document.getElementById('maxCorridorDistanceDisplay').textContent = currentMaxCorridorDistance + ' miles';

            // If corridors are currently visible, recalculate them
            if (smartCorridorsVisible) {{
                // Hide current corridors
                smartCorridorLayers.forEach(layer => map.removeLayer(layer));
                smartCorridorLayers = [];

                // Show new corridors with updated parameters
                var corridorZips = calculateSmartCorridors(currentMaxCorridorDistance, currentBaseCorridorWidth);
                if (corridorZips.length > 0) {{
                    corridorZips.forEach(zip => {{
                        var layer = L.geoJSON(zip, {{
                            style: {{
                                fillColor: '#FFA500',
                                color: '#FF8C00',
                                weight: 2,
                                opacity: 0.8,
                                fillOpacity: 0.6
                            }},
                            onEachFeature: function(feature, layer) {{
                                layer.bindPopup(`
                                    <strong>🚀 SMART Corridor Zip: ${{feature.properties.zipcode_clean}}</strong><br>
                                    <strong>Quotes:</strong> ${{feature.properties.total_quotes}}<br>
                                    <strong>CBSA:</strong> ${{feature.properties.cbsa_name || 'Non-CBSA'}}<br>
                                    <hr style="margin: 5px 0;">
                                    <em>Part of optimized corridor pathway to nearest CBSA coverage</em>
                                `);
                                layer.bindTooltip(
                                    '🚀 SMART Corridor: ' + feature.properties.zipcode_clean + ' | ' + feature.properties.total_quotes + ' quotes',
                                    {{ permanent: false, direction: 'top' }}
                                );
                            }}
                        }}).addTo(map);
                        smartCorridorLayers.push(layer);
                    }});
                }}
            }}

            // Update statistics to reflect corridor changes
            updateMapDisplayStats();
        }});

        document.getElementById('baseCorridorWidth').addEventListener('input', function() {{
            currentBaseCorridorWidth = parseInt(this.value);
            document.getElementById('baseCorridorWidthDisplay').textContent = currentBaseCorridorWidth + ' miles';

            // If corridors are currently visible, recalculate them
            if (smartCorridorsVisible) {{
                // Hide current corridors
                smartCorridorLayers.forEach(layer => map.removeLayer(layer));
                smartCorridorLayers = [];

                // Show new corridors with updated parameters
                var corridorZips = calculateSmartCorridors(currentMaxCorridorDistance, currentBaseCorridorWidth);
                if (corridorZips.length > 0) {{
                    corridorZips.forEach(zip => {{
                        var layer = L.geoJSON(zip, {{
                            style: {{
                                fillColor: '#FFA500',
                                color: '#FF8C00',
                                weight: 2,
                                opacity: 0.8,
                                fillOpacity: 0.6
                            }},
                            onEachFeature: function(feature, layer) {{
                                layer.bindPopup(`
                                    <strong>🚀 SMART Corridor Zip: ${{feature.properties.zipcode_clean}}</strong><br>
                                    <strong>Quotes:</strong> ${{feature.properties.total_quotes}}<br>
                                    <strong>CBSA:</strong> ${{feature.properties.cbsa_name || 'Non-CBSA'}}<br>
                                    <hr style="margin: 5px 0;">
                                    <em>Part of optimized corridor pathway to nearest CBSA coverage</em>
                                `);
                                layer.bindTooltip(
                                    '🚀 SMART Corridor: ' + feature.properties.zipcode_clean + ' | ' + feature.properties.total_quotes + ' quotes',
                                    {{ permanent: false, direction: 'top' }}
                                );
                            }}
                        }}).addTo(map);
                        smartCorridorLayers.push(layer);
                    }});
                }}
            }}

            // Update statistics to reflect corridor changes
            updateMapDisplayStats();
        }});

        // Initial map display
        updateMapDisplay();

    </script>
</body>
</html>"""

    return html_content

def main():
    """Main function to create the LTL CBSA color-coded corridor map"""
    print("🎨 Creating LTL CBSA Color-Coded Corridor Map...")
    print("=" * 60)

    try:
        # Check if we should create a shareable version
        if len(sys.argv) > 1 and sys.argv[1] == '--shareable':
            print("Creating shareable version...")
            zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js = create_shareable_version()
        else:
            # Load and prepare data
            merged_data, zip_shapes, unassigned_zips = load_and_prepare_data()
            crossdock_data = load_crossdock_data()

            # Calculate weighted centroids and distances - MODIFIED to use LTL CBSAs
            map_data, cbsa_centroids_df = calculate_weighted_cbsa_centroids(merged_data, zip_shapes)
            map_data = calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df)

            # Process crossdock locations
            crossdock_locations = process_crossdock_locations(crossdock_data, zip_shapes)

            # Load ALL USA data for corridor analysis
            all_usa_data = load_all_usa_zip_data_for_corridors(merged_data, zip_shapes)
            all_usa_data_js = prepare_all_usa_data_for_js(all_usa_data)

            # Create the map with CBSA-based coloring
            zip_features, cbsa_centroid_features = create_cbsa_proximity_map(
                map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data
            )

        print(f"✅ Map preparation complete:")
        print(f"   - {len(zip_features)} zip code features")
        print(f"   - {len(cbsa_centroid_features)} LTL CBSA centroids")
        print(f"   - {len(crossdock_locations)} crossdock locations")

        # Create the complete HTML map
        html_content = create_complete_html_map(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js)

        # Save the HTML file
        output_file = BASE_DIR / 'ltl_cbsa_color_coded_corridor_map.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ LTL CBSA Color-Coded Corridor Map saved to: {output_file}")
        print(f"📊 Map Features:")
        print(f"   🎨 Uses CBSAs from LTL Rate File zip codes (not top 75 by population)")
        print(f"   🌈 Each CBSA has a unique color for easy distinction")
        print(f"   🔴 Red color for non-CBSA zip codes")
        print(f"   🚀 Smart corridor analysis with distance and width limits")
        print(f"   📏 Distance calculations to nearest CBSA zip codes")
        print(f"   📈 Quote performance filtering and statistics")
        print(f"   📥 Export functionality for filtered data")

        # Open in browser
        try:
            webbrowser.open(f'file://{output_file.absolute()}')
            print(f"🌐 Map opened in default browser")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please open {output_file} manually in your browser")

    except Exception as e:
        print(f"❌ Error creating map: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main()
