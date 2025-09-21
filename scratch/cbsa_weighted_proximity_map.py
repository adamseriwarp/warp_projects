import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from pathlib import Path
from sklearn.neighbors import BallTree

# FIX 4: Configurable paths instead of hard-coded absolute paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DOWNLOADS_DIR = Path.home() / 'Downloads'  # More portable than hard-coded path

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

def calculate_weighted_cbsa_centroids(data, zip_shapes):
    """Calculate quote-weighted centroids for each CBSA"""
    print("Calculating quote-weighted CBSA centroids...")

    # FIX 1: Calculate top 75 CBSAs by population BEFORE filtering by quotes
    # This prevents bias where CBSAs without quotes get excluded
    print("Determining top 75 CBSAs by population (unbiased by quotes)...")

    # First, get all CBSA populations from the full dataset (no quote filter)
    all_cbsa_data = zip_shapes.merge(
        data,
        left_on='ZCTA5CE20',
        right_on='Zipcode_clean',
        how='inner'
    )

    # Calculate total population for each CBSA from ALL zip codes (not just those with quotes)
    cbsa_populations = all_cbsa_data[all_cbsa_data['Primary CBSA Name'].notna()].groupby('Primary CBSA Name')['ZCTA Population (2020)'].sum().reset_index()
    cbsa_populations = cbsa_populations.sort_values('ZCTA Population (2020)', ascending=False)
    top_75_cbsas = cbsa_populations.head(75)['Primary CBSA Name'].tolist()

    print(f"Top 75 CBSAs by population (unbiased selection):")
    for i, cbsa in enumerate(top_75_cbsas[:10], 1):
        pop = cbsa_populations[cbsa_populations['Primary CBSA Name'] == cbsa]['ZCTA Population (2020)'].iloc[0]
        print(f"  {i}. {cbsa}: {pop:,.0f}")
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

    # Filter map data to only include top 75 CBSAs (but keep all non-CBSA zip codes)
    map_data = map_data[
        (map_data['Primary CBSA Name'].isin(top_75_cbsas)) |
        (map_data['Primary CBSA Name'].isna())
    ]
    
    # Calculate centroids for each zip code
    map_data_projected = map_data.to_crs('EPSG:3857')  # Web Mercator
    centroids = map_data_projected.geometry.centroid.to_crs('EPSG:4326')  # Back to lat/lon
    
    map_data['centroid_lat'] = centroids.y
    map_data['centroid_lon'] = centroids.x
    
    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    non_cbsa_zips = map_data[map_data['Primary CBSA Name'].isna()].copy()
    
    print(f"Calculating weighted centroids for {cbsa_zips['Primary CBSA Name'].nunique()} CBSAs...")
    
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
    
    print(f"Calculated weighted centroids for {len(cbsa_centroids_df)} CBSAs")
    print(f"Quote range across CBSAs: {cbsa_centroids_df['total_quotes'].min():,.0f} - {cbsa_centroids_df['total_quotes'].max():,.0f}")
    
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
    std_deviations_below = []
    quote_percentiles_in_nearest_cbsa = []
    cbsa_mean_quotes = []  # Add CBSA mean quotes for tooltips

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
    cbsa_zips['std_dev_below_cbsa_mean'] = 0  # CBSA zips are part of their own CBSA
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

def load_all_zip_data_for_corridors(merged_data, zip_shapes):
    """Load ALL zip codes in the US for universal corridor analysis"""

    # Merge all zip codes with shapes (ALL zip codes, not just CBSAs)
    all_data = zip_shapes.merge(merged_data, left_on='ZCTA5CE20', right_on='Zipcode_clean', how='left')
    all_data['Total Quotes'] = all_data['Total Quotes'].fillna(0)

    # Keep ALL zip codes (both CBSA and non-CBSA)
    print(f"Loaded {len(all_data)} zip codes from entire US for universal corridor analysis")
    return all_data

def prepare_all_cbsa_data_for_js(all_cbsa_data):
    """Prepare all CBSA data for JavaScript corridor analysis"""
    if all_cbsa_data is None:
        return []

    # Convert to WGS84 for web display
    all_cbsa_wgs84 = all_cbsa_data.to_crs('EPSG:4326')

    features = []
    for idx, row in all_cbsa_wgs84.iterrows():
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
                "has_cbsa": pd.notna(row['Primary CBSA Name'])
            },
            "geometry": json.loads(gpd.GeoSeries([simplified_geom]).to_json())['features'][0]['geometry']
        })

    return features

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

def create_cbsa_proximity_map(map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data=None):
    """Create a CBSA proximity map with weighted centroids"""
    print("Creating CBSA proximity map with weighted centroids...")
    
    # Convert to WGS84 for web display
    map_data_wgs84 = map_data.to_crs('EPSG:4326')
    
    # Calculate separate percentiles for better color sensitivity
    all_quotes = map_data_wgs84['Total Quotes']
    cbsa_quotes = map_data_wgs84[map_data_wgs84['Primary CBSA Name'].notna()]['Total Quotes']
    non_cbsa_quotes = map_data_wgs84[map_data_wgs84['Primary CBSA Name'].isna()]['Total Quotes']
    
    # Calculate separate percentiles for CBSA and non-CBSA zip codes for better color distribution
    cbsa_percentiles = np.percentile(cbsa_quotes, [0, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 100]) if len(cbsa_quotes) > 0 else [0]
    non_cbsa_percentiles = np.percentile(non_cbsa_quotes, [0, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 100]) if len(non_cbsa_quotes) > 0 else [0]
    
    print(f"CBSA quote range: {cbsa_percentiles[0]:.0f} - {cbsa_percentiles[-1]:.0f}")
    print(f"Non-CBSA quote range: {non_cbsa_percentiles[0]:.0f} - {non_cbsa_percentiles[-1]:.0f}")
    
    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        has_cbsa = pd.notna(row['Primary CBSA Name'])
        quotes = row['Total Quotes']
        
        # Calculate percentile position using appropriate group and logarithmic scaling
        if has_cbsa and len(cbsa_quotes) > 0:
            percentiles = cbsa_percentiles
        elif not has_cbsa and len(non_cbsa_quotes) > 0:
            percentiles = non_cbsa_percentiles
        else:
            percentiles = [0, max(1, quotes)]
        
        # Apply logarithmic scaling for better visual distribution
        if quotes > 0 and percentiles[-1] > 0:
            log_quotes = np.log10(quotes + 1)
            log_max = np.log10(percentiles[-1] + 1)
            log_min = np.log10(max(1, percentiles[0]) + 1)
            
            if log_max > log_min:
                percentile = (log_quotes - log_min) / (log_max - log_min)
            else:
                percentile = 0.5
        else:
            percentile = 0
        
        percentile = max(0, min(1, percentile))
        
        # FIX 5: Add column guards to prevent crashes when columns are missing
        # FIX 3: Simplify geometry to reduce file size (partial fix)
        simplified_geom = row['geometry']
        if hasattr(simplified_geom, 'simplify'):
            # Simplify polygon to reduce complexity (tolerance in degrees, ~100m)
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
                "quote_percentile": percentile,
                "was_unassigned": row['Zipcode_clean'] in unassigned_zips,
                "is_unassigned_cbsa": row['Zipcode_clean'] in unassigned_zips and has_cbsa,
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
    
    # Create the HTML map
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Top 75 CBSAs Proximity Map with Weighted Centroids</title>
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
        #distanceSlider::-moz-range-thumb {{
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
        .disabled {{
            opacity: 0.5;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="controls">
        <h4>🎯 Top 75 CBSAs Proximity Filter</h4>
        
        <div class="slider-container">
            <label for="distanceSlider">Distance to Nearest CBSA Coverage (Zip-to-Zip):</label>
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
            <label for="showAllCBSA">Show all CBSA zip codes</label>
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

        <div class="slider-container" style="margin-top: 15px;" id="stdDevContainer">
            <label for="stdDevSlider">Min Quote Performance vs Nearest CBSA:</label>
            <div class="slider-wrapper">
                <input type="range" id="stdDevSlider" min="-3" max="2" value="-3" step="0.5">
            </div>
            <div class="distance-display" id="stdDevDisplay" style="font-size: 16px;">≥ -3.0 std dev</div>
            <div class="range-labels">
                <span>-3 std dev</span>
                <span>+2 std dev</span>
            </div>
            <div style="font-size: 10px; color: #666; margin-top: 5px; line-height: 1.3;">
                Filters non-CBSA zip codes by quote volume relative to their nearest CBSA.<br>
                <strong>-1.0</strong> = 1 std dev below CBSA average<br>
                <strong>0.0</strong> = at CBSA average<br>
                <strong>+1.0</strong> = 1 std dev above CBSA average
            </div>
        </div>


        <div class="stats" id="filterStats">
            Loading...
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCorridors">
            <label for="showCorridors">Show ALL corridor zip codes</label>
        </div>

        <div style="margin-top: 8px;">
            <label for="corridorWidth">Corridor Width (miles):</label>
            <input type="range" id="corridorWidth" min="10" max="100" value="25" step="5" style="width: 100%;">
            <div id="corridorWidthDisplay" style="text-align: center; font-weight: bold;">25 miles</div>
        </div>

        <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 12px; margin: 10px 0;">
            <strong>💡 Corridor Analysis:</strong> Shows zip code shapes for all areas "on the way" between non-CBSA zip codes and their nearest CBSA coverage
        </div>

        <div class="checkbox-container">
            <input type="checkbox" id="showCorridors">
            <label for="showCorridors">Show ALL corridor zip codes</label>
        </div>

        <div style="margin-top: 8px;">
            <label for="corridorWidth">Corridor Width (miles):</label>
            <input type="range" id="corridorWidth" min="10" max="100" value="25" step="5" style="width: 100%;">
            <div id="corridorWidthDisplay" style="text-align: center; font-weight: bold;">25 miles</div>
        </div>

        <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 12px; margin: 10px 0;">
            <strong>💡 Corridor Analysis:</strong> Shows zip code shapes for all areas "on the way" between non-CBSA zip codes and their nearest CBSA coverage. Uses ALL US zip codes.
        </div>

        <button onclick="exportFilteredData()" style="width: 100%; padding: 10px; margin-top: 15px; background: #007cba; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
            📥 Export Filtered Zip Codes
        </button>
    </div>
    
    <div class="legend">
        <h4>🔥 Top 75 CBSAs Quote Volume Heat Map</h4>
        <div><strong>CBSA-Assigned (Enhanced Blue Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #f8fbff;"></div>
            <div class="color-box" style="background: #c6dbed;"></div>
            <div class="color-box" style="background: #6baed6;"></div>
            <div class="color-box" style="background: #3182bd;"></div>
            <div class="color-box" style="background: #08306b;"></div>
            <span>Very Low → Very High Quotes (Log Scale)</span>
        </div>
        <div style="margin-top: 10px;"><strong>Non-CBSA (Enhanced Red Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #fff5f0;"></div>
            <div class="color-box" style="background: #fee0d2;"></div>
            <div class="color-box" style="background: #fc9272;"></div>
            <div class="color-box" style="background: #de2d26;"></div>
            <div class="color-box" style="background: #a50f15;"></div>
            <span>Very Low → Very High Quotes (Log Scale)</span>
        </div>
        <div style="margin-top: 8px; font-size: 10px; color: #666; font-style: italic;">
            ✨ Distances calculated to nearest CBSA zip codes (adjacency analysis)<br>
            🎯 Red circles = CBSA weighted centroids (business activity centers)<br>
            🏭 Green diamonds = Crossdock locations<br>
            �️ Orange dashed lines = Market corridors (isolated to CBSA)<br>
            �📏 Filter uses zip-to-zip distance, not centroid distance
        </div>

    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
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
        var allUsaData = {json.dumps(prepare_all_usa_data_for_js(all_usa_data) if all_usa_data is not None else [], indent=2)};
        var zipLayers = {{}};
        var centroidLayers = [];
        var crossdockLayers = [];
        var corridorLayers = [];
        var currentDistanceThreshold = 50;
        var currentStdDevThreshold = -3.0;
        var corridorsVisible = false;
        var currentCorridorWidth = 25;
        var stdDevFilterEnabled = true;
        
        console.log('Loaded', allZipData.length, 'zip codes,', cbsaCentroids.length, 'CBSA centroids,', crossdockLocations.length, 'crossdocks, and', allUsaData.length, 'USA zip codes for corridor analysis');

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

            console.log('getFilteredNonCbsaZips called with allUsaData.length =', allUsaData.length);
            console.log('Filter settings: maxDistance =', maxDistance, 'minStdDev =', minStdDev);

            // Use allUsaData instead of allZipData for corridor analysis
            var filtered = allUsaData.filter(zip => {{
                // Must be non-CBSA
                if (zip.properties.has_cbsa) return false;

                // Apply distance filter
                if (zip.properties.distance_to_nearest_cbsa > maxDistance) return false;

                // Apply standard deviation filter
                if (zip.properties.std_dev_from_cbsa < minStdDev) return false;

                return true;
            }});

            console.log('getFilteredNonCbsaZips found', filtered.length, 'filtered non-CBSA zip codes');
            if (filtered.length > 0) {{
                console.log('Sample filtered zip:', filtered[0].properties.zipcode_clean, 'distance:', filtered[0].properties.distance_to_nearest_cbsa, 'stddev:', filtered[0].properties.std_dev_from_cbsa);
            }}
            return filtered;
        }}

        // Calculate corridors for FILTERED non-CBSA zip codes only
        function calculateFilteredCorridors(corridorWidth) {{
            console.log('Calculating corridors for FILTERED non-CBSA zip codes with width:', corridorWidth, 'miles');

            // Find only the FILTERED non-CBSA zip codes that are currently visible
            var filteredNonCbsaZips = getFilteredNonCbsaZips();
            var assignedCbsaZips = allUsaData.filter(zip => zip.properties.has_cbsa);

            if (assignedCbsaZips.length === 0) {{
                console.log('No assigned CBSA zips available for corridor analysis');
                return [];
            }}

            console.log('Analyzing corridors for', filteredNonCbsaZips.length, 'filtered non-CBSA zips to', assignedCbsaZips.length, 'CBSA zips');

            if (filteredNonCbsaZips.length === 0) {{
                console.log('No filtered non-CBSA zip codes found - no corridors to calculate');
                return [];
            }}

            var allCorridorZips = new Set(); // Use Set to avoid duplicates
            var processedCount = 0;

            // For each FILTERED non-CBSA zip, find its corridor to nearest CBSA
            filteredNonCbsaZips.forEach(targetZip => {{
                processedCount++;
                if (processedCount % 100 === 0) {{
                    console.log('Processed', processedCount, 'of', filteredNonCbsaZips.length, 'filtered non-CBSA zips');
                }}

                var targetCentroid = getZipCentroid(targetZip);

                // Find nearest assigned CBSA zip code
                var nearestCbsaZip = null;
                var minDistance = Infinity;

                assignedCbsaZips.forEach(cbsaZip => {{
                    var cbsaCentroid = getZipCentroid(cbsaZip);
                    var distance = calculateDistance(
                        targetCentroid[0], targetCentroid[1],
                        cbsaCentroid[0], cbsaCentroid[1]
                    );

                    if (distance < minDistance) {{
                        minDistance = distance;
                        nearestCbsaZip = cbsaZip;
                    }}
                }});

                if (!nearestCbsaZip) return;

                var nearestCentroid = getZipCentroid(nearestCbsaZip);

                // Find all zip codes within corridor width of the line between target and nearest assigned CBSA
                allUsaData.forEach(zip => {{
                    // Skip if already processed or if it's the target/destination
                    if (allCorridorZips.has(zip.properties.zipcode_clean) ||
                        zip.properties.zipcode_clean === targetZip.properties.zipcode_clean ||
                        zip.properties.zipcode_clean === nearestCbsaZip.properties.zipcode_clean) {{
                        return;
                    }}

                    var zipCentroid = getZipCentroid(zip);
                    var distanceToLine = pointToLineDistance(
                        zipCentroid[0], zipCentroid[1],
                        targetCentroid[0], targetCentroid[1],
                        nearestCentroid[0], nearestCentroid[1]
                    );

                    if (distanceToLine <= corridorWidth) {{
                        // Also check if the zip is roughly between the target and assigned CBSA
                        var distanceToTarget = calculateDistance(
                            zipCentroid[0], zipCentroid[1],
                            targetCentroid[0], targetCentroid[1]
                        );
                        var distanceToCbsa = calculateDistance(
                            zipCentroid[0], zipCentroid[1],
                            nearestCentroid[0], nearestCentroid[1]
                        );

                        // Only include if the zip is roughly on the path (not too far out of the way)
                        if (distanceToTarget + distanceToCbsa <= minDistance * 1.3) {{
                            allCorridorZips.add(zip.properties.zipcode_clean);
                        }}
                    }}
                }});
            }});

            // Convert Set back to array of zip objects
            var corridorZipObjects = [];
            allCorridorZips.forEach(zipCode => {{
                var zipObj = allUsaData.find(zip => zip.properties.zipcode_clean === zipCode);
                if (zipObj) {{
                    corridorZipObjects.push(zipObj);
                }}
            }});

            console.log('Total corridor zip codes found:', corridorZipObjects.length);
            console.log('Sample corridor zip codes:', corridorZipObjects.slice(0, 3));

            console.log('Found', corridorZipObjects.length, 'total corridor zip codes across USA');
            console.log('Sample corridor zip codes:', corridorZipObjects.slice(0, 5).map(z => z.properties.zipcode_clean));
            return corridorZipObjects;
        }}

        // Toggle ALL corridors on/off
        function toggleAllCorridors() {{
            console.log('toggleAllCorridors called, corridorsVisible =', corridorsVisible);
            if (corridorsVisible) {{
                // Hide all corridors
                console.log('Hiding all corridors');
                corridorLayers.forEach(layer => map.removeLayer(layer));
                corridorLayers = [];
                corridorsVisible = false;
            }} else {{
                // Show corridors for filtered non-CBSA zip codes
                console.log('Showing corridors for filtered non-CBSA zip codes');
                console.log('Current corridor width:', currentCorridorWidth);
                var corridorZips = calculateFilteredCorridors(currentCorridorWidth);
                console.log('Corridor calculation returned', corridorZips.length, 'zip codes');

                console.log('Attempting to display', corridorZips.length, 'corridor zip codes');
                if (corridorZips.length > 0) {{
                    corridorZips.forEach((zip, index) => {{
                        if (index < 3) {{
                            console.log('Corridor zip', index, ':', zip.properties?.zipcode_clean, 'has geometry:', !!zip.geometry);
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
                                    <strong>🛣️ Corridor Zip: ${{feature.properties.zipcode_clean}}</strong><br>
                                    <strong>Quotes:</strong> ${{feature.properties.total_quotes}}<br>
                                    <strong>CBSA:</strong> ${{feature.properties.cbsa_name || 'Non-CBSA'}}<br>
                                    <hr style="margin: 5px 0;">
                                    <em>Part of corridor pathway to nearest CBSA coverage</em>
                                `);

                                layer.bindTooltip(
                                    '🛣️ Corridor: ' + feature.properties.zipcode_clean + ' | ' + feature.properties.total_quotes + ' quotes',
                                    {{ permanent: false, direction: 'top' }}
                                );
                            }}
                        }}).addTo(map);

                        corridorLayers.push(layer);
                    }});

                    corridorsVisible = true;
                    console.log('Successfully added', corridorZips.length, 'corridor zip shapes to map');
                }} else {{
                    console.log('No corridor zip codes found to display');
                }}
            }}
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


        
        // Color functions (same enhanced color scheme as before)
        function interpolateColor(color1, color2, factor) {{
            var result = color1.slice();
            for (var i = 0; i < 3; i++) {{
                result[i] = Math.round(result[i] + factor * (color2[i] - result[i]));
            }}
            return result;
        }}
        
        function rgbToHex(rgb) {{
            return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
        }}
        
        function getQuoteColor(feature) {{
            var props = feature.properties;
            var percentile = props.quote_percentile;

            // Special color for unassigned CBSA zip codes
            if (props.is_unassigned_cbsa) {{
                return '#ffd700'; // Gold/yellow for unassigned CBSA zips
            }}

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
        
        function getZipStyle(feature) {{
            var props = feature.properties;
            return {{
                fillColor: getQuoteColor(feature),
                weight: props.was_unassigned ? 3 : 1,
                opacity: 1,
                color: props.was_unassigned ? '#ff8c00' : '#333',
                dashArray: props.was_unassigned ? '5,5' : null,
                fillOpacity: 0.7 + 0.3 * props.quote_percentile
            }};
        }}
        
        function createPopupContent(feature) {{
            var props = feature.properties;
            var cbsaStatus = props.has_cbsa ? 'CBSA-Assigned' : 'Non-CBSA';
            var cbsaAnalysisInfo = '';
            if (props.has_cbsa) {{
                // For CBSA-assigned zip codes
                cbsaAnalysisInfo = '<strong>CBSA:</strong> ' + (props.cbsa_name || 'Unknown') + '<br>' +
                                  '<strong>CBSA average quotes:</strong> ' + (props.cbsa_mean_quotes || 0).toFixed(0) + '<br>';
            }} else {{
                // For non-CBSA zip codes
                cbsaAnalysisInfo = '<strong>Distance to nearest CBSA zip:</strong> ' + props.distance_to_cbsa.toFixed(1) + ' miles<br>' +
                                  '<strong>Nearest CBSA zip code:</strong> ' + (props.nearest_cbsa_zip || 'Unknown') + '<br>' +
                                  '<strong>Nearest CBSA:</strong> ' + (props.nearest_cbsa_name || 'Unknown') + '<br>' +
                                  '<strong>Nearest CBSA average quotes:</strong> ' + (props.cbsa_mean_quotes || 0).toFixed(0) + '<br>' +
                                  '<strong>Quote performance vs CBSA:</strong> ' + props.std_dev_below_cbsa_mean.toFixed(1) + ' std dev below mean<br>' +
                                  '<strong>Percentile in nearest CBSA:</strong> ' + props.quote_percentile_in_nearest_cbsa.toFixed(1) + '%<br>';
            }}
            
            return `
                <div style="font-family: Arial; max-width: 300px;">
                    <h4 style="margin: 0 0 10px 0;">Zip: ${{props.zipcode}} (${{cbsaStatus}})</h4>
                    <hr style="margin: 5px 0;">
                    <strong>Location:</strong> ${{props.city}}, ${{props.state}}<br>
                    ${{cbsaAnalysisInfo}}
                    <strong>Total Quotes:</strong> ${{props.total_quotes.toLocaleString()}}<br>
                    <strong>Population:</strong> ${{props.population.toLocaleString()}}<br>
                    <strong>Quote Percentile:</strong> ${{(props.quote_percentile * 100).toFixed(1)}}%
                </div>
            `;
        }}
        
        function addCentroidMarkers() {{
            // Clear existing centroid markers
            centroidLayers.forEach(function(layer) {{
                map.removeLayer(layer);
            }});
            centroidLayers = [];
            
            if (!document.getElementById('showCentroids').checked) {{
                return;
            }}
            
            cbsaCentroids.forEach(function(centroid) {{
                var marker = L.circleMarker([centroid.lat, centroid.lon], {{
                    radius: Math.min(15, Math.max(5, Math.log10(centroid.total_quotes + 1) * 2)),
                    fillColor: '#ff4444',
                    color: '#cc0000',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.7
                }});
                
                marker.bindPopup(`
                    <div style="font-family: Arial;">
                        <h4 style="margin: 0 0 10px 0;">CBSA Weighted Centroid</h4>
                        <hr style="margin: 5px 0;">
                        <strong>CBSA:</strong> ${{centroid.cbsa_name}}<br>
                        <strong>Total Quotes:</strong> ${{centroid.total_quotes.toLocaleString()}}<br>
                        <strong>Zip Codes:</strong> ${{centroid.zip_count}}<br>
                        <small><i>Centroid weighted by quote volume</i></small>
                    </div>
                `);
                
                marker.bindTooltip(
                    centroid.cbsa_name.substring(0, 30) + '... | ' + centroid.total_quotes.toLocaleString() + ' quotes',
                    {{ permanent: false, direction: 'top' }}
                );
                
                marker.addTo(map);
                centroidLayers.push(marker);
            }});
        }}

        function addCrossdockMarkers() {{
            // Clear existing crossdock markers
            crossdockLayers.forEach(function(layer) {{
                map.removeLayer(layer);
            }});
            crossdockLayers = [];

            if (!document.getElementById('showCrossdocks').checked) {{
                return;
            }}

            crossdockLocations.forEach(function(crossdock) {{
                // Create diamond-shaped marker using CSS
                var crossdockIcon = L.divIcon({{
                    className: 'crossdock-marker',
                    html: '<div style="width: 20px; height: 20px; background: #28a745; transform: rotate(45deg); border: 2px solid #155724; margin: -10px 0 0 -10px;"></div>',
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                }});

                var marker = L.marker([crossdock.lat, crossdock.lon], {{
                    icon: crossdockIcon
                }});

                marker.bindPopup(`
                    <div style="font-family: Arial;">
                        <h4 style="margin: 0 0 10px 0; color: #28a745;">🏭 Crossdock Location</h4>
                        <hr style="margin: 5px 0;">
                        <strong>Name:</strong> ${{crossdock.name}}<br>
                        <strong>Zip Code:</strong> ${{crossdock.zip_code}}<br>
                        <strong>Coordinates:</strong> ${{crossdock.lat.toFixed(4)}}, ${{crossdock.lon.toFixed(4)}}
                    </div>
                `);

                marker.bindTooltip(
                    '🏭 ' + crossdock.name + ' (' + crossdock.zip_code + ')',
                    {{ permanent: false, direction: 'top' }}
                );

                marker.addTo(map);
                crossdockLayers.push(marker);
            }});
        }}



        // Export filtered data function
        function exportFilteredData() {{
            console.log('Exporting filtered data...');

            // Get currently displayed zip codes
            var displayedZipCodes = [];

            allZipData.forEach(function(feature) {{
                var props = feature.properties;
                var shouldInclude = false;

                if (props.has_cbsa && document.getElementById('showAllCBSA').checked) {{
                    shouldInclude = true;
                }} else if (!props.has_cbsa) {{
                    // Check distance threshold
                    var meetsDistanceThreshold = props.distance_to_cbsa <= currentDistanceThreshold;

                    // Check statistical threshold only if filter is enabled
                    var meetsStdDevThreshold = true;
                    if (stdDevFilterEnabled) {{
                        meetsStdDevThreshold = (-props.std_dev_below_cbsa_mean) >= currentStdDevThreshold;
                    }}

                    if (document.getElementById('showNearbyNonCBSA').checked && meetsDistanceThreshold && meetsStdDevThreshold) {{
                        shouldInclude = true;
                    }}
                }}

                if (shouldInclude) {{
                    // Calculate nearest crossdock if we have crossdocks
                    var nearestCrossdockInfo = {{ distance: null, crossdock: null }};
                    if (crossdockLocations.length > 0) {{
                        // Get zip code centroid (approximate from geometry center)
                        var bounds = L.geoJSON(feature).getBounds();
                        var centerLat = (bounds.getNorth() + bounds.getSouth()) / 2;
                        var centerLon = (bounds.getEast() + bounds.getWest()) / 2;
                        nearestCrossdockInfo = findNearestCrossdock(centerLat, centerLon);
                    }}

                    displayedZipCodes.push({{
                        zip_code: props.zipcode,
                        city: props.city,
                        state: props.state,
                        total_quotes: props.total_quotes,
                        pickup_count: props.pickup_count,
                        dropoff_count: props.dropoff_count,
                        population: props.population,
                        assigned_cbsa: props.assigned_cbsa || '',
                        closest_cbsa: props.closest_cbsa || '',
                        distance_to_nearest_cbsa_miles: props.distance_to_cbsa.toFixed(2),
                        nearest_cbsa_zip_code: props.nearest_cbsa_zip || '',
                        std_dev_vs_nearest_cbsa: props.std_dev_below_cbsa_mean.toFixed(2),
                        quote_percentile_in_nearest_cbsa: props.quote_percentile_in_nearest_cbsa.toFixed(1),
                        nearest_crossdock_zip: nearestCrossdockInfo.crossdock ? nearestCrossdockInfo.crossdock.zip_code : '',
                        nearest_crossdock_name: nearestCrossdockInfo.crossdock ? nearestCrossdockInfo.crossdock.name : '',
                        distance_to_nearest_crossdock_miles: nearestCrossdockInfo.distance ? nearestCrossdockInfo.distance.toFixed(2) : '',
                        was_recently_unassigned: props.was_unassigned ? 'Yes' : 'No'
                    }});
                }}
            }});

            if (displayedZipCodes.length === 0) {{
                alert('No zip codes are currently displayed to export.');
                return;
            }}

            // Create CSV content
            var csvContent = [
                'Zip_Code',
                'City',
                'State',
                'Total_Quotes',
                'Pickup_Count',
                'Dropoff_Count',
                'Population',
                'Assigned_CBSA',
                'Closest_CBSA',
                'Distance_to_Nearest_CBSA_Miles',
                'Nearest_CBSA_Zip_Code',
                'Std_Dev_vs_Nearest_CBSA',
                'Quote_Percentile_in_Nearest_CBSA',
                'Nearest_Crossdock_Zip',
                'Nearest_Crossdock_Name',
                'Distance_to_Nearest_Crossdock_Miles',
                'Was_Recently_Unassigned'
            ].join(',') + '\\n';

            displayedZipCodes.forEach(function(zip) {{
                csvContent += [
                    zip.zip_code,
                    '"' + zip.city + '"',
                    zip.state,
                    zip.total_quotes,
                    zip.pickup_count,
                    zip.dropoff_count,
                    zip.population,
                    '"' + zip.assigned_cbsa + '"',
                    '"' + zip.closest_cbsa + '"',
                    zip.distance_to_nearest_cbsa_miles,
                    zip.nearest_cbsa_zip_code,
                    zip.std_dev_vs_nearest_cbsa,
                    zip.quote_percentile_in_nearest_cbsa,
                    zip.nearest_crossdock_zip,
                    '"' + zip.nearest_crossdock_name + '"',
                    zip.distance_to_nearest_crossdock_miles,
                    zip.was_recently_unassigned
                ].join(',') + '\\n';
            }});

            // Create and download file
            var blob = new Blob([csvContent], {{ type: 'text/csv' }});
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;

            var timestamp = new Date().toISOString().slice(0,19).replace(/:/g, '-');
            var filterDesc = stdDevFilterEnabled ?
                '_dist' + currentDistanceThreshold + '_stddev' + currentStdDevThreshold.toFixed(1) :
                '_dist' + currentDistanceThreshold;
            a.download = 'filtered_zip_analysis' + filterDesc + '_' + timestamp + '.csv';

            a.click();
            window.URL.revokeObjectURL(url);

            alert('Exported ' + displayedZipCodes.length + ' zip codes to CSV file!\\n\\nColumns included:\\n' +
                  '• Basic info (zip, city, state, quotes, population)\\n' +
                  '• CBSA analysis (assigned vs closest CBSA, distance, std dev)\\n' +
                  '• Crossdock analysis (nearest crossdock, distance)\\n' +
                  '• Filter status (recently unassigned)');
        }}
        
        function updateMap() {{
            console.log('Updating map with distance threshold:', currentDistanceThreshold);
            
            // Clear existing layers
            Object.values(zipLayers).forEach(function(layer) {{
                map.removeLayer(layer);
            }});
            zipLayers = {{}};
            
            var showAllCBSA = document.getElementById('showAllCBSA').checked;
            var showNearbyNonCBSA = document.getElementById('showNearbyNonCBSA').checked;

            var cbsaCount = 0;
            var nonCBSACount = 0;
            var filteredNonCBSACount = 0;
            
            allZipData.forEach(function(feature) {{
                var props = feature.properties;
                var shouldShow = false;
                
                if (props.has_cbsa && showAllCBSA) {{
                    shouldShow = true;
                    cbsaCount++;
                }} else if (!props.has_cbsa) {{
                    nonCBSACount++;
                    // Check distance threshold
                    var meetsDistanceThreshold = props.distance_to_cbsa <= currentDistanceThreshold;

                    // Check statistical threshold only if filter is enabled
                    var meetsStdDevThreshold = true;
                    if (stdDevFilterEnabled) {{
                        meetsStdDevThreshold = (-props.std_dev_below_cbsa_mean) >= currentStdDevThreshold;
                    }}

                    if (showNearbyNonCBSA && meetsDistanceThreshold && meetsStdDevThreshold) {{
                        shouldShow = true;
                        filteredNonCBSACount++;
                    }}
                }}
                
                if (shouldShow) {{
                    var layer = L.geoJSON(feature, {{
                        style: getZipStyle,
                        onEachFeature: function(feature, layer) {{
                            layer.bindPopup(createPopupContent(feature));



                            var tooltip = 'Zip: ' + feature.properties.zipcode +
                                         ' | Quotes: ' + feature.properties.total_quotes.toLocaleString();

                            // Add CBSA mean quotes information
                            if (feature.properties.cbsa_mean_quotes > 0) {{
                                var cbsaType = feature.properties.has_cbsa ? 'CBSA' : 'Nearest CBSA';
                                tooltip += ' | ' + cbsaType + ' Avg: ' + feature.properties.cbsa_mean_quotes.toFixed(0);
                            }}

                            if (!feature.properties.has_cbsa) {{
                                tooltip += ' | Dist: ' + feature.properties.distance_to_cbsa.toFixed(1) + 'mi';
                            }}



                            layer.bindTooltip(tooltip, {{ permanent: false, direction: 'top' }});
                        }}
                    }}).addTo(map);
                    
                    zipLayers[props.zipcode_clean] = layer;
                }}
            }});
            
            // Update centroid and crossdock markers
            addCentroidMarkers();
            addCrossdockMarkers();
            
            // Update stats
            var filterDescription = '';
            if (stdDevFilterEnabled) {{
                var stdDevSign = currentStdDevThreshold >= 0 ? '+' : '';
                filterDescription = '≤ ' + currentDistanceThreshold + ' miles & ≥ ' + stdDevSign + currentStdDevThreshold.toFixed(1) + ' std dev';
            }} else {{
                filterDescription = '≤ ' + currentDistanceThreshold + ' miles';
            }}

            document.getElementById('filterStats').innerHTML =
                '<strong>Currently Displayed:</strong><br>' +
                '🔵 CBSA zip codes: ' + cbsaCount.toLocaleString() + '<br>' +
                '🔴 Non-CBSA (' + filterDescription + '): ' + filteredNonCBSACount.toLocaleString() + '<br>' +
                '📊 Total non-CBSA zip codes: ' + nonCBSACount.toLocaleString() + '<br>' +
                '🎯 CBSA centroids: ' + cbsaCentroids.length + '<br>' +
                '<strong>📍 Total displayed: ' + (cbsaCount + filteredNonCBSACount).toLocaleString() + '</strong>';
            
            console.log('Map updated - CBSA:', cbsaCount, 'Non-CBSA shown:', filteredNonCBSACount, 'of', nonCBSACount);
        }}
        
        // Event listeners
        function updateDistance() {{
            var slider = document.getElementById('distanceSlider');
            currentDistanceThreshold = parseInt(slider.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            updateMap();

            // If corridors are currently visible, refresh them with new filtered data
            if (corridorsVisible) {{
                toggleAllCorridors(); // Hide
                toggleAllCorridors(); // Show with new filtered data
            }}
        }}

        function updateStdDev() {{
            var slider = document.getElementById('stdDevSlider');
            currentStdDevThreshold = parseFloat(slider.value);
            var sign = currentStdDevThreshold >= 0 ? '+' : '';
            document.getElementById('stdDevDisplay').textContent = '≥ ' + sign + currentStdDevThreshold.toFixed(1) + ' std dev';
            updateMap();

            // If corridors are currently visible, refresh them with new filtered data
            if (corridorsVisible) {{
                toggleAllCorridors(); // Hide
                toggleAllCorridors(); // Show with new filtered data
            }}
        }}

        function toggleStdDevFilter() {{
            stdDevFilterEnabled = document.getElementById('enableStdDevFilter').checked;
            var container = document.getElementById('stdDevContainer');

            if (stdDevFilterEnabled) {{
                container.classList.remove('disabled');
            }} else {{
                container.classList.add('disabled');
            }}

            updateMap();

            // If corridors are currently visible, refresh them with new filtered data
            if (corridorsVisible) {{
                toggleAllCorridors(); // Hide
                toggleAllCorridors(); // Show with new filtered data
            }}
        }}



        document.getElementById('distanceSlider').addEventListener('input', updateDistance);
        document.getElementById('distanceSlider').addEventListener('change', updateDistance);

        document.getElementById('stdDevSlider').addEventListener('input', updateStdDev);
        document.getElementById('stdDevSlider').addEventListener('change', updateStdDev);

        document.getElementById('enableStdDevFilter').addEventListener('change', toggleStdDevFilter);

        document.getElementById('showAllCBSA').addEventListener('change', updateMap);
        document.getElementById('showNearbyNonCBSA').addEventListener('change', updateMap);
        document.getElementById('showCentroids').addEventListener('change', updateMap);
        document.getElementById('showCrossdocks').addEventListener('change', updateMap);

        // Corridor controls
        document.getElementById('showCorridors').addEventListener('change', function(e) {{
            toggleAllCorridors();
        }});

        document.getElementById('corridorWidth').addEventListener('input', function(e) {{
            currentCorridorWidth = parseInt(e.target.value);
            document.getElementById('corridorWidthDisplay').textContent = currentCorridorWidth + ' miles';

            // If corridors are currently visible, refresh them with new width
            if (corridorsVisible) {{
                toggleAllCorridors(); // Hide
                toggleAllCorridors(); // Show with new width
            }}
        }});

        // Initial load
        setTimeout(function() {{
            updateMap();
            console.log('CBSA proximity map initialized successfully');
        }}, 500);
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    """Main function"""
    try:
        # Load data
        merged_data, zip_shapes, unassigned_zips = load_and_prepare_data()

        # Load crossdock data
        crossdock_data = load_crossdock_data()
        crossdock_locations = process_crossdock_locations(crossdock_data, zip_shapes)

        # Calculate weighted CBSA centroids
        map_data, cbsa_centroids_df = calculate_weighted_cbsa_centroids(merged_data, zip_shapes)

        # Calculate distances to nearest CBSA zip codes
        map_data = calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df)

        # Load ALL USA zip codes for corridor analysis
        print("Loading ALL USA zip codes for corridor analysis...")
        all_usa_data = load_all_usa_zip_data_for_corridors(merged_data, zip_shapes)

        # Create map with corridor functionality
        html_content = create_cbsa_proximity_map(map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data)
        
        if html_content:
            # Save map
            map_file = 'cbsa_weighted_proximity_map.html'
            with open(map_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\nCBSA weighted proximity map saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("CBSA WEIGHTED PROXIMITY MAP FEATURES:")
            print("✅ Quote-weighted CBSA centroids for accurate distance calculation")
            print("✅ Working distance slider with enhanced styling")
            print("✅ Distances calculated to business activity centers")
            print("✅ Visual CBSA centroid markers (red circles)")
            print("✅ Enhanced color sensitivity with logarithmic scaling")
            print("✅ Real-time statistics and filtering")
            print("="*60)
            
        else:
            print("Failed to create map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
