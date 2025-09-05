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
    """Create a CBSA proximity map with improved corridor analysis"""
    print("Creating CBSA proximity map with improved corridor analysis...")

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

    return zip_features, cbsa_centroid_features

def generate_html_map(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js):
    """Generate the HTML map with improved corridor analysis"""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IMPROVED Top 75 CBSAs Proximity Map with Smart Corridors</title>
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
        <h4>🚀 <span class="improvement-badge">IMPROVED</span> Smart Corridor Analysis</h4>

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

        <div class="slider-container" id="stdDevContainer">
            <label for="stdDevSlider">Min Quote Performance vs Nearest CBSA:</label>
            <div class="slider-wrapper">
                <input type="range" id="stdDevSlider" min="-3" max="2" value="-3" step="0.5">
            </div>
            <div class="distance-display" id="stdDevDisplay" style="font-size: 16px;">≥ -3.0 std dev</div>
            <div class="range-labels">
                <span>-3 std dev</span>
                <span>+2 std dev</span>
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
            <strong>🚀 SMART Corridor Improvements:</strong><br>
            ✅ Distance limits prevent overly long corridors<br>
            ✅ Dynamic width: shorter = wider, longer = narrower<br>
            ✅ Only shows corridors for filtered non-CBSA zips<br>
            ✅ Dramatically reduces visual clutter
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
            🎯 Red circles = CBSA weighted centroids<br>
            🏭 Green diamonds = Crossdock locations<br>
            🚀 Orange areas = SMART corridor zones (distance + width limited)<br>
            📏 Distances calculated to nearest CBSA zip codes
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>"""

    return html_content

def create_complete_html_map(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js):
    """Create the complete HTML map with all JavaScript functionality"""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IMPROVED Top 75 CBSAs Proximity Map with Smart Corridors</title>
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
        <h4>🚀 <span class="improvement-badge">IMPROVED</span> Smart Corridor Analysis</h4>

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

        <div class="slider-container" id="stdDevContainer">
            <label for="stdDevSlider">Min Quote Performance vs Nearest CBSA:</label>
            <div class="slider-wrapper">
                <input type="range" id="stdDevSlider" min="-3" max="2" value="-3" step="0.5">
            </div>
            <div class="distance-display" id="stdDevDisplay" style="font-size: 16px;">≥ -3.0 std dev</div>
            <div class="range-labels">
                <span>-3 std dev</span>
                <span>+2 std dev</span>
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
            <strong>🚀 SMART Corridor Improvements:</strong><br>
            ✅ Distance limits prevent overly long corridors<br>
            ✅ Dynamic width: shorter = wider, longer = narrower<br>
            ✅ Only shows corridors for filtered non-CBSA zips<br>
            ✅ Dramatically reduces visual clutter
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
            🎯 Red circles = CBSA weighted centroids<br>
            🏭 Green diamonds = Crossdock locations<br>
            🚀 Orange areas = SMART corridor zones (distance + width limited)<br>
            📏 Distances calculated to nearest CBSA zip codes
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

        console.log('🚀 IMPROVED Smart Corridor Map loaded with:');
        console.log('   -', allZipData.length, 'zip codes');
        console.log('   -', cbsaCentroids.length, 'CBSA centroids');
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
                if (stdDevFilterEnabled && zip.properties.std_dev_below_cbsa_mean > minStdDev) return false;

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
                    console.log('   🛣️  Creating corridor:', targetZip.properties.zipcode_clean, '→ CBSA', nearestCbsaCentroid.cbsa_title, '(' + minDistance.toFixed(1) + ' mi, width: ' + dynamicWidth.toFixed(1) + ' mi)');

                    var corridorZipsForThisPath = 0;
                    var zipsWithinWidth = 0;
                    var zipsFailingPathCheck = 0;

                    // Find all zip codes within dynamic corridor width of the line
                    allUsaData.forEach(zip => {{
                        // Skip if already processed or if it's the target
                        var zipCode = zip.properties.zipcode_clean;
                        if (allCorridorZips.has(zipCode) ||
                            zipCode === targetZip.properties.zipcode_clean) {{
                            return;
                        }}

                        var zipCentroid = getZipCentroid(zip);
                        var distanceToLine = pointToLineDistance(
                            zipCentroid[0], zipCentroid[1], // lat, lon for point
                            targetCentroid[0], targetCentroid[1], // lat, lon for line start
                            nearestCbsaCentroid.lat, nearestCbsaCentroid.lon // lat, lon for line end (CBSA centroid)
                        );

                        if (distanceToLine <= dynamicWidth) {{
                            zipsWithinWidth++;

                            // Check if the zip is roughly between the target and CBSA centroid (not too far out of the way)
                            var distanceToTarget = calculateDistance(
                                zipCentroid[0], zipCentroid[1],
                                targetCentroid[0], targetCentroid[1]
                            );
                            var distanceToCbsa = calculateDistance(
                                zipCentroid[0], zipCentroid[1],
                                nearestCbsaCentroid.lat, nearestCbsaCentroid.lon
                            );

                            var totalDetourDistance = distanceToTarget + distanceToCbsa;
                            // For short corridors, use a more generous detour allowance
                            var detourMultiplier = minDistance < 20 ? 3.0 : (minDistance < 50 ? 2.0 : 1.5);
                            var maxAllowedDistance = minDistance * detourMultiplier;

                            // Only include if the zip is roughly on the path
                            if (totalDetourDistance <= maxAllowedDistance) {{
                                allCorridorZips.add(zipCode);
                                corridorZipsForThisPath++;
                            }} else {{
                                zipsFailingPathCheck++;
                            }}
                        }}
                    }});

                    var detourMultiplier = minDistance < 20 ? 3.0 : (minDistance < 50 ? 2.0 : 1.5);
                    console.log('     Zips within', dynamicWidth.toFixed(1), 'mile width:', zipsWithinWidth);
                    console.log('     Zips failing path check (>' + (minDistance * detourMultiplier).toFixed(1) + ' mile detour, ' + detourMultiplier + 'x):', zipsFailingPathCheck);

                    console.log('     Found', corridorZipsForThisPath, 'corridor zips for this path');
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
                var zipObj = allUsaData.find(zip => zip.properties.zipcode_clean === zipCode);
                if (zipObj) {{
                    corridorZipObjects.push(zipObj);
                }}
            }});

            console.log('✅ SMART Corridor Analysis Results:');
            console.log('   - Total potential corridors:', corridorStats.total);
            console.log('   - Corridors within', maxCorridorDistance, 'mile limit:', corridorStats.withinLimit);
            console.log('   - Average corridor distance:', corridorStats.avgDistance.toFixed(1), 'miles');
            console.log('   - Average corridor width:', corridorStats.avgWidth.toFixed(1), 'miles');
            console.log('   - Total corridor zip codes found:', corridorZipObjects.length);

            return corridorZipObjects;
        }}

        // Toggle SMART corridors on/off
        function toggleSmartCorridors() {{
            console.log('toggleSmartCorridors called, smartCorridorsVisible =', smartCorridorsVisible);
            if (smartCorridorsVisible) {{
                // Hide all corridors
                console.log('Hiding all SMART corridors');
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
        }}

        // Color functions
        function interpolateColor(color1, color2, factor) {{
            var result = color1.slice();
            for (var i = 0; i < 3; i++) {{
                result[i] = Math.round(result[i] + factor * (color2[i] - result[i]));
            }}
            return result;
        }}

        function rgbToHex(r, g, b) {{
            return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }}

        function getQuoteColor(percentile, hasCbsa) {{
            if (hasCbsa) {{
                // Enhanced blue scale for CBSA zip codes
                var colors = [
                    [248, 251, 255], // Very light blue
                    [198, 219, 237], // Light blue
                    [107, 174, 214], // Medium blue
                    [49, 130, 189],  // Dark blue
                    [8, 48, 107]     // Very dark blue
                ];
            }} else {{
                // Enhanced red scale for non-CBSA zip codes
                var colors = [
                    [255, 245, 240], // Very light red
                    [254, 224, 210], // Light red
                    [252, 146, 114], // Medium red
                    [222, 45, 38],   // Dark red
                    [165, 15, 21]    // Very dark red
                ];
            }}

            var index = percentile * (colors.length - 1);
            var lowerIndex = Math.floor(index);
            var upperIndex = Math.ceil(index);
            var factor = index - lowerIndex;

            if (lowerIndex === upperIndex) {{
                return rgbToHex(colors[lowerIndex][0], colors[lowerIndex][1], colors[lowerIndex][2]);
            }}

            var interpolated = interpolateColor(colors[lowerIndex], colors[upperIndex], factor);
            return rgbToHex(interpolated[0], interpolated[1], interpolated[2]);
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
                        if (!stdDevFilterEnabled || props.std_dev_below_cbsa_mean <= minStdDev) {{
                            shouldShow = true;
                            nonCbsaCount++;
                        }}
                    }}
                }}

                if (shouldShow) {{
                    totalDisplayed++;
                    var color = getQuoteColor(props.quote_percentile, props.has_cbsa);

                    var layer = L.geoJSON(zipData, {{
                        style: {{
                            fillColor: color,
                            color: props.has_cbsa ? '#2171b5' : '#cb181d',
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
                                     <strong>Performance vs CBSA:</strong> ${{feature.properties.std_dev_below_cbsa_mean.toFixed(1)}} std dev<br>` : ''}}
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

            // Update statistics
            document.getElementById('filterStats').innerHTML = `
                <strong>Currently Displayed:</strong><br>
                🔵 CBSA zip codes: ${{cbsaCount.toLocaleString()}}<br>
                🔴 Non-CBSA (≤ ${{maxDistance}} miles & ≥ ${{minStdDev.toFixed(1)}} std dev): ${{nonCbsaCount.toLocaleString()}}<br>
                📊 Total non-CBSA zip codes: ${{allZipData.filter(z => !z.properties.has_cbsa).length.toLocaleString()}}<br>
                🎯 CBSA centroids: ${{showCentroids ? cbsaCentroids.length : 0}}<br>
                🏭 Crossdocks: ${{showCrossdocks ? crossdockLocations.length : 0}}<br>
                📍 Total displayed: ${{totalDisplayed.toLocaleString()}}
            `;
        }}

        // Export filtered data
        function exportFilteredData() {{
            var maxDistance = parseFloat(document.getElementById('distanceSlider').value);
            var minStdDev = parseFloat(document.getElementById('stdDevSlider').value);

            var exportData = [];

            allZipData.forEach(function(zipData) {{
                var props = zipData.properties;
                var shouldInclude = false;

                if (props.has_cbsa) {{
                    shouldInclude = document.getElementById('showAllCBSA').checked;
                }} else {{
                    if (props.distance_to_cbsa <= maxDistance) {{
                        if (!stdDevFilterEnabled || props.std_dev_below_cbsa_mean <= minStdDev) {{
                            shouldInclude = document.getElementById('showNearbyNonCBSA').checked;
                        }}
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
                        'Std_Dev_vs_CBSA_Mean': props.std_dev_below_cbsa_mean,
                        'Quote_Percentile_in_Nearest_CBSA': props.quote_percentile_in_nearest_cbsa,
                        'CBSA_Mean_Quotes': props.cbsa_mean_quotes,
                        'Was_Unassigned': props.was_unassigned,
                        'Nearest_Crossdock': nearestCrossdock.crossdock ? nearestCrossdock.crossdock.name : '',
                        'Distance_to_Crossdock_Miles': nearestCrossdock.crossdock ? nearestCrossdock.distance.toFixed(1) : ''
                    }});
                }}
            }});

            // Convert to CSV
            if (exportData.length === 0) {{
                alert('No data to export with current filters');
                return;
            }}

            var csv = Object.keys(exportData[0]).join(',') + '\\n';
            exportData.forEach(function(row) {{
                csv += Object.values(row).map(function(value) {{
                    return typeof value === 'string' && value.includes(',') ? '"' + value + '"' : value;
                }}).join(',') + '\\n';
            }});

            // Download CSV
            var blob = new Blob([csv], {{ type: 'text/csv' }});
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'improved_cbsa_proximity_filtered_data_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            console.log('📥 Exported', exportData.length, 'zip codes to CSV');
        }}

        // Event handlers
        document.getElementById('distanceSlider').addEventListener('input', function() {{
            currentDistanceThreshold = parseFloat(this.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            updateMapDisplay();

            // Update corridors if they're visible
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new settings
            }}
        }});

        document.getElementById('stdDevSlider').addEventListener('input', function() {{
            currentStdDevThreshold = parseFloat(this.value);
            document.getElementById('stdDevDisplay').textContent = '≥ ' + currentStdDevThreshold.toFixed(1) + ' std dev';
            updateMapDisplay();

            // Update corridors if they're visible
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new settings
            }}
        }});

        document.getElementById('enableStdDevFilter').addEventListener('change', function() {{
            stdDevFilterEnabled = this.checked;
            document.getElementById('stdDevContainer').style.opacity = stdDevFilterEnabled ? '1' : '0.5';
            updateMapDisplay();

            // Update corridors if they're visible
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new settings
            }}
        }});

        document.getElementById('showAllCBSA').addEventListener('change', updateMapDisplay);
        document.getElementById('showNearbyNonCBSA').addEventListener('change', updateMapDisplay);
        document.getElementById('showCentroids').addEventListener('change', updateMapDisplay);
        document.getElementById('showCrossdocks').addEventListener('change', updateMapDisplay);

        document.getElementById('showSmartCorridors').addEventListener('change', function() {{
            toggleSmartCorridors();
        }});

        document.getElementById('maxCorridorDistance').addEventListener('input', function() {{
            currentMaxCorridorDistance = parseFloat(this.value);
            document.getElementById('maxCorridorDistanceDisplay').textContent = currentMaxCorridorDistance + ' miles';

            // Update corridors if they're visible
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new settings
            }}
        }});

        document.getElementById('baseCorridorWidth').addEventListener('input', function() {{
            currentBaseCorridorWidth = parseFloat(this.value);
            document.getElementById('baseCorridorWidthDisplay').textContent = currentBaseCorridorWidth + ' miles';

            // Update corridors if they're visible
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new settings
            }}
        }});

        // Initialize the map
        updateMapDisplay();

        console.log('🚀 IMPROVED Smart Corridor Map fully loaded and ready!');
    </script>
</body>
</html>"""

    return html_content

def main():
    """Main execution function"""
    print("🚀 Starting IMPROVED CBSA Proximity Map with Smart Corridor Analysis...")

    # Load and prepare data
    merged_data, zip_shapes, unassigned_zips = load_and_prepare_data()
    crossdock_data = load_crossdock_data()

    # Calculate weighted centroids and distances
    map_data, cbsa_centroids_df = calculate_weighted_cbsa_centroids(merged_data, zip_shapes)
    map_data = calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df)

    # Process crossdock locations
    crossdock_locations = process_crossdock_locations(crossdock_data, zip_shapes)

    # Load ALL USA data for corridor analysis
    all_usa_data = load_all_usa_zip_data_for_corridors(merged_data, zip_shapes)
    all_usa_data_js = prepare_all_usa_data_for_js(all_usa_data)

    # Create the map
    zip_features, cbsa_centroid_features = create_cbsa_proximity_map(
        map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data
    )

    print(f"✅ Map preparation complete:")
    print(f"   - {len(zip_features)} zip code features")
    print(f"   - {len(cbsa_centroid_features)} CBSA centroids")
    print(f"   - {len(crossdock_locations)} crossdock locations")
    print(f"   - {len(all_usa_data_js)} USA zip codes for corridor analysis")

    # Generate complete HTML with JavaScript
    html_content = create_complete_html_map(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js)

    # Save the HTML file
    output_file = BASE_DIR / 'cbsa_improved_corridor_map.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ IMPROVED CBSA Proximity Map generated successfully!")
    print(f"📁 Saved to: {output_file}")
    print(f"🌐 Open the file in your browser to view the interactive map")

    # Optionally open in browser
    try:
        import webbrowser
        webbrowser.open(f'file://{output_file.absolute()}')
        print("🚀 Opening map in your default browser...")
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
        print(f"Please manually open: {output_file}")

    print("\n🚀 IMPROVED Features Summary:")
    print("✅ Distance limits prevent overly long corridors")
    print("✅ Dynamic width: shorter corridors = wider, longer = narrower")
    print("✅ Only shows corridors for filtered non-CBSA zip codes")
    print("✅ Dramatically reduces visual clutter")
    print("✅ Real-time corridor parameter adjustment")
    print("✅ Enhanced performance statistics and filtering")

if __name__ == "__main__":
    main()

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
