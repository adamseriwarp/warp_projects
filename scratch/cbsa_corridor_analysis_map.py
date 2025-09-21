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
DOWNLOADS_DIR = Path.home() / 'Downloads'  # More portable than hard-coded path

def load_and_prepare_data():
    """Load and prepare the data for heat mapping"""
    print("Loading data...")

    # Load quote data
    quote_data = pd.read_csv(DATA_DIR / 'raw' / 'quote_data.csv')
    print(f"Loaded {len(quote_data)} quote records")

    # Load zip to CSA mapping
    zip_csa_mapping = pd.read_csv(DATA_DIR / 'raw' / 'zip_to_csa_mapping.csv', encoding='latin-1')
    print(f"Loaded {len(zip_csa_mapping)} zip-to-CSA mappings")

    # Load CSA population data
    csa_population = pd.read_excel(DATA_DIR / 'raw' / 'csa_population.xlsx')
    print(f"Loaded {len(csa_population)} CSA population records")

    # Clean and prepare data for merging
    quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
    quote_data['Total Quotes'] = quote_data['Pickup Count'] + quote_data['Dropoff Count']

    # Prepare zip-to-CSA mapping
    zip_csa_mapping['Zipcode_clean'] = zip_csa_mapping['Zip Code'].astype(str).str.zfill(5)

    # Merge quote data with zip-to-CSA mapping
    merged_data = quote_data.merge(zip_csa_mapping, on='Zipcode_clean', how='left')
    print(f"Merged data: {len(merged_data)} records")

    # Load unassigned zip codes if available
    unassigned_file = DOWNLOADS_DIR / 'unassigned_zip_codes.csv'
    if unassigned_file.exists():
        unassigned_df = pd.read_csv(unassigned_file)
        unassigned_zips = set(unassigned_df['Zipcode_clean'].astype(str))
        print(f"Loaded {len(unassigned_zips)} unassigned zip codes from CSV file")

        # Unassign these zip codes from their CBSAs
        print(f"Unassigning {len(unassigned_zips)} zip codes from their CBSAs...")
        mask = merged_data['Zipcode_clean'].astype(str).isin(unassigned_zips)
        merged_data.loc[mask, 'Primary CBSA Name'] = None
        print(f"Updated {mask.sum()} zip codes to remove CBSA assignments")
    else:
        unassigned_zips = set()
        print("No unassigned zip codes file found")

    # Load shapefiles
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file(DATA_DIR / 'shapefiles' / 'cb_2020_us_zcta520_500k.shp')
    zip_shapes = zip_shapes.to_crs('EPSG:3857')  # Web Mercator for distance calculations

    return merged_data, zip_shapes, unassigned_zips

def load_crossdock_data():
    """Load crossdock location data"""
    print("Loading crossdock data...")
    crossdock_file = DOWNLOADS_DIR / 'crossdock_locations.csv'
    if crossdock_file.exists():
        crossdock_data = pd.read_csv(crossdock_file)
        print(f"Crossdock CSV columns: {list(crossdock_data.columns)}")
        print(f"Loaded {len(crossdock_data)} crossdock zip codes from CSV file")
        return crossdock_data
    else:
        print("No crossdock file found, creating empty dataset")
        return pd.DataFrame(columns=['Name', 'Zip Code'])

def process_crossdock_locations(crossdock_data, zip_shapes):
    """Process crossdock locations and get their coordinates"""
    print("Processing crossdock locations...")
    crossdock_locations = []
    
    if len(crossdock_data) == 0:
        print("No crossdock data to process")
        return crossdock_locations
    
    # Convert zip shapes to WGS84 for coordinate extraction
    zip_shapes_wgs84 = zip_shapes.to_crs('EPSG:4326')
    
    for idx, row in crossdock_data.iterrows():
        zip_code = str(row.get('Zip Code', '')).zfill(5)
        name = row.get('Name', f'Crossdock-{zip_code}')
        
        # Find matching zip code shape
        matching_shape = zip_shapes_wgs84[zip_shapes_wgs84['ZCTA5CE20'] == zip_code]
        
        if not matching_shape.empty:
            # Get centroid of the zip code
            centroid = matching_shape.geometry.centroid.iloc[0]
            lat, lon = centroid.y, centroid.x
            
            crossdock_locations.append({
                'name': name,
                'zip_code': zip_code,
                'lat': lat,
                'lon': lon
            })
            print(f"  Found crossdock: {name} at {zip_code} ({lat:.4f}, {lon:.4f})")
    
    print(f"Successfully processed {len(crossdock_locations)} crossdock locations")
    return crossdock_locations

def calculate_weighted_cbsa_centroids(merged_data, zip_shapes):
    """Calculate quote-weighted centroids for CBSAs and prepare map data"""
    print("Calculating quote-weighted CBSA centroids...")
    
    # Merge data with shapes
    map_data = zip_shapes.merge(merged_data, left_on='ZCTA5CE20', right_on='Zipcode_clean', how='left')
    map_data['Total Quotes'] = map_data['Total Quotes'].fillna(0)
    
    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    
    print(f"Calculating weighted centroids for {cbsa_zips['Primary CBSA Name'].nunique()} CBSAs...")
    
    # Get top 75 CBSAs by population (unbiased selection)
    print("Determining top 75 CBSAs by population (unbiased by quotes)...")
    cbsa_populations = merged_data[merged_data['Primary CBSA Name'].notna()].groupby('Primary CBSA Name')['ZCTA Population (2020)'].sum().sort_values(ascending=False)
    top_75_cbsas = cbsa_populations.head(75).index.tolist()
    
    print("Top 75 CBSAs by population (unbiased selection):")
    for i, (cbsa, pop) in enumerate(cbsa_populations.head(75).items(), 1):
        if i <= 10:
            print(f"  {i}. {cbsa}: {pop:,}")
        elif i == 11:
            print(f"  ... and {len(top_75_cbsas) - 10} more")
    
    # Filter to top 75 CBSAs
    map_data = map_data[
        (map_data['Primary CBSA Name'].isin(top_75_cbsas)) | 
        (map_data['Primary CBSA Name'].isna())
    ].copy()
    
    # Calculate weighted centroids for each CBSA
    cbsa_centroids = []
    cbsa_zips_filtered = map_data[map_data['Primary CBSA Name'].notna()]
    
    for cbsa_name in top_75_cbsas:
        cbsa_data = cbsa_zips_filtered[cbsa_zips_filtered['Primary CBSA Name'] == cbsa_name]
        
        if len(cbsa_data) == 0:
            continue
            
        # Calculate quote-weighted centroid
        total_quotes = cbsa_data['Total Quotes'].sum()
        if total_quotes > 0:
            # Weight by quotes
            weights = cbsa_data['Total Quotes'] / total_quotes
        else:
            # Equal weights if no quotes
            weights = pd.Series([1/len(cbsa_data)] * len(cbsa_data), index=cbsa_data.index)
        
        # Get centroids of each zip code geometry
        centroids = cbsa_data.geometry.centroid
        
        # Calculate weighted average coordinates (in Web Mercator)
        weighted_x = (centroids.x * weights).sum()
        weighted_y = (centroids.y * weights).sum()
        
        # Convert to WGS84 for display
        from shapely.geometry import Point
        point_mercator = Point(weighted_x, weighted_y)
        gdf_temp = gpd.GeoDataFrame([1], geometry=[point_mercator], crs='EPSG:3857')
        point_wgs84 = gdf_temp.to_crs('EPSG:4326').geometry.iloc[0]
        
        cbsa_centroids.append({
            'cbsa_name': cbsa_name,
            'weighted_lat': point_wgs84.y,
            'weighted_lon': point_wgs84.x,
            'total_quotes': int(total_quotes),
            'zip_count': len(cbsa_data)
        })
    
    cbsa_centroids_df = pd.DataFrame(cbsa_centroids)
    print(f"Calculated weighted centroids for {len(cbsa_centroids_df)} CBSAs")
    
    # Print quote statistics
    quote_range = cbsa_centroids_df['total_quotes']
    print(f"Quote range across CBSAs: {quote_range.min():,} - {quote_range.max():,}")
    
    return map_data, cbsa_centroids_df

def load_all_usa_zip_data(zip_shapes, merged_data):
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

def calculate_distances_to_nearest_cbsa_zips(map_data, cbsa_centroids_df):
    """Calculate distances from non-CBSA zip codes to nearest CBSA zip codes"""
    print("Calculating distances to nearest CBSA zip codes (adjacency analysis)...")

    # Separate CBSA and non-CBSA zip codes
    cbsa_zips = map_data[map_data['Primary CBSA Name'].notna()].copy()
    non_cbsa_zips = map_data[map_data['Primary CBSA Name'].isna()].copy()

    if len(non_cbsa_zips) == 0:
        print("No non-CBSA zip codes found")
        return map_data

    print(f"Calculating distances from {len(non_cbsa_zips)} non-CBSA zip codes to {len(cbsa_zips)} CBSA zip codes...")

    # Get centroids for distance calculation (in Web Mercator for accuracy)
    cbsa_centroids = cbsa_zips.geometry.centroid
    non_cbsa_centroids = non_cbsa_zips.geometry.centroid

    # Convert to coordinate arrays
    cbsa_coords = np.array([[point.x, point.y] for point in cbsa_centroids])
    non_cbsa_coords = np.array([[point.x, point.y] for point in non_cbsa_centroids])

    # Use BallTree for efficient nearest neighbor search
    tree = BallTree(cbsa_coords, metric='euclidean')
    distances, indices = tree.query(non_cbsa_coords, k=1)

    # Convert distances from meters to miles (Web Mercator is in meters)
    distances_miles = distances.flatten() * 3.28084 / 5280  # meters to feet to miles

    # Get information about nearest CBSA zip codes
    nearest_cbsa_info = cbsa_zips.iloc[indices.flatten()]

    # Add distance and nearest CBSA information to non-CBSA zip codes
    non_cbsa_zips = non_cbsa_zips.copy()
    non_cbsa_zips['min_distance_to_cbsa'] = distances_miles
    non_cbsa_zips['nearest_cbsa_zip'] = nearest_cbsa_info['Zipcode_clean'].values
    non_cbsa_zips['nearest_cbsa_name'] = nearest_cbsa_info['Primary CBSA Name'].values

    # Calculate quote statistics for each CBSA
    print("Calculating quote statistics for each CBSA...")
    cbsa_stats = cbsa_zips.groupby('Primary CBSA Name')['Total Quotes'].agg(['mean', 'std']).reset_index()
    cbsa_stats.columns = ['Primary CBSA Name', 'cbsa_mean_quotes', 'cbsa_std_quotes']
    cbsa_stats['cbsa_std_quotes'] = cbsa_stats['cbsa_std_quotes'].fillna(0)

    # Add CBSA statistics to non-CBSA zip codes
    non_cbsa_zips = non_cbsa_zips.merge(
        cbsa_stats.rename(columns={'Primary CBSA Name': 'nearest_cbsa_name'}),
        on='nearest_cbsa_name',
        how='left'
    )

    # Calculate standard deviations below CBSA mean
    non_cbsa_zips['std_dev_below_cbsa_mean'] = np.where(
        non_cbsa_zips['cbsa_std_quotes'] > 0,
        (non_cbsa_zips['cbsa_mean_quotes'] - non_cbsa_zips['Total Quotes']) / non_cbsa_zips['cbsa_std_quotes'],
        0
    )

    # Calculate percentile within nearest CBSA
    def calculate_percentile(row):
        if pd.isna(row['nearest_cbsa_name']):
            return 0
        cbsa_quotes = cbsa_zips[cbsa_zips['Primary CBSA Name'] == row['nearest_cbsa_name']]['Total Quotes']
        if len(cbsa_quotes) == 0:
            return 0
        return (cbsa_quotes < row['Total Quotes']).mean() * 100

    non_cbsa_zips['quote_percentile_in_nearest_cbsa'] = non_cbsa_zips.apply(calculate_percentile, axis=1)

    # Add CBSA mean quotes to CBSA zip codes as well
    cbsa_zips = cbsa_zips.merge(cbsa_stats, on='Primary CBSA Name', how='left')
    cbsa_zips['min_distance_to_cbsa'] = 0  # They are already in a CBSA
    cbsa_zips['nearest_cbsa_zip'] = cbsa_zips['Zipcode_clean']
    cbsa_zips['nearest_cbsa_name'] = cbsa_zips['Primary CBSA Name']
    cbsa_zips['std_dev_below_cbsa_mean'] = 0
    cbsa_zips['quote_percentile_in_nearest_cbsa'] = cbsa_zips.apply(calculate_percentile, axis=1)

    # Combine back together
    map_data_updated = pd.concat([cbsa_zips, non_cbsa_zips], ignore_index=True)

    print(f"Adjacency analysis complete. Distance range: {distances_miles.min():.1f} - {distances_miles.max():.1f} miles")

    return map_data_updated

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

def create_corridor_analysis_map(map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data):
    """Create a CBSA proximity map with comprehensive corridor analysis"""
    print("Creating CBSA corridor analysis map...")

    # Convert to WGS84 for web display
    map_data_wgs84 = map_data.to_crs('EPSG:4326')

    # Calculate separate percentiles for better color sensitivity
    cbsa_quotes = map_data_wgs84[map_data_wgs84['Primary CBSA Name'].notna()]['Total Quotes']
    non_cbsa_quotes = map_data_wgs84[map_data_wgs84['Primary CBSA Name'].isna()]['Total Quotes']

    # Calculate percentiles for color mapping
    if len(cbsa_quotes) > 0:
        cbsa_percentiles = np.percentile(cbsa_quotes[cbsa_quotes > 0], [20, 50, 80, 95])
    else:
        cbsa_percentiles = [0, 1, 10, 100]

    if len(non_cbsa_quotes) > 0:
        non_cbsa_percentiles = np.percentile(non_cbsa_quotes[non_cbsa_quotes > 0], [20, 50, 80, 95])
    else:
        non_cbsa_percentiles = [0, 1, 10, 100]

    print(f"CBSA quote range: {cbsa_quotes.min()} - {cbsa_quotes.max()}")
    print(f"Non-CBSA quote range: {non_cbsa_quotes.min()} - {non_cbsa_quotes.max()}")

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

    # Prepare ALL USA data for corridor analysis
    all_usa_features = prepare_all_usa_data_for_js(all_usa_data)

    # Create the HTML map
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CBSA Corridor Analysis Map - Universal USA Coverage</title>
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
            color: #007cba;
        }}
        .slider-wrapper {{
            position: relative;
            margin: 10px 0;
        }}
        .slider-wrapper input[type="range"] {{
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }}
        .slider-wrapper input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .slider-wrapper input[type="range"]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .distance-display {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #007cba;
            margin: 10px 0;
            padding: 8px;
            background: #e3f2fd;
            border-radius: 5px;
        }}
        .range-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }}
        .checkbox-container {{
            margin: 10px 0;
            display: flex;
            align-items: center;
        }}
        .checkbox-container input[type="checkbox"] {{
            margin-right: 8px;
            transform: scale(1.2);
        }}
        .checkbox-container label {{
            margin: 0;
            font-weight: normal;
        }}
        .stats {{
            margin: 15px 0;
            padding: 15px;
            background: #f0f8ff;
            border-radius: 8px;
            border: 2px solid #4a90e2;
            font-size: 12px;
            line-height: 1.4;
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
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
            font-size: 12px;
        }}
        .legend h4 {{ margin: 0 0 10px 0; color: #333; }}
        .color-scale {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .color-box {{
            width: 20px;
            height: 15px;
            margin-right: 3px;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="controls">
        <h4>🎯 CBSA Corridor Analysis</h4>

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

        <div class="slider-container">
            <label for="stdDevSlider">Min Quote Performance vs Nearest CBSA:</label>
            <div class="slider-wrapper">
                <input type="range" id="stdDevSlider" min="-3.0" max="2.0" value="-3.0" step="0.1">
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

        <div class="corridor-controls">
            <h5>🛣️ Individual Corridor Analysis</h5>
            <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; font-size: 12px; margin-bottom: 8px;">
                <strong>💡 How to use:</strong> Click any zip code to open its popup, then use the corridor checkbox to toggle its corridor on/off
            </div>
            <div style="font-size: 10px; color: #856404; line-height: 1.2;">
                Shows corridor connections between selected zip codes and their nearest CBSA coverage.
                Uses entire USA zip code database for comprehensive analysis.
            </div>
            <div style="margin-top: 8px;">
                <label for="corridorWidth">Corridor Width (miles):</label>
                <input type="range" id="corridorWidth" min="10" max="100" value="25" step="5" style="width: 100%;">
                <div id="corridorWidthDisplay" style="text-align: center; font-weight: bold;">25 miles</div>
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
        <h4>🔥 CBSA Corridor Analysis Heat Map</h4>
        <div><strong>CBSA-Assigned (Blue Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #f8fbff;"></div>
            <div class="color-box" style="background: #c6dbed;"></div>
            <div class="color-box" style="background: #6baed6;"></div>
            <div class="color-box" style="background: #3182bd;"></div>
            <div class="color-box" style="background: #08306b;"></div>
            <span>Very Low → Very High Quotes</span>
        </div>
        <div style="margin-top: 10px;"><strong>Non-CBSA (Red Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #fff5f0;"></div>
            <div class="color-box" style="background: #fee0d2;"></div>
            <div class="color-box" style="background: #fc9272;"></div>
            <div class="color-box" style="background: #de2d26;"></div>
            <div class="color-box" style="background: #a50f15;"></div>
            <span>Very Low → Very High Quotes</span>
        </div>
        <div style="margin-top: 10px;"><strong>Special Features:</strong></div>
        <div style="margin-top: 5px; font-size: 12px;">
            <div style="margin: 3px 0;"><span style="background: #ffd700; padding: 2px 6px; border-radius: 3px; color: #000;">Yellow</span> Unassigned CBSA</div>
            <div style="margin: 3px 0;"><span style="background: #FFA500; padding: 2px 6px; border-radius: 3px; color: #000;">Orange</span> Corridor zip codes</div>
            <div style="margin: 3px 0; font-style: italic;">🛣️ Click zip codes for corridor toggle</div>
        </div>
        <div style="margin-top: 8px; font-size: 10px; color: #666; font-style: italic;">
            Red circles = CBSA weighted centroids<br>
            Green diamonds = Crossdock locations<br>
            Orange markers = Corridor connections<br>
            Uses ALL USA zip codes for corridor analysis
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
        var allUsaData = {json.dumps(all_usa_features, indent=2)};
        var zipLayers = {{}};
        var centroidLayers = [];
        var crossdockLayers = [];
        var currentDistanceThreshold = 50;
        var currentStdDevThreshold = -3.0;
        var stdDevFilterEnabled = true;
        var activeCorridors = new Map(); // Track active corridors by zip code
        var corridorLayers = new Map(); // Track corridor layer groups by zip code
        var currentCorridorWidth = 25;

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

        // Calculate corridor for a specific zip code
        function calculateCorridorForZip(targetZip, corridorWidth) {{
            console.log('Calculating corridor for zip:', targetZip.properties.zipcode_clean, 'with width:', corridorWidth, 'miles');

            // Find all assigned CBSA zip codes from ALL USA data
            var assignedCbsaZips = allUsaData.filter(zip =>
                zip.properties.has_cbsa &&
                zip.properties.zipcode_clean !== targetZip.properties.zipcode_clean
            );

            if (assignedCbsaZips.length === 0) {{
                console.log('No assigned CBSA zips available for corridor analysis');
                return [];
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

            if (!nearestCbsaZip) {{
                console.log('No nearest assigned CBSA zip found');
                return [];
            }}

            console.log('Nearest assigned CBSA zip:', nearestCbsaZip.properties.zipcode_clean,
                       'in', nearestCbsaZip.properties.cbsa_name,
                       '(' + minDistance.toFixed(1) + ' miles)');

            var nearestCentroid = getZipCentroid(nearestCbsaZip);
            var corridorZips = [];

            // Find all zip codes within corridor width of the line between target and nearest assigned CBSA
            allUsaData.forEach(zip => {{
                // Skip the target zip itself and the nearest assigned CBSA zip
                if (zip.properties.zipcode_clean === targetZip.properties.zipcode_clean ||
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
                    var directDistance = minDistance;

                    // Only include if the zip is roughly on the path (not too far out of the way)
                    if (distanceToTarget + distanceToCbsa <= directDistance * 1.3) {{
                        corridorZips.push(zip);
                    }}
                }}
            }});

            console.log('Found', corridorZips.length, 'corridor zips for', targetZip.properties.zipcode_clean);
            return corridorZips;
        }}

        // Toggle corridor for a specific zip code
        function toggleCorridorForZip(targetZip, isChecked) {{
            var zipCode = targetZip.properties.zipcode_clean;
            console.log('toggleCorridorForZip called for:', zipCode, 'isChecked:', isChecked);

            if (!isChecked || activeCorridors.has(zipCode)) {{
                // Remove existing corridor
                console.log('Removing corridor for zip:', zipCode);
                var layerGroup = corridorLayers.get(zipCode);
                if (layerGroup) {{
                    map.removeLayer(layerGroup);
                    corridorLayers.delete(zipCode);
                }}
                activeCorridors.delete(zipCode);
            }}

            if (isChecked && !activeCorridors.has(zipCode)) {{
                // Add new corridor
                console.log('Adding corridor for zip:', zipCode, 'with width:', currentCorridorWidth);
                var corridorZips = calculateCorridorForZip(targetZip, currentCorridorWidth);
                console.log('Calculated', corridorZips.length, 'corridor zips for', zipCode);

                if (corridorZips.length > 0) {{
                    var layerGroup = L.layerGroup();

                    corridorZips.forEach(zip => {{
                        var coords = getZipCentroid(zip);
                        if (coords[0] !== 0 || coords[1] !== 0) {{
                            var marker = L.circleMarker([coords[0], coords[1]], {{
                                radius: 6,
                                fillColor: '#FFA500', // Orange for corridor zips
                                color: '#FF4500',
                                weight: 2,
                                opacity: 1.0,
                                fillOpacity: 0.8
                            }});

                            marker.bindPopup(`
                                <strong>🛣️ Corridor Zip: ${{zip.properties.zipcode_clean}}</strong><br>
                                <strong>Location:</strong> ${{zip.properties.cbsa_name || 'Non-CBSA'}}<br>
                                <strong>Quotes:</strong> ${{zip.properties.total_quotes}}<br>
                                <strong>Connecting to:</strong> ${{zipCode}}<br>
                                <hr style="margin: 5px 0;">
                                <em>Part of corridor pathway to nearest CBSA coverage</em>
                            `);

                            layerGroup.addLayer(marker);
                        }}
                    }});

                    layerGroup.addTo(map);
                    corridorLayers.set(zipCode, layerGroup);
                    activeCorridors.set(zipCode, corridorZips);
                    console.log('Successfully added', corridorZips.length, 'corridor markers to map for zip', zipCode);
                }}
            }}
        }}

        // Clear all corridors
        function clearAllCorridors() {{
            console.log('Clearing all corridors');
            corridorLayers.forEach((layerGroup, zipCode) => {{
                map.removeLayer(layerGroup);
            }});
            corridorLayers.clear();
            activeCorridors.clear();
        }}

        // Initialize corridor controls
        document.getElementById('corridorWidth').addEventListener('input', function(e) {{
            currentCorridorWidth = parseInt(e.target.value);
            document.getElementById('corridorWidthDisplay').textContent = currentCorridorWidth + ' miles';

            // Refresh any active corridors with new width
            if (activeCorridors.size > 0) {{
                console.log('Refreshing', activeCorridors.size, 'active corridors with new width');
                var activeZipCodes = Array.from(activeCorridors.keys());

                // Clear existing corridors
                clearAllCorridors();

                // Recreate corridors with new width
                activeZipCodes.forEach(zipCode => {{
                    var targetZip = allZipData.find(zip => zip.properties.zipcode_clean === zipCode);
                    if (targetZip) {{
                        toggleCorridorForZip(targetZip, true);
                    }}
                }});
            }}
        }});

        // Color functions
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
                // Blue scale for CBSA zip codes
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
                // Red scale for non-CBSA zip codes
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
            var popupContent = `
                <div style="font-family: Arial; max-width: 300px;">
                    <h4 style="margin: 0 0 10px 0; color: #333;">Zip Code: ${{props.zipcode}}</h4>
                    <hr style="margin: 5px 0;">
                    <strong>Location:</strong> ${{props.city}}, ${{props.state}}<br>
                    <strong>Population:</strong> ${{props.population.toLocaleString()}}<br>
                    <strong>Total Quotes:</strong> ${{props.total_quotes.toLocaleString()}}<br>
                    <strong>Pickup Count:</strong> ${{props.pickup_count.toLocaleString()}}<br>
                    <strong>Dropoff Count:</strong> ${{props.dropoff_count.toLocaleString()}}<br>
            `;

            if (props.has_cbsa) {{
                popupContent += `<strong>CBSA:</strong> ${{props.cbsa_name}}<br>`;
                if (props.was_unassigned) {{
                    popupContent += `<div style="background: #fff3cd; padding: 5px; margin: 5px 0; border-radius: 3px; border: 1px solid #ffc107;">
                        <strong>⚠️ Previously Unassigned</strong><br>
                        This zip code was manually unassigned from its CBSA
                    </div>`;
                }}
            }} else {{
                popupContent += `<strong>Status:</strong> Non-CBSA<br>`;
                if (props.distance_to_cbsa > 0) {{
                    popupContent += `<strong>Distance to nearest CBSA:</strong> ${{props.distance_to_cbsa.toFixed(1)}} miles<br>`;
                    popupContent += `<strong>Nearest CBSA:</strong> ${{props.nearest_cbsa_name}}<br>`;
                    if (props.std_dev_below_cbsa_mean !== undefined) {{
                        var stdDevText = props.std_dev_below_cbsa_mean >= 0 ?
                            `${{props.std_dev_below_cbsa_mean.toFixed(1)}} std dev below` :
                            `${{Math.abs(props.std_dev_below_cbsa_mean).toFixed(1)}} std dev above`;
                        popupContent += `<strong>Quote performance vs CBSA:</strong> ${{stdDevText}} mean<br>`;
                    }}
                }}
            }}

            popupContent += `
                    <hr style="margin: 10px 0;">
                    <div style="background: #fff3cd; padding: 8px; border-radius: 4px; border: 1px solid #ffc107;">
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <input type="checkbox" id="corridor_${{props.zipcode_clean}}" style="margin-right: 8px; transform: scale(1.2);">
                            <label for="corridor_${{props.zipcode_clean}}" style="font-weight: bold; color: #856404;">🛣️ Show Corridor</label>
                        </div>
                        <div style="font-size: 11px; color: #856404;">
                            Toggle to show/hide corridor connections to nearest CBSA coverage
                        </div>
                    </div>
                </div>
            `;

            return popupContent;
        }}

        // Initialize event handlers and render map
        updateMap();

        // Event handlers for controls
        document.getElementById('distanceSlider').addEventListener('input', function(e) {{
            currentDistanceThreshold = parseInt(e.target.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            updateMap();
        }});

        document.getElementById('stdDevSlider').addEventListener('input', function(e) {{
            currentStdDevThreshold = parseFloat(e.target.value);
            var sign = currentStdDevThreshold >= 0 ? '≥ +' : '≥ ';
            document.getElementById('stdDevDisplay').textContent = sign + currentStdDevThreshold.toFixed(1) + ' std dev';
            updateMap();
        }});

        document.getElementById('showAllCBSA').addEventListener('change', updateMap);
        document.getElementById('showNearbyNonCBSA').addEventListener('change', updateMap);
        document.getElementById('showCentroids').addEventListener('change', updateMap);
        document.getElementById('showCrossdocks').addEventListener('change', updateMap);
        document.getElementById('enableStdDevFilter').addEventListener('change', function(e) {{
            stdDevFilterEnabled = e.target.checked;
            updateMap();
        }});

        function updateMap() {{
            // Clear existing zip layers
            Object.values(zipLayers).forEach(layer => map.removeLayer(layer));
            zipLayers = {{}};

            // Clear existing centroid layers
            centroidLayers.forEach(layer => map.removeLayer(layer));
            centroidLayers = [];

            // Clear existing crossdock layers
            crossdockLayers.forEach(layer => map.removeLayer(layer));
            crossdockLayers = [];

            var showAllCBSA = document.getElementById('showAllCBSA').checked;
            var showNearbyNonCBSA = document.getElementById('showNearbyNonCBSA').checked;
            var showCentroids = document.getElementById('showCentroids').checked;
            var showCrossdocks = document.getElementById('showCrossdocks').checked;

            var visibleZips = 0;
            var cbsaZips = 0;
            var nonCbsaZips = 0;

            // Add zip code layers with corridor click handlers
            allZipData.forEach(function(feature) {{
                var props = feature.properties;
                var shouldShow = false;

                if (props.has_cbsa && showAllCBSA) {{
                    shouldShow = true;
                    cbsaZips++;
                }} else if (!props.has_cbsa && showNearbyNonCBSA) {{
                    var meetsDistanceThreshold = props.distance_to_cbsa <= currentDistanceThreshold;
                    var meetsStdDevThreshold = true;
                    if (stdDevFilterEnabled) {{
                        meetsStdDevThreshold = (-props.std_dev_below_cbsa_mean) >= currentStdDevThreshold;
                    }}
                    if (meetsDistanceThreshold && meetsStdDevThreshold) {{
                        shouldShow = true;
                        nonCbsaZips++;
                    }}
                }}

                if (shouldShow) {{
                    visibleZips++;
                    var layer = L.geoJSON(feature, {{
                        style: getZipStyle,
                        onEachFeature: function(feature, layer) {{
                            layer.bindPopup(createPopupContent(feature));

                            // Add event listener for when popup opens
                            layer.on('popupopen', function(e) {{
                                var zipCode = feature.properties.zipcode_clean;
                                var checkbox = document.getElementById('corridor_' + zipCode);

                                if (checkbox) {{
                                    // Set initial checkbox state
                                    checkbox.checked = activeCorridors.has(zipCode);

                                    // Add event listener to checkbox
                                    checkbox.addEventListener('change', function(event) {{
                                        toggleCorridorForZip(feature, event.target.checked);
                                    }});
                                }}
                            }});

                            var tooltip = 'Zip: ' + feature.properties.zipcode +
                                         ' | Quotes: ' + feature.properties.total_quotes.toLocaleString();

                            if (feature.properties.cbsa_mean_quotes > 0) {{
                                var cbsaType = feature.properties.has_cbsa ? 'CBSA' : 'Nearest CBSA';
                                tooltip += ' | ' + cbsaType + ' Avg: ' + feature.properties.cbsa_mean_quotes.toFixed(0);
                            }}

                            if (!feature.properties.has_cbsa) {{
                                tooltip += ' | Dist: ' + feature.properties.distance_to_cbsa.toFixed(1) + 'mi';
                            }}

                            tooltip += ' | 🛣️ Click for corridor toggle';

                            layer.bindTooltip(tooltip, {{ permanent: false, direction: 'top' }});
                        }}
                    }}).addTo(map);

                    zipLayers[props.zipcode_clean] = layer;
                }}
            }});

            // Add centroid and crossdock markers
            if (showCentroids) addCentroidMarkers();
            if (showCrossdocks) addCrossdockMarkers();

            // Update stats
            updateStats(visibleZips, cbsaZips, nonCbsaZips);
        }}

        function addCentroidMarkers() {{
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
                        <strong>Coordinates:</strong> ${{centroid.lat.toFixed(4)}}, ${{centroid.lon.toFixed(4)}}
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
            crossdockLocations.forEach(function(crossdock) {{
                var marker = L.marker([crossdock.lat, crossdock.lon], {{
                    icon: L.divIcon({{
                        className: 'crossdock-marker',
                        html: '<div style="background: #28a745; color: white; width: 12px; height: 12px; transform: rotate(45deg); border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                        iconSize: [12, 12],
                        iconAnchor: [6, 6]
                    }})
                }});

                marker.bindPopup(`
                    <div style="font-family: Arial;">
                        <h4 style="margin: 0 0 10px 0;">🏭 Crossdock Location</h4>
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

        function updateStats(visibleZips, cbsaZips, nonCbsaZips) {{
            var filterDescription = '';
            if (stdDevFilterEnabled) {{
                var stdDevSign = currentStdDevThreshold >= 0 ? '+' : '';
                filterDescription = ` (≥ ${{stdDevSign}}${{currentStdDevThreshold.toFixed(1)}} std dev)`;
            }}

            var activeCorridorCount = activeCorridors.size;
            var totalCorridorZips = 0;
            activeCorridors.forEach(corridorZips => {{
                totalCorridorZips += corridorZips.length;
            }});

            document.getElementById('filterStats').innerHTML = `
                <strong>📊 Current View Statistics:</strong><br>
                • Total visible zip codes: ${{visibleZips.toLocaleString()}}<br>
                • CBSA zip codes: ${{cbsaZips.toLocaleString()}}<br>
                • Non-CBSA zip codes: ${{nonCbsaZips.toLocaleString()}}<br>
                • Distance threshold: ≤ ${{currentDistanceThreshold}} miles<br>
                • Quote performance filter${{filterDescription}}: ${{stdDevFilterEnabled ? 'ON' : 'OFF'}}<br>
                <hr style="margin: 8px 0;">
                <strong>🛣️ Individual Corridor Analysis:</strong><br>
                • Active corridor analyses: ${{activeCorridorCount}}<br>
                • Total corridor zip codes: ${{totalCorridorZips.toLocaleString()}}<br>
                • Corridor width: ${{currentCorridorWidth}} miles<br>
                • Click zip codes to toggle corridors
            `;
        }}

        function exportFilteredData() {{
            var exportData = [];

            allZipData.forEach(function(feature) {{
                var props = feature.properties;
                var shouldInclude = false;

                if (props.has_cbsa && document.getElementById('showAllCBSA').checked) {{
                    shouldInclude = true;
                }} else if (!props.has_cbsa && document.getElementById('showNearbyNonCBSA').checked) {{
                    var meetsDistanceThreshold = props.distance_to_cbsa <= currentDistanceThreshold;
                    var meetsStdDevThreshold = true;
                    if (stdDevFilterEnabled) {{
                        meetsStdDevThreshold = (-props.std_dev_below_cbsa_mean) >= currentStdDevThreshold;
                    }}
                    if (meetsDistanceThreshold && meetsStdDevThreshold) {{
                        shouldInclude = true;
                    }}
                }}

                if (shouldInclude) {{
                    exportData.push({{
                        zipcode: props.zipcode_clean,
                        city: props.city,
                        state: props.state,
                        total_quotes: props.total_quotes,
                        pickup_count: props.pickup_count,
                        dropoff_count: props.dropoff_count,
                        population: props.population,
                        has_cbsa: props.has_cbsa,
                        assigned_cbsa: props.assigned_cbsa || '',
                        closest_cbsa: props.closest_cbsa || '',
                        distance_to_cbsa: props.distance_to_cbsa,
                        std_dev_below_cbsa_mean: props.std_dev_below_cbsa_mean,
                        quote_percentile_in_nearest_cbsa: props.quote_percentile_in_nearest_cbsa,
                        was_unassigned: props.was_unassigned,
                        nearest_cbsa_zip: props.nearest_cbsa_zip || '',
                        cbsa_mean_quotes: props.cbsa_mean_quotes
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
                    return typeof value === 'string' ? '"' + value.replace(/"/g, '""') + '"' : value;
                }}).join(',') + '\\n';
            }});

            // Download CSV
            var blob = new Blob([csv], {{ type: 'text/csv' }});
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'cbsa_corridor_analysis_filtered_data.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }}

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
        all_usa_data = load_all_usa_zip_data(zip_shapes, merged_data)

        # Create map with corridor functionality
        html_content = create_corridor_analysis_map(map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data)

        # Save and open map
        output_file = 'cbsa_corridor_analysis_map.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\nCBSA Corridor Analysis map saved as '{output_file}'")
        print("Opening map in browser...")

        # Open in browser
        file_path = os.path.abspath(output_file)
        webbrowser.open(f'file://{file_path}')

        print("\n" + "="*60)
        print("CBSA CORRIDOR ANALYSIS MAP FEATURES:")
        print("✅ Universal corridor analysis using ALL USA zip codes")
        print("✅ Click any zip code to show corridor connections")
        print("✅ Adjustable corridor width (10-100 miles)")
        print("✅ Quote-weighted CBSA centroids")
        print("✅ Distance and performance filtering")
        print("✅ Clear all corridors functionality")
        print("✅ Comprehensive USA zip code coverage")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
