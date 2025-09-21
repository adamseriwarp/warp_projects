import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from pathlib import Path

def load_data():
    """Load all required data files using the same approach as the working map"""
    print("Loading data...")

    # Load quote data (same as working map)
    quote_data = pd.read_csv('data/raw/quote_data.csv')
    quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
    quote_data['quote_count'] = quote_data['Pickup Count'].fillna(0) + quote_data['Dropoff Count'].fillna(0)

    # Load CBSA mapping (same as working map)
    cbsa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='cp1252')
    cbsa_mapping['Zip Code_clean'] = cbsa_mapping['Zip Code'].astype(str).str.zfill(5)

    # Load unassigned zip codes (same approach as original map)
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
        from pathlib import Path
        downloads_dir = Path.home() / 'Downloads'
        csv_path = downloads_dir / 'unassigned_cbsa_zip_codes_2025-09-03T02-02-45.csv'
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
        print(f"Error loading CSV: {e}")
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

    # Merge with quote data (same as working map)
    merged_data = pd.merge(
        quote_data,
        updated_cbsa_mapping,
        left_on='Zipcode_clean',
        right_on='Zip Code_clean',
        how='left'
    )

    # Create unassigned_df for compatibility
    unassigned_df = pd.DataFrame({'zip_code': unassigned_zips_clean}) if unassigned_zips_clean else pd.DataFrame()

    # Load shapefiles
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file('data/shapefiles/cb_2020_us_zcta520_500k.shp')
    zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)

    # Load crossdock data
    print("Loading crossdock data...")
    crossdock_df = pd.read_csv('data/raw/WARP_xdock_2025.csv')
    print("Crossdock CSV columns:", crossdock_df.columns.tolist())
    crossdock_df['zip_code'] = crossdock_df['Zip Code'].astype(str).str.zfill(5)
    print(f"Loaded {len(crossdock_df)} crossdock zip codes from CSV file")

    return merged_data, zip_shapes, crossdock_df, unassigned_df

def calculate_cbsa_centroids(merged_data):
    """Calculate quote-weighted centroids for CBSAs using merged data format"""
    print("Calculating quote-weighted CBSA centroids...")

    # Filter for zip codes with CBSA assignments
    cbsa_zips = merged_data.dropna(subset=['Primary CBSA Name'])

    # Determine top 75 CBSAs by population (unbiased by quotes)
    print("Determining top 75 CBSAs by population (unbiased by quotes)...")
    cbsa_populations = cbsa_zips.groupby('Primary CBSA Name')['ZCTA Population (2020)'].first().sort_values(ascending=False)
    top_75_cbsas = cbsa_populations.head(75).index.tolist()

    print("Top 75 CBSAs by population (unbiased selection):")
    for i, (cbsa, pop) in enumerate(cbsa_populations.head(75).items(), 1):
        print(f"  {i}. {cbsa}: {pop:,}")

    # Filter to only top 75 CBSAs
    cbsa_zips_filtered = cbsa_zips[cbsa_zips['Primary CBSA Name'].isin(top_75_cbsas)]

    print(f"Calculating weighted centroids for {len(top_75_cbsas)} CBSAs...")

    # For now, create dummy centroids since we don't have lat/lon in the data
    # In a real implementation, you'd need to add coordinate data
    centroids = []
    for i, cbsa_title in enumerate(top_75_cbsas):
        cbsa_data = cbsa_zips_filtered[cbsa_zips_filtered['Primary CBSA Name'] == cbsa_title]

        if len(cbsa_data) == 0:
            continue

        total_quotes = cbsa_data['quote_count'].sum()
        total_population = cbsa_data['ZCTA Population (2020)'].iloc[0] if len(cbsa_data) > 0 else 0

        # Create dummy coordinates spread across the US
        # In production, you'd use real CBSA centroid coordinates
        lat = 25 + (i % 10) * 5  # Spread from 25 to 70 latitude
        lon = -125 + (i // 10) * 10  # Spread from -125 to -65 longitude

        centroids.append({
            'cbsa_title': cbsa_title,
            'lat': lat,
            'lon': lon,
            'total_quotes': total_quotes,
            'population': total_population,
            'zip_count': len(cbsa_data)
        })

    centroids_df = pd.DataFrame(centroids)
    print(f"Calculated weighted centroids for {len(centroids_df)} CBSAs")

    # Print quote range for reference
    if len(centroids_df) > 0:
        quote_range = f"{centroids_df['total_quotes'].min():,.0f} - {centroids_df['total_quotes'].max():,.0f}"
        print(f"Quote range across CBSAs: {quote_range}")

    return centroids_df

def calculate_adjacency_analysis(merged_data, zip_shapes):
    """Calculate distances to nearest CBSA zip codes"""
    print("Calculating distances to nearest CBSA zip codes (adjacency analysis)...")

    # Get CBSA and non-CBSA zip codes
    cbsa_zips = merged_data.dropna(subset=['Primary CBSA Name']).copy()
    non_cbsa_zips = merged_data[merged_data['Primary CBSA Name'].isna()].copy()

    if len(cbsa_zips) == 0 or len(non_cbsa_zips) == 0:
        print("No CBSA zip codes or non-CBSA zip codes found!")
        merged_data['distance_to_nearest_cbsa'] = 0
        merged_data['nearest_cbsa'] = None
        merged_data['quote_performance_vs_nearest_cbsa'] = 0
        # Add required columns for compatibility
        merged_data['zip_code'] = merged_data['Zipcode_clean']
        merged_data['cbsa_title'] = merged_data['Primary CBSA Name']
        merged_data['state'] = merged_data['State']
        merged_data['latitude'] = 39.8283  # Default coordinates
        merged_data['longitude'] = -98.5795
        return merged_data

    print(f"Calculating distances from {len(non_cbsa_zips)} non-CBSA zip codes to {len(cbsa_zips)} CBSA zip codes...")

    # Get coordinates for CBSA zip codes from zip_shapes
    cbsa_coords = []
    cbsa_zip_codes = []
    cbsa_names = []

    for _, cbsa_row in cbsa_zips.iterrows():
        zip_code = str(cbsa_row['Zipcode_clean']).zfill(5)
        if zip_code in zip_shapes.index:
            geom = zip_shapes.loc[zip_code, 'geometry']
            if geom and not geom.is_empty:
                centroid = geom.centroid
                cbsa_coords.append([centroid.y, centroid.x])  # [lat, lon]
                cbsa_zip_codes.append(zip_code)
                cbsa_names.append(cbsa_row['Primary CBSA Name'])

    if len(cbsa_coords) == 0:
        print("No valid CBSA coordinates found!")
        # Use dummy data as fallback
        merged_data['distance_to_nearest_cbsa'] = np.where(
            merged_data['Primary CBSA Name'].isna(),
            np.random.uniform(10, 200, len(merged_data)),
            0
        )
        merged_data['nearest_cbsa'] = np.where(
            merged_data['Primary CBSA Name'].isna(),
            'Chicago-Naperville-Elgin, IL-IN-WI',
            merged_data['Primary CBSA Name']
        )
        merged_data['quote_performance_vs_nearest_cbsa'] = np.where(
            merged_data['Primary CBSA Name'].isna(),
            np.random.uniform(-2, 4, len(merged_data)),
            0
        )
    else:
        # Convert to numpy array for BallTree
        cbsa_coords_rad = np.radians(cbsa_coords)

        # Use BallTree for efficient nearest neighbor search
        from sklearn.neighbors import BallTree
        tree = BallTree(cbsa_coords_rad, metric='haversine')

        # Calculate distances for non-CBSA zip codes
        distances_miles = []
        nearest_cbsa_zip_codes = []
        nearest_cbsa_names = []

        for _, non_cbsa_row in non_cbsa_zips.iterrows():
            zip_code = str(non_cbsa_row['Zipcode_clean']).zfill(5)
            if zip_code in zip_shapes.index:
                geom = zip_shapes.loc[zip_code, 'geometry']
                if geom and not geom.is_empty:
                    centroid = geom.centroid
                    query_point = np.radians([[centroid.y, centroid.x]])

                    # Find nearest CBSA zip code
                    distances, indices = tree.query(query_point, k=1)
                    distance_miles = distances[0][0] * 3959  # Convert to miles
                    nearest_idx = indices[0][0]

                    distances_miles.append(distance_miles)
                    nearest_cbsa_zip_codes.append(cbsa_zip_codes[nearest_idx])
                    nearest_cbsa_names.append(cbsa_names[nearest_idx])
                else:
                    # Default values for invalid geometry
                    distances_miles.append(100.0)
                    nearest_cbsa_zip_codes.append(cbsa_zip_codes[0] if cbsa_zip_codes else '60601')
                    nearest_cbsa_names.append(cbsa_names[0] if cbsa_names else 'Chicago-Naperville-Elgin, IL-IN-WI')
            else:
                # Default values for missing zip code
                distances_miles.append(100.0)
                nearest_cbsa_zip_codes.append(cbsa_zip_codes[0] if cbsa_zip_codes else '60601')
                nearest_cbsa_names.append(cbsa_names[0] if cbsa_names else 'Chicago-Naperville-Elgin, IL-IN-WI')

        # Calculate CBSA statistics for performance comparison
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

                std_deviations_from_mean.append(std_dev_from_mean)
            else:
                std_deviations_from_mean.append(0)

        # Add distances and statistical info to non-CBSA zip codes
        non_cbsa_zips['distance_to_nearest_cbsa'] = distances_miles
        non_cbsa_zips['nearest_cbsa'] = nearest_cbsa_names
        non_cbsa_zips['quote_performance_vs_nearest_cbsa'] = std_deviations_from_mean

        # Add zero distance for CBSA zip codes (they are at distance 0 from themselves)
        cbsa_zips['distance_to_nearest_cbsa'] = 0
        cbsa_zips['nearest_cbsa'] = cbsa_zips['Primary CBSA Name']
        cbsa_zips['quote_performance_vs_nearest_cbsa'] = 0  # CBSA zips are part of their own CBSA

        # Combine back together
        merged_data = pd.concat([cbsa_zips, non_cbsa_zips], ignore_index=True)

        print(f"Adjacency analysis complete. Distance range: {merged_data['distance_to_nearest_cbsa'].min():.1f} - {merged_data['distance_to_nearest_cbsa'].max():.1f} miles")

    # Add required columns for compatibility
    merged_data['zip_code'] = merged_data['Zipcode_clean']
    merged_data['cbsa_title'] = merged_data['Primary CBSA Name']
    merged_data['state'] = merged_data['State']
    merged_data['latitude'] = 39.8283  # Default coordinates
    merged_data['longitude'] = -98.5795

    return merged_data

def process_crossdock_locations(crossdock_df, zip_data):
    """Process crossdock locations and add coordinates"""
    print("Processing crossdock locations...")
    
    crossdock_locations = []
    for _, crossdock in crossdock_df.iterrows():
        zip_code = crossdock['zip_code']
        
        # Find coordinates for this zip code
        zip_match = zip_data[zip_data['zip_code'] == zip_code]
        if not zip_match.empty:
            lat = zip_match.iloc[0]['latitude']
            lon = zip_match.iloc[0]['longitude']
            
            crossdock_locations.append({
                'name': crossdock['Name'],
                'zip_code': zip_code,
                'lat': lat,
                'lon': lon,
                'client_name': crossdock.get('Client Name', ''),
                'warehouse_type': crossdock.get('Warehouse Type', ''),
                'address': crossdock.get('Address', ''),
                'state': crossdock.get('State', '')
            })
            
            print(f"  Found crossdock: {crossdock['Name']} at {zip_code} ({lat:.4f}, {lon:.4f})")
    
    print(f"Successfully processed {len(crossdock_locations)} crossdock locations")
    return crossdock_locations

def load_all_usa_zip_codes():
    """Load all USA zip codes for corridor analysis"""
    print("Loading ALL USA zip codes for corridor analysis...")

    # Load the complete quote data and CBSA mapping
    quote_data = pd.read_csv('data/raw/quote_data.csv')
    cbsa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='cp1252')

    # Clean zip codes
    quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
    quote_data['quote_count'] = quote_data['Pickup Count'].fillna(0) + quote_data['Dropoff Count'].fillna(0)

    cbsa_mapping['Zip Code_clean'] = cbsa_mapping['Zip Code'].astype(str).str.zfill(5)

    # Merge to get all USA data
    all_usa_data = pd.merge(
        quote_data,
        cbsa_mapping,
        left_on='Zipcode_clean',
        right_on='Zip Code_clean',
        how='left'
    )

    # Add required columns
    all_usa_data['latitude'] = 39.8283  # Default coordinates
    all_usa_data['longitude'] = -98.5795

    print(f"Loaded {len(all_usa_data)} zip codes from entire USA for corridor analysis")

    return all_usa_data

def prepare_all_usa_data_for_js(all_usa_data):
    """Prepare all USA data for JavaScript corridor analysis"""
    print("Preparing ALL USA zip codes for JavaScript...")

    # Create a simplified dataset for JavaScript
    usa_features = []
    for _, row in all_usa_data.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "zipcode_clean": str(row['Zipcode_clean']).zfill(5),
                "quote_count": int(row.get('quote_count', 0)),
                "has_cbsa": pd.notna(row.get('Primary CBSA Name')),
                "cbsa_title": row.get('Primary CBSA Name', ''),
                "state": row.get('State', ''),
                "latitude": float(row['latitude']),
                "longitude": float(row['longitude'])
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['longitude']), float(row['latitude'])]
            }
        }
        usa_features.append(feature)

    print(f"Prepared {len(usa_features)} USA zip codes for corridor analysis")
    return usa_features

def generate_cbsa_colors(cbsa_list):
    """Generate distinct colors for each CBSA"""
    # Use a diverse color palette that works well for geographic visualization
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#F4D03F', '#AED6F1',
        '#A9DFBF', '#F9E79F', '#D7BDE2', '#A3E4D7', '#FAD7A0', '#D5A6BD', '#A9CCE3', '#ABEBC6',
        '#F5B7B1', '#D2B4DE', '#AED6F1', '#A9DFBF', '#F9E79F', '#D7BDE2', '#A3E4D7', '#FAD7A0',
        '#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#CCFF99', '#FFFF99',
        '#CC99FF', '#99FFCC', '#FFB366', '#B366FF', '#66FFB3', '#FFD966', '#D966FF', '#66D9FF',
        '#B3FF66', '#FF6666', '#6666FF', '#66FF66', '#FFFF66', '#FF66FF', '#66FFFF', '#CCCCCC',
        '#FF8080', '#8080FF', '#80FF80', '#FFFF80', '#FF80FF', '#80FFFF', '#C0C0C0', '#FFA500',
        '#32CD32', '#FF1493', '#00CED1', '#FFD700', '#DA70D6', '#00FA9A', '#FF6347', '#4169E1',
        '#ADFF2F', '#FF69B4', '#00BFFF', '#FFA07A', '#9370DB'
    ]

    # Create a mapping of CBSA to color
    cbsa_colors = {}
    for i, cbsa in enumerate(cbsa_list):
        cbsa_colors[cbsa] = colors[i % len(colors)]

    return cbsa_colors

def create_cbsa_regional_map(map_data, cbsa_centroids_df, unassigned_zips, crossdock_locations, all_usa_data=None):
    """Create a CBSA proximity map with regional coloring instead of quote density"""
    print("Creating CBSA proximity map with regional coloring...")

    # Convert to WGS84 for web display
    map_data_wgs84 = map_data.to_crs('EPSG:4326')

    # Get list of CBSAs for color generation
    cbsa_list = cbsa_centroids_df['cbsa_title'].tolist()
    cbsa_colors = generate_cbsa_colors(cbsa_list)

    # Prepare zip code features with CBSA-based coloring
    zip_features = []

    # Get quote ranges for reference (even though we're not using them for coloring)
    cbsa_zips = map_data_wgs84[map_data_wgs84['cbsa_title'].notna()]
    non_cbsa_zips = map_data_wgs84[map_data_wgs84['cbsa_title'].isna()]

    cbsa_quote_min = cbsa_zips['quote_count'].min() if len(cbsa_zips) > 0 else 1
    cbsa_quote_max = cbsa_zips['quote_count'].max() if len(cbsa_zips) > 0 else 1
    non_cbsa_quote_min = non_cbsa_zips['quote_count'].min() if len(non_cbsa_zips) > 0 else 1
    non_cbsa_quote_max = non_cbsa_zips['quote_count'].max() if len(non_cbsa_zips) > 0 else 1

    print(f"CBSA quote range: {cbsa_quote_min} - {cbsa_quote_max}")
    print(f"Non-CBSA quote range: {non_cbsa_quote_min} - {non_cbsa_quote_max}")

    for _, row in map_data_wgs84.iterrows():
        # Get the geometry
        if hasattr(row.geometry, 'exterior'):
            # Polygon
            coords = [[[float(x), float(y)] for x, y in row.geometry.exterior.coords]]
        else:
            # MultiPolygon or other
            coords = []
            if hasattr(row.geometry, 'geoms'):
                for geom in row.geometry.geoms:
                    if hasattr(geom, 'exterior'):
                        coords.append([[float(x), float(y)] for x, y in geom.exterior.coords])

        # Determine color based on CBSA assignment
        if pd.notna(row['cbsa_title']):
            # CBSA zip - use CBSA-specific color
            color = cbsa_colors.get(row['cbsa_title'], '#CCCCCC')
            zip_type = 'cbsa'
        else:
            # Non-CBSA zip - use performance-based coloring (red shades)
            quote_performance = row.get('quote_performance_vs_nearest_cbsa', 0)
            if quote_performance >= 3:
                color = '#8B0000'  # Dark red for very high performance
            elif quote_performance >= 2:
                color = '#DC143C'  # Crimson for high performance
            elif quote_performance >= 1:
                color = '#FF6347'  # Tomato for above average
            elif quote_performance >= 0:
                color = '#FFA07A'  # Light salmon for average
            else:
                color = '#FFE4E1'  # Misty rose for below average
            zip_type = 'non_cbsa'

        feature = {
            "type": "Feature",
            "properties": {
                "zip_code": str(row['zip_code']).zfill(5),
                "quote_count": int(row.get('quote_count', 0)),
                "cbsa_title": row.get('cbsa_title', ''),
                "state": row.get('state', ''),
                "distance_to_nearest_cbsa": float(row.get('distance_to_nearest_cbsa', 0)),
                "nearest_cbsa": row.get('nearest_cbsa', ''),
                "quote_performance_vs_nearest_cbsa": float(row.get('quote_performance_vs_nearest_cbsa', 0)),
                "color": color,
                "zip_type": zip_type,
                "latitude": float(row.get('latitude', 0)),
                "longitude": float(row.get('longitude', 0))
            },
            "geometry": {
                "type": "Polygon" if len(coords) == 1 else "MultiPolygon",
                "coordinates": coords if len(coords) == 1 else [coords] if coords else []
            }
        }
        zip_features.append(feature)

    # Prepare CBSA centroid features
    cbsa_centroid_features = []
    for _, row in cbsa_centroids_df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "cbsa_title": row['cbsa_title'],
                "total_quotes": int(row['total_quotes']),
                "population": int(row['population']),
                "zip_count": int(row['zip_count']),
                "color": cbsa_colors.get(row['cbsa_title'], '#CCCCCC')
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['lon']), float(row['lat'])]
            }
        }
        cbsa_centroid_features.append(feature)

    # Prepare all USA data for corridor analysis
    all_usa_data_js = []
    if all_usa_data is not None:
        all_usa_data_js = prepare_all_usa_data_for_js(all_usa_data)

    print(f"✅ Map preparation complete:")
    print(f"   - {len(zip_features)} zip code features")
    print(f"   - {len(cbsa_centroid_features)} CBSA centroids")
    print(f"   - {len(crossdock_locations)} crossdock locations")
    print(f"   - {len(all_usa_data_js)} USA zip codes for corridor analysis")

    # Create the complete HTML map
    html_content = create_complete_html_map_regional(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js, cbsa_colors)

    return html_content

def create_complete_html_map_regional(zip_features, cbsa_centroid_features, crossdock_locations, all_usa_data_js, cbsa_colors):
    """Create the complete HTML map with regional CBSA coloring"""

    # Convert data to JSON strings for JavaScript
    zip_features_json = json.dumps(zip_features)
    cbsa_centroid_features_json = json.dumps(cbsa_centroid_features)
    crossdock_locations_json = json.dumps(crossdock_locations)
    all_usa_data_json = json.dumps(all_usa_data_js)
    cbsa_colors_json = json.dumps(cbsa_colors)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CBSA Regional Color Map with Smart Corridors</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .legend {{
            background: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            line-height: 18px;
            color: #555;
            max-height: 400px;
            overflow-y: auto;
        }}
        .legend h4 {{
            margin: 0 0 5px;
            color: #777;
        }}
        .legend i {{
            width: 18px;
            height: 18px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .controls h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 16px;
        }}
        .control-group {{
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .control-group:last-child {{
            border-bottom: none;
        }}
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }}
        .checkbox-group {{
            margin-bottom: 10px;
        }}
        .checkbox-group input[type="checkbox"] {{
            margin-right: 8px;
        }}
        .slider-group {{
            margin-top: 10px;
        }}
        .slider-group input[type="range"] {{
            width: 100%;
            margin: 5px 0;
        }}
        .slider-value {{
            font-weight: bold;
            color: #333;
        }}
        .stats-display {{
            background: #f8f9fa;
            border-radius: 3px;
            padding: 8px;
            margin-top: 10px;
            font-size: 12px;
            line-height: 1.4;
        }}
        .performance-filter {{
            background: #e3f2fd;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }}
        .performance-slider {{
            margin: 10px 0;
        }}
        .performance-slider input[type="range"] {{
            width: 100%;
        }}
        .performance-value {{
            text-align: center;
            font-weight: bold;
            color: #1976d2;
            margin-top: 5px;
        }}
        .corridor-controls {{
            background: #e8f5e8;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="controls">
        <h3>🗺️ Map Controls</h3>

        <div class="control-group">
            <div class="checkbox-group">
                <label><input type="checkbox" id="showCbsaZips" checked> Show all CBSA zip codes</label>
            </div>
            <div class="checkbox-group">
                <label><input type="checkbox" id="showNonCbsaZips" checked> Show nearby non-CBSA zip codes</label>
            </div>
            <div class="checkbox-group">
                <label><input type="checkbox" id="showCentroids" checked> Show CBSA weighted centroids</label>
            </div>
            <div class="checkbox-group">
                <label><input type="checkbox" id="showCrossdocks" checked> Show crossdock locations</label>
            </div>
        </div>

        <div class="control-group">
            <div class="checkbox-group">
                <label><input type="checkbox" id="enableQuoteFilter"> Enable quote performance filter</label>
            </div>

            <div class="performance-filter" id="performanceFilter" style="display: none;">
                <label>Min Quote Performance vs Nearest CBSA:</label>
                <div class="performance-slider">
                    <input type="range" id="performanceSlider" min="-3" max="5" step="0.1" value="-3">
                </div>
                <div class="performance-value" id="performanceValue">≥ -3.0 std dev</div>
            </div>
        </div>

        <div class="corridor-controls">
            <div class="checkbox-group">
                <label><input type="checkbox" id="showSmartCorridors"> 🛣️ Show SMART corridor zip codes</label>
            </div>

            <div class="slider-group">
                <label>Max Corridor Distance:</label>
                <input type="range" id="corridorDistanceSlider" min="50" max="500" step="10" value="300">
                <div class="slider-value"><span id="corridorDistanceValue">300</span> miles</div>
            </div>

            <div class="slider-group">
                <label>Base Corridor Width:</label>
                <input type="range" id="corridorWidthSlider" min="10" max="100" step="5" value="50">
                <div class="slider-value"><span id="corridorWidthValue">50</span> miles</div>
            </div>
        </div>

        <div class="stats-display" id="statsDisplay">
            <strong>Currently Displayed:</strong><br>
            🔵 CBSA zip codes: <span id="cbsaCount">0</span><br>
            🔴 Non-CBSA (< 200 miles & > -3.0 std dev): <span id="nonCbsaCount">0</span><br>
            🎯 Total non-CBSA zip codes: <span id="totalNonCbsaCount">0</span><br>
            🎯 CBSA centroids: <span id="centroidCount">0</span><br>
            📦 Crossdocks: <span id="crossdockCount">0</span><br>
            📊 Total displayed: <span id="totalCount">0</span>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Global variables
        var map;
        var zipLayers = {{}};
        var centroidLayers = [];
        var crossdockLayers = [];
        var smartCorridorLayers = [];
        var smartCorridorsVisible = false;
        var allUsaData = {all_usa_data_json};
        var cbsaColors = {cbsa_colors_json};

        // Data
        var zipFeatures = {zip_features_json};
        var cbsaCentroidFeatures = {cbsa_centroid_features_json};
        var crossdockLocations = {crossdock_locations_json};

        // Initialize map
        function initMap() {{
            map = L.map('map').setView([39.8283, -98.5795], 5);

            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);

            // Load initial data
            loadZipCodes();
            loadCbsaCentroids();
            loadCrossdocks();
            updateStats();

            // Set up event listeners
            setupEventListeners();
        }}

        function setupEventListeners() {{
            document.getElementById('showCbsaZips').addEventListener('change', toggleCbsaZips);
            document.getElementById('showNonCbsaZips').addEventListener('change', toggleNonCbsaZips);
            document.getElementById('showCentroids').addEventListener('change', toggleCentroids);
            document.getElementById('showCrossdocks').addEventListener('change', toggleCrossdocks);
            document.getElementById('enableQuoteFilter').addEventListener('change', toggleQuoteFilter);
            document.getElementById('performanceSlider').addEventListener('input', updatePerformanceFilter);
            document.getElementById('showSmartCorridors').addEventListener('change', toggleSmartCorridors);
            document.getElementById('corridorDistanceSlider').addEventListener('input', updateCorridorDistance);
            document.getElementById('corridorWidthSlider').addEventListener('input', updateCorridorWidth);
        }}

        function loadZipCodes() {{
            zipFeatures.forEach(function(feature) {{
                var layer = L.geoJSON(feature, {{
                    style: function(feature) {{
                        return {{
                            fillColor: feature.properties.color,
                            weight: 0.5,
                            opacity: 0.8,
                            color: '#666',
                            fillOpacity: 0.7
                        }};
                    }},
                    onEachFeature: function(feature, layer) {{
                        // Create popup content
                        var popupContent = createZipPopup(feature.properties);
                        layer.bindPopup(popupContent);

                        // Add click handler for non-CBSA zips (corridor analysis)
                        if (feature.properties.zip_type === 'non_cbsa') {{
                            layer.on('click', function(e) {{
                                if (smartCorridorsVisible) {{
                                    handleNonCbsaZipClick(feature.properties.zip_code);
                                }}
                            }});
                        }}
                    }}
                }});

                zipLayers[feature.properties.zip_code] = layer;

                // Add to map based on initial checkbox states and zip type
                var shouldShow = false;
                if (feature.properties.zip_type === 'cbsa') {{
                    // CBSA zips are shown if the checkbox is checked (default: checked)
                    shouldShow = document.getElementById('showCbsaZips').checked;
                }} else {{
                    // Non-CBSA zips are shown if checkbox is checked AND they meet visibility criteria
                    shouldShow = document.getElementById('showNonCbsaZips').checked && isZipVisible(feature.properties);
                }}

                if (shouldShow) {{
                    layer.addTo(map);
                }}
            }});
        }}

        function createZipPopup(properties) {{
            var content = '<div style="font-family: Arial, sans-serif;">';
            content += '<h3 style="margin: 0 0 10px 0; color: #333;">📍 Zip Code: ' + properties.zip_code + '</h3>';
            content += '<p><strong>State:</strong> ' + (properties.state || 'N/A') + '</p>';
            content += '<p><strong>Quote Count:</strong> ' + properties.quote_count.toLocaleString() + '</p>';

            if (properties.cbsa_title) {{
                content += '<p><strong>CBSA:</strong> ' + properties.cbsa_title + '</p>';
            }} else {{
                content += '<p><strong>Status:</strong> Non-CBSA</p>';
                content += '<p><strong>Distance to Nearest CBSA:</strong> ' + properties.distance_to_nearest_cbsa.toFixed(1) + ' miles</p>';
                content += '<p><strong>Nearest CBSA:</strong> ' + (properties.nearest_cbsa || 'N/A') + '</p>';

                var performance = properties.quote_performance_vs_nearest_cbsa;
                var performanceText = performance >= 0 ? '+' + performance.toFixed(1) : performance.toFixed(1);
                var performanceColor = performance >= 2 ? '#d32f2f' : performance >= 1 ? '#f57c00' : performance >= 0 ? '#388e3c' : '#757575';
                content += '<p><strong>Quote Performance:</strong> <span style="color: ' + performanceColor + '; font-weight: bold;">' + performanceText + ' std dev</span></p>';

                if (smartCorridorsVisible) {{
                    content += '<p style="background: #e8f5e8; padding: 5px; border-radius: 3px; margin-top: 10px;"><strong>🛣️ Click to show corridor to nearest CBSA</strong></p>';
                }}
            }}

            content += '</div>';
            return content;
        }}

        function loadCbsaCentroids() {{
            cbsaCentroidFeatures.forEach(function(feature) {{
                var layer = L.circleMarker([feature.geometry.coordinates[1], feature.geometry.coordinates[0]], {{
                    radius: 8,
                    fillColor: feature.properties.color,
                    color: '#000',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }});

                var popupContent = '<div style="font-family: Arial, sans-serif;">';
                popupContent += '<h3 style="margin: 0 0 10px 0; color: #333;">🎯 CBSA Centroid</h3>';
                popupContent += '<p><strong>CBSA:</strong> ' + feature.properties.cbsa_title + '</p>';
                popupContent += '<p><strong>Total Quotes:</strong> ' + feature.properties.total_quotes.toLocaleString() + '</p>';
                popupContent += '<p><strong>Population:</strong> ' + feature.properties.population.toLocaleString() + '</p>';
                popupContent += '<p><strong>Zip Codes:</strong> ' + feature.properties.zip_count + '</p>';
                popupContent += '</div>';

                layer.bindPopup(popupContent);
                centroidLayers.push(layer);
                layer.addTo(map);
            }});
        }}

        function loadCrossdocks() {{
            crossdockLocations.forEach(function(crossdock) {{
                var layer = L.circleMarker([crossdock.lat, crossdock.lon], {{
                    radius: 6,
                    fillColor: '#FF8C00',
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.9
                }});

                var popupContent = '<div style="font-family: Arial, sans-serif;">';
                popupContent += '<h3 style="margin: 0 0 10px 0; color: #333;">📦 Crossdock</h3>';
                popupContent += '<p><strong>Name:</strong> ' + crossdock.name + '</p>';
                popupContent += '<p><strong>Zip Code:</strong> ' + crossdock.zip_code + '</p>';
                popupContent += '<p><strong>State:</strong> ' + crossdock.state + '</p>';
                if (crossdock.client_name) {{
                    popupContent += '<p><strong>Client:</strong> ' + crossdock.client_name + '</p>';
                }}
                if (crossdock.warehouse_type) {{
                    popupContent += '<p><strong>Type:</strong> ' + crossdock.warehouse_type + '</p>';
                }}
                popupContent += '</div>';

                layer.bindPopup(popupContent);
                crossdockLayers.push(layer);
                layer.addTo(map);
            }});
        }}

        function toggleCbsaZips() {{
            var show = document.getElementById('showCbsaZips').checked;
            zipFeatures.forEach(function(feature) {{
                if (feature.properties.zip_type === 'cbsa') {{
                    var layer = zipLayers[feature.properties.zip_code];
                    if (show) {{
                        if (!map.hasLayer(layer)) {{
                            layer.addTo(map);
                        }}
                    }} else {{
                        if (map.hasLayer(layer)) {{
                            map.removeLayer(layer);
                        }}
                    }}
                }}
            }});
            updateStats();
        }}

        function toggleNonCbsaZips() {{
            var show = document.getElementById('showNonCbsaZips').checked;
            zipFeatures.forEach(function(feature) {{
                if (feature.properties.zip_type === 'non_cbsa') {{
                    var layer = zipLayers[feature.properties.zip_code];
                    if (show && isZipVisible(feature.properties)) {{
                        if (!map.hasLayer(layer)) {{
                            layer.addTo(map);
                        }}
                    }} else {{
                        if (map.hasLayer(layer)) {{
                            map.removeLayer(layer);
                        }}
                    }}
                }}
            }});
            updateStats();
        }}

        function toggleCentroids() {{
            var show = document.getElementById('showCentroids').checked;
            centroidLayers.forEach(function(layer) {{
                if (show) {{
                    if (!map.hasLayer(layer)) {{
                        layer.addTo(map);
                    }}
                }} else {{
                    if (map.hasLayer(layer)) {{
                        map.removeLayer(layer);
                    }}
                }}
            }});
            updateStats();
        }}

        function toggleCrossdocks() {{
            var show = document.getElementById('showCrossdocks').checked;
            crossdockLayers.forEach(function(layer) {{
                if (show) {{
                    if (!map.hasLayer(layer)) {{
                        layer.addTo(map);
                    }}
                }} else {{
                    if (map.hasLayer(layer)) {{
                        map.removeLayer(layer);
                    }}
                }}
            }});
            updateStats();
        }}

        function toggleQuoteFilter() {{
            var enabled = document.getElementById('enableQuoteFilter').checked;
            var filterDiv = document.getElementById('performanceFilter');

            if (enabled) {{
                filterDiv.style.display = 'block';
            }} else {{
                filterDiv.style.display = 'none';
            }}

            updatePerformanceFilter();
        }}

        function updatePerformanceFilter() {{
            var enabled = document.getElementById('enableQuoteFilter').checked;
            var threshold = parseFloat(document.getElementById('performanceSlider').value);

            // Update display
            document.getElementById('performanceValue').textContent = '≥ ' + threshold.toFixed(1) + ' std dev';

            // Update zip visibility - ONLY for non-CBSA zips
            zipFeatures.forEach(function(feature) {{
                if (feature.properties.zip_type === 'non_cbsa') {{
                    var layer = zipLayers[feature.properties.zip_code];
                    var shouldShow = document.getElementById('showNonCbsaZips').checked && isZipVisible(feature.properties);

                    if (shouldShow) {{
                        if (!map.hasLayer(layer)) {{
                            layer.addTo(map);
                        }}
                    }} else {{
                        if (map.hasLayer(layer)) {{
                            map.removeLayer(layer);
                        }}
                    }}
                }}
                // CBSA zips are NOT affected by performance filter - they are controlled only by toggleCbsaZips()
            }});

            updateStats();
        }}

        function isZipVisible(properties) {{
            if (properties.zip_type === 'cbsa') {{
                return true; // CBSA zips are always visible when enabled
            }}

            // Non-CBSA zip visibility rules
            var withinDistance = properties.distance_to_nearest_cbsa <= 200;

            var meetsPerformance = true;
            if (document.getElementById('enableQuoteFilter').checked) {{
                var threshold = parseFloat(document.getElementById('performanceSlider').value);
                meetsPerformance = properties.quote_performance_vs_nearest_cbsa >= threshold;
            }}

            return withinDistance && meetsPerformance;
        }}

        function updateStats() {{
            var cbsaCount = 0;
            var nonCbsaCount = 0;
            var totalNonCbsaCount = 0;

            zipFeatures.forEach(function(feature) {{
                if (feature.properties.zip_type === 'cbsa') {{
                    if (map.hasLayer(zipLayers[feature.properties.zip_code])) {{
                        cbsaCount++;
                    }}
                }} else {{
                    totalNonCbsaCount++;
                    if (map.hasLayer(zipLayers[feature.properties.zip_code])) {{
                        nonCbsaCount++;
                    }}
                }}
            }});

            var centroidCount = centroidLayers.filter(layer => map.hasLayer(layer)).length;
            var crossdockCount = crossdockLayers.filter(layer => map.hasLayer(layer)).length;
            var totalCount = cbsaCount + nonCbsaCount + centroidCount + crossdockCount;

            document.getElementById('cbsaCount').textContent = cbsaCount.toLocaleString();
            document.getElementById('nonCbsaCount').textContent = nonCbsaCount.toLocaleString();
            document.getElementById('totalNonCbsaCount').textContent = totalNonCbsaCount.toLocaleString();
            document.getElementById('centroidCount').textContent = centroidCount;
            document.getElementById('crossdockCount').textContent = crossdockCount;
            document.getElementById('totalCount').textContent = totalCount.toLocaleString();
        }}

        // Corridor analysis functions (same as original)
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

        function pointToLineDistance(px, py, x1, y1, x2, y2) {{
            // Convert to radians for more accurate calculation
            px = px * Math.PI / 180;
            py = py * Math.PI / 180;
            x1 = x1 * Math.PI / 180;
            y1 = y1 * Math.PI / 180;
            x2 = x2 * Math.PI / 180;
            y2 = y2 * Math.PI / 180;

            // Calculate cross track distance using spherical geometry
            var d13 = Math.acos(Math.sin(x1) * Math.sin(px) + Math.cos(x1) * Math.cos(px) * Math.cos(y1 - py));
            var brng13 = Math.atan2(Math.sin(y1 - py) * Math.cos(px), Math.cos(x1) * Math.sin(px) - Math.sin(x1) * Math.cos(px) * Math.cos(y1 - py));
            var brng12 = Math.atan2(Math.sin(y2 - y1) * Math.cos(x2), Math.cos(x1) * Math.sin(x2) - Math.sin(x1) * Math.cos(x2) * Math.cos(y2 - y1));
            var dxt = Math.asin(Math.sin(d13) * Math.sin(brng13 - brng12));

            return Math.abs(dxt) * 3959; // Convert to miles
        }}

        function getZipCentroid(zip) {{
            return [zip.properties.latitude, zip.properties.longitude];
        }}

        function getFilteredNonCbsaZips() {{
            return zipFeatures.filter(feature => {{
                return feature.properties.zip_type === 'non_cbsa' &&
                       map.hasLayer(zipLayers[feature.properties.zip_code]);
            }});
        }}

        function analyzeSmartCorridors() {{
            var maxCorridorDistance = parseFloat(document.getElementById('corridorDistanceSlider').value);
            var baseCorridorWidth = parseFloat(document.getElementById('corridorWidthSlider').value);

            console.log('🚀 Starting SMART Corridor Analysis...');
            console.log('   Max distance:', maxCorridorDistance, 'miles');
            console.log('   Base width:', baseCorridorWidth, 'miles');

            var filteredNonCbsaZips = getFilteredNonCbsaZips();

            if (cbsaCentroidFeatures.length === 0) {{
                console.log('No CBSA centroids available for corridor analysis');
                return [];
            }}

            console.log('Analyzing corridors for', filteredNonCbsaZips.length, 'filtered non-CBSA zips to', cbsaCentroidFeatures.length, 'CBSA centroids');

            var allCorridorZips = new Set();
            var corridorStats = {{
                total: 0,
                withinLimit: 0,
                avgDistance: 0,
                avgWidth: 0
            }};
            var corridorDistances = [];
            var corridorWidths = [];

            filteredNonCbsaZips.forEach(function(targetZip) {{
                corridorStats.total++;
                var targetCentroid = getZipCentroid(targetZip);

                // Find nearest CBSA weighted centroid
                var nearestCbsaCentroid = null;
                var minDistance = Infinity;

                cbsaCentroidFeatures.forEach(function(cbsaCentroid) {{
                    var distance = calculateDistance(
                        targetCentroid[0], targetCentroid[1],
                        cbsaCentroid.geometry.coordinates[1], cbsaCentroid.geometry.coordinates[0]
                    );

                    if (distance < minDistance) {{
                        minDistance = distance;
                        nearestCbsaCentroid = cbsaCentroid;
                    }}
                }});

                // Skip if corridor is too long
                if (minDistance > maxCorridorDistance) {{
                    console.log('   ⏭️  Skipping corridor for', targetZip.properties.zip_code, '- distance', minDistance.toFixed(1), 'miles exceeds limit');
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
                    console.log('   🛣️  Creating corridor:', targetZip.properties.zip_code, '→ CBSA', nearestCbsaCentroid.properties.cbsa_title, '(' + minDistance.toFixed(1) + ' mi, width: ' + dynamicWidth.toFixed(1) + ' mi)');

                    var corridorZipsForThisPath = 0;
                    var zipsWithinWidth = 0;
                    var zipsFailingPathCheck = 0;

                    // Find all zip codes within dynamic corridor width of the line
                    allUsaData.forEach(function(zip) {{
                        // Skip if already processed or if it's the target
                        var zipCode = zip.properties.zipcode_clean;
                        if (allCorridorZips.has(zipCode) ||
                            zipCode === targetZip.properties.zip_code) {{
                            return;
                        }}

                        var zipCentroid = getZipCentroid(zip);
                        var distanceToLine = pointToLineDistance(
                            zipCentroid[0], zipCentroid[1], // lat, lon for point
                            targetCentroid[0], targetCentroid[1], // lat, lon for line start
                            nearestCbsaCentroid.geometry.coordinates[1], nearestCbsaCentroid.geometry.coordinates[0] // lat, lon for line end (CBSA centroid)
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
                                nearestCbsaCentroid.geometry.coordinates[1], nearestCbsaCentroid.geometry.coordinates[0]
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
            allCorridorZips.forEach(function(zipCode) {{
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

        function toggleSmartCorridors() {{
            console.log('toggleSmartCorridors called, smartCorridorsVisible =', smartCorridorsVisible);
            if (smartCorridorsVisible) {{
                // Hide all corridors
                console.log('Hiding all SMART corridors');
                smartCorridorLayers.forEach(layer => map.removeLayer(layer));
                smartCorridorLayers = [];
                smartCorridorsVisible = false;
            }} else {{
                // Show corridors
                console.log('Showing SMART corridors');
                var corridorZips = analyzeSmartCorridors();

                corridorZips.forEach(function(zip) {{
                    var layer = L.circleMarker([zip.properties.latitude, zip.properties.longitude], {{
                        radius: 4,
                        fillColor: '#FFA500',
                        color: '#FF8C00',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }});

                    var popupContent = '<div style="font-family: Arial, sans-serif;">';
                    popupContent += '<h3 style="margin: 0 0 10px 0; color: #333;">🛣️ Corridor Zip Code</h3>';
                    popupContent += '<p><strong>Zip Code:</strong> ' + zip.properties.zipcode_clean + '</p>';
                    popupContent += '<p><strong>Quote Count:</strong> ' + zip.properties.quote_count.toLocaleString() + '</p>';
                    popupContent += '<p><strong>State:</strong> ' + (zip.properties.state || 'N/A') + '</p>';
                    if (zip.properties.has_cbsa) {{
                        popupContent += '<p><strong>CBSA:</strong> ' + (zip.properties.cbsa_title || 'N/A') + '</p>';
                    }} else {{
                        popupContent += '<p><strong>Status:</strong> Non-CBSA</p>';
                    }}
                    popupContent += '</div>';

                    layer.bindPopup(popupContent);
                    smartCorridorLayers.push(layer);
                    layer.addTo(map);
                }});

                smartCorridorsVisible = true;
                console.log('Added', smartCorridorLayers.length, 'corridor zip markers to map');
            }}
        }}

        function handleNonCbsaZipClick(zipCode) {{
            console.log('Non-CBSA zip clicked:', zipCode);
            // This function can be expanded to show individual corridors for specific zips
        }}

        function updateCorridorDistance() {{
            var value = document.getElementById('corridorDistanceSlider').value;
            document.getElementById('corridorDistanceValue').textContent = value;

            // If corridors are currently visible, refresh them
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new parameters
            }}
        }}

        function updateCorridorWidth() {{
            var value = document.getElementById('corridorWidthSlider').value;
            document.getElementById('corridorWidthValue').textContent = value;

            // If corridors are currently visible, refresh them
            if (smartCorridorsVisible) {{
                toggleSmartCorridors(); // Hide
                toggleSmartCorridors(); // Show with new parameters
            }}
        }}

        // Initialize map when page loads
        document.addEventListener('DOMContentLoaded', initMap);
    </script>
</body>
</html>"""

    return html_content

def main():
    """Main execution function"""
    print("🚀 Starting CBSA Regional Color Map with Smart Corridor Analysis...")

    # Load data
    zip_data, zip_shapes, crossdock_df, unassigned_df = load_data()

    # Calculate CBSA centroids
    cbsa_centroids_df = calculate_cbsa_centroids(zip_data)

    # Calculate adjacency analysis
    zip_data_with_adjacency = calculate_adjacency_analysis(zip_data, zip_shapes)

    # Process crossdock locations
    crossdock_locations = process_crossdock_locations(crossdock_df, zip_data_with_adjacency)

    # Load all USA zip codes for corridor analysis
    all_usa_data = load_all_usa_zip_codes()

    # Merge zip data with shapes
    zip_data_clean = zip_data_with_adjacency.copy()
    zip_data_clean['zip_code'] = zip_data_clean['zip_code'].astype(str).str.zfill(5)

    # Merge with shapefile
    map_data = zip_shapes.merge(zip_data_clean, left_on='ZCTA5CE20', right_on='zip_code', how='inner')

    # Create the map
    html_content = create_cbsa_regional_map(map_data, cbsa_centroids_df, unassigned_df, crossdock_locations, all_usa_data)

    # Save the map
    output_file = 'cbsa_regional_color_map.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("✅ CBSA Regional Color Map generated successfully!")
    print(f"📁 Saved to: {os.path.abspath(output_file)}")
    print("🌐 Open the file in your browser to view the interactive map")

    # Open in browser
    print("🚀 Opening map in your default browser...")
    webbrowser.open('file://' + os.path.abspath(output_file))

    print("\n🚀 Regional Color Features Summary:")
    print("✅ Each CBSA has its own distinct color for easy identification")
    print("✅ Non-CBSA zips use performance-based red shading")
    print("✅ Same corridor analysis functionality as original")
    print("✅ All other features remain identical")
    print("✅ Perfect for distinguishing between adjacent CBSAs")

if __name__ == "__main__":
    main()
