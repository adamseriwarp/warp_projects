import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from sklearn.neighbors import BallTree

def load_and_prepare_data():
    """Load and prepare the data for heat mapping"""
    print("Loading data...")
    
    # Load quote data
    quote_data = pd.read_csv('data/raw/quote_data.csv')
    quote_data['Pickup Count'] = pd.to_numeric(quote_data['Pickup Count'], errors='coerce').fillna(0)
    quote_data['Dropoff Count'] = pd.to_numeric(quote_data['Dropoff Count'], errors='coerce').fillna(0)
    quote_data['Total Quotes'] = quote_data['Pickup Count'] + quote_data['Dropoff Count']
    quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
    
    # Load CSA mapping data
    try:
        csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='latin-1')
        except UnicodeDecodeError:
            csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='cp1252')
    
    csa_mapping['Zip Code_clean'] = csa_mapping['Zip Code'].astype(str).str.zfill(5)
    
    # List of unassigned zip codes from your export
    unassigned_zips = [
        '92363', '92267', '92242', '92332', '92364', '92225', '92280', '92309', 
        '92310', '89049', '89045', '89409', '84034', '84022', '85542', '79837', 
        '27936', '27920', '27943', '27915', '27982', '27968', '27953', '27978', 
        '27959', '27981', '27954', '78851', '78839', '89007', '89003', '89020', 
        '89034', '89027'
    ]
    
    # Clean the unassigned zip codes to 5 digits
    unassigned_zips_clean = [zip_code.zfill(5) for zip_code in unassigned_zips]
    
    print(f"Unassigning {len(unassigned_zips_clean)} zip codes from their CSAs...")
    
    # Create updated CSA mapping by removing CSA assignments for unassigned zip codes
    updated_csa_mapping = csa_mapping.copy()
    mask = updated_csa_mapping['Zip Code_clean'].isin(unassigned_zips_clean)
    updated_csa_mapping.loc[mask, 'Primary CSA'] = None
    updated_csa_mapping.loc[mask, 'Primary CSA Name'] = None
    
    print(f"Updated {mask.sum()} zip codes to remove CSA assignments")
    
    # Merge with quote data
    merged_data = pd.merge(
        quote_data, 
        updated_csa_mapping, 
        left_on='Zipcode_clean', 
        right_on='Zip Code_clean', 
        how='left'
    )
    
    # Load shapefiles
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file('data/shapefiles/cb_2020_us_zcta520_500k.shp')
    zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)
    
    return merged_data, zip_shapes, unassigned_zips_clean

def calculate_distances_efficient(data, zip_shapes):
    """Efficiently calculate distances using BallTree"""
    print("Efficiently calculating distances...")
    
    # Filter for zip codes with quotes
    quote_data = data[data['Total Quotes'] > 0]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        quote_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    # Calculate centroids efficiently
    map_data_projected = map_data.to_crs('EPSG:3857')  # Web Mercator
    centroids = map_data_projected.geometry.centroid.to_crs('EPSG:4326')  # Back to lat/lon
    
    map_data['centroid_lat'] = centroids.y
    map_data['centroid_lon'] = centroids.x
    
    # Separate CSA and non-CSA zip codes
    csa_zips = map_data[map_data['Primary CSA Name'].notna()].copy()
    non_csa_zips = map_data[map_data['Primary CSA Name'].isna()].copy()
    
    print(f"Calculating distances for {len(non_csa_zips)} non-CSA zip codes to {len(csa_zips)} CSA zip codes...")
    
    if len(csa_zips) == 0 or len(non_csa_zips) == 0:
        print("No CSA or non-CSA zip codes found!")
        map_data['min_distance_to_csa'] = 0
        return map_data
    
    # Convert coordinates to radians for BallTree
    csa_coords_rad = np.radians(csa_zips[['centroid_lat', 'centroid_lon']].values)
    non_csa_coords_rad = np.radians(non_csa_zips[['centroid_lat', 'centroid_lon']].values)
    
    # Build BallTree for efficient nearest neighbor search
    tree = BallTree(csa_coords_rad, metric='haversine')
    
    # Find nearest CSA zip code for each non-CSA zip code
    distances_rad, indices = tree.query(non_csa_coords_rad, k=1)
    
    # Convert distances from radians to miles
    earth_radius_miles = 3959
    distances_miles = distances_rad.flatten() * earth_radius_miles
    
    # Add distances to non-CSA zip codes
    non_csa_zips['min_distance_to_csa'] = distances_miles
    
    # Add zero distance for CSA zip codes (they are at distance 0 from themselves)
    csa_zips['min_distance_to_csa'] = 0
    
    # Combine back together
    map_data = pd.concat([csa_zips, non_csa_zips], ignore_index=True)
    
    print(f"Distance calculation complete. Distance range: {map_data['min_distance_to_csa'].min():.1f} - {map_data['min_distance_to_csa'].max():.1f} miles")
    
    return map_data

def create_efficient_proximity_map(map_data, unassigned_zips):
    """Create an efficient proximity map"""
    print("Creating efficient proximity map...")
    
    # Convert to WGS84 for web display
    map_data_wgs84 = map_data.to_crs('EPSG:4326')
    
    # Calculate separate percentiles for better color sensitivity
    all_quotes = map_data_wgs84['Total Quotes']
    csa_quotes = map_data_wgs84[map_data_wgs84['Primary CSA Name'].notna()]['Total Quotes']
    non_csa_quotes = map_data_wgs84[map_data_wgs84['Primary CSA Name'].isna()]['Total Quotes']

    # Calculate separate percentiles for CSA and non-CSA zip codes for better color distribution
    csa_percentiles = np.percentile(csa_quotes, [0, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 100]) if len(csa_quotes) > 0 else [0]
    non_csa_percentiles = np.percentile(non_csa_quotes, [0, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 100]) if len(non_csa_quotes) > 0 else [0]

    print(f"CSA quote range: {csa_percentiles[0]:.0f} - {csa_percentiles[-1]:.0f}")
    print(f"Non-CSA quote range: {non_csa_percentiles[0]:.0f} - {non_csa_percentiles[-1]:.0f}")

    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        has_csa = pd.notna(row['Primary CSA Name'])
        quotes = row['Total Quotes']

        # Calculate percentile position using appropriate group and logarithmic scaling
        if has_csa and len(csa_quotes) > 0:
            # Use CSA-specific percentiles
            percentiles = csa_percentiles
            group_quotes = csa_quotes
        elif not has_csa and len(non_csa_quotes) > 0:
            # Use non-CSA specific percentiles
            percentiles = non_csa_percentiles
            group_quotes = non_csa_quotes
        else:
            percentiles = [0, max(1, quotes)]
            group_quotes = [quotes]

        # Apply logarithmic scaling for better visual distribution
        if quotes > 0 and percentiles[-1] > 0:
            # Use log scale to spread out the colors better
            log_quotes = np.log10(quotes + 1)  # +1 to handle quotes = 0
            log_max = np.log10(percentiles[-1] + 1)
            log_min = np.log10(max(1, percentiles[0]) + 1)

            if log_max > log_min:
                percentile = (log_quotes - log_min) / (log_max - log_min)
            else:
                percentile = 0.5
        else:
            percentile = 0

        # Ensure percentile is between 0 and 1
        percentile = max(0, min(1, percentile))
        
        feature = {
            "type": "Feature",
            "properties": {
                "zipcode": row['Zipcode'],
                "zipcode_clean": row['Zipcode_clean'],
                "csa_name": row['Primary CSA Name'] if has_csa else None,
                "city": row['City'],
                "state": row['State'],
                "total_quotes": float(row['Total Quotes']),
                "pickup_count": float(row['Pickup Count']),
                "dropoff_count": float(row['Dropoff Count']),
                "population": int(row['ZCTA Population (2020)']) if pd.notna(row['ZCTA Population (2020)']) else 0,
                "has_csa": has_csa,
                "quote_percentile": percentile,
                "was_unassigned": row['Zipcode_clean'] in unassigned_zips,
                "distance_to_csa": float(row['min_distance_to_csa'])
            },
            "geometry": json.loads(gpd.GeoSeries([row['geometry']]).to_json())['features'][0]['geometry']
        }
        zip_features.append(feature)
    
    # Create the HTML map
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Efficient Proximity-Filtered Heat Map</title>
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
            font-size: 12px;
            color: #495057;
            margin-top: 15px;
            padding: 15px;
            background: #f1f3f4;
            border-radius: 8px;
            border-left: 4px solid #007cba;
            line-height: 1.5;
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
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="controls">
        <h4>🎯 Proximity Filter</h4>
        
        <div class="slider-container">
            <label for="distanceSlider">Distance Threshold:</label>
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
            <input type="checkbox" id="showAllCSA" checked>
            <label for="showAllCSA">Show all CSA zip codes</label>
        </div>
        
        <div class="checkbox-container">
            <input type="checkbox" id="showNearbyNonCSA" checked>
            <label for="showNearbyNonCSA">Show nearby non-CSA zip codes</label>
        </div>
        
        <div class="stats" id="filterStats">
            Loading...
        </div>
    </div>
    
    <div class="legend">
        <h4>🔥 Quote Volume Heat Map</h4>
        <div><strong>CSA-Assigned (Enhanced Blue Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #f8fbff;"></div>
            <div class="color-box" style="background: #c6dbed;"></div>
            <div class="color-box" style="background: #6baed6;"></div>
            <div class="color-box" style="background: #3182bd;"></div>
            <div class="color-box" style="background: #08306b;"></div>
            <span>Very Low → Very High Quotes (Log Scale)</span>
        </div>
        <div style="margin-top: 10px;"><strong>Non-CSA (Enhanced Red Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #fff5f0;"></div>
            <div class="color-box" style="background: #fee0d2;"></div>
            <div class="color-box" style="background: #fc9272;"></div>
            <div class="color-box" style="background: #de2d26;"></div>
            <div class="color-box" style="background: #a50f15;"></div>
            <span>Very Low → Very High Quotes (Log Scale)</span>
        </div>
        <div style="margin-top: 8px; font-size: 10px; color: #666; font-style: italic;">
            ✨ Enhanced sensitivity: Logarithmic scaling shows more color variation
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
        var zipLayers = {{}};
        var currentDistanceThreshold = 50;
        
        console.log('Loaded', allZipData.length, 'zip codes with pre-calculated distances');
        
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

            if (props.has_csa) {{
                // Enhanced blue scale with more variation
                if (percentile < 0.2) {{
                    // Very light blue for low quotes
                    var color1 = [248, 251, 255];  // Very light blue
                    var color2 = [198, 219, 239];  // Light blue
                    var localPercentile = percentile / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.5) {{
                    // Light to medium blue
                    var color1 = [198, 219, 239];  // Light blue
                    var color2 = [107, 174, 214];  // Medium blue
                    var localPercentile = (percentile - 0.2) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.8) {{
                    // Medium to dark blue
                    var color1 = [107, 174, 214];  // Medium blue
                    var color2 = [49, 130, 189];   // Dark blue
                    var localPercentile = (percentile - 0.5) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else {{
                    // Very dark blue for high quotes
                    var color1 = [49, 130, 189];   // Dark blue
                    var color2 = [8, 48, 107];     // Very dark blue
                    var localPercentile = (percentile - 0.8) / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }}
            }} else {{
                // Enhanced red scale with more variation
                if (percentile < 0.2) {{
                    // Very light red for low quotes
                    var color1 = [255, 245, 240];  // Very light red
                    var color2 = [254, 224, 210];  // Light red
                    var localPercentile = percentile / 0.2;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.5) {{
                    // Light to medium red
                    var color1 = [254, 224, 210];  // Light red
                    var color2 = [252, 146, 114];  // Medium red
                    var localPercentile = (percentile - 0.2) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else if (percentile < 0.8) {{
                    // Medium to dark red
                    var color1 = [252, 146, 114];  // Medium red
                    var color2 = [222, 45, 38];    // Dark red
                    var localPercentile = (percentile - 0.5) / 0.3;
                    var color = interpolateColor(color1, color2, localPercentile);
                    return rgbToHex(color);
                }} else {{
                    // Very dark red for high quotes
                    var color1 = [222, 45, 38];    // Dark red
                    var color2 = [165, 15, 21];    // Very dark red
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
            var csaStatus = props.has_csa ? 'CSA-Assigned' : 'Non-CSA';
            var distanceInfo = props.has_csa ? '' : 
                '<strong>Distance to nearest CSA:</strong> ' + props.distance_to_csa.toFixed(1) + ' miles<br>';
            
            return `
                <div style="font-family: Arial; max-width: 280px;">
                    <h4 style="margin: 0 0 10px 0;">Zip: ${{props.zipcode}} (${{csaStatus}})</h4>
                    <hr style="margin: 5px 0;">
                    <strong>Location:</strong> ${{props.city}}, ${{props.state}}<br>
                    ${{distanceInfo}}
                    <strong>Total Quotes:</strong> ${{props.total_quotes.toLocaleString()}}<br>
                    <strong>Population:</strong> ${{props.population.toLocaleString()}}<br>
                    <strong>Quote Percentile:</strong> ${{(props.quote_percentile * 100).toFixed(1)}}%
                </div>
            `;
        }}
        
        function updateMap() {{
            console.log('Updating map with distance threshold:', currentDistanceThreshold);
            
            // Clear existing layers
            Object.values(zipLayers).forEach(function(layer) {{
                map.removeLayer(layer);
            }});
            zipLayers = {{}};
            
            var showAllCSA = document.getElementById('showAllCSA').checked;
            var showNearbyNonCSA = document.getElementById('showNearbyNonCSA').checked;
            
            var csaCount = 0;
            var nonCSACount = 0;
            var filteredNonCSACount = 0;
            
            allZipData.forEach(function(feature) {{
                var props = feature.properties;
                var shouldShow = false;
                
                if (props.has_csa && showAllCSA) {{
                    shouldShow = true;
                    csaCount++;
                }} else if (!props.has_csa) {{
                    nonCSACount++;
                    if (showNearbyNonCSA && props.distance_to_csa <= currentDistanceThreshold) {{
                        shouldShow = true;
                        filteredNonCSACount++;
                    }}
                }}
                
                if (shouldShow) {{
                    var layer = L.geoJSON(feature, {{
                        style: getZipStyle,
                        onEachFeature: function(feature, layer) {{
                            layer.bindPopup(createPopupContent(feature));
                            
                            var tooltip = 'Zip: ' + feature.properties.zipcode + 
                                         ' | Quotes: ' + feature.properties.total_quotes.toLocaleString();
                            if (!feature.properties.has_csa) {{
                                tooltip += ' | Dist: ' + feature.properties.distance_to_csa.toFixed(1) + 'mi';
                            }}
                            layer.bindTooltip(tooltip, {{ permanent: false, direction: 'top' }});
                        }}
                    }}).addTo(map);
                    
                    zipLayers[props.zipcode_clean] = layer;
                }}
            }});
            
            // Update stats
            document.getElementById('filterStats').innerHTML = 
                '<strong>Currently Displayed:</strong><br>' +
                '🔵 CSA zip codes: ' + csaCount.toLocaleString() + '<br>' +
                '🔴 Non-CSA (≤ ' + currentDistanceThreshold + ' miles): ' + filteredNonCSACount.toLocaleString() + '<br>' +
                '📊 Total non-CSA zip codes: ' + nonCSACount.toLocaleString() + '<br>' +
                '<strong>📍 Total displayed: ' + (csaCount + filteredNonCSACount).toLocaleString() + '</strong>';
            
            console.log('Map updated - CSA:', csaCount, 'Non-CSA shown:', filteredNonCSACount, 'of', nonCSACount);
        }}
        
        // Event listeners with better handling
        function updateDistance() {{
            var slider = document.getElementById('distanceSlider');
            currentDistanceThreshold = parseInt(slider.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            updateMap();
        }}
        
        document.getElementById('distanceSlider').addEventListener('input', updateDistance);
        document.getElementById('distanceSlider').addEventListener('change', updateDistance);
        
        document.getElementById('showAllCSA').addEventListener('change', updateMap);
        document.getElementById('showNearbyNonCSA').addEventListener('change', updateMap);
        
        // Initial load
        setTimeout(function() {{
            updateMap();
            console.log('Map initialized successfully');
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
        
        # Calculate distances efficiently
        map_data = calculate_distances_efficient(merged_data, zip_shapes)
        
        # Create map
        html_content = create_efficient_proximity_map(map_data, unassigned_zips)
        
        if html_content:
            # Save map
            map_file = 'efficient_proximity_map.html'
            with open(map_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\nEfficient proximity map saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("EFFICIENT PROXIMITY MAP FEATURES:")
            print("✅ Fast distance calculation using BallTree algorithm")
            print("✅ Working distance slider with enhanced styling")
            print("✅ Accurate filtering based on pre-calculated distances")
            print("✅ Real-time statistics and visual feedback")
            print("✅ Enhanced UI with better slider visibility")
            print("="*60)
            
        else:
            print("Failed to create map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
