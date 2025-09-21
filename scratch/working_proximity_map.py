import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from geopy.distance import geodesic

def load_and_prepare_data():
    """Load and prepare the data for heat mapping with proximity calculations"""
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

def calculate_distances_python(data, zip_shapes):
    """Pre-calculate distances in Python for better performance and accuracy"""
    print("Pre-calculating distances between zip codes...")
    
    # Filter for zip codes with quotes
    quote_data = data[data['Total Quotes'] > 0]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        quote_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    # Calculate centroids
    map_data_projected = map_data.to_crs('EPSG:3857')  # Web Mercator for accurate centroids
    centroids = map_data_projected.geometry.centroid.to_crs('EPSG:4326')  # Back to lat/lon
    
    map_data['centroid_lat'] = centroids.y
    map_data['centroid_lon'] = centroids.x
    
    # Separate CSA and non-CSA zip codes
    csa_zips = map_data[map_data['Primary CSA Name'].notna()]
    non_csa_zips = map_data[map_data['Primary CSA Name'].isna()]
    
    print(f"Calculating distances for {len(non_csa_zips)} non-CSA zip codes to {len(csa_zips)} CSA zip codes...")
    
    # Calculate minimum distance for each non-CSA zip code to any CSA zip code
    distance_results = []
    
    for idx, non_csa_row in non_csa_zips.iterrows():
        non_csa_coord = (non_csa_row['centroid_lat'], non_csa_row['centroid_lon'])
        min_distance = float('inf')
        nearest_csa_zip = None
        
        for csa_idx, csa_row in csa_zips.iterrows():
            csa_coord = (csa_row['centroid_lat'], csa_row['centroid_lon'])
            distance = geodesic(non_csa_coord, csa_coord).miles
            
            if distance < min_distance:
                min_distance = distance
                nearest_csa_zip = csa_row['Zipcode_clean']
        
        distance_results.append({
            'zipcode_clean': non_csa_row['Zipcode_clean'],
            'min_distance_to_csa': min_distance,
            'nearest_csa_zip': nearest_csa_zip
        })
    
    # Convert to DataFrame and merge back
    distance_df = pd.DataFrame(distance_results)
    map_data = map_data.merge(distance_df, on='zipcode_clean', how='left')
    
    # Fill NaN distances for CSA zip codes (they have distance 0 to themselves)
    map_data['min_distance_to_csa'] = map_data['min_distance_to_csa'].fillna(0)
    
    print(f"Distance calculation complete. Distance range: {map_data['min_distance_to_csa'].min():.1f} - {map_data['min_distance_to_csa'].max():.1f} miles")
    
    return map_data

def create_working_proximity_map(map_data, unassigned_zips):
    """Create a working proximity map with pre-calculated distances"""
    print("Creating working proximity map...")
    
    # Convert to WGS84 for web display
    map_data_wgs84 = map_data.to_crs('EPSG:4326')
    
    # Calculate percentiles for color scaling
    all_quotes = map_data_wgs84['Total Quotes']
    quote_percentiles = np.percentile(all_quotes, [0, 25, 50, 75, 90, 95, 99, 100])
    
    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        has_csa = pd.notna(row['Primary CSA Name'])
        
        # Calculate percentile position
        quotes = row['Total Quotes']
        percentile = (quotes - quote_percentiles[0]) / (quote_percentiles[-1] - quote_percentiles[0])
        percentile = max(0, min(1, percentile))  # Clamp between 0 and 1
        
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
                "distance_to_csa": float(row['min_distance_to_csa']),
                "nearest_csa_zip": row.get('nearest_csa_zip', '')
            },
            "geometry": json.loads(gpd.GeoSeries([row['geometry']]).to_json())['features'][0]['geometry']
        }
        zip_features.append(feature)
    
    # Create the HTML map
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Working Proximity-Filtered Heat Map</title>
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
            min-width: 280px;
            font-size: 13px;
        }}
        .controls h4 {{ margin: 0 0 15px 0; color: #333; }}
        .slider-container {{
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .slider-container label {{
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #495057;
        }}
        #distanceSlider {{
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }}
        #distanceSlider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
        }}
        #distanceSlider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #007cba;
            cursor: pointer;
            border: none;
        }}
        .distance-display {{
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: #007cba;
            margin: 8px 0;
            padding: 5px;
            background: #e3f2fd;
            border-radius: 3px;
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
        }}
        .checkbox-container label {{
            font-weight: normal;
            margin-left: 5px;
        }}
        .stats {{
            font-size: 11px;
            color: #666;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            line-height: 1.4;
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
            <input type="range" id="distanceSlider" min="5" max="200" value="50" step="5">
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
        <div><strong>CSA-Assigned (Blue Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #f0f8ff;"></div>
            <div class="color-box" style="background: #87ceeb;"></div>
            <div class="color-box" style="background: #4682b4;"></div>
            <div class="color-box" style="background: #191970;"></div>
            <span>Low → High Quotes</span>
        </div>
        <div style="margin-top: 10px;"><strong>Non-CSA (Red Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #ffe4e1;"></div>
            <div class="color-box" style="background: #ffa07a;"></div>
            <div class="color-box" style="background: #ff6347;"></div>
            <div class="color-box" style="background: #8b0000;"></div>
            <span>Low → High Quotes</span>
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
        
        console.log('Loaded', allZipData.length, 'zip codes');
        
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
                var lightBlue = [240, 248, 255];
                var darkBlue = [25, 25, 112];
                var color = interpolateColor(lightBlue, darkBlue, percentile);
                return rgbToHex(color);
            }} else {{
                var lightRed = [255, 228, 225];
                var darkRed = [139, 0, 0];
                var color = interpolateColor(lightRed, darkRed, percentile);
                return rgbToHex(color);
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
                'CSA zip codes: ' + csaCount.toLocaleString() + '<br>' +
                'Non-CSA (within ' + currentDistanceThreshold + ' miles): ' + filteredNonCSACount.toLocaleString() + '<br>' +
                'Total non-CSA zip codes: ' + nonCSACount.toLocaleString() + '<br>' +
                '<strong>Total displayed: ' + (csaCount + filteredNonCSACount).toLocaleString() + '</strong>';
            
            console.log('Map updated - Distance:', currentDistanceThreshold, 'CSA:', csaCount, 'Non-CSA shown:', filteredNonCSACount);
        }}
        
        // Event listeners
        document.getElementById('distanceSlider').addEventListener('input', function(e) {{
            currentDistanceThreshold = parseInt(e.target.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            updateMap();
        }});
        
        document.getElementById('showAllCSA').addEventListener('change', updateMap);
        document.getElementById('showNearbyNonCSA').addEventListener('change', updateMap);
        
        // Initial load
        updateMap();
        console.log('Map initialized');
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
        
        # Calculate distances
        map_data = calculate_distances_python(merged_data, zip_shapes)
        
        # Create map
        html_content = create_working_proximity_map(map_data, unassigned_zips)
        
        if html_content:
            # Save map
            map_file = 'working_proximity_map.html'
            with open(map_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\nWorking proximity map saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("WORKING PROXIMITY MAP FEATURES:")
            print("✅ Pre-calculated accurate distances using geopy")
            print("✅ Working distance slider (5-200 miles)")
            print("✅ Real-time filtering based on actual distances")
            print("✅ Distance shown in tooltips and popups")
            print("✅ Live statistics with accurate counts")
            print("="*60)
            
        else:
            print("Failed to create map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
