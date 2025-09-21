import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from datetime import datetime
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

def calculate_centroids(map_data):
    """Calculate centroids for distance calculations"""
    print("Calculating zip code centroids for distance calculations...")
    
    # Convert to a projected coordinate system for accurate centroid calculation
    map_data_projected = map_data.to_crs('EPSG:3857')  # Web Mercator
    centroids = map_data_projected.geometry.centroid
    
    # Convert back to lat/lon for distance calculations
    centroids_wgs84 = centroids.to_crs('EPSG:4326')
    
    # Extract lat/lon coordinates
    map_data['centroid_lat'] = centroids_wgs84.y
    map_data['centroid_lon'] = centroids_wgs84.x
    
    return map_data

def create_proximity_heatmap(data, zip_shapes, unassigned_zips):
    """Create a heat map with dynamic proximity filtering"""
    print("Creating proximity-filtered heat map...")
    
    # Filter for zip codes with quotes
    quote_data = data[data['Total Quotes'] > 0]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        quote_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    if map_data.empty:
        print("No data to display!")
        return None
    
    # Calculate centroids for distance calculations
    map_data = calculate_centroids(map_data)
    
    print(f"Processing {len(map_data)} zip codes with quotes")
    
    # Separate CSA-assigned and unassigned zip codes
    csa_assigned = map_data[map_data['Primary CSA Name'].notna()]
    csa_unassigned = map_data[map_data['Primary CSA Name'].isna()]
    
    print(f"CSA-assigned zip codes: {len(csa_assigned)}")
    print(f"Unassigned zip codes: {len(csa_unassigned)}")
    
    # Convert geometries to GeoJSON
    map_data_wgs84 = map_data.to_crs('EPSG:4326')
    
    # Calculate percentiles for color scaling
    all_quotes = map_data_wgs84['Total Quotes']
    quote_percentiles = np.percentile(all_quotes, [0, 25, 50, 75, 90, 95, 99, 100])
    
    # Prepare CSA-assigned zip code coordinates for proximity calculations
    csa_coords = []
    for idx, row in csa_assigned.iterrows():
        csa_coords.append({
            'lat': row['centroid_lat'],
            'lon': row['centroid_lon'],
            'zipcode': row['Zipcode_clean']
        })
    
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
                "centroid_lat": float(row['centroid_lat']),
                "centroid_lon": float(row['centroid_lon'])
            },
            "geometry": json.loads(gpd.GeoSeries([row['geometry']]).to_json())['features'][0]['geometry']
        }
        zip_features.append(feature)
    
    # Create the HTML heat map with proximity filtering
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Proximity-Filtered Quote Heat Map</title>
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
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 250px;
            font-size: 12px;
        }}
        .controls h4 {{ margin: 0 0 10px 0; }}
        .distance-control {{
            margin: 10px 0;
        }}
        .distance-control label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .distance-control input[type="range"] {{
            width: 100%;
            margin: 5px 0;
            cursor: pointer;
            -webkit-appearance: none;
            appearance: none;
            height: 8px;
            background: #ddd;
            border-radius: 5px;
            outline: none;
        }}
        .distance-control input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #007cba;
            border-radius: 50%;
            cursor: pointer;
        }}
        .distance-control input[type="range"]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            background: #007cba;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .distance-display {{
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            color: #007cba;
            margin: 5px 0;
        }}
        .filter-stats {{
            font-size: 10px;
            color: #666;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
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
        .legend h4 {{ margin: 0 0 10px 0; }}
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
        .popup-content {{
            font-family: Arial, sans-serif;
            max-width: 280px;
        }}
        .popup-content h4 {{ margin: 0 0 10px 0; color: #333; }}
        .popup-content hr {{ margin: 5px 0; }}
        .csa-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            margin-left: 5px;
        }}
        .csa-assigned {{ background: #28a745; color: white; }}
        .csa-unassigned {{ background: #dc3545; color: white; }}
        .recently-unassigned {{ background: #ff8c00; color: white; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="controls">
        <h4>🎯 Proximity Filter</h4>
        <div class="distance-control">
            <label for="distanceSlider">Distance Threshold:</label>
            <input type="range" id="distanceSlider" min="5" max="200" value="50" step="5"
                   oninput="updateDistance(this.value)" onchange="updateDistance(this.value)">
            <div class="distance-display" id="distanceDisplay">50 miles</div>
            <div style="font-size: 10px; color: #666;">
                5 miles ←→ 200 miles
            </div>
        </div>
        <div>
            <label>
                <input type="checkbox" id="showAllCSA" checked> Show all CSA zip codes
            </label>
        </div>
        <div>
            <label>
                <input type="checkbox" id="showNearbyNonCSA" checked> Show nearby non-CSA zip codes
            </label>
        </div>
        <div class="filter-stats" id="filterStats">
            Calculating...
        </div>
    </div>
    
    <div class="legend">
        <h4>🔥 Quote Volume Heat Map</h4>
        
        <div><strong>CSA-Assigned Zip Codes (Blue Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #f0f8ff;"></div>
            <div class="color-box" style="background: #87ceeb;"></div>
            <div class="color-box" style="background: #4682b4;"></div>
            <div class="color-box" style="background: #191970;"></div>
            <span>Low → High Quotes</span>
        </div>
        
        <div style="margin-top: 10px;"><strong>Non-CSA Zip Codes (Red Scale):</strong></div>
        <div class="color-scale">
            <div class="color-box" style="background: #ffe4e1;"></div>
            <div class="color-box" style="background: #ffa07a;"></div>
            <div class="color-box" style="background: #ff6347;"></div>
            <div class="color-box" style="background: #8b0000;"></div>
            <span>Low → High Quotes</span>
        </div>
        
        <div style="margin-top: 10px; font-size: 10px; color: #666;">
            🟠 Orange borders = Recently unassigned<br>
            🎯 Filter shows non-CSA zips within distance threshold
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Initialize map
        var map = L.map('map').setView([39.8283, -98.5795], 4);
        
        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Store all zip code data and layers
        var allZipData = {json.dumps(zip_features, indent=2)};
        var zipLayers = {{}};
        var currentDistanceThreshold = 50;
        
        // Separate CSA and non-CSA zip codes
        var csaZips = allZipData.filter(function(feature) {{
            return feature.properties.has_csa;
        }});
        
        var nonCSAZips = allZipData.filter(function(feature) {{
            return !feature.properties.has_csa;
        }});
        
        console.log('Loaded', csaZips.length, 'CSA zip codes and', nonCSAZips.length, 'non-CSA zip codes');

        // Simple function to update distance threshold
        window.updateDistance = function(value) {{
            currentDistanceThreshold = parseInt(value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            console.log('Distance updated to:', currentDistanceThreshold, 'miles');
            updateMap();
        }};
        
        // Haversine distance calculation
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
        
        // Find minimum distance from a non-CSA zip to any CSA zip
        function findMinDistanceToCSA(nonCSAZip) {{
            var minDistance = Infinity;
            var nonCSALat = nonCSAZip.properties.centroid_lat;
            var nonCSALon = nonCSAZip.properties.centroid_lon;
            
            for (var i = 0; i < csaZips.length; i++) {{
                var csaLat = csaZips[i].properties.centroid_lat;
                var csaLon = csaZips[i].properties.centroid_lon;
                var distance = calculateDistance(nonCSALat, nonCSALon, csaLat, csaLon);
                if (distance < minDistance) {{
                    minDistance = distance;
                }}
            }}
            
            return minDistance;
        }}
        
        // Color interpolation functions
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
        
        // Function to get color based on quote volume and CSA status
        function getQuoteColor(feature) {{
            var props = feature.properties;
            var percentile = props.quote_percentile;
            
            if (props.has_csa) {{
                // Blue scale for CSA-assigned zip codes
                var lightBlue = [240, 248, 255];  // AliceBlue
                var darkBlue = [25, 25, 112];     // MidnightBlue
                var color = interpolateColor(lightBlue, darkBlue, percentile);
                return rgbToHex(color);
            }} else {{
                // Red scale for non-CSA zip codes
                var lightRed = [255, 228, 225];   // MistyRose
                var darkRed = [139, 0, 0];        // DarkRed
                var color = interpolateColor(lightRed, darkRed, percentile);
                return rgbToHex(color);
            }}
        }}
        
        // Function to get style for each zip code
        function getZipStyle(feature) {{
            var props = feature.properties;
            var fillColor = getQuoteColor(feature);
            var opacity = 0.7 + 0.3 * props.quote_percentile;
            
            return {{
                fillColor: fillColor,
                weight: props.was_unassigned ? 3 : 1,
                opacity: 1,
                color: props.was_unassigned ? '#ff8c00' : '#333',
                dashArray: props.was_unassigned ? '5,5' : null,
                fillOpacity: opacity
            }};
        }}
        
        // Function to create popup content
        function createPopupContent(feature) {{
            var props = feature.properties;
            var csaStatus = props.has_csa ? 'CSA-Assigned' : 'Non-CSA';
            var csaBadgeClass = props.has_csa ? 'csa-assigned' : 'csa-unassigned';
            var recentlyUnassigned = props.was_unassigned ? '<span class="csa-badge recently-unassigned">Recently Unassigned</span>' : '';
            
            var distanceInfo = '';
            if (!props.has_csa) {{
                var minDist = findMinDistanceToCSA(feature);
                distanceInfo = '<strong>Distance to nearest CSA zip:</strong> ' + minDist.toFixed(1) + ' miles<br>';
            }}
            
            return `
                <div class="popup-content">
                    <h4>Zip Code: ${{props.zipcode}} <span class="csa-badge ${{csaBadgeClass}}">${{csaStatus}}</span>${{recentlyUnassigned}}</h4>
                    <hr>
                    <strong>CSA:</strong> ${{props.csa_name || 'None'}}<br>
                    <strong>Location:</strong> ${{props.city}}, ${{props.state}}<br>
                    ${{distanceInfo}}
                    <strong>Total Quotes:</strong> ${{props.total_quotes.toLocaleString()}}<br>
                    <strong>Pickup Count:</strong> ${{props.pickup_count.toLocaleString()}}<br>
                    <strong>Dropoff Count:</strong> ${{props.dropoff_count.toLocaleString()}}<br>
                    <strong>Population:</strong> ${{props.population.toLocaleString()}}<br>
                    <strong>Quote Percentile:</strong> ${{(props.quote_percentile * 100).toFixed(1)}}%
                </div>
            `;
        }}
        
        // Function to add a zip code layer to the map
        function addZipLayer(feature) {{
            var layer = L.geoJSON(feature, {{
                style: getZipStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindPopup(createPopupContent(feature));
                    
                    var csaInfo = feature.properties.has_csa ? 
                        (feature.properties.csa_name || 'CSA').substring(0, 25) + '...' : 
                        'Non-CSA';
                    var recentTag = feature.properties.was_unassigned ? ' [Recently Unassigned]' : '';
                    
                    layer.bindTooltip(
                        'Zip: ' + feature.properties.zipcode + 
                        ' | ' + csaInfo + 
                        ' | Quotes: ' + feature.properties.total_quotes.toLocaleString() +
                        recentTag,
                        {{ permanent: false, direction: 'top' }}
                    );
                }}
            }}).addTo(map);
            
            zipLayers[feature.properties.zipcode_clean] = layer;
            return layer;
        }}
        
        // Function to update the map based on current filters
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
            
            // Add CSA zip codes if enabled
            if (showAllCSA) {{
                csaZips.forEach(function(feature) {{
                    addZipLayer(feature);
                    csaCount++;
                }});
            }}
            
            // Add non-CSA zip codes within distance threshold if enabled
            if (showNearbyNonCSA) {{
                nonCSAZips.forEach(function(feature) {{
                    var minDistance = findMinDistanceToCSA(feature);
                    if (minDistance <= currentDistanceThreshold) {{
                        addZipLayer(feature);
                        filteredNonCSACount++;
                    }}
                    nonCSACount++;
                }});
            }}
            
            // Update stats
            document.getElementById('filterStats').innerHTML = 
                'CSA zip codes: ' + csaCount.toLocaleString() + '<br>' +
                'Non-CSA zip codes (total): ' + nonCSACount.toLocaleString() + '<br>' +
                'Non-CSA zip codes (within ' + currentDistanceThreshold + ' miles): ' + filteredNonCSACount.toLocaleString() + '<br>' +
                'Total displayed: ' + (csaCount + filteredNonCSACount).toLocaleString();
        }}
        
        // Event listeners for controls
        document.getElementById('distanceSlider').addEventListener('input', function(e) {{
            currentDistanceThreshold = parseInt(e.target.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            console.log('Distance threshold changed to:', currentDistanceThreshold);
            updateMap();
        }});

        document.getElementById('distanceSlider').addEventListener('change', function(e) {{
            currentDistanceThreshold = parseInt(e.target.value);
            document.getElementById('distanceDisplay').textContent = currentDistanceThreshold + ' miles';
            console.log('Distance threshold changed to:', currentDistanceThreshold);
            updateMap();
        }});

        document.getElementById('showAllCSA').addEventListener('change', function() {{
            console.log('CSA checkbox changed');
            updateMap();
        }});

        document.getElementById('showNearbyNonCSA').addEventListener('change', function() {{
            console.log('Non-CSA checkbox changed');
            updateMap();
        }});

        // Wait for DOM to be fully loaded before initial map load
        setTimeout(function() {{
            console.log('Loading initial map...');
            updateMap();
        }}, 100);
        
        console.log('Proximity-filtered heat map loaded');
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    """Main function to create and display the proximity-filtered heat map"""
    try:
        # Load data
        merged_data, zip_shapes, unassigned_zips = load_and_prepare_data()
        
        # Create proximity heat map
        html_content = create_proximity_heatmap(merged_data, zip_shapes, unassigned_zips)
        
        if html_content:
            # Save map
            map_file = 'proximity_filtered_heatmap.html'
            with open(map_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\nProximity-filtered heat map saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("PROXIMITY-FILTERED HEAT MAP FEATURES:")
            print("🎯 Dynamic distance slider (5-200 miles)")
            print("🔵 Blue scale: CSA-assigned zip codes")
            print("🔴 Red scale: Non-CSA zip codes within distance threshold") 
            print("🟠 Orange borders: Recently unassigned zip codes")
            print("📊 Real-time filtering and statistics")
            print("💡 Click zip codes for distance information")
            print("="*60)
            
        else:
            print("Failed to create heat map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
