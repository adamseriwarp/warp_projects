import pandas as pd
import numpy as np
import geopandas as gpd
import json
import webbrowser
import os
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare the data for mapping"""
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
    
    # Merge datasets
    merged_data = pd.merge(
        quote_data, 
        csa_mapping, 
        left_on='Zipcode_clean', 
        right_on='Zip Code_clean', 
        how='left'
    )
    
    # Load shapefiles
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file('data/shapefiles/cb_2020_us_zcta520_500k.shp')
    zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)
    
    return merged_data, zip_shapes

def generate_colors(n_colors):
    """Generate distinct colors for CSAs"""
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (i % 2) * 0.1       # Vary brightness slightly
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def create_dynamic_map(data, zip_shapes):
    """Create a dynamic map with real-time color changes"""
    print("Creating dynamic map...")
    
    # Filter for CSAs
    csa_data = data[data['Primary CSA Name'].notna()]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        csa_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    if map_data.empty:
        print("No data to display!")
        return None
    
    print(f"Displaying {len(map_data)} zip codes across {map_data['Primary CSA Name'].nunique()} CSAs")
    
    # Generate colors for CSAs
    unique_csas = sorted(map_data['Primary CSA Name'].unique())
    colors = generate_colors(len(unique_csas))
    csa_colors = dict(zip(unique_csas, colors))
    
    # Convert geometries to GeoJSON
    map_data_wgs84 = map_data.to_crs('EPSG:4326')
    
    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        # Calculate opacity based on quote volume within CSA
        csa_zips = map_data_wgs84[map_data_wgs84['Primary CSA Name'] == row['Primary CSA Name']]
        max_quotes = csa_zips['Total Quotes'].max()
        opacity = 0.4 + 0.6 * (row['Total Quotes'] / max_quotes) if max_quotes > 0 else 0.4
        
        feature = {
            "type": "Feature",
            "properties": {
                "zipcode": row['Zipcode'],
                "zipcode_clean": row['Zipcode_clean'],
                "csa_name": row['Primary CSA Name'],
                "city": row['City'],
                "state": row['State'],
                "total_quotes": float(row['Total Quotes']),
                "pickup_count": float(row['Pickup Count']),
                "dropoff_count": float(row['Dropoff Count']),
                "population": int(row['ZCTA Population (2020)']),
                "color": csa_colors[row['Primary CSA Name']],
                "opacity": opacity,
                "assigned": True
            },
            "geometry": json.loads(gpd.GeoSeries([row['geometry']]).to_json())['features'][0]['geometry']
        }
        zip_features.append(feature)
    
    # Create the HTML map
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dynamic CSA Zip Code Analysis</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; width: 100vw; }}
        .legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
            font-size: 12px;
        }}
        .legend h4 {{ margin: 0 0 10px 0; }}
        .legend button {{
            width: 100%;
            padding: 8px;
            margin: 3px 0;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        }}
        .export-btn {{ background: #007cba; color: white; }}
        .reset-btn {{ background: #28a745; color: white; }}
        .stats {{ font-size: 10px; color: #666; margin-top: 10px; }}
        .popup-content {{
            font-family: Arial, sans-serif;
            max-width: 250px;
        }}
        .popup-content h4 {{ margin: 0 0 10px 0; color: #333; }}
        .popup-content hr {{ margin: 5px 0; }}
        .unassign-btn {{
            background: #dc3545;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 3px;
            cursor: pointer;
            margin-top: 10px;
            width: 100%;
        }}
        .unassign-btn:hover {{ background: #c82333; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="legend">
        <h4>🗺️ CSA Zip Code Analysis</h4>
        <p><strong>All {len(unique_csas)} CSAs</strong></p>
        <p>• Click zip codes to unassign from CSA</p>
        <p>• Unassigned zips turn gray immediately</p>
        <button class="export-btn" onclick="exportUnassignedZips()">📥 Export Unassigned Zips</button>
        <button class="reset-btn" onclick="resetAllChanges()">🔄 Reset All Changes</button>
        <div class="stats">
            <div>Total zip codes: {len(map_data):,}</div>
            <div>Total quotes: {map_data['Total Quotes'].sum():,.0f}</div>
            <div id="unassigned-count">Unassigned: 0</div>
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
        
        // Store zip code layers and unassigned zips
        var zipLayers = {{}};
        var unassignedZips = [];
        
        // GeoJSON data
        var zipData = {json.dumps(zip_features, indent=2)};
        
        // Function to get style for each zip code
        function getZipStyle(feature) {{
            var props = feature.properties;
            if (!props.assigned) {{
                return {{
                    fillColor: '#808080',
                    weight: 2,
                    opacity: 1,
                    color: '#404040',
                    dashArray: '5,5',
                    fillOpacity: 0.3
                }};
            }}
            return {{
                fillColor: props.color,
                weight: 1,
                opacity: 1,
                color: 'black',
                fillOpacity: props.opacity
            }};
        }}
        
        // Function to create popup content
        function createPopupContent(feature) {{
            var props = feature.properties;
            var status = props.assigned ? 'Assigned to CSA' : 'Unassigned';
            var buttonText = props.assigned ? '🗑️ Unassign from CSA' : '✅ Already Unassigned';
            var buttonDisabled = props.assigned ? '' : 'disabled';
            
            return `
                <div class="popup-content">
                    <h4>Zip Code: ${{props.zipcode}}</h4>
                    <hr>
                    <strong>Status:</strong> ${{status}}<br>
                    <strong>CSA:</strong> ${{props.csa_name || 'None'}}<br>
                    <strong>Location:</strong> ${{props.city}}, ${{props.state}}<br>
                    <strong>Total Quotes:</strong> ${{props.total_quotes.toLocaleString()}}<br>
                    <strong>Pickup Count:</strong> ${{props.pickup_count.toLocaleString()}}<br>
                    <strong>Dropoff Count:</strong> ${{props.dropoff_count.toLocaleString()}}<br>
                    <strong>Population:</strong> ${{props.population.toLocaleString()}}<br>
                    <button class="unassign-btn" onclick="unassignZip('${{props.zipcode_clean}}')" ${{buttonDisabled}}>
                        ${{buttonText}}
                    </button>
                </div>
            `;
        }}
        
        // Function to unassign a zip code
        function unassignZip(zipcode) {{
            if (unassignedZips.includes(zipcode)) {{
                alert('Zip code ' + zipcode + ' is already unassigned.');
                return;
            }}
            
            // Add to unassigned list
            unassignedZips.push(zipcode);
            
            // Update the feature properties
            var layer = zipLayers[zipcode];
            if (layer) {{
                layer.feature.properties.assigned = false;
                
                // Update the layer style immediately
                layer.setStyle(getZipStyle(layer.feature));
                
                // Update popup content
                layer.setPopupContent(createPopupContent(layer.feature));
            }}
            
            // Update counter
            document.getElementById('unassigned-count').textContent = 'Unassigned: ' + unassignedZips.length;
            
            alert('Zip code ' + zipcode + ' has been unassigned!\\nIt should now appear gray on the map.');
        }}
        
        // Function to export unassigned zip codes
        function exportUnassignedZips() {{
            if (unassignedZips.length === 0) {{
                alert('No zip codes have been unassigned yet.');
                return;
            }}
            
            var csvContent = "Zipcode,Action,Timestamp\\n";
            var timestamp = new Date().toISOString();
            unassignedZips.forEach(function(zip) {{
                csvContent += zip + ",Unassigned," + timestamp + "\\n";
            }});
            
            var blob = new Blob([csvContent], {{ type: 'text/csv' }});
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'unassigned_zip_codes_' + new Date().toISOString().slice(0,19).replace(/:/g, '-') + '.csv';
            a.click();
            window.URL.revokeObjectURL(url);
            
            alert('Downloaded CSV with ' + unassignedZips.length + ' unassigned zip codes!');
        }}
        
        // Function to reset all changes
        function resetAllChanges() {{
            if (unassignedZips.length === 0) {{
                alert('No changes to reset.');
                return;
            }}
            
            if (confirm('Reset all ' + unassignedZips.length + ' unassigned zip codes back to their original CSA assignments?')) {{
                // Reset all zip codes
                unassignedZips.forEach(function(zipcode) {{
                    var layer = zipLayers[zipcode];
                    if (layer) {{
                        layer.feature.properties.assigned = true;
                        layer.setStyle(getZipStyle(layer.feature));
                        layer.setPopupContent(createPopupContent(layer.feature));
                    }}
                }});
                
                unassignedZips = [];
                document.getElementById('unassigned-count').textContent = 'Unassigned: 0';
                alert('All changes have been reset!');
            }}
        }}
        
        // Add zip codes to map
        zipData.forEach(function(feature) {{
            var layer = L.geoJSON(feature, {{
                style: getZipStyle,
                onEachFeature: function(feature, layer) {{
                    // Store layer reference
                    zipLayers[feature.properties.zipcode_clean] = layer;
                    
                    // Add popup
                    layer.bindPopup(createPopupContent(feature));
                    
                    // Add tooltip
                    layer.bindTooltip(
                        'Zip: ' + feature.properties.zipcode + 
                        ' | CSA: ' + (feature.properties.csa_name || 'None').substring(0, 30) + 
                        ' | Quotes: ' + feature.properties.total_quotes.toLocaleString(),
                        {{ permanent: false, direction: 'top' }}
                    );
                }}
            }}).addTo(map);
        }});
        
        console.log('Map loaded with', Object.keys(zipLayers).length, 'zip codes');
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    """Main function to create and display the dynamic map"""
    try:
        # Load data
        merged_data, zip_shapes = load_and_prepare_data()
        
        # Create dynamic map
        html_content = create_dynamic_map(merged_data, zip_shapes)
        
        if html_content:
            # Save map
            map_file = 'dynamic_csa_map.html'
            with open(map_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\nDynamic map saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("DYNAMIC MAP FEATURES:")
            print("✅ Real-time color changes when unassigning zip codes")
            print("✅ Immediate visual feedback (gray + dashed border)")
            print("✅ Click any zip code to unassign from CSA")
            print("✅ Export functionality for unassigned zip codes")
            print("✅ Reset button to restore all changes")
            print("✅ Live counter of unassigned zip codes")
            print("="*60)
            
        else:
            print("Failed to create map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
