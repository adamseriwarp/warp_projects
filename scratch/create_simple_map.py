import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import json
import webbrowser
import os

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

def create_interactive_map(data, zip_shapes, show_all_csas=True):
    """Create an interactive map with CSA clusters and clickable zip codes"""
    print("Creating interactive map...")

    # Filter for all CSAs (not just those with quotes)
    if show_all_csas:
        csa_data = data[data['Primary CSA Name'].notna()]
        print(f"Showing ALL {csa_data['Primary CSA Name'].nunique()} CSAs")
    else:
        # Original logic for top CSAs with quotes
        csa_data = data[data['Primary CSA Name'].notna() & (data['Total Quotes'] > 0)]
        top_csas = csa_data.groupby('Primary CSA Name')['Total Quotes'].sum().sort_values(ascending=False).head(10)
        csa_data = csa_data[csa_data['Primary CSA Name'].isin(top_csas.index)]
    
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
    
    unique_csas_count = map_data['Primary CSA Name'].nunique()
    print(f"Displaying {len(map_data)} zip codes across {unique_csas_count} CSAs")
    
    # Create base map
    center_lat = map_data.geometry.centroid.y.mean()
    center_lon = map_data.geometry.centroid.x.mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Generate colors for all CSAs
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    unique_csas = map_data['Primary CSA Name'].unique()

    # Use a colormap to generate enough distinct colors
    if len(unique_csas) <= 20:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                  'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
                  'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'yellow']
        csa_colors = dict(zip(unique_csas, colors[:len(unique_csas)]))
    else:
        # Generate colors using matplotlib colormap for many CSAs
        cmap = plt.cm.get_cmap('tab20')
        colors = [mcolors.rgb2hex(cmap(i / len(unique_csas))) for i in range(len(unique_csas))]
        csa_colors = dict(zip(unique_csas, colors))
    
    # Add zip codes to map grouped by CSA
    for csa in unique_csas:
        csa_zips = map_data[map_data['Primary CSA Name'] == csa]
        
        # Create feature group for this CSA
        csa_group = folium.FeatureGroup(name=f"{csa} ({len(csa_zips)} zips)")
        
        for idx, row in csa_zips.iterrows():
            # Create popup with zip code info
            popup_html = f"""
            <div style="width: 250px; font-family: Arial;" data-zipcode="{row['Zipcode_clean']}">
                <h4 style="margin: 0; color: #333;">Zip Code: {row['Zipcode']}</h4>
                <hr style="margin: 5px 0;">
                <b>CSA:</b> {row['Primary CSA Name']}<br>
                <b>Location:</b> {row['City']}, {row['State']}<br>
                <b>Total Quotes:</b> {row['Total Quotes']:,.0f}<br>
                <b>Pickup Count:</b> {row['Pickup Count']:,.0f}<br>
                <b>Dropoff Count:</b> {row['Dropoff Count']:,.0f}<br>
                <b>Population:</b> {row['ZCTA Population (2020)']:,}<br>
                <hr style="margin: 5px 0;">
                <small><i>Click to unassign from CSA</i></small>
                <br>
                <button onclick="unassignZip('{row['Zipcode_clean']}')"
                        style="margin-top: 5px; padding: 3px 8px; background: #ff4444; color: white; border: none; border-radius: 3px; cursor: pointer;">
                    🗑️ Unassign from CSA
                </button>
            </div>
            """
            
            # Color intensity based on quote volume within CSA
            max_quotes = csa_zips['Total Quotes'].max()
            opacity = 0.4 + 0.6 * (row['Total Quotes'] / max_quotes) if max_quotes > 0 else 0.4
            
            folium.GeoJson(
                row['geometry'],
                style_function=lambda feature, color=csa_colors[csa], op=opacity: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': op,
                },
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Zip: {row['Zipcode']} | CSA: {csa[:30]}... | Quotes: {row['Total Quotes']:,.0f}"
            ).add_to(csa_group)
        
        csa_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add custom JavaScript for unassigning zip codes with visual feedback
    unassign_js = """
    <script>
    var unassignedZips = [];
    var zipElements = {};

    // Store references to zip code elements when the map loads
    document.addEventListener('DOMContentLoaded', function() {
        // Wait a bit for the map to fully load
        setTimeout(function() {
            // Find all path elements (zip code shapes) and store them
            var paths = document.querySelectorAll('path');
            paths.forEach(function(path) {
                // Try to extract zip code from the path's popup or tooltip
                var popup = path.getAttribute('data-popup') || '';
                var tooltip = path.getAttribute('title') || '';
                var zipMatch = (popup + tooltip).match(/Zip:\\s*(\\d{5})/);
                if (zipMatch) {
                    zipElements[zipMatch[1]] = path;
                }
            });
        }, 2000);
    });

    function unassignZip(zipCode) {
        if (!unassignedZips.includes(zipCode)) {
            unassignedZips.push(zipCode);

            // Update visual styling immediately
            updateZipCodeStyling(zipCode);

            alert('Zip code ' + zipCode + ' has been unassigned from its CSA!\\n\\nUnassigned zips: ' + unassignedZips.length + '\\n\\nThe zip code is now shown in gray.');
            console.log('Unassigned zip codes:', unassignedZips);
        } else {
            alert('Zip code ' + zipCode + ' is already unassigned.');
        }
    }

    function updateZipCodeStyling(zipCode) {
        // Find all SVG path elements that might represent this zip code
        var allPaths = document.querySelectorAll('svg path');

        allPaths.forEach(function(path) {
            // Check if this path belongs to the unassigned zip code
            var parentElement = path.closest('.folium-popup, .leaflet-tooltip, [title*="' + zipCode + '"]');
            var nextSibling = path.parentElement.nextElementSibling;
            var prevSibling = path.parentElement.previousElementSibling;

            // Check various ways the zip code might be associated with this path
            if (parentElement ||
                (nextSibling && nextSibling.textContent && nextSibling.textContent.includes(zipCode)) ||
                (prevSibling && prevSibling.textContent && prevSibling.textContent.includes(zipCode))) {

                // Style as unassigned (gray and semi-transparent)
                path.style.fill = '#808080';
                path.style.fillOpacity = '0.3';
                path.style.stroke = '#404040';
                path.style.strokeWidth = '2';
                path.style.strokeDasharray = '5,5';
            }
        });

        // Alternative approach: find by checking popup content
        var popups = document.querySelectorAll('.leaflet-popup-content');
        popups.forEach(function(popup) {
            if (popup.textContent.includes('Zip Code: ' + zipCode)) {
                // Find the associated path element
                var mapContainer = popup.closest('.leaflet-container');
                if (mapContainer) {
                    var paths = mapContainer.querySelectorAll('path');
                    paths.forEach(function(path) {
                        // This is a heuristic - style paths that might be associated
                        if (path.style.fillOpacity && parseFloat(path.style.fillOpacity) > 0.3) {
                            path.style.fill = '#808080';
                            path.style.fillOpacity = '0.3';
                            path.style.stroke = '#404040';
                            path.style.strokeWidth = '2';
                            path.style.strokeDasharray = '5,5';
                        }
                    });
                }
            }
        });
    }

    function exportUnassignedZips() {
        if (unassignedZips.length === 0) {
            alert('No zip codes have been unassigned yet.');
            return;
        }

        var csvContent = "Zipcode,Action\\n";
        unassignedZips.forEach(function(zip) {
            csvContent += zip + ",Unassigned\\n";
        });

        var blob = new Blob([csvContent], { type: 'text/csv' });
        var url = window.URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'unassigned_zip_codes.csv';
        a.click();
        window.URL.revokeObjectURL(url);

        alert('Downloaded CSV with ' + unassignedZips.length + ' unassigned zip codes!');
    }

    // Add a reset function to restore original styling
    function resetAllZips() {
        if (confirm('Reset all unassigned zip codes back to their original CSA assignments?')) {
            unassignedZips = [];
            location.reload(); // Reload the page to restore original styling
        }
    }
    </script>
    """
    
    # Add legend and controls
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 300px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
    <h4 style="margin: 0 0 10px 0;">CSA Zip Code Analysis</h4>
    <p style="margin: 5px 0;"><b>All {unique_csas_count} CSAs</b></p>
    <p style="margin: 5px 0;">• Zip codes colored by CSA</p>
    <p style="margin: 5px 0;">• Opacity indicates quote volume</p>
    <p style="margin: 5px 0;">• Click zip codes to unassign from CSA</p>
    <hr style="margin: 10px 0;">
    <button onclick="exportUnassignedZips()"
            style="padding: 8px 12px; background: #007cba; color: white; border: none; border-radius: 3px; cursor: pointer; width: 100%; margin-bottom: 5px;">
        📥 Export Unassigned Zip Codes
    </button>
    <button onclick="resetAllZips()"
            style="padding: 6px 12px; background: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer; width: 100%;">
        🔄 Reset All Changes
    </button>
    <p style="margin: 5px 0; font-size: 10px; color: #666;">
        Total zip codes displayed: {len(map_data)}<br>
        Total quotes: {map_data['Total Quotes'].sum():,.0f}
    </p>
    </div>
    {unassign_js}
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def main():
    """Main function to create and display the map"""
    try:
        # Load data
        merged_data, zip_shapes = load_and_prepare_data()
        
        # Create map showing ALL CSAs
        map_obj = create_interactive_map(merged_data, zip_shapes, show_all_csas=True)
        
        if map_obj:
            # Save map
            map_file = 'csa_zip_analysis_map.html'
            map_obj.save(map_file)
            print(f"\nMap saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("INSTRUCTIONS:")
            print("1. The map shows ALL CSAs with assigned zip codes")
            print("2. Each CSA is color-coded with its zip codes")
            print("3. Click on any zip code to unassign it from its CSA")
            print("4. Use the 'Export Unassigned Zip Codes' button to download changes")
            print("5. Use layer controls to show/hide specific CSAs")
            print("="*60)
            
        else:
            print("Failed to create map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
