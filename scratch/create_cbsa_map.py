import pandas as pd
import numpy as np
import geopandas as gpd
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
    
    # Load CBSA mapping data
    try:
        cbsa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            cbsa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='latin-1')
        except UnicodeDecodeError:
            cbsa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='cp1252')
    
    cbsa_mapping['Zip Code_clean'] = cbsa_mapping['Zip Code'].astype(str).str.zfill(5)
    
    # Merge datasets
    merged_data = pd.merge(
        quote_data, 
        cbsa_mapping, 
        left_on='Zipcode_clean', 
        right_on='Zip Code_clean', 
        how='left'
    )
    
    # Load shapefiles
    print("Loading shapefiles...")
    zip_shapes = gpd.read_file('data/shapefiles/cb_2020_us_zcta520_500k.shp')
    zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)
    
    return merged_data, zip_shapes

def create_cbsa_map():
    """Create CBSA map by copying and modifying the CSA map"""
    
    # Load data
    merged_data, zip_shapes = load_and_prepare_data()
    
    # Filter for CBSAs
    cbsa_data = merged_data[merged_data['Primary CBSA Name'].notna()]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        cbsa_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    if map_data.empty:
        print("No data to display!")
        return None
    
    print(f"Displaying {len(map_data)} zip codes across {map_data['Primary CBSA Name'].nunique()} CBSAs")
    
    # Convert geometries to GeoJSON
    map_data_wgs84 = map_data.to_crs('EPSG:4326')
    
    # Generate colors for CBSAs
    unique_cbsas = sorted(map_data_wgs84['Primary CBSA Name'].unique())
    
    # Use matplotlib colormap to generate enough distinct colors
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    if len(unique_cbsas) <= 20:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                  'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
                  'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'yellow']
        cbsa_colors = dict(zip(unique_cbsas, colors[:len(unique_cbsas)]))
    else:
        # Generate colors using matplotlib colormap for many CBSAs
        cmap = plt.cm.get_cmap('tab20')
        colors = [mcolors.rgb2hex(cmap(i / len(unique_cbsas))) for i in range(len(unique_cbsas))]
        cbsa_colors = dict(zip(unique_cbsas, colors))
    
    # Prepare data for JavaScript
    zip_features = []
    for idx, row in map_data_wgs84.iterrows():
        # Calculate opacity based on quote volume within CBSA
        cbsa_zips = map_data_wgs84[map_data_wgs84['Primary CBSA Name'] == row['Primary CBSA Name']]
        max_quotes = cbsa_zips['Total Quotes'].max()
        opacity = 0.4 + 0.6 * (row['Total Quotes'] / max_quotes) if max_quotes > 0 else 0.4
        
        feature = {
            "type": "Feature",
            "properties": {
                "zipcode": row['Zipcode'],
                "zipcode_clean": row['Zipcode_clean'],
                "cbsa_name": row['Primary CBSA Name'],
                "city": row['City'],
                "state": row['State'],
                "total_quotes": float(row['Total Quotes']),
                "pickup_count": float(row['Pickup Count']),
                "dropoff_count": float(row['Dropoff Count']),
                "population": int(row['ZCTA Population (2020)']),
                "color": cbsa_colors[row['Primary CBSA Name']],
                "opacity": opacity,
                "assigned": True
            },
            "geometry": json.loads(gpd.GeoSeries([row['geometry']]).to_json())['features'][0]['geometry']
        }
        zip_features.append(feature)
    
    # Read the existing CSA map and modify it for CBSA
    try:
        with open('dynamic_csa_map.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace CSA references with CBSA
        html_content = html_content.replace('CSA Zip Code Analysis', 'CBSA Zip Code Analysis')
        html_content = html_content.replace('CSA zip codes', 'CBSA zip codes')
        html_content = html_content.replace('from CSA', 'from CBSA')
        html_content = html_content.replace('to CSA', 'to CBSA')
        html_content = html_content.replace('CSA:', 'CBSA:')
        html_content = html_content.replace('csa_name', 'cbsa_name')
        html_content = html_content.replace('All CSAs', f'All {len(unique_cbsas)} CBSAs')
        
        # Replace the data
        # Find the zipData assignment and replace it
        start_marker = 'var zipData = '
        end_marker = ';'
        
        start_idx = html_content.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = html_content.find('\n        \n        console.log', start_idx)
            if end_idx != -1:
                # Replace the data
                new_data = json.dumps(zip_features, indent=8)
                html_content = (html_content[:start_idx] + 
                              new_data + 
                              html_content[end_idx:])
        
        return html_content
        
    except FileNotFoundError:
        print("Could not find dynamic_csa_map.html file")
        return None

def main():
    """Main function"""
    try:
        # Create CBSA map
        html_content = create_cbsa_map()
        
        if html_content:
            # Save map
            map_file = 'dynamic_cbsa_map.html'
            with open(map_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\nDynamic CBSA map saved as '{map_file}'")
            print("Opening map in browser...")
            
            # Open in browser
            webbrowser.open('file://' + os.path.realpath(map_file))
            
            print("\n" + "="*60)
            print("DYNAMIC CBSA MAP FEATURES:")
            print("✅ Real-time color changes when unassigning zip codes")
            print("✅ Immediate visual feedback (gray + dashed border)")
            print("✅ Click any zip code to unassign from CBSA")
            print("✅ Export functionality for unassigned zip codes")
            print("✅ Reset button to restore all changes")
            print("✅ Live counter of unassigned zip codes")
            print("="*60)
            
        else:
            print("Failed to create CBSA map!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
