import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

# Load data
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

# Load quote data and mapping
quote_data = pd.read_csv(DATA_DIR / 'raw' / 'quote_data.csv')
zip_csa_mapping = pd.read_csv(DATA_DIR / 'raw' / 'zip_to_csa_mapping.csv', encoding='latin-1')

# Clean and prepare data
quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
quote_data['Total Quotes'] = quote_data['Pickup Count'] + quote_data['Dropoff Count']

zip_csa_mapping['Zipcode_clean'] = zip_csa_mapping['Zip Code'].astype(str).str.zfill(5)

# Merge data
merged_data = quote_data.merge(zip_csa_mapping, on='Zipcode_clean', how='left')

# Load shapefiles
zip_shapes = gpd.read_file(DATA_DIR / 'shapefiles' / 'cb_2020_us_zcta520_500k.shp')
zip_shapes = zip_shapes.to_crs('EPSG:4326')  # Use WGS84 for easier debugging

# Merge with shapes
all_data = zip_shapes.merge(merged_data, left_on='ZCTA5CE20', right_on='Zipcode_clean', how='left')
all_data['Total Quotes'] = all_data['Total Quotes'].fillna(0)

print("Debugging corridor calculation for zip 51247 to Omaha-Council Bluffs...")

# Find zip 51247
target_zip = all_data[all_data['ZCTA5CE20'] == '51247']
if len(target_zip) == 0:
    print("ERROR: Zip 51247 not found in shapefile data")
    exit()

target_zip = target_zip.iloc[0]
print(f"Target zip 51247: {target_zip['City']}, {target_zip['State']}")
print(f"Has CBSA: {pd.notna(target_zip['Primary CBSA Name'])}")

# Get target centroid
target_geom = target_zip['geometry']
if hasattr(target_geom, 'centroid'):
    target_centroid = target_geom.centroid
    target_lat, target_lon = target_centroid.y, target_centroid.x
    print(f"Target coordinates: {target_lat:.4f}, {target_lon:.4f}")
else:
    print("ERROR: Could not get target centroid")
    exit()

# Find Omaha-Council Bluffs CBSA zip codes
omaha_zips = all_data[all_data['Primary CBSA Name'] == 'Omaha-Council Bluffs, NE-IA']
print(f"Found {len(omaha_zips)} zip codes in Omaha-Council Bluffs CBSA")

if len(omaha_zips) == 0:
    print("ERROR: No Omaha-Council Bluffs zip codes found")
    exit()

# Calculate distance from 51247 to each Omaha zip
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    R = 3959  # Earth's radius in miles
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

distances = []
for idx, omaha_zip in omaha_zips.iterrows():
    omaha_geom = omaha_zip['geometry']
    if hasattr(omaha_geom, 'centroid'):
        omaha_centroid = omaha_geom.centroid
        omaha_lat, omaha_lon = omaha_centroid.y, omaha_centroid.x
        distance = calculate_distance(target_lat, target_lon, omaha_lat, omaha_lon)
        distances.append({
            'zip': omaha_zip['ZCTA5CE20'],
            'city': omaha_zip['City'],
            'distance': distance,
            'lat': omaha_lat,
            'lon': omaha_lon
        })

# Sort by distance and find nearest
distances.sort(key=lambda x: x['distance'])
nearest_omaha = distances[0]

print(f"\nNearest Omaha zip: {nearest_omaha['zip']} ({nearest_omaha['city']})")
print(f"Distance: {nearest_omaha['distance']:.1f} miles")
print(f"Nearest coordinates: {nearest_omaha['lat']:.4f}, {nearest_omaha['lon']:.4f}")

# Now test corridor calculation with different widths
corridor_widths = [10, 25, 50, 100]

for corridor_width in corridor_widths:
    print(f"\n--- Testing corridor width: {corridor_width} miles ---")
    
    # Calculate corridor
    corridor_count = 0
    sample_corridors = []
    
    for idx, zip_row in all_data.iterrows():
        if zip_row['ZCTA5CE20'] in ['51247', nearest_omaha['zip']]:
            continue  # Skip target and destination
            
        zip_geom = zip_row['geometry']
        if not hasattr(zip_geom, 'centroid'):
            continue
            
        zip_centroid = zip_geom.centroid
        zip_lat, zip_lon = zip_centroid.y, zip_centroid.x
        
        # Calculate distance from zip to line between target and nearest Omaha
        # Using point-to-line distance formula
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1
            
            dot = A * C + B * D
            lenSq = C * C + D * D
            
            if lenSq == 0:
                return calculate_distance(px, py, x1, y1)
            
            t = max(0, min(1, dot / lenSq))
            projX = x1 + t * C
            projY = y1 + t * D
            
            return calculate_distance(px, py, projX, projY)
        
        distance_to_line = point_to_line_distance(
            zip_lat, zip_lon,
            target_lat, target_lon,
            nearest_omaha['lat'], nearest_omaha['lon']
        )
        
        if distance_to_line <= corridor_width:
            # Check if zip is roughly on the path
            dist_to_target = calculate_distance(zip_lat, zip_lon, target_lat, target_lon)
            dist_to_omaha = calculate_distance(zip_lat, zip_lon, nearest_omaha['lat'], nearest_omaha['lon'])
            direct_distance = nearest_omaha['distance']
            
            if dist_to_target + dist_to_omaha <= direct_distance * 1.3:
                corridor_count += 1
                if len(sample_corridors) < 5:  # Keep first 5 for debugging
                    sample_corridors.append({
                        'zip': zip_row['ZCTA5CE20'],
                        'city': zip_row['City'],
                        'state': zip_row['State'],
                        'distance_to_line': distance_to_line,
                        'dist_to_target': dist_to_target,
                        'dist_to_omaha': dist_to_omaha
                    })
    
    print(f"Found {corridor_count} corridor zip codes")
    if sample_corridors:
        print("Sample corridor zip codes:")
        for corridor in sample_corridors:
            print(f"  {corridor['zip']} ({corridor['city']}, {corridor['state']}) - {corridor['distance_to_line']:.1f}mi from line")
    else:
        print("No corridor zip codes found!")

print(f"\nDirect distance from 51247 to nearest Omaha zip: {nearest_omaha['distance']:.1f} miles")
print("If no corridors are found, the issue might be:")
print("1. Corridor width too narrow")
print("2. Path constraint too strict (1.3x direct distance)")
print("3. Geographic alignment issues")
