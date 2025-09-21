import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import json
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="CSA Zip Code Analysis", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
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
    
    return merged_data, quote_data, csa_mapping

@st.cache_data
def load_shapefiles():
    # Load zip code shapefiles
    zip_shapes = gpd.read_file('data/shapefiles/cb_2020_us_zcta520_500k.shp')
    zip_shapes['ZCTA5CE20'] = zip_shapes['ZCTA5CE20'].astype(str)
    return zip_shapes

def create_interactive_map(data, zip_shapes, selected_csa=None):
    # Filter data for CSAs with quotes
    csa_data = data[data['Primary CSA Name'].notna() & (data['Total Quotes'] > 0)]
    
    if selected_csa:
        csa_data = csa_data[csa_data['Primary CSA Name'] == selected_csa]
    
    # Get top CSAs by quote volume for initial display
    top_csas = csa_data.groupby('Primary CSA Name')['Total Quotes'].sum().sort_values(ascending=False).head(10)
    
    if not selected_csa:
        csa_data = csa_data[csa_data['Primary CSA Name'].isin(top_csas.index)]
    
    # Merge with shapefiles
    map_data = zip_shapes.merge(
        csa_data, 
        left_on='ZCTA5CE20', 
        right_on='Zipcode_clean', 
        how='inner'
    )
    
    if map_data.empty:
        st.warning("No data to display for the selected filters.")
        return None
    
    # Create base map
    center_lat = map_data.geometry.centroid.y.mean()
    center_lon = map_data.geometry.centroid.x.mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Color palette for CSAs
    unique_csas = map_data['Primary CSA Name'].unique()
    colors = px.colors.qualitative.Set3[:len(unique_csas)]
    csa_colors = dict(zip(unique_csas, colors))
    
    # Add zip codes to map with CSA clustering
    for csa in unique_csas:
        csa_zips = map_data[map_data['Primary CSA Name'] == csa]
        
        # Create feature group for this CSA
        csa_group = folium.FeatureGroup(name=f"{csa} ({len(csa_zips)} zips)")
        
        for idx, row in csa_zips.iterrows():
            # Create popup with zip code info
            popup_html = f"""
            <div style="width: 200px;">
                <b>Zip Code:</b> {row['Zipcode']}<br>
                <b>CSA:</b> {row['Primary CSA Name']}<br>
                <b>City:</b> {row['City']}, {row['State']}<br>
                <b>Total Quotes:</b> {row['Total Quotes']:,.0f}<br>
                <b>Pickup:</b> {row['Pickup Count']:,.0f}<br>
                <b>Dropoff:</b> {row['Dropoff Count']:,.0f}<br>
                <b>Population:</b> {row['ZCTA Population (2020)']:,}<br>
                <hr>
                <small>Click to unassign from CSA</small>
            </div>
            """
            
            # Color intensity based on quote volume
            max_quotes = csa_zips['Total Quotes'].max()
            opacity = 0.3 + 0.7 * (row['Total Quotes'] / max_quotes) if max_quotes > 0 else 0.3
            
            folium.GeoJson(
                row['geometry'],
                style_function=lambda feature, color=csa_colors[csa], op=opacity: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': op,
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Zip: {row['Zipcode']} | Quotes: {row['Total Quotes']:,.0f}"
            ).add_to(csa_group)
        
        csa_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>CSA Zip Code Map</b></p>
    <p>• Zip codes colored by CSA</p>
    <p>• Opacity = Quote volume</p>
    <p>• Click zip to unassign from CSA</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def main():
    st.title("🗺️ CSA Zip Code Analysis - Interactive Map")
    st.markdown("**Analyze CSA assignments and identify expansion opportunities**")
    
    # Load data
    with st.spinner("Loading data..."):
        merged_data, quote_data, csa_mapping = load_data()
        zip_shapes = load_shapefiles()
    
    # Initialize session state for modified data
    if 'modified_data' not in st.session_state:
        st.session_state.modified_data = merged_data.copy()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # CSA selection
    csa_options = ['All CSAs'] + sorted(merged_data[merged_data['Primary CSA Name'].notna()]['Primary CSA Name'].unique())
    selected_csa = st.sidebar.selectbox("Select CSA to focus on:", csa_options)
    selected_csa = None if selected_csa == 'All CSAs' else selected_csa
    
    # Display summary statistics
    st.sidebar.subheader("📊 Summary")
    total_quotes = st.session_state.modified_data['Total Quotes'].sum()
    total_zips_with_csa = st.session_state.modified_data['Primary CSA Name'].notna().sum()
    total_csas = st.session_state.modified_data['Primary CSA Name'].nunique()
    
    st.sidebar.metric("Total Quotes", f"{total_quotes:,.0f}")
    st.sidebar.metric("Zip Codes with CSA", f"{total_zips_with_csa:,}")
    st.sidebar.metric("Unique CSAs", f"{total_csas}")
    
    # Export functionality
    if st.sidebar.button("📥 Export Modified Data"):
        export_data = st.session_state.modified_data[['Zip Code', 'Primary CSA Name']].copy()
        export_data = export_data.rename(columns={'Zip Code': 'Zipcode', 'Primary CSA Name': 'CSA_Assignment'})
        
        csv = export_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="modified_csa_assignments.csv",
            mime="text/csv"
        )
    
    # Reset data button
    if st.sidebar.button("🔄 Reset to Original Data"):
        st.session_state.modified_data = merged_data.copy()
        st.rerun()
    
    # Create and display map
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Interactive CSA Map")
        
        # Create map
        map_obj = create_interactive_map(st.session_state.modified_data, zip_shapes, selected_csa)
        
        if map_obj:
            # Display map with click handling
            map_data = st_folium(map_obj, width=800, height=600, returned_objects=["last_object_clicked"])
            
            # Handle zip code clicks for unassignment
            if map_data['last_object_clicked']:
                clicked_data = map_data['last_object_clicked']
                if clicked_data and 'tooltip' in clicked_data:
                    # Extract zip code from tooltip
                    tooltip = clicked_data['tooltip']
                    if 'Zip:' in tooltip:
                        zip_code = tooltip.split('Zip: ')[1].split(' |')[0]
                        
                        # Unassign zip code from CSA
                        mask = st.session_state.modified_data['Zipcode_clean'] == zip_code.zfill(5)
                        if mask.any():
                            st.session_state.modified_data.loc[mask, 'Primary CSA Name'] = None
                            st.success(f"Unassigned zip code {zip_code} from its CSA!")
                            st.rerun()
    
    with col2:
        st.subheader("📈 Top CSAs by Quotes")
        
        # Display top CSAs
        csa_summary = st.session_state.modified_data[
            st.session_state.modified_data['Primary CSA Name'].notna()
        ].groupby('Primary CSA Name').agg({
            'Total Quotes': 'sum',
            'Zipcode_clean': 'count'
        }).sort_values('Total Quotes', ascending=False).head(10)
        
        csa_summary.columns = ['Total Quotes', 'Zip Count']
        csa_summary['Total Quotes'] = csa_summary['Total Quotes'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(csa_summary, use_container_width=True)

if __name__ == "__main__":
    main()
