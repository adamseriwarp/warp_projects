import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def export_full_dataset():
    """
    Export the complete dataset without browser limitations
    """
    print("🚀 Exporting Full Dataset Without Browser Limits")
    print("=" * 60)
    
    # Load the main analysis data (this should be the complete dataset)
    print("📊 Loading complete analysis data...")
    
    # You'll need to run the main analysis to get the full dataset
    # For now, let's work with what we have and see the scope
    
    try:
        # Try to load the existing export to see what we're working with
        existing_export = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/ltl_zipcode_candidates.csv')
        print(f"✅ Current export has: {len(existing_export):,} rows")
        
        # Load the original quote data to see full scope
        quote_data = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/data/raw/quote_data.csv')
        print(f"✅ Original quote data has: {len(quote_data):,} rows")
        
        # Load CBSA mapping with encoding handling
        try:
            cbsa_mapping = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/data/raw/zip_to_csa_mapping.csv', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                cbsa_mapping = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/data/raw/zip_to_csa_mapping.csv', encoding='latin-1')
            except UnicodeDecodeError:
                cbsa_mapping = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/data/raw/zip_to_csa_mapping.csv', encoding='cp1252')
        print(f"✅ CBSA mapping has: {len(cbsa_mapping):,} rows")
        
        # Check for potential data size issues
        print(f"\n📊 DATA SCOPE ANALYSIS")
        print("-" * 30)
        
        # Unique zip codes in each dataset
        export_zips = set(existing_export['Zipcode'].astype(str).str.zfill(5))
        quote_zips = set(quote_data['Zipcode'].astype(str).str.zfill(5))
        cbsa_zips = set(cbsa_mapping['Zip Code'].astype(str).str.zfill(5))
        
        print(f"Unique zip codes in export: {len(export_zips):,}")
        print(f"Unique zip codes in quote data: {len(quote_zips):,}")
        print(f"Unique zip codes in CBSA mapping: {len(cbsa_zips):,}")
        
        # Find what's missing
        quote_not_in_export = quote_zips - export_zips
        print(f"\n❗ Quote zip codes NOT in export: {len(quote_not_in_export):,}")
        
        if len(quote_not_in_export) > 0:
            print("🔍 This suggests export limitations!")
            
            # Sample of missing zip codes
            missing_sample = list(quote_not_in_export)[:20]
            print(f"\nSample of missing zip codes:")
            for zip_code in missing_sample:
                quote_info = quote_data[quote_data['Zipcode'].astype(str).str.zfill(5) == zip_code]
                if not quote_info.empty:
                    total_quotes = quote_info['Pickup Count'].fillna(0).sum() + quote_info['Dropoff Count'].fillna(0).sum()
                    city = quote_info['City'].iloc[0] if 'City' in quote_info.columns else 'Unknown'
                    state = quote_info['State'].iloc[0] if 'State' in quote_info.columns else 'Unknown'
                    print(f"  {zip_code} - {city}, {state} | {total_quotes} quotes")
        
        # Check if we're hitting the 10K limit
        if len(existing_export) > 10000 and len(existing_export) < 11000:
            print(f"\n🚨 LIKELY EXPORT LIMIT DETECTED!")
            print(f"Export has {len(existing_export):,} rows - suspiciously close to 10,000 limit")
        
        # Analyze what criteria might be filtering the data
        print(f"\n🔍 EXPORT FILTER ANALYSIS")
        print("-" * 30)
        
        # Check quote thresholds
        min_quotes = existing_export['Total_Quotes'].min()
        max_quotes = existing_export['Total_Quotes'].max()
        print(f"Quote range in export: {min_quotes} - {max_quotes}")
        
        # Check distance thresholds
        if 'Distance_to_CBSA_Miles' in existing_export.columns:
            max_distance = existing_export['Distance_to_CBSA_Miles'].max()
            print(f"Max distance in export: {max_distance:.1f} miles")
        
        # Check CBSA vs non-CBSA
        cbsa_count = len(existing_export[existing_export['Has_CBSA'] == True])
        non_cbsa_count = len(existing_export[existing_export['Has_CBSA'] == False])
        print(f"CBSA zips: {cbsa_count:,}, Non-CBSA zips: {non_cbsa_count:,}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 20)
        
        if len(quote_not_in_export) > 1000:
            print("1. ✅ You're likely hitting export limits")
            print("2. 🔧 Try exporting with more restrictive filters:")
            print("   - Smaller distance radius (25 miles)")
            print("   - Higher quote minimums")
            print("   - Single state at a time")
            print("3. 🚀 Use Python script to export full dataset")
            print("4. 📊 Export in chunks by state or region")
        
        return {
            'export_rows': len(existing_export),
            'quote_zips': len(quote_zips),
            'missing_zips': len(quote_not_in_export),
            'likely_limit': len(existing_export) > 10000 and len(existing_export) < 11000
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_chunked_export():
    """
    Create exports by state to avoid limits
    """
    print("\n🔄 Creating Chunked Export by State")
    print("-" * 40)
    
    try:
        existing_export = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/ltl_zipcode_candidates.csv')
        
        # Group by state and export separately
        states = existing_export['State'].unique()
        print(f"Found {len(states)} states in export")
        
        output_dir = Path('/Users/adamseri/Documents/augment-projects/quotes_analysis/exports_by_state')
        output_dir.mkdir(exist_ok=True)
        
        for state in sorted(states):
            state_data = existing_export[existing_export['State'] == state]
            filename = f"ltl_candidates_{state}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = output_dir / filename
            
            state_data.to_csv(filepath, index=False)
            print(f"  {state}: {len(state_data):,} zip codes → {filename}")
        
        print(f"\n✅ Exported {len(states)} state files to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error creating chunked export: {e}")

if __name__ == "__main__":
    results = export_full_dataset()
    if results and results['missing_zips'] > 1000:
        create_chunked_export()
