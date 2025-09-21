import pandas as pd
import numpy as np
from pathlib import Path

def calculate_quote_coverage():
    """Calculate what percentage of total quotes would be captured by servicing filtered zip codes"""
    
    print("📊 Calculating Quote Coverage Analysis...")
    print("=" * 60)
    
    # Define file paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    
    # File 1: Serviced CBSAs filtered zip codes
    filtered_file = BASE_DIR / 'serviced_cbsas_filtered_zip_codes_2025-09-05T03-36-22 (1).csv'
    
    # File 2: Quote data
    quote_file = DATA_DIR / 'raw' / 'quote_data.csv'
    
    print("📁 Loading files...")
    
    # Load the filtered zip codes file
    try:
        filtered_df = pd.read_csv(filtered_file)
        # Clean zip codes to 5-digit format
        filtered_df['Zipcode_clean'] = filtered_df['Zipcode'].astype(str).str.zfill(5)
        filtered_zips = set(filtered_df['Zipcode_clean'].unique())
        print(f"✅ Loaded {len(filtered_df)} zip codes from filtered file")
        print(f"   Unique zip codes: {len(filtered_zips)}")
        print(f"   Sample zip codes: {list(filtered_zips)[:5]}")
    except Exception as e:
        print(f"❌ Error loading filtered file: {e}")
        return
    
    # Load the quote data file
    try:
        quote_df = pd.read_csv(quote_file)
        print(f"✅ Loaded {len(quote_df)} rows from quote data file")
        print(f"   Columns: {quote_df.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error loading quote data file: {e}")
        return
    
    print("\n📊 Processing quote data...")
    
    # Clean and process quote data
    quote_df['Zipcode_clean'] = quote_df['Zipcode'].astype(str).str.zfill(5)
    
    # Convert pickup and dropoff counts to numeric, filling NaN with 0
    quote_df['Pickup Count'] = pd.to_numeric(quote_df['Pickup Count'], errors='coerce').fillna(0)
    quote_df['Dropoff Count'] = pd.to_numeric(quote_df['Dropoff Count'], errors='coerce').fillna(0)
    
    # Calculate total quotes per zip code
    quote_df['Total Quotes'] = quote_df['Pickup Count'] + quote_df['Dropoff Count']
    
    # Group by zip code to get total quotes per zip
    quote_summary = quote_df.groupby('Zipcode_clean').agg({
        'Pickup Count': 'sum',
        'Dropoff Count': 'sum',
        'Total Quotes': 'sum'
    }).reset_index()
    
    print(f"📍 Processed quote data:")
    print(f"   • Total rows in quote data: {len(quote_df):,}")
    print(f"   • Unique zip codes with quotes: {len(quote_summary):,}")
    print(f"   • Total pickup quotes: {quote_summary['Pickup Count'].sum():,}")
    print(f"   • Total dropoff quotes: {quote_summary['Dropoff Count'].sum():,}")
    print(f"   • Total quotes (pickup + dropoff): {quote_summary['Total Quotes'].sum():,}")
    
    print("\n🎯 Calculating coverage...")
    
    # Calculate total quotes across all zip codes
    total_quotes_all = quote_summary['Total Quotes'].sum()
    total_pickup_all = quote_summary['Pickup Count'].sum()
    total_dropoff_all = quote_summary['Dropoff Count'].sum()
    
    # Find quotes from zip codes that are in the filtered file
    covered_quotes = quote_summary[quote_summary['Zipcode_clean'].isin(filtered_zips)]
    
    if len(covered_quotes) > 0:
        total_quotes_covered = covered_quotes['Total Quotes'].sum()
        total_pickup_covered = covered_quotes['Pickup Count'].sum()
        total_dropoff_covered = covered_quotes['Dropoff Count'].sum()
        
        # Calculate percentages
        quote_coverage_pct = (total_quotes_covered / total_quotes_all) * 100 if total_quotes_all > 0 else 0
        pickup_coverage_pct = (total_pickup_covered / total_pickup_all) * 100 if total_pickup_all > 0 else 0
        dropoff_coverage_pct = (total_dropoff_covered / total_dropoff_all) * 100 if total_dropoff_all > 0 else 0
        
        print("=" * 60)
        print("📊 QUOTE COVERAGE ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"🌍 TOTAL MARKET (All Zip Codes with Quotes):")
        print(f"   • Zip codes: {len(quote_summary):,}")
        print(f"   • Total quotes: {total_quotes_all:,}")
        print(f"   • Pickup quotes: {total_pickup_all:,}")
        print(f"   • Dropoff quotes: {total_dropoff_all:,}")
        
        print(f"\n🎯 COVERED BY FILTERED ZIP CODES:")
        print(f"   • Zip codes covered: {len(covered_quotes):,}")
        print(f"   • Total quotes covered: {total_quotes_covered:,}")
        print(f"   • Pickup quotes covered: {total_pickup_covered:,}")
        print(f"   • Dropoff quotes covered: {total_dropoff_covered:,}")
        
        print(f"\n📈 COVERAGE PERCENTAGES:")
        print(f"   • Total quote coverage: {quote_coverage_pct:.2f}%")
        print(f"   • Pickup quote coverage: {pickup_coverage_pct:.2f}%")
        print(f"   • Dropoff quote coverage: {dropoff_coverage_pct:.2f}%")
        
        print(f"\n🔍 ANALYSIS:")
        zip_coverage_pct = (len(covered_quotes) / len(quote_summary)) * 100
        print(f"   • Zip code coverage: {zip_coverage_pct:.2f}%")
        print(f"   • Average quotes per covered zip: {total_quotes_covered / len(covered_quotes):.1f}")
        print(f"   • Average quotes per all zips: {total_quotes_all / len(quote_summary):.1f}")
        
        # Find top uncovered zip codes
        uncovered_quotes = quote_summary[~quote_summary['Zipcode_clean'].isin(filtered_zips)]
        if len(uncovered_quotes) > 0:
            top_uncovered = uncovered_quotes.nlargest(10, 'Total Quotes')
            print(f"\n❌ TOP 10 UNCOVERED ZIP CODES (Missing Opportunities):")
            for idx, row in top_uncovered.iterrows():
                print(f"   • {row['Zipcode_clean']}: {row['Total Quotes']:,} quotes ({row['Pickup Count']:,} pickup, {row['Dropoff Count']:,} dropoff)")
        
        # Save detailed results
        results_file = BASE_DIR / 'quote_coverage_analysis.csv'
        
        # Create summary results
        summary_data = {
            'Metric': [
                'Total Zip Codes with Quotes',
                'Covered Zip Codes',
                'Total Quotes (All)',
                'Covered Quotes',
                'Total Quote Coverage %',
                'Pickup Coverage %',
                'Dropoff Coverage %'
            ],
            'Value': [
                len(quote_summary),
                len(covered_quotes),
                total_quotes_all,
                total_quotes_covered,
                round(quote_coverage_pct, 2),
                round(pickup_coverage_pct, 2),
                round(dropoff_coverage_pct, 2)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")
        
        return {
            'total_quote_coverage': quote_coverage_pct,
            'pickup_coverage': pickup_coverage_pct,
            'dropoff_coverage': dropoff_coverage_pct,
            'covered_quotes': total_quotes_covered,
            'total_quotes': total_quotes_all
        }
    
    else:
        print("❌ No matching zip codes found between the two files!")
        return None

if __name__ == "__main__":
    results = calculate_quote_coverage()
