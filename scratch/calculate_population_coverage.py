import pandas as pd
import numpy as np
from pathlib import Path

def calculate_population_coverage():
    """Calculate population coverage for two sets of zip codes"""
    
    print("🔍 Calculating Population Coverage Analysis...")
    print("=" * 60)
    
    # Define file paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    
    # File 1: Serviced CBSAs filtered zip codes (has population data)
    filtered_file = BASE_DIR / 'serviced_cbsas_filtered_zip_codes_2025-09-05T03-36-22 (1).csv'
    
    # File 2: LTL Rate File (needs population lookup)
    ltl_file = DATA_DIR / 'raw' / 'LTL Rate File 07-07-2025 - Zips.csv'
    
    # Population lookup file
    cbsa_mapping_file = DATA_DIR / 'raw' / 'zip_to_csa_mapping.csv'
    
    print("📁 Loading files...")
    
    # Load the filtered zip codes file (already has population data)
    try:
        filtered_df = pd.read_csv(filtered_file)
        print(f"✅ Loaded {len(filtered_df)} zip codes from filtered file")
        print(f"   Sample zip codes: {filtered_df['Zipcode'].head().tolist()}")
    except Exception as e:
        print(f"❌ Error loading filtered file: {e}")
        return
    
    # Load the LTL Rate File
    try:
        ltl_df = pd.read_csv(ltl_file)
        ltl_df['Zip_clean'] = ltl_df['Zip'].astype(str).str.zfill(5)
        print(f"✅ Loaded {len(ltl_df)} zip codes from LTL Rate File")
        print(f"   Sample zip codes: {ltl_df['Zip_clean'].head().tolist()}")
    except Exception as e:
        print(f"❌ Error loading LTL Rate File: {e}")
        return
    
    # Load CBSA mapping for population data
    try:
        cbsa_mapping = pd.read_csv(cbsa_mapping_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            cbsa_mapping = pd.read_csv(cbsa_mapping_file, encoding='latin-1')
        except UnicodeDecodeError:
            cbsa_mapping = pd.read_csv(cbsa_mapping_file, encoding='cp1252')
    
    cbsa_mapping['Zip Code_clean'] = cbsa_mapping['Zip Code'].astype(str).str.zfill(5)
    print(f"✅ Loaded CBSA mapping with {len(cbsa_mapping)} zip codes for population lookup")
    
    # Calculate total USA population from CBSA mapping
    total_usa_population = cbsa_mapping['ZCTA Population (2020)'].sum()
    print(f"📊 Total USA Population (2020): {total_usa_population:,}")
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS 1: Serviced CBSAs Filtered Zip Codes")
    print("=" * 60)
    
    # Analysis 1: Filtered zip codes (already has population data)
    filtered_population = filtered_df['Population'].sum()
    filtered_coverage = (filtered_population / total_usa_population) * 100
    
    print(f"📍 Number of zip codes: {len(filtered_df):,}")
    print(f"👥 Total population covered: {filtered_population:,}")
    print(f"📈 USA population coverage: {filtered_coverage:.2f}%")
    
    # Show breakdown by CBSA status
    cbsa_zips = filtered_df[filtered_df['Has_CBSA'] == 'Yes']
    non_cbsa_zips = filtered_df[filtered_df['Has_CBSA'] == 'No']
    corridor_zips = filtered_df[filtered_df['Is_Corridor_Zip'] == 'Yes']
    
    print(f"\n📋 Breakdown:")
    print(f"   • CBSA zip codes: {len(cbsa_zips):,} ({cbsa_zips['Population'].sum():,} people)")
    print(f"   • Non-CBSA zip codes: {len(non_cbsa_zips):,} ({non_cbsa_zips['Population'].sum():,} people)")
    print(f"   • Corridor zip codes: {len(corridor_zips):,} ({corridor_zips['Population'].sum():,} people)")
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS 2: LTL Rate File Zip Codes")
    print("=" * 60)
    
    # Analysis 2: LTL Rate File zip codes (need to lookup population)
    ltl_with_population = pd.merge(
        ltl_df,
        cbsa_mapping[['Zip Code_clean', 'ZCTA Population (2020)']],
        left_on='Zip_clean',
        right_on='Zip Code_clean',
        how='left'
    )
    
    # Fill missing population with 0
    ltl_with_population['ZCTA Population (2020)'] = ltl_with_population['ZCTA Population (2020)'].fillna(0)
    
    ltl_population = ltl_with_population['ZCTA Population (2020)'].sum()
    ltl_coverage = (ltl_population / total_usa_population) * 100
    
    # Count zip codes with and without population data
    zips_with_pop = len(ltl_with_population[ltl_with_population['ZCTA Population (2020)'] > 0])
    zips_without_pop = len(ltl_with_population[ltl_with_population['ZCTA Population (2020)'] == 0])
    
    print(f"📍 Number of zip codes: {len(ltl_df):,}")
    print(f"👥 Total population covered: {ltl_population:,}")
    print(f"📈 USA population coverage: {ltl_coverage:.2f}%")
    print(f"\n📋 Data quality:")
    print(f"   • Zip codes with population data: {zips_with_pop:,}")
    print(f"   • Zip codes without population data: {zips_without_pop:,}")
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"🎯 Serviced CBSAs Filtered Zip Codes:")
    print(f"   • Zip codes: {len(filtered_df):,}")
    print(f"   • Population: {filtered_population:,}")
    print(f"   • USA coverage: {filtered_coverage:.2f}%")
    
    print(f"\n🎯 LTL Rate File Zip Codes:")
    print(f"   • Zip codes: {len(ltl_df):,}")
    print(f"   • Population: {ltl_population:,}")
    print(f"   • USA coverage: {ltl_coverage:.2f}%")
    
    difference = filtered_coverage - ltl_coverage
    print(f"\n📊 Difference:")
    print(f"   • Coverage difference: {difference:+.2f} percentage points")
    print(f"   • Population difference: {filtered_population - ltl_population:+,} people")
    
    if difference > 0:
        print(f"   ✅ Filtered zip codes cover {difference:.2f}% MORE of USA population")
    elif difference < 0:
        print(f"   ⚠️  LTL Rate File covers {abs(difference):.2f}% MORE of USA population")
    else:
        print(f"   ⚖️  Both files cover the same percentage of USA population")
    
    # Save detailed results
    results_file = BASE_DIR / 'population_coverage_analysis.csv'
    
    # Create summary results
    summary_data = {
        'Analysis': ['Serviced CBSAs Filtered', 'LTL Rate File'],
        'Zip_Code_Count': [len(filtered_df), len(ltl_df)],
        'Total_Population': [filtered_population, ltl_population],
        'USA_Coverage_Percent': [filtered_coverage, ltl_coverage]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    return {
        'filtered_coverage': filtered_coverage,
        'ltl_coverage': ltl_coverage,
        'filtered_population': filtered_population,
        'ltl_population': ltl_population,
        'total_usa_population': total_usa_population
    }

if __name__ == "__main__":
    results = calculate_population_coverage()
