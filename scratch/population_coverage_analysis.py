import pandas as pd
import numpy as np

def analyze_population_coverage():
    """
    Compare population coverage between current service areas and expansion candidates
    """
    print("🔍 Analyzing Population Coverage: Current vs. Expansion Candidates")
    print("=" * 70)
    
    # Load the expansion candidates dataset
    print("📊 Loading expansion candidates dataset...")
    expansion_candidates = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/ltl_zipcode_candidates.csv')
    
    # Load the current service areas dataset
    print("📊 Loading current service areas dataset...")
    current_service = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/data/raw/2025 Zip Look Up - Sheet1 (1).csv')
    
    # Clean and prepare the data
    print("🧹 Cleaning and preparing data...")
    
    # Clean expansion candidates
    expansion_candidates['Zipcode_clean'] = expansion_candidates['Zipcode'].astype(str).str.zfill(5)
    expansion_candidates['Population'] = pd.to_numeric(expansion_candidates['Population'], errors='coerce').fillna(0)
    
    # Clean current service data
    current_service['Zip_clean'] = current_service['Zip'].astype(str).str.zfill(5)
    
    # Remove any invalid zip codes (less than 5 digits when cleaned)
    expansion_candidates = expansion_candidates[expansion_candidates['Zipcode_clean'].str.len() == 5]
    current_service = current_service[current_service['Zip_clean'].str.len() == 5]
    
    print(f"✅ Expansion candidates: {len(expansion_candidates):,} zip codes")
    print(f"✅ Current service areas: {len(current_service):,} zip codes")
    
    # Calculate population metrics for expansion candidates
    total_expansion_population = expansion_candidates['Population'].sum()
    cbsa_expansion_population = expansion_candidates[expansion_candidates['Has_CBSA'] == True]['Population'].sum()
    non_cbsa_expansion_population = expansion_candidates[expansion_candidates['Has_CBSA'] == False]['Population'].sum()
    corridor_expansion_population = expansion_candidates[expansion_candidates['Is_Corridor_Zip'] == True]['Population'].sum()
    
    print("\n📈 EXPANSION CANDIDATES ANALYSIS")
    print("-" * 40)
    print(f"Total Population in Expansion Dataset: {total_expansion_population:,}")
    print(f"  • CBSA zip codes: {cbsa_expansion_population:,} ({cbsa_expansion_population/total_expansion_population*100:.1f}%)")
    print(f"  • Non-CBSA zip codes: {non_cbsa_expansion_population:,} ({non_cbsa_expansion_population/total_expansion_population*100:.1f}%)")
    print(f"  • Corridor zip codes: {corridor_expansion_population:,} ({corridor_expansion_population/total_expansion_population*100:.1f}%)")
    
    # For current service areas, we need to get population data
    # We'll merge with the expansion dataset to get population info where available
    print("\n🔄 Matching current service areas with population data...")
    
    # Create a population lookup from expansion candidates
    population_lookup = expansion_candidates[['Zipcode_clean', 'Population']].drop_duplicates()
    
    # Merge current service with population data
    current_with_pop = current_service.merge(
        population_lookup, 
        left_on='Zip_clean', 
        right_on='Zipcode_clean', 
        how='left'
    )
    
    # Fill missing population with 0 (zip codes not in our expansion analysis)
    current_with_pop['Population'] = current_with_pop['Population'].fillna(0)
    
    # Calculate current service population
    current_service_population = current_with_pop['Population'].sum()
    current_service_with_data = current_with_pop[current_with_pop['Population'] > 0]
    
    print(f"✅ Current service zip codes with population data: {len(current_service_with_data):,}")
    print(f"✅ Current service zip codes without population data: {len(current_with_pop) - len(current_service_with_data):,}")
    
    # Find overlap between current service and expansion candidates
    overlap_zips = set(current_service['Zip_clean']) & set(expansion_candidates['Zipcode_clean'])
    overlap_population = expansion_candidates[expansion_candidates['Zipcode_clean'].isin(overlap_zips)]['Population'].sum()
    
    print(f"🔄 Overlapping zip codes: {len(overlap_zips):,}")
    print(f"🔄 Overlapping population: {overlap_population:,}")
    
    # Calculate net new population from expansion
    expansion_only_zips = set(expansion_candidates['Zipcode_clean']) - set(current_service['Zip_clean'])
    net_new_population = expansion_candidates[expansion_candidates['Zipcode_clean'].isin(expansion_only_zips)]['Population'].sum()
    
    print(f"🆕 New zip codes in expansion: {len(expansion_only_zips):,}")
    print(f"🆕 Net new population from expansion: {net_new_population:,}")
    
    # US total population (2020 Census: ~331.4 million)
    us_total_population = 331449281
    
    print("\n📊 POPULATION COVERAGE COMPARISON")
    print("=" * 50)
    print(f"🇺🇸 Total US Population (2020 Census): {us_total_population:,}")
    print()
    print(f"📍 Current Service Coverage:")
    print(f"   Population: {current_service_population:,}")
    print(f"   % of US Population: {current_service_population/us_total_population*100:.2f}%")
    print()
    print(f"🎯 Expansion Candidates Coverage:")
    print(f"   Population: {total_expansion_population:,}")
    print(f"   % of US Population: {total_expansion_population/us_total_population*100:.2f}%")
    print()
    print(f"🚀 Combined Coverage (Current + Net New):")
    combined_population = current_service_population + net_new_population
    print(f"   Population: {combined_population:,}")
    print(f"   % of US Population: {combined_population/us_total_population*100:.2f}%")
    print()
    print(f"📈 Population Increase from Expansion:")
    print(f"   Additional Population: {net_new_population:,}")
    if current_service_population > 0:
        increase_percentage = (net_new_population / current_service_population) * 100
        print(f"   % Increase: {increase_percentage:.1f}%")
    
    # Breakdown by zip code types in expansion
    print("\n🎨 EXPANSION BREAKDOWN BY TYPE")
    print("-" * 40)
    
    # CBSA vs Non-CBSA breakdown
    cbsa_count = len(expansion_candidates[expansion_candidates['Has_CBSA'] == True])
    non_cbsa_count = len(expansion_candidates[expansion_candidates['Has_CBSA'] == False])
    corridor_count = len(expansion_candidates[expansion_candidates['Is_Corridor_Zip'] == True])
    
    print(f"📊 CBSA Zip Codes: {cbsa_count:,} ({cbsa_count/len(expansion_candidates)*100:.1f}%)")
    print(f"   Population: {cbsa_expansion_population:,}")
    print(f"   Avg Population per Zip: {cbsa_expansion_population/cbsa_count:,.0f}")
    print()
    print(f"🔴 Non-CBSA Zip Codes: {non_cbsa_count:,} ({non_cbsa_count/len(expansion_candidates)*100:.1f}%)")
    print(f"   Population: {non_cbsa_expansion_population:,}")
    print(f"   Avg Population per Zip: {non_cbsa_expansion_population/non_cbsa_count:,.0f}")
    print()
    print(f"🛣️ Corridor Zip Codes: {corridor_count:,} ({corridor_count/len(expansion_candidates)*100:.1f}%)")
    print(f"   Population: {corridor_expansion_population:,}")
    print(f"   Avg Population per Zip: {corridor_expansion_population/corridor_count:,.0f}")
    
    # Top states by population in expansion
    print("\n🗺️ TOP STATES IN EXPANSION CANDIDATES")
    print("-" * 40)
    state_analysis = expansion_candidates.groupby('State').agg({
        'Population': 'sum',
        'Zipcode': 'count',
        'Total_Quotes': 'sum'
    }).sort_values('Population', ascending=False).head(10)
    
    for state, row in state_analysis.iterrows():
        print(f"{state}: {row['Population']:,} people, {row['Zipcode']:,} zips, {row['Total_Quotes']:,} quotes")
    
    return {
        'current_population': current_service_population,
        'expansion_population': total_expansion_population,
        'net_new_population': net_new_population,
        'combined_population': combined_population,
        'us_total_population': us_total_population,
        'current_coverage_pct': current_service_population/us_total_population*100,
        'expansion_coverage_pct': total_expansion_population/us_total_population*100,
        'combined_coverage_pct': combined_population/us_total_population*100
    }

if __name__ == "__main__":
    results = analyze_population_coverage()
