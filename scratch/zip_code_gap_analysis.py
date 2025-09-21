import pandas as pd
import numpy as np

def analyze_zip_code_gaps():
    """
    Analyze which current service zip codes don't appear in expansion candidates
    """
    print("🔍 Analyzing Zip Code Gaps: Current Service vs. Expansion Candidates")
    print("=" * 70)
    
    # Load both datasets
    print("📊 Loading datasets...")
    expansion_candidates = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/ltl_zipcode_candidates.csv')
    current_service = pd.read_csv('/Users/adamseri/Documents/augment-projects/quotes_analysis/data/raw/2025 Zip Look Up - Sheet1 (1).csv')
    
    # Clean zip codes
    expansion_candidates['Zipcode_clean'] = expansion_candidates['Zipcode'].astype(str).str.zfill(5)
    current_service['Zip_clean'] = current_service['Zip'].astype(str).str.zfill(5)
    
    # Remove invalid zip codes
    expansion_candidates = expansion_candidates[expansion_candidates['Zipcode_clean'].str.len() == 5]
    current_service = current_service[current_service['Zip_clean'].str.len() == 5]
    
    print(f"✅ Expansion candidates: {len(expansion_candidates):,} zip codes")
    print(f"✅ Current service areas: {len(current_service):,} zip codes")
    
    # Create sets for analysis
    expansion_zips = set(expansion_candidates['Zipcode_clean'])
    current_zips = set(current_service['Zip_clean'])
    
    # Find gaps
    current_not_in_expansion = current_zips - expansion_zips
    expansion_not_in_current = expansion_zips - current_zips
    overlap_zips = current_zips & expansion_zips
    
    print("\n📊 ZIP CODE OVERLAP ANALYSIS")
    print("=" * 50)
    print(f"🔄 Zip codes in BOTH datasets: {len(overlap_zips):,}")
    print(f"📍 Current service zip codes NOT in expansion: {len(current_not_in_expansion):,}")
    print(f"🎯 Expansion zip codes NOT in current service: {len(expansion_not_in_current):,}")
    
    # Calculate percentages
    current_coverage_pct = len(overlap_zips) / len(current_zips) * 100
    expansion_coverage_pct = len(overlap_zips) / len(expansion_zips) * 100
    
    print(f"\n📈 COVERAGE PERCENTAGES")
    print("-" * 30)
    print(f"% of current service zips in expansion: {current_coverage_pct:.1f}%")
    print(f"% of expansion zips in current service: {expansion_coverage_pct:.1f}%")
    
    # Analyze the gaps - current service zips not in expansion
    print(f"\n🔍 ANALYZING {len(current_not_in_expansion):,} CURRENT SERVICE ZIPS NOT IN EXPANSION")
    print("-" * 60)
    
    # Get details for current service zips not in expansion
    current_gaps = current_service[current_service['Zip_clean'].isin(current_not_in_expansion)]
    
    # Analyze by state
    if 'State' in current_gaps.columns:
        state_gaps = current_gaps['State'].value_counts().head(10)
        print("Top 10 states with current service zips not in expansion:")
        for state, count in state_gaps.items():
            print(f"  {state}: {count:,} zip codes")
    
    # Analyze by zone if available
    if 'Zone' in current_gaps.columns:
        zone_gaps = current_gaps['Zone'].value_counts().head(10)
        print(f"\nTop zones with current service zips not in expansion:")
        for zone, count in zone_gaps.items():
            if pd.notna(zone) and zone != '':
                print(f"  {zone}: {count:,} zip codes")
    
    # Sample of current service zips not in expansion
    print(f"\n📋 SAMPLE OF CURRENT SERVICE ZIPS NOT IN EXPANSION (first 20):")
    sample_gaps = current_gaps.head(20)
    for _, row in sample_gaps.iterrows():
        city = row.get('City', 'Unknown')
        state = row.get('State', 'Unknown')
        zip_code = row['Zip_clean']
        print(f"  {zip_code} - {city}, {state}")
    
    # Analyze the new expansion zips
    print(f"\n🎯 ANALYZING {len(expansion_not_in_current):,} NEW EXPANSION ZIPS NOT IN CURRENT SERVICE")
    print("-" * 60)
    
    new_expansion_zips = expansion_candidates[expansion_candidates['Zipcode_clean'].isin(expansion_not_in_current)]
    
    # Population analysis of new expansion zips
    new_expansion_population = new_expansion_zips['Population'].sum()
    print(f"Total population in new expansion zips: {new_expansion_population:,}")
    
    # CBSA vs Non-CBSA breakdown
    new_cbsa_count = len(new_expansion_zips[new_expansion_zips['Has_CBSA'] == True])
    new_non_cbsa_count = len(new_expansion_zips[new_expansion_zips['Has_CBSA'] == False])
    
    print(f"  • CBSA zip codes: {new_cbsa_count:,}")
    print(f"  • Non-CBSA zip codes: {new_non_cbsa_count:,}")
    
    # Top states for new expansion
    new_state_analysis = new_expansion_zips['State'].value_counts().head(10)
    print(f"\nTop states for new expansion zips:")
    for state, count in new_state_analysis.items():
        state_pop = new_expansion_zips[new_expansion_zips['State'] == state]['Population'].sum()
        print(f"  {state}: {count:,} zip codes, {state_pop:,} population")
    
    # Quote analysis for new expansion zips
    total_quotes_new = new_expansion_zips['Total_Quotes'].sum()
    print(f"\nTotal quotes in new expansion zips: {total_quotes_new:,}")
    
    # Sample of new expansion zips
    print(f"\n📋 SAMPLE OF NEW EXPANSION ZIPS (first 20):")
    sample_new = new_expansion_zips.head(20)
    for _, row in sample_new.iterrows():
        city = row.get('City', 'Unknown')
        state = row.get('State', 'Unknown')
        zip_code = row['Zipcode_clean']
        population = row.get('Population', 0)
        quotes = row.get('Total_Quotes', 0)
        cbsa_status = "CBSA" if row.get('Has_CBSA') else "Non-CBSA"
        print(f"  {zip_code} - {city}, {state} | Pop: {population:,} | Quotes: {quotes} | {cbsa_status}")
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"Current service zip codes: {len(current_zips):,}")
    print(f"Expansion candidate zip codes: {len(expansion_zips):,}")
    print(f"Overlap: {len(overlap_zips):,}")
    print(f"Current service zips NOT in expansion: {len(current_not_in_expansion):,}")
    print(f"New expansion zips: {len(expansion_not_in_current):,}")
    
    return {
        'current_total': len(current_zips),
        'expansion_total': len(expansion_zips),
        'overlap': len(overlap_zips),
        'current_not_in_expansion': len(current_not_in_expansion),
        'expansion_not_in_current': len(expansion_not_in_current),
        'current_coverage_pct': current_coverage_pct,
        'expansion_coverage_pct': expansion_coverage_pct
    }

if __name__ == "__main__":
    results = analyze_zip_code_gaps()
