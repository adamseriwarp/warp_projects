import pandas as pd
import json
import sys
from datetime import datetime

def load_original_data():
    """Load the original CSA mapping data"""
    try:
        csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='latin-1')
        except UnicodeDecodeError:
            csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='cp1252')
    
    return csa_mapping

def create_export_functionality(unassigned_zips_list):
    """Create updated dataset with unassigned zip codes"""
    
    # Load original data
    original_data = load_original_data()
    
    # Create a copy for modifications
    updated_data = original_data.copy()
    
    # Clean zip codes for matching
    updated_data['Zip Code_clean'] = updated_data['Zip Code'].astype(str).str.zfill(5)
    
    # Unassign specified zip codes
    unassigned_count = 0
    for zip_code in unassigned_zips_list:
        zip_clean = str(zip_code).zfill(5)
        mask = updated_data['Zip Code_clean'] == zip_clean
        if mask.any():
            # Store original CSA info before unassigning
            original_csa = updated_data.loc[mask, 'Primary CSA Name'].iloc[0]
            
            # Unassign from CSA
            updated_data.loc[mask, 'Primary CSA'] = None
            updated_data.loc[mask, 'Primary CSA Name'] = None
            
            unassigned_count += 1
            print(f"Unassigned zip {zip_code} from CSA: {original_csa}")
    
    # Remove the helper column
    updated_data = updated_data.drop('Zip Code_clean', axis=1)
    
    return updated_data, unassigned_count

def export_data(updated_data, unassigned_count):
    """Export the updated data to CSV"""
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export full updated dataset
    full_export_file = f'data/processed/updated_csa_mapping_{timestamp}.csv'
    updated_data.to_csv(full_export_file, index=False)
    
    # Create summary of changes
    summary_data = {
        'export_timestamp': timestamp,
        'total_zip_codes': len(updated_data),
        'zip_codes_with_csa': updated_data['Primary CSA Name'].notna().sum(),
        'zip_codes_without_csa': updated_data['Primary CSA Name'].isna().sum(),
        'unassigned_in_this_session': unassigned_count
    }
    
    # Export summary
    summary_file = f'data/processed/export_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Create a simple unassigned zips list
    unassigned_only = updated_data[updated_data['Primary CSA Name'].isna()][['Zip Code', 'City', 'State']]
    unassigned_file = f'data/processed/unassigned_zip_codes_{timestamp}.csv'
    unassigned_only.to_csv(unassigned_file, index=False)
    
    return full_export_file, summary_file, unassigned_file, summary_data

def main():
    """Main function for command line usage"""
    
    print("CSA Zip Code Export Tool")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # If zip codes provided as command line arguments
        unassigned_zips = sys.argv[1:]
        print(f"Processing {len(unassigned_zips)} zip codes from command line...")
    else:
        # Interactive mode
        print("Enter zip codes to unassign from CSAs (comma-separated):")
        user_input = input("Zip codes: ").strip()
        
        if not user_input:
            print("No zip codes provided. Exiting.")
            return
        
        unassigned_zips = [zip_code.strip() for zip_code in user_input.split(',')]
    
    print(f"\nProcessing {len(unassigned_zips)} zip codes...")
    
    try:
        # Create updated dataset
        updated_data, unassigned_count = create_export_functionality(unassigned_zips)
        
        # Export data
        full_file, summary_file, unassigned_file, summary = export_data(updated_data, unassigned_count)
        
        print(f"\n✅ Export completed successfully!")
        print(f"📁 Files created:")
        print(f"   • Full updated dataset: {full_file}")
        print(f"   • Unassigned zip codes only: {unassigned_file}")
        print(f"   • Export summary: {summary_file}")
        
        print(f"\n📊 Summary:")
        print(f"   • Total zip codes: {summary['total_zip_codes']:,}")
        print(f"   • Zip codes with CSA: {summary['zip_codes_with_csa']:,}")
        print(f"   • Zip codes without CSA: {summary['zip_codes_without_csa']:,}")
        print(f"   • Unassigned in this session: {summary['unassigned_in_this_session']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
