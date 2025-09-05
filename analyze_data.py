import pandas as pd
import numpy as np

# Load the data
print("Loading quote data...")
quote_data = pd.read_csv('data/raw/quote_data.csv')
print(f"Quote data shape: {quote_data.shape}")
print("Quote data columns:", quote_data.columns.tolist())
print("\nFirst few rows of quote data:")
print(quote_data.head())

# Check for missing values and data types
print("\nQuote data info:")
print(quote_data.info())

print("\n" + "="*50)

print("Loading CSA mapping data...")
# Try different encodings to handle special characters
try:
    csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='latin-1')
    except UnicodeDecodeError:
        csa_mapping = pd.read_csv('data/raw/zip_to_csa_mapping.csv', encoding='cp1252')
print(f"CSA mapping shape: {csa_mapping.shape}")
print("CSA mapping columns:", csa_mapping.columns.tolist())
print("\nFirst few rows of CSA mapping:")
print(csa_mapping.head())

print("\nCSA mapping info:")
print(csa_mapping.info())

print("\n" + "="*50)

# Analyze the quote data - combine pickup and dropoff counts
print("Analyzing quote data...")
quote_data['Pickup Count'] = pd.to_numeric(quote_data['Pickup Count'], errors='coerce').fillna(0)
quote_data['Dropoff Count'] = pd.to_numeric(quote_data['Dropoff Count'], errors='coerce').fillna(0)
quote_data['Total Quotes'] = quote_data['Pickup Count'] + quote_data['Dropoff Count']

print("Quote data with total quotes:")
print(quote_data[quote_data['Total Quotes'] > 0].head(10))

print(f"\nTotal zip codes with quotes: {len(quote_data[quote_data['Total Quotes'] > 0])}")
print(f"Total quotes across all zip codes: {quote_data['Total Quotes'].sum()}")

print("\n" + "="*50)

# Analyze CSA data
print("Analyzing CSA data...")
print(f"Unique CSAs: {csa_mapping['Primary CSA Name'].nunique()}")
print(f"Zip codes with CSA assignment: {csa_mapping['Primary CSA Name'].notna().sum()}")
print(f"Zip codes without CSA assignment: {csa_mapping['Primary CSA Name'].isna().sum()}")

print("\nTop 10 CSAs by zip code count:")
csa_counts = csa_mapping['Primary CSA Name'].value_counts().head(10)
print(csa_counts)

print("\n" + "="*50)

# Merge the datasets to see overlap
print("Merging datasets...")
# Clean zip codes for matching
quote_data['Zipcode_clean'] = quote_data['Zipcode'].astype(str).str.zfill(5)
csa_mapping['Zip Code_clean'] = csa_mapping['Zip Code'].astype(str).str.zfill(5)

merged_data = pd.merge(
    quote_data, 
    csa_mapping, 
    left_on='Zipcode_clean', 
    right_on='Zip Code_clean', 
    how='outer',
    indicator=True
)

print(f"Merged data shape: {merged_data.shape}")
print("\nMerge statistics:")
print(merged_data['_merge'].value_counts())

# Analyze zip codes with quotes that have CSA assignments
quotes_with_csa = merged_data[
    (merged_data['Total Quotes'] > 0) & 
    (merged_data['Primary CSA Name'].notna())
]

print(f"\nZip codes with quotes AND CSA assignment: {len(quotes_with_csa)}")
print(f"Total quotes in CSA-assigned zip codes: {quotes_with_csa['Total Quotes'].sum()}")

print("\nTop CSAs by total quotes:")
csa_quote_summary = quotes_with_csa.groupby('Primary CSA Name')['Total Quotes'].agg(['count', 'sum']).sort_values('sum', ascending=False)
print(csa_quote_summary.head(10))
