# WARP CBSA Market Expansion Analysis

A comprehensive geospatial analysis tool for identifying strategic market expansion opportunities using Core Based Statistical Area (CBSA) data, quote demand patterns, and intelligent corridor optimization.

## 🎯 Project Overview

This project provides Warp with a data-driven framework for expanding logistics services beyond current CBSA coverage areas. By analyzing quote demand patterns in non-CBSA zip codes and implementing smart corridor analysis, the tool identifies optimal expansion routes that maximize market penetration while minimizing operational complexity.

### Key Features
- **Interactive CBSA Visualization**: Color-coded map showing all 75 top CBSAs by population
- **Quote Demand Analysis**: Statistical analysis of demand patterns relative to nearest CBSA markets
- **Smart Corridor Optimization**: Dynamic route planning for efficient market expansion
- **Real-time Filtering**: Configurable distance and performance thresholds
- **Export Capabilities**: Comprehensive CSV export for operational planning

## 🏗️ Technical Architecture

### Core Components
1. **`cbsa_color_coded_corridor_map.py`** - Main analysis engine and map generator
2. **Data Processing Pipeline** - Handles geographic data, quote analysis, and statistical calculations
3. **Interactive Web Interface** - HTML/JavaScript frontend with real-time filtering
4. **Export System** - CSV generation with comprehensive market analysis data

### Dependencies
```python
pandas>=1.3.0
numpy>=1.20.0
geopandas>=0.10.0
scikit-learn>=1.0.0
```

## 📊 Analysis Logic & Methodology

### 1. CBSA Population-Based Prioritization
- Identifies top 75 CBSAs by population (unbiased by current quote volume)
- Ensures coverage of major metropolitan areas regardless of temporary demand fluctuations
- Prevents exclusion of high-potential markets due to current low activity

### 2. Quote Demand Analysis
```python
# Calculate total quote volume per zip code
total_quotes = pickup_count + dropoff_count

# Statistical comparison against nearest CBSA
std_dev_from_mean = (zip_quotes - cbsa_mean) / cbsa_std
```

### 3. Proximity-Based Market Assessment
- Calculates distances from non-CBSA zip codes to nearest CBSA coverage
- Applies configurable distance thresholds (5-200 miles) for expansion feasibility
- Integrates performance metrics relative to nearest CBSA characteristics

### 4. Smart Corridor Analysis
**Innovation**: Dynamic corridor identification algorithm featuring:
- **Distance Limits**: Prevents overextension (50-300 mile maximum)
- **Dynamic Width**: Shorter corridors = wider inclusion, longer corridors = narrower focus
- **Intermediate Inclusion**: Identifies zip codes along optimal routes to high-demand targets
- **Performance Filtering**: Ensures corridor efficiency through intelligent selection

#### Corridor Calculation Logic
```python
# Dynamic width based on corridor distance
distance_factor = corridor_distance / max_distance
dynamic_width = base_width * (1.5 - distance_factor)

# Include intermediate zip codes within corridor
if point_to_line_distance <= dynamic_width:
    if total_detour_distance <= direct_distance * detour_multiplier:
        include_in_corridor = True
```

### 5. Color-Coded Visualization System
- **CBSA Zip Codes**: Unique color per CBSA (hash-based generation, avoiding red)
- **Non-CBSA Zip Codes**: Red highlighting for immediate recognition
- **Corridor Zones**: Orange overlay for expansion route visualization
- **Infrastructure**: Crossdock locations and CBSA centroids marked

## 🎛️ Interactive Controls

### Distance Filtering
- **Range**: 5-200 miles from nearest CBSA coverage
- **Purpose**: Define expansion feasibility based on operational constraints

### Performance Filtering
- **Metric**: Standard deviations from nearest CBSA mean quote volume
- **Range**: -3 to +2 standard deviations
- **Logic**: Higher values = better performance relative to CBSA benchmark

### Corridor Parameters
- **Max Distance**: 50-300 miles (prevents overextension)
- **Base Width**: 10-50 miles (corridor inclusion radius)
- **Dynamic Adjustment**: Real-time recalculation based on parameter changes

## 📈 Export Dataset Column Descriptions

### Geographic Identifiers
| Column | Description |
|--------|-------------|
| `Zipcode` | 5-digit ZIP code (zero-padded) |
| `City` | Primary city name for the ZIP code |
| `State` | Two-letter state abbreviation |
| `Population` | 2020 Census population for the ZIP code area |

### Quote Demand Metrics
| Column | Description |
|--------|-------------|
| `Total_Quotes` | Combined pickup and dropoff quote volume |
| `Pickup_Count` | Number of pickup quotes originated from this ZIP |
| `Dropoff_Count` | Number of delivery quotes destined for this ZIP |

### CBSA Assignment & Analysis
| Column | Description |
|--------|-------------|
| `Has_CBSA` | Boolean: Whether ZIP is assigned to a CBSA |
| `Assigned_CBSA` | Name of assigned CBSA (if applicable) |
| `Closest_CBSA` | Name of nearest CBSA for non-CBSA ZIPs |
| `Distance_to_CBSA_Miles` | Distance to nearest CBSA coverage (miles) |
| `Nearest_CBSA_Zip` | Specific ZIP code of nearest CBSA coverage |

### Performance Analytics
| Column | Description |
|--------|-------------|
| `Std_Dev_vs_CBSA_Mean` | Standard deviations from nearest CBSA mean quote volume<br>• Positive = Above CBSA average<br>• Negative = Below CBSA average |
| `Quote_Percentile_in_Nearest_CBSA` | Percentile ranking within nearest CBSA market |
| `CBSA_Mean_Quotes` | Average quote volume for the nearest CBSA |

### Corridor & Infrastructure Analysis
| Column | Description |
|--------|-------------|
| `Is_Corridor_Zip` | Boolean: Whether ZIP is part of smart corridor analysis<br>• `true` = Included in expansion corridor<br>• `false` = Not part of corridor route |
| `Nearest_Crossdock_Name` | Name of closest Warp crossdock facility |
| `Distance_to_Nearest_Crossdock_Miles` | Distance to nearest crossdock (miles) |

## 🚀 Usage Instructions

### 1. Data Preparation
Ensure the following files are in the `data/raw/` directory:
- `quote_data.csv` - Historical quote volume by ZIP code
- `zip_to_csa_mapping.csv` - ZIP to CBSA mapping data
- `WARP_xdock_2025.csv` - Crossdock location data

### 2. Run Analysis
```bash
python3 cbsa_color_coded_corridor_map.py
```

### 3. Interactive Analysis
- Open the generated HTML file in your browser
- Adjust filters using the control panel
- Enable corridor analysis for expansion planning
- Export filtered results for operational use

### 4. Export Data
- Configure desired filters and corridor parameters
- Click "Export Filtered Zip Codes" button
- CSV file includes all analysis metrics and corridor status

## 📋 Key Insights & Applications

### Market Opportunity Identification
- **9,005 non-CBSA ZIP codes** identified as potential expansion targets
- **Distance-based prioritization** for phased rollout planning
- **Performance benchmarking** against established CBSA markets

### Strategic Expansion Planning
- **Corridor optimization** reduces per-stop expansion costs
- **Infrastructure integration** with existing crossdock network
- **Risk assessment** through statistical performance analysis

### Operational Benefits
- **Route efficiency** through intermediate ZIP inclusion
- **Market penetration** along expansion corridors
- **Competitive advantage** in underserved markets

## 🔄 Future Enhancements

- **Automated data refresh** capabilities
- **Predictive demand modeling** for market forecasting
- **Cost-benefit analysis** integration
- **Multi-modal transportation** optimization

---

*This analysis framework represents a significant advancement in logistics market analysis, combining traditional geographic methods with modern data science techniques to drive strategic business growth.*
