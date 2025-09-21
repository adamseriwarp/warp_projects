# Warp CBSA Expansion Analysis Report
## Strategic Market Expansion Through Data-Driven Geographic Analysis

**Prepared for:** Warp (wearewarp.com)  
**Analysis Period:** Q3 2025 (3-month quote data)  
**Report Date:** September 3, 2025  

---

## Executive Summary

This report presents a comprehensive analysis of Warp's current Core Based Statistical Area (CBSA) coverage and identifies strategic expansion opportunities based on quote demand, geographic proximity, and market performance metrics. Through advanced geospatial analysis of 182 recently unassigned zip codes and evaluation of 19,478 total zip codes across the top 75 CBSAs by population, we have developed a data-driven framework for optimizing Warp's service area expansion.

**Key Findings:**
- **182 zip codes** were strategically unassigned from CBSA coverage due to geographic isolation
- **Top 75 CBSAs** represent the most viable markets for expansion, covering 555,556 total quotes
- **9,005 non-CBSA zip codes** present potential expansion opportunities with varying proximity to current coverage
- **77 crossdock facilities** provide strategic infrastructure context for expansion planning

---

## Business Context & Methodology

### Current Service Model
Warp operates within a CBSA-based service model, focusing on contiguous geographic coverage within major metropolitan statistical areas. This approach ensures:
- **Operational efficiency** through concentrated service areas
- **Cost-effective logistics** by avoiding isolated service points
- **Scalable infrastructure** aligned with population density

### The Challenge: Geographic Optimization
While CBSAs provide logical market boundaries, not all zip codes within a CBSA are operationally viable to serve. Warp identified **182 zip codes** that, despite being officially assigned to CBSAs, were:
- **Geographically isolated** from main CBSA coverage areas
- **Operationally inefficient** to service due to distance
- **Strategically disconnected** from core business activity centers

### Analytical Approach
Our analysis employs a multi-layered methodology combining:

1. **Population-Based CBSA Selection**
   - Focused on top 75 CBSAs by population (unbiased by quote volume)
   - Ensures coverage of major metropolitan markets
   - Prevents exclusion of high-potential markets with limited current quotes

2. **Quote-Weighted Business Activity Centers**
   - Calculated weighted centroids for each CBSA based on quote volume
   - Identifies actual business activity concentration within CBSAs
   - Provides more accurate representation than geographic centers

3. **Adjacency-Based Distance Analysis**
   - Measures distance to nearest CBSA zip code (not centroid)
   - Focuses on market adjacency and expansion feasibility
   - Enables identification of natural expansion corridors

4. **Statistical Performance Benchmarking**
   - Compares non-CBSA zip codes to their nearest CBSA market performance
   - Uses standard deviation analysis for relative performance assessment
   - Identifies high-opportunity areas relative to local market conditions

---

## Data Sources & Infrastructure

### Quote Demand Data
- **Source:** 3-month historical quote data (Q3 2025)
- **Metrics:** Pickup counts, dropoff counts, total quote volume
- **Coverage:** 19,478 zip codes with quote activity
- **Range:** 1 - 202,975 quotes per zip code

### Geographic Data
- **CBSA Mapping:** Official zip code to CBSA assignments
- **Shapefiles:** US Census ZCTA (Zip Code Tabulation Areas)
- **Population Data:** 2020 Census population by zip code
- **Infrastructure:** 77 Warp crossdock facility locations

### Market Segmentation
- **CBSA-Assigned Zip Codes:** 10,473 zip codes across 75 major CBSAs
- **Non-CBSA Zip Codes:** 9,005 zip codes representing expansion opportunities
- **Recently Unassigned:** 182 zip codes removed from CBSA coverage for optimization

---

## Key Findings & Analysis

### 1. CBSA Market Performance
**Top-Performing CBSAs by Quote Volume:**
- Quote range: 3,096 - 555,556 total quotes per CBSA
- Average quotes per CBSA zip code: 1 - 202,975
- Geographic distribution spans all major US metropolitan areas

**Market Concentration Insights:**
- Business activity within CBSAs is not uniformly distributed
- Quote-weighted centroids often differ significantly from geographic centers
- High-density quote areas create natural expansion targets

### 2. Expansion Opportunity Assessment
**Non-CBSA Market Potential:**
- 9,005 zip codes outside current CBSA coverage
- Quote range: 1 - 6,408 quotes per zip code
- Distance range: 0.0 - 3,791.9 miles from nearest CBSA coverage

**Proximity Analysis:**
- **0-25 miles:** Immediate expansion candidates (adjacent markets)
- **25-50 miles:** Strategic expansion opportunities
- **50+ miles:** Long-term or specialized market considerations

### 3. Statistical Performance Benchmarking
**Relative Performance Metrics:**
- Standard deviation analysis compares non-CBSA zip codes to nearest CBSA averages
- Identifies zip codes performing at or above local market standards
- Enables prioritization based on market-relative opportunity

**Performance Categories:**
- **Above Average (≥0 std dev):** High-priority expansion targets
- **Near Average (-1 to 0 std dev):** Moderate expansion opportunities
- **Below Average (<-1 std dev):** Lower priority or specialized consideration

### 4. Infrastructure Context
**Crossdock Network Analysis:**
- 77 facilities provide strategic infrastructure backbone
- Geographic distribution aligns with major CBSA coverage
- Distance to nearest crossdock influences operational feasibility

---

## Strategic Recommendations

### Immediate Actions (0-3 months)
1. **High-Priority Expansion Targets**
   - Focus on non-CBSA zip codes within 25 miles of current coverage
   - Prioritize zip codes with ≥0 standard deviations above nearest CBSA average
   - Target areas with >100 quotes and <15 miles from existing coverage

2. **Recently Unassigned Zip Code Review**
   - Evaluate the 182 unassigned zip codes for potential re-inclusion
   - Focus on those with high quote volume and improved connectivity
   - Consider operational changes that might make service viable

### Medium-Term Strategy (3-12 months)
1. **Strategic Market Expansion**
   - Develop expansion plans for 25-50 mile radius opportunities
   - Prioritize markets with strong quote performance relative to local CBSAs
   - Consider infrastructure investments to support expanded coverage

2. **Market Corridor Development**
   - Identify natural expansion paths between existing CBSAs
   - Focus on contiguous coverage to maintain operational efficiency
   - Evaluate crossdock network expansion needs

### Long-Term Planning (12+ months)
1. **New Market Entry**
   - Evaluate CBSAs beyond the current top 75 for expansion potential
   - Consider markets with strong growth indicators
   - Assess infrastructure requirements for new market entry

2. **Service Model Optimization**
   - Continuously refine CBSA boundaries based on operational data
   - Implement dynamic service area adjustments
   - Develop predictive models for expansion opportunity identification

---

## Implementation Framework

### Data-Driven Decision Making
The interactive analysis tool provides real-time capability to:
- **Filter opportunities** by distance and performance thresholds
- **Export targeted datasets** for operational planning
- **Visualize expansion scenarios** with immediate feedback
- **Assess infrastructure gaps** relative to expansion plans

### Success Metrics
- **Quote volume capture** from newly serviced zip codes
- **Operational efficiency** maintenance during expansion
- **Market penetration** rates in expanded areas
- **Infrastructure utilization** optimization

### Risk Mitigation
- **Phased expansion** approach to minimize operational disruption
- **Performance monitoring** of newly added zip codes
- **Rollback procedures** for underperforming expansions
- **Continuous optimization** based on operational data

---

## Technical Methodology Details

### Geographic Analysis Techniques

**1. Quote-Weighted Centroid Calculation**
```
Weighted Latitude = Σ(zip_lat × zip_quotes) / Σ(zip_quotes)
Weighted Longitude = Σ(zip_lon × zip_quotes) / Σ(zip_quotes)
```
This approach ensures that CBSA "centers" reflect actual business activity rather than geographic midpoints.

**2. Haversine Distance Formula**
Accurate great-circle distances calculated using:
- Earth radius: 3,959 miles
- Accounts for Earth's curvature
- Provides precise mile-based measurements for strategic planning

**3. Statistical Performance Analysis**
```
Standard Deviation Score = (CBSA_Mean - Zip_Quotes) / CBSA_StdDev
Percentile Ranking = (Zips_Below_Target / Total_CBSA_Zips) × 100
```

### Data Quality & Validation

**Population Bias Correction:**
- CBSA selection based on total population, not quote volume
- Prevents exclusion of high-potential markets with limited current data
- Ensures comprehensive coverage of major metropolitan areas

**Geographic Accuracy:**
- US Census ZCTA shapefiles for precise boundaries
- Polygon simplification for performance optimization
- Centroid calculations using projected coordinate systems

**Quote Data Validation:**
- 3-month aggregation for seasonal stability
- Pickup and dropoff count verification
- Outlier detection and validation procedures

---

## Competitive Advantage & Market Intelligence

### Strategic Positioning
This analysis provides Warp with several competitive advantages:

1. **Data-Driven Expansion:** Unlike competitors who may expand based on intuition or basic demographics, Warp's approach uses actual demand data and sophisticated geographic analysis.

2. **Operational Efficiency Focus:** The adjacency-based distance analysis ensures expansion maintains operational viability, avoiding the trap of scattered, inefficient service areas.

3. **Market-Relative Performance:** Statistical benchmarking against local market conditions provides more nuanced opportunity identification than absolute metrics alone.

### Market Intelligence Insights

**Demand Pattern Analysis:**
- Quote distribution reveals actual market demand vs. theoretical market size
- Seasonal patterns can be incorporated for more accurate planning
- Geographic clustering identifies natural market boundaries

**Infrastructure Optimization:**
- Crossdock network analysis reveals coverage gaps and optimization opportunities
- Distance-to-infrastructure metrics inform expansion feasibility
- Strategic facility placement can unlock new market opportunities

**Competitive Landscape:**
- High-quote areas outside current coverage may indicate competitor strength
- Market performance relative to CBSA averages suggests competitive positioning
- Expansion timing can be optimized based on market conditions

---

## Risk Assessment & Mitigation

### Operational Risks

**1. Service Quality Dilution**
- **Risk:** Expanding too quickly may strain operational capacity
- **Mitigation:** Phased expansion with performance monitoring
- **Metrics:** Service level maintenance in existing areas

**2. Infrastructure Strain**
- **Risk:** Expansion beyond crossdock network capacity
- **Mitigation:** Infrastructure investment planning aligned with expansion
- **Metrics:** Crossdock utilization rates and service times

**3. Market Cannibalization**
- **Risk:** New zip codes may reduce demand in existing areas
- **Mitigation:** Market analysis to identify complementary vs. competitive areas
- **Metrics:** Total market growth vs. redistribution

### Strategic Risks

**1. Market Timing**
- **Risk:** Entering markets at suboptimal times
- **Mitigation:** Continuous market monitoring and flexible expansion plans
- **Metrics:** Market growth rates and competitive activity

**2. Resource Allocation**
- **Risk:** Misallocating expansion resources to lower-opportunity areas
- **Mitigation:** Data-driven prioritization and regular reassessment
- **Metrics:** ROI per expanded zip code and market penetration rates

**3. Competitive Response**
- **Risk:** Competitors may respond to Warp's expansion moves
- **Mitigation:** Strategic timing and differentiated service offerings
- **Metrics:** Market share changes and competitive pricing pressure

---

## Technology & Tools

### Interactive Analysis Platform
The developed analysis tool provides:

**Real-Time Filtering:**
- Distance thresholds (5-200 miles)
- Performance benchmarks (-3 to +2 standard deviations)
- Visual feedback with immediate map updates

**Export Capabilities:**
- 17-column comprehensive dataset
- Filtered results based on current analysis parameters
- Integration-ready format for operational systems

**Visual Intelligence:**
- Heat map visualization with logarithmic scaling
- Enhanced color sensitivity for opportunity identification
- Multi-layer display (CBSAs, crossdocks, centroids)

### Data Integration Framework
- **Automated data refresh** capabilities for ongoing analysis
- **API integration** potential for real-time quote data
- **Scalable architecture** for additional data sources
- **Export compatibility** with existing business systems

---

## Conclusion

This analysis provides Warp with a robust, data-driven framework for strategic market expansion. By combining quote demand data, geographic proximity analysis, and statistical performance benchmarking, we have identified clear opportunities for growth while maintaining operational efficiency.

The interactive analysis tool enables ongoing strategic planning and real-time decision making, ensuring that expansion efforts are both data-driven and operationally viable. With 9,005 potential expansion zip codes and a clear methodology for prioritization, Warp is well-positioned for strategic growth in high-opportunity markets.

**Immediate Action Items:**
1. **Week 1-2:** Export and review top 50 expansion targets (≤25 miles, ≥0 std dev)
2. **Week 3-4:** Operational feasibility assessment for immediate targets
3. **Month 1:** Pilot expansion with 10-15 highest-priority zip codes
4. **Month 2-3:** Performance monitoring and expansion optimization
5. **Quarterly:** Full analysis refresh with updated quote data

---

## Appendices

### Appendix A: Key Metrics Definitions

**Distance Metrics:**
- **Distance to Nearest CBSA:** Straight-line distance to closest zip code within any of the top 75 CBSAs
- **Distance to Crossdock:** Straight-line distance to nearest Warp facility

**Performance Metrics:**
- **Standard Deviation Score:** How many standard deviations a zip code's quotes are above/below the nearest CBSA's average
- **Quote Percentile:** Where a zip code would rank within its nearest CBSA (0-100%)
- **CBSA Mean Quotes:** Average quote volume across all zip codes in the nearest CBSA

**Geographic Classifications:**
- **CBSA-Assigned:** Zip codes currently within Warp's service CBSAs
- **Non-CBSA:** Zip codes outside current CBSA coverage (expansion opportunities)
- **Recently Unassigned:** 182 zip codes removed from CBSA coverage for optimization

### Appendix B: Analysis Tool Usage Guide

**Filter Configuration:**
1. **Distance Slider:** Adjust maximum distance from CBSA coverage (5-200 miles)
2. **Performance Filter:** Set minimum performance threshold (-3 to +2 std dev)
3. **Toggle Controls:** Show/hide CBSAs, crossdocks, centroids

**Export Process:**
1. Configure filters to desired expansion criteria
2. Click "Export Filtered Data" button
3. CSV file includes all currently visible zip codes
4. 17 columns of comprehensive market intelligence data

**Interpretation Guidelines:**
- **Blue areas:** Current CBSA coverage
- **Red heat map:** Non-CBSA opportunities (darker = higher quotes)
- **Red circles:** Quote-weighted CBSA business centers
- **Green diamonds:** Crossdock facility locations

### Appendix C: Sample Expansion Scenarios

**Scenario 1: Conservative Expansion**
- Distance: ≤25 miles
- Performance: ≥0 standard deviations
- Expected result: ~200-500 high-confidence zip codes

**Scenario 2: Moderate Expansion**
- Distance: ≤50 miles
- Performance: ≥-0.5 standard deviations
- Expected result: ~800-1,500 moderate-opportunity zip codes

**Scenario 3: Aggressive Expansion**
- Distance: ≤100 miles
- Performance: ≥-1.0 standard deviations
- Expected result: ~2,000-4,000 total expansion opportunities

### Appendix D: Data Sources & References

**Primary Data Sources:**
- Warp internal quote database (Q3 2025)
- US Census Bureau ZCTA shapefiles (2020)
- Office of Management and Budget CBSA definitions
- Warp crossdock facility database

**Analysis Tools:**
- Python geospatial analysis (GeoPandas, Shapely)
- Interactive mapping (Leaflet.js)
- Statistical analysis (NumPy, Pandas)
- Distance calculations (Haversine formula)

**Update Frequency:**
- Quote data: Real-time capability, recommended quarterly refresh
- Geographic boundaries: Annual review for CBSA definition changes
- Infrastructure data: Monthly updates for new crossdock facilities

---

*This comprehensive analysis report provides Warp with the strategic intelligence and operational framework necessary for data-driven market expansion. The combination of sophisticated geospatial analysis, statistical benchmarking, and interactive decision-making tools positions Warp for optimal growth in high-opportunity markets while maintaining operational excellence.*


