# 311 Service Request Analysis

Mining municipal 311 data to uncover patterns, predict demand, and optimize city operations.

**Stack:** Python · pandas · scikit-learn · statsmodels · mlxtend · matplotlib/plotly

---

## What this does

Analyzes Winnipeg's 311 service requests (2008–2024) to extract actionable insights:

- **Frequent patterns** – recurring issue types and their geographic clustering
- **Seasonal trends** – yearly cycles in service demand (e.g., snow removal, waste collection)
- **Anomaly detection** – unusually long case durations or sudden request spikes

**Why it matters:** Data-driven triage → smarter staffing, preventive maintenance, faster response.

---

## Repository structure

```
311_Service_Request_Analysis/
├── Data/                    # Raw/cleaned datasets (large files git-ignored)
├── Documents/               # Scripts, notebooks, methodology
├── Final_Report/            # Executive summary, findings, recommendations
└── Visualizations/          # Charts, heatmaps, anomaly timelines
```

---

## Key methods

| Technique | Purpose | Tool |
|-----------|---------|------|
| **FP-Growth** | Identify co-occurring request types and locations | mlxtend / SPMF |
| **STL Decomposition** | Separate trend, seasonal, and residual signals | statsmodels |
| **Median Absolute Deviation (MAD)** | Detect unusually long service durations | NumPy / custom logic |
| **Visualization** | Generate heatmaps, time-series, and density plots | matplotlib / plotly |

---

### View outputs

- **Charts:** `Visualizations/` → heatmaps, seasonal plots, anomaly timelines
- **Report:** `Final_Report/` → PDF with full results and recommendations

---

## Methods (in detail)

### Frequent Pattern Mining

- Cleaned dataset to retain only *Service Requests* with valid `Neighborhood` entries
- Used columns: `Reason`, `Type`, `Neighborhood`, `Ward`
- Encoded as categorical IDs (e.g., `Water and Waste_R`, `Mynarski_W`)
- Applied **FP-Growth (SPMF)** with minimum support tuned to **0.007**, producing multi-attribute itemsets

**Purpose:** Identify frequently co-occurring issues, departments, and wards to uncover systemic service needs.

---

### Seasonal & Anomaly Detection

- Calculated **case duration** = `Closed Date` − `Open Date` (in hours)
- Aggregated by request type and date, applied **zero imputation** for missing days
- Decomposed each series via **STL (Seasonal-Trend Decomposition)**:
  ```
  x(t) = T(t) + S(t) + R(t)
  ```
- Applied **Median Absolute Deviation (MAD)** on residuals to flag anomalies where **Z > 3**
- Computed **anomaly rate** = anomalous days ÷ total days per request type

**Purpose:** Reveal inefficiencies or bottlenecks causing abnormal service delays.

---

## Key findings

### Frequent Patterns

- **Dominant Department:** *Water and Waste* — >1.6M service requests citywide
- **Top co-occurrences:**
  - `Water and Waste_R` × `Missed Garbage Collection_T`
  - `Water and Waste_R` × `Carts Damaged by Collection Crew_T`
  - `Community Services_R` × `Housing Complaint – Yard and Accessory Buildings_T`
- **Hotspot wards:** *Mynarski*, *Daniel McIntyre*, *Point Douglas*, *William Whyte*
- **Interpretation:** Water/Waste management and property maintenance dominate recurring citizen concerns

---

### Anomaly Detection

| Request Type | Anomaly Rate | Implication |
|--------------|--------------|-------------|
| Dutch Elm Disease Tree Inspection | 43% | Resource shortage / inspection delay |
| Ditch Overflowing (Priority 3) | 43% | Maintenance inefficiency |
| Park Maint – Boat Dock/Launch Hazardous | 42% | Safety-critical backlog |
| Watermain Cleaning Concerns | 41% | Operational delay |
| Sewer Charge Adjustment Requests | 39% | Administrative bottleneck |

**Takeaway:** High anomaly rates reveal systemic inefficiencies, not random outliers — especially in environmental and infrastructure-related services.

---

## For stakeholders

| Insight Type | Operational Use |
|--------------|-----------------|
| **Frequent itemsets** | Bundle related services to reduce truck rolls |
| **Seasonal patterns** | Pre-position crews ahead of predictable surges |
| **Anomaly alerts** | Investigate causes of abnormal duration spikes |

---

## Limitations & future work

- **Time decay model:** Current analysis treats all historical data equally; incorporating time decay would prioritize recent patterns
- **Data reliability:** Dataset includes requests resolved immediately alongside multi-year cases, making verification challenging
- **Documentation gaps:** Limited context for service request types hampers result interpretation

**Future work:** Integrate time-decay weighting, improve data validation, enhance service category documentation, and automate dashboard reporting.

---
