# 🚲 Copenhagenize Index 2025 Dashboard

A comprehensive interactive dashboard for analyzing bicycle-friendly cities based on the Copenhagenize Index 2025 dataset. Explore infrastructure investments, policy impacts, safety metrics, and modal share across the globe's top 100 cycling cities.

## Overview

The Copenhagenize Index measures how bicycle-friendly cities are across three core pillars:
- **Safe & Connected Infrastructure**: Physical cycling investments and design standards
- **Usage & Reach**: How much cycling is practiced in daily life
- **Policy & Support**: Governance, funding, and public perception driving change

This dashboard enables city planners, researchers, and stakeholders to explore these metrics interactively.

---

## Features

✅ **Regional Analytics** - Overview statistics and geographic visualization  
✅ **City Profiles** - Detailed performance analysis with PDF export  
✅ **Correlation Explorer** - Analyze policy impacts on cycling outcomes  
✅ **City Benchmarking** - Compare peer cities with intelligent matching  
✅ **Indicator Distribution** - Regional statistics and outlier analysis  
✅ **PDF Reports** - Download detailed city benchmark cards  
✅ **SVG Export** - Save all maps, graphs and charts as vector graphics  

---

## Installation

### Prerequisites
- Python 3.14+
- Virtual environment (`venv`)

### Setup Steps

1. **Clone/Open the Project**
   ```bash
   cd "C:\Users\lauri\OneDrive - TU Eindhoven\Q3\Internship\Copenhagenize_index_dashboard"
   ```

2. **Activate Virtual Environment**
   ```powershell
   & ".\.venv\Scripts\Activate.ps1"
   ```
   *(For Command Prompt, use: `.venv\Scripts\activate.bat`)*

3. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - streamlit
   - pandas
   - plotly
   - numpy
   - fpdf (for PDF generation)
   - geopy (for coordinates)

4. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

---

## Usage

### Tab 1: 📊 Regional Overview
- **Metrics**: Total cities, average scores, modal share, protected km
- **Bubble Chart**: Explore relationship between infrastructure density and ridership
- **Geographic Map**: View city locations and performance scores globally
- **Data Viewer**: Browse the complete filtered dataset

**Filter by Region**: Use the sidebar to select continents or view all regions.

### Tab 2: 🏙️ City Profile
- **Search**: Select any city from the dropdown (filtered by selected region)
- **Gauge Chart**: Visual display of overall Index Score (0-100)
- **3 Core Pillars**: Progress bars showing performance in each pillar
- **Diagnostic Indicators**: Top 3 strengths and top 3 areas for improvement
- **PDF Report**: Download a benchmark card with analysis and strategic recommendations

### Tab 3: 📈 Correlation Explorer
- **Impact Analysis**: Scatter plot showing relationships between interventions and outcomes
- **Correlation Insights**: Automatic interpretation of correlation strength
- **Heatmap**: Multi-metric correlation matrix to identify policy synergies

**Select metrics** to explore:
- Interventions: Infrastructure density, budget, traffic calming, etc.
- Outcomes: Modal share, cycling trips by women, safety rates, etc.

### Tab 4: ⚖️ City Benchmarking
- **Target City**: Select a city to benchmark
- **Smart Peer Matching**: Automatically suggests comparable cities in the same region
- **Radar Chart**: Multi-metric comparison visualization
- **Diagnostic Table**: Raw data → Normalized data → Final scores with comparison arrows

**Interpretation**:
- 🟢 Benchmark outperforms target city
- 🔴 Benchmark underperforms target city
- ⚪ No difference

### Tab 5: 📏 Indicator Metrics
- **Category Selection**: Choose from 11 indicator categories
- **Binary Metrics** (Yes/No policies):
  - Geographic heatmap showing adoption rates by city
- **Continuous Metrics** (km, %, rates):
  - Box plots with outlier detection
  - Statistical summaries by region

---

## Data Files

- **`master_copenhagenize_data.csv`**: Main dataset with all cities, scores, and indicators
- **`coordinates.py`**: Script to fetch and update latitude/longitude coordinates
- **`data_cleaning.py`**: Data preprocessing 

---

## Project Structure

```
Copenhagenize_index_dashboard/
│
├── app.py                              # Main Streamlit application
├── coordinates.py                      # Coordinate fetching utility
├── data_cleaning.py                    # Data cleaning functions
├── master_copenhagenize_data.csv       # Main dataset
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .venv/                              # Virtual environment (do not commit)
└── .vscode/                            # VS Code settings
```

---

## Key Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| **Index Score** | Overall bicycle-friendliness rating | 0-100 |
| **Protected_km** | Total kilometers of protected cycling infrastructure | km |
| **Infra_density** | Protected infrastructure per 100 km of roadway | % |
| **Modal_share_2024** | Percentage of trips by bicycle | % |
| **Safety_rate** | Cyclist fatalities per 100,000 population | rate/100K |
| **Spending_per_capita** | Annual cycling infrastructure spending | €/capita/year |
| **Traffic_30** | Percentage of roadway with speed limits ≤30 km/h | % |

---

## PDF Report Contents

Each exported PDF includes:
1. **City KPIs**: Rank, score, population
2. **3 Core Pillars**: Detailed breakdowns
3. **Diagnostics**: Top 3 strengths and weaknesses
4. **Strategic Leverage Points**: Policy recommendations

---

## Troubleshooting

### "Module 'fpdf' not found"
```bash
# Activate virtual environment and install
& ".\.venv\Scripts\Activate.ps1"
pip install fpdf
```

### Wrong Python environment in use
Ensure you activated your virtual environment:
```powershell
& ".\.venv\Scripts\Activate.ps1"
```
Not: `.venv\Scripts\activate` (this causes PowerShell module errors)

### Map shows "Data Unavailable"
Run `coordinates.py` to fetch/update latitude and longitude data:
```bash
python coordinates.py
```

### CSV file not found
Ensure `master_copenhagenize_data.csv` is in the same directory as `app.py`.

---

## Browser Compatibility

- Chrome/Chromium ✅
- Firefox ✅
- Safari ✅
- Edge ✅

Recommend full-screen or widescreen display (1920x1080+) for optimal dashboard experience.

---

## Notes

- **Deprecation Warnings**: The warnings about `use_container_width` are cosmetic and do not affect functionality. These will be updated in future versions.
- **SVG Export**: All Plotly charts can be exported as vector graphics using the camera icon in the toolbar
- **Data Updates**: When refreshing data, remember to update `master_copenhagenize_data.csv` and rerun `coordinates.py`

---

## Contact & Support

For questions about the Copenhagenize Index methodology, visit: https://copenhagenizeindex.eu

---

**Last Updated**: March 3, 2026  
**Dashboard Version**: 1.0  
**Python Version**: 3.14+
