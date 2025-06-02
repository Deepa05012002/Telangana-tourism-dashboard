# Telangana Tourism Dashboard

A data visualization dashboard for analyzing tourism trends in Telangana state, India. This dashboard provides insights into domestic and foreign tourism patterns from 2014 to 2023.

## Features

- Monthly and yearly visitor analysis
- Comparison between domestic and foreign tourists
- District-wise visitor statistics
- Hyderabad-specific tourism data analysis
- Tourism forecast using SARIMA model 
- CAGR (Compound Annual Growth Rate) analysis
- Footfall ratio analysis based on population data
- Interactive visualizations using Plotly Dash

## Data Sources

The dashboard uses three primary datasets:
- Domestic visitor data by district and month (2014-2023)
- Foreign visitor data by district and month (2014-2023)
- Telangana district-wise population data

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/telangana-tourism-dashboard.git
cd telangana-tourism-dashboard
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Ensure the data files are in the project directory:
   - Telangana_Visitors_Domestic.xlsx
   - Telangana_Visitors_Foreign.xlsx
   - Telangana_Population_Districtwise.xlsx

2. Run the dashboard application:
```
python dashboard.py
```

3. Open your web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

## Dashboard Sections

1. **Monthly Visitors**: Analyze monthly tourism trends with options to view domestic, foreign, or combined data
2. **Yearly Visitors**: Compare yearly tourism statistics  
3. **Hyderabad Monthly**: Focus on tourism patterns in the capital city
4. **Forecast**: View tourism predictions until 2030 using time series forecasting
5. **CAGR Analysis**: Calculate compound annual growth rates for top districts
6. **Footfall Ratio**: Compare visitor numbers against district population
7. **Top 10 Districts**: View the districts with the highest visitor counts

## Requirements

- Python 3.7+
- Dash
- Plotly
- Pandas
- NumPy
- Statsmodels

## License

This project is licensed under the MIT License - see the LICENSE file for details.
