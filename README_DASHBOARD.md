# Aquaculture Survey Dashboard

## Overview
This package contains two interactive dashboard implementations for analyzing aquaculture survey data.

## Files Included
1. **dashboard.py** - Streamlit-based dashboard
2. **dashboard_dash.py** - Plotly Dash-based dashboard
3. **eda_script.py** - Complete EDA script
4. **survey-data.csv** - Original data
5. **survey-data-cleaned.csv** - Cleaned data

## Installation

### For Streamlit Dashboard
```bash
pip install streamlit plotly pandas numpy
```

### For Dash Dashboard
```bash
pip install dash plotly pandas numpy
```

## Running the Dashboards

### Streamlit Dashboard (Recommended for beginners)
```bash
streamlit run dashboard.py
```
- Opens automatically in browser at http://localhost:8501
- Very user-friendly interface
- Better for quick exploration

### Dash Dashboard (More customizable)
```bash
python dashboard_dash.py
```
- Opens at http://localhost:8050
- More control over styling
- Better for production deployment

## Dashboard Features

### Interactive Filters
- Duration range (months)
- Number of pens
- Farm area (hectares)

### Key Performance Indicators (KPIs)
- Total number of farms
- Average production (kg)
- Average production per hectare
- Average fish weight
- Average cycle duration

### Visualizations
1. **Production Distribution** - Histogram showing production/hectare distribution
2. **Production vs Duration** - Scatter plot with area sizing and weight coloring
3. **Density Analysis** - Stocking density vs production with trend line
4. **Pens Comparison** - Box plot showing production by number of pens
5. **Environmental Factors** - Vegetation coverage impact on production
6. **Distance Analysis** - Distance from source vs production
7. **Growth Rate** - Top performing farms by growth rate
8. **Weight Analysis** - Initial vs final weight comparison
9. **Correlation Heatmap** - Relationships between all variables

### Data Export
- Download filtered data as CSV
- View detailed statistics table
- Highlighted maximum values

## Dashboard Comparison

| Feature | Streamlit | Dash |
|---------|-----------|------|
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Customization | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Setup Time | Fast | Medium |
| Best For | Quick analysis | Production apps |

## Troubleshooting

### Port Already in Use
- Streamlit: Change port with `streamlit run dashboard.py --server.port 8502`
- Dash: Edit `app.run_server(port=8050)` in the code

### Missing Data Errors
- Ensure survey-data.csv is in the same directory
- Check file permissions

### Package Import Errors
- Reinstall packages: `pip install --upgrade streamlit plotly pandas`

## Support
For issues or questions, check the console output for error messages.
