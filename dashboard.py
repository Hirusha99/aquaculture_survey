import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

# Page configuration
st.set_page_config(
    page_title="Aquaculture Survey Dashboard",
    page_icon="🐟",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('D:/Research/madhavii/implementation/survey data.csv')
    # Clean DENSITY column
    df['DENSITY 1 (total stock/Ha)'] = df['DENSITY 1 (total stock/Ha)'].replace('#REF!', np.nan)
    df['DENSITY 1 (total stock/Ha)'] = pd.to_numeric(df['DENSITY 1 (total stock/Ha)'])
    return df

df = load_data()

# Title and description
st.title("🐟 Aquaculture Production Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("📊 Filters")
duration_filter = st.sidebar.slider(
    "Duration (Months)",
    int(df['DURATIONMONTHS'].min()),
    int(df['DURATIONMONTHS'].max()),
    (int(df['DURATIONMONTHS'].min()), int(df['DURATIONMONTHS'].max()))
)

pens_filter = st.sidebar.multiselect(
    "Number of Pens",
    options=sorted(df['PENS'].unique()),
    default=sorted(df['PENS'].unique())
)

area_filter = st.sidebar.slider(
    "Area (Hectares)",
    float(df['Area (Ha 1)'].min()),
    float(df['Area (Ha 1)'].max()),
    (float(df['Area (Ha 1)'].min()), float(df['Area (Ha 1)'].max()))
)

# Apply filters
filtered_df = df[
    (df['DURATIONMONTHS'] >= duration_filter[0]) &
    (df['DURATIONMONTHS'] <= duration_filter[1]) &
    (df['PENS'].isin(pens_filter)) &
    (df['Area (Ha 1)'] >= area_filter[0]) &
    (df['Area (Ha 1)'] <= area_filter[1])
]

# Key Metrics
st.header("📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Farms", len(filtered_df))
with col2:
    st.metric("Avg Production", f"{filtered_df['PRODUCTION 1'].mean():.0f} kg")
with col3:
    st.metric("Avg Production/Ha", f"{filtered_df['PRODUCTION 1/Ha'].mean():.0f} kg/Ha")
with col4:
    st.metric("Avg Fish Weight", f"{filtered_df['AVG WEIGHT 1'].mean():.2f} kg")
with col5:
    st.metric("Avg Duration", f"{filtered_df['DURATIONMONTHS'].mean():.1f} months")

st.markdown("---")

# Row 1: Production Charts
st.header("🎯 Production Analysis")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        filtered_df,
        x='PRODUCTION 1/Ha',
        nbins=20,
        title="Distribution of Production per Hectare",
        labels={'PRODUCTION 1/Ha': 'Production (kg/Ha)', 'count': 'Number of Farms'},
        color_discrete_sequence=['#1f77b4']
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        filtered_df,
        x='DURATIONMONTHS',
        y='PRODUCTION 1',
        size='Area (Ha 1)',
        color='AVG WEIGHT 1',
        title="Production vs Duration (sized by Area)",
        labels={
            'DURATIONMONTHS': 'Duration (Months)',
            'PRODUCTION 1': 'Total Production (kg)',
            'AVG WEIGHT 1': 'Avg Weight (kg)'
        },
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Density and Environmental Factors
st.header("🌊 Environmental & Operational Factors")
col1, col2 = st.columns(2)

with col1:
    fig3 = px.scatter(
        filtered_df.dropna(subset=['DENSITY 1 (total stock/Ha)']),
        x='DENSITY 1 (total stock/Ha)',
        y='PRODUCTION 1/Ha',
        trendline="ols",
        title="Stocking Density vs Production per Hectare",
        labels={
            'DENSITY 1 (total stock/Ha)': 'Stocking Density (fish/Ha)',
            'PRODUCTION 1/Ha': 'Production (kg/Ha)'
        },
        color_discrete_sequence=['#2ca02c']
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    fig4 = px.box(
        filtered_df,
        x='PENS',
        y='PRODUCTION 1/Ha',
        title="Production per Hectare by Number of Pens",
        labels={
            'PENS': 'Number of Pens',
            'PRODUCTION 1/Ha': 'Production (kg/Ha)'
        },
        color='PENS',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig4, use_container_width=True)

# Row 3: Environmental Conditions
col1, col2 = st.columns(2)

with col1:
    fig5 = px.scatter(
        filtered_df,
        x='VEGETATION%',
        y='PRODUCTION 1/Ha',
        color='AVERAGE_DEPTH/ft',
        size='Area (Ha 1)',
        title="Vegetation Coverage vs Production (colored by Depth)",
        labels={
            'VEGETATION%': 'Vegetation Coverage (%)',
            'PRODUCTION 1/Ha': 'Production (kg/Ha)',
            'AVERAGE_DEPTH/ft': 'Depth (ft)'
        },
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    fig6 = px.scatter(
        filtered_df,
        x='DISTANCE/m',
        y='PRODUCTION 1/Ha',
        color='DURATIONMONTHS',
        title="Distance from Source vs Production",
        labels={
            'DISTANCE/m': 'Distance (meters)',
            'PRODUCTION 1/Ha': 'Production (kg/Ha)',
            'DURATIONMONTHS': 'Duration (months)'
        },
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig6, use_container_width=True)

# Row 4: Growth Analysis
st.header("📊 Growth & Harvest Analysis")
col1, col2 = st.columns(2)

with col1:
    # Calculate growth rate
    filtered_df['Growth Rate'] = (filtered_df['AVG WEIGHT 1'] - filtered_df['INITIAL WEIGHT/kg']) / filtered_df['DURATIONMONTHS']
    fig7 = px.bar(
        filtered_df.sort_values('Growth Rate', ascending=False).head(15),
        x='ID',
        y='Growth Rate',
        title="Top 15 Farms by Growth Rate (kg/month)",
        labels={'Growth Rate': 'Growth Rate (kg/month)', 'ID': 'Farm ID'},
        color='Growth Rate',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    fig8 = px.scatter(
        filtered_df,
        x='INITIAL WEIGHT/kg',
        y='AVG WEIGHT 1',
        size='DURATIONMONTHS',
        color='PRODUCTION 1/Ha',
        title="Initial vs Final Weight (sized by Duration)",
        labels={
            'INITIAL WEIGHT/kg': 'Initial Weight (kg)',
            'AVG WEIGHT 1': 'Final Weight (kg)',
            'PRODUCTION 1/Ha': 'Production/Ha'
        },
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig8, use_container_width=True)

# Correlation Heatmap
st.header("🔗 Correlation Analysis")
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
corr_matrix = filtered_df[numeric_cols].corr()

fig9 = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect='auto',
    title="Correlation Heatmap",
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1
)
fig9.update_layout(height=600)
st.plotly_chart(fig9, use_container_width=True)

# Data Table
st.header("📋 Filtered Data")
st.dataframe(
    filtered_df.style.highlight_max(axis=0, subset=['PRODUCTION 1', 'PRODUCTION 1/Ha', 'AVG WEIGHT 1']),
    use_container_width=True
)

# Download button
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_survey_data.csv',
    mime='text/csv'
)

# Summary Statistics
st.header("📊 Summary Statistics")
st.dataframe(filtered_df.describe().T, use_container_width=True)
