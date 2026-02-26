import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import cast, Literal

# 1. Page Configuration
st.set_page_config(page_title="Copenhagenize Index 2025 Dashboard", page_icon="🚲", layout="wide")

# 2. Data Loading
@st.cache_data
def load_data():
    # Read dataset
    df = pd.read_csv("master_copenhagenize_data.csv")
    return df

df = load_data()

# --- INDICATOR DICTIONARY ---
# Grouping the raw columns into your requested categories
indicator_categories = {
    "Infrastructure": ["Protected_km", "Infra_density (km of bicycle infra/100 km of roadway)"],
    "Parking": ["Public_spaces", "Enclosed_spaces"],
    "Traffic Calming": ["Street_km_total", "Street_Km_30", "Traffic_30 (% of km of roadway)"],
    "Modal Share": ["Bike_trips_women_%", "Modal_share_2024_% \n(or nearest post-Covid)", "Modal_share_2019_% \n(or nearest pre-Covid)", "Modal_delta (percentage points)"],
    "Safety": ["Cyclist_deaths", "Safety_rate (rate/100K pop)"],
    "Image of the Bicycle": ["Media_tone_70%_positive", "Bicycle_brand_network_identity", "School_cycling_education"],
    "Cargo Bikes": ["Household_purchase_subsidy_yes_no", "Logistics_business_subsidy_yes_no", "Cargo_commercial_adoption_yes_no", "Supportive_infra_standards_yes_no"],
    "Advocacy": ["NGO_exists_yes_no", "NGO_events_yes_no", "NGO_policy_yes_no"],
    "Political Commitment": ["Bicycle_budget_5yr", "Spending_per_capita (€/capita/year)"],
    "Bike Share": ["Bike_share_fleet", "Bike_share_trips", "PT_integration_yes_no", "Bike_share_cov_density (bikes/1K pop)", "Bike_share_usage (trips/bike/day)"],
    "Urban Planning": ["3yr_new_lanes_km", "Cycling_masterplan_yes_no", "Cycling_unit_yes_no", "Cycling_standards_yes_no", "Cycling_monitoring_yes_no", "Infra_increase (km of bicycle infra/100 km of roadway)"]
}

# 3. Sidebar Filtering
st.sidebar.title("🚲 Copenhagenize Index 2025")
st.sidebar.markdown("**Dashboard Analytics**")
st.sidebar.divider()

# Dynamic list of regions for the dropdown
regions = ["All Regions"] + sorted(df['Continent'].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("🌍 Select Region", regions)

# Filter the data based on the sidebar selection
if selected_region != "All Regions":
    df_filtered = df[df['Continent'] == selected_region]
else:
    df_filtered = df.copy()

st.sidebar.divider()
st.sidebar.info("Navigate tabs to explore rankings, correlations, and comparisons.")

# 4. Main App Layout
st.title("🚲 Copenhagenize Index 2025")
st.markdown("Explore infrastructure, policies, and cycling usage across the globe's top 100 cities.")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Regional Overview", 
    "🏙️ City Profile",      
    "📈 Correlation Explorer", 
    "⚖️ City Comparison", 
    "📏 Indicator Metrics"
    ])

with tab1:
    st.subheader(f"State of Cycling: {selected_region}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cities in View", len(df_filtered))
    col2.metric("Avg Overall Score", round(df_filtered['Index Score'].mean(), 1))
    
    modal_share_col = None
    try:
        modal_share_col = [c for c in df.columns if 'Modal_share_2024' in c][0]
        col3.metric("Avg Modal Share (%)", round(df_filtered[modal_share_col].mean(), 1))
    except Exception:
        col3.metric("Avg Modal Share (%)", "Data Unavailable")
    
    col4.metric("Avg Protected Km", round(df_filtered['Protected_km'].mean(), 1))
    
    st.markdown("### 📈 Infrastructure vs. Usage (Bubble Chart)")
    if modal_share_col:
        fig = px.scatter(
            df_filtered, 
            x='Infra_density (km of bicycle infra/100 km of roadway)', 
            y=modal_share_col,
            size='Population',
            color='Continent',
            hover_name='City',
            hover_data=['Country', 'Index Score', 'Rank'],
            title="Does infrastructure density increase ridership?",
            labels={
                'Infra_density (km of bicycle infra/100 km of roadway)': 'Infra Density (km per 100km roadway)',
                modal_share_col: 'Modal Share (%)'
            },
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # ---> MAP SECTION <---
    st.markdown("### 🗺️ Geographic Viewer")
    if 'Lat' in df_filtered.columns and 'Lon' in df_filtered.columns:
        fig_map = px.scatter_geo(
            df_filtered,
            lat='Lat',
            lon='Lon',
            color='Index Score', # Color the dots by how high their score is
            size='Population',   # Size the dots by population
            hover_name='City',
            hover_data=['Country', 'Rank', 'Index Score'],
            projection="natural earth", # Gives curved globe effect
            color_continuous_scale="Viridis",
            title="City Locations & Performance"
        )
        
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            geo=dict(
                showland=True, landcolor="lightgray",
                showcoastlines=True, coastlinecolor="white",
                showcountries=True, countrycolor="white"
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Map data is updating. Please ensure you ran the coordinate fetching script.")
    
    st.markdown("---")
    st.markdown("### Data Viewer")
    st.dataframe(df_filtered, use_container_width=True)
    
# --- TAB 2: CITY PROFILE ---

with tab2:
    st.subheader("🏙️ City Report")
    st.markdown("Summary of city's performance, highlighting strengths and critical areas for improvement.")
    
    # 1. City Selector (Filtered by the sidebar region)
    cities_in_region = sorted(df_filtered['City'].unique().tolist())
    selected_city = st.selectbox("Select a City to view its profile:", cities_in_region)
    
    if selected_city:
        # Extract the exact row of data for this city
        city_data = df_filtered[df_filtered['City'] == selected_city].iloc[0]
        
        st.markdown("---")
        
        # 2. KPIs & Gauge Chart
        col_gauge, col_stats = st.columns([1, 1.5])
        
        with col_gauge:
            # Build Plotly Gauge Chart for the Overall Score
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = city_data['Index Score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"<b>{selected_city}</b><br><span style='font-size:0.8em;color:gray'>Overall Index Score</span>"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1f77b4"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#ffcccb'},   # Red tint for poor
                        {'range': [40, 70], 'color': '#ffffcc'},  # Yellow tint for average
                        {'range': [70, 100], 'color': '#ccffcc'}  # Green tint for excellent
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_stats:
            st.markdown(f"### Global Rank: **#{city_data['Rank']}**")
            st.markdown(f"**Country:** {city_data['Country']} | **Population:** {city_data['Population']:,}")
            
            st.markdown("#### The 3 Core Pillars")

            # Use Streamlit's native progress bars
            st.write("**Safe & Connected Infrastructure**")
            st.progress(int(city_data['Safe and Connected Infrastructure']), text=f"{city_data['Safe and Connected Infrastructure']:.1f} / 100")
            
            st.write("**Usage & Reach**")
            st.progress(int(city_data['Usage and Reach']), text=f"{city_data['Usage and Reach']:.1f} / 100")
            
            st.write("**Policy & Support**")
            st.progress(int(city_data['Policy and Support']), text=f"{city_data['Policy and Support']:.1f} / 100")
            
        st.markdown("---")
        
        # 3. Strengths & Weaknesses
        st.markdown("### 🔍 Indicator Diagnostics")
        st.markdown("Based on the 13 sub-indicators, here is where this city excels and where it falls behind.")
        
        # 13 score columns
        score_cols = [c for c in df.columns if 'Score ' in c and c not in ['Index Score', 'Score per Pillar']]
        
        # Create a dictionary of the city's scores
        city_scores = {c.replace('Score ', ''): city_data[c] for c in score_cols if not pd.isna(city_data[c])}
        
        # Sort them from highest to lowest
        sorted_scores = sorted(city_scores.items(), key=lambda x: x[1], reverse=True)
        
        col_strength, col_weak = st.columns(2)
        
        with col_strength:
            st.success("#### 🌟 Top 3 Strengths")
            # Get the top 3
            for metric, score in sorted_scores[:3]:
                st.markdown(f"**{metric}:** {score:.1f} / 100")
                
        with col_weak:
            st.error("#### ⚠️ Top 3 Areas for Improvement")
            # Get the bottom 3
            for metric, score in reversed(sorted_scores[-3:]):
                st.markdown(f"**{metric}:** {score:.1f} / 100")

# --- TAB 3: CORRELATION EXPLORER ---
with tab3:
    st.subheader("📈 Correlation & Policy Impact Explorer")
    st.markdown("Select a City Input (X) and observe its historical relationship with a City Outcome (Y).")
    
    # 1. Categorize variables (Interventions vs Outcomes)
    interventions = [
        'Infra_density (km of bicycle infra/100 km of roadway)',
        'Protected_km',
        'Spending_per_capita (€/capita/year)',
        'Bicycle_budget_5yr',
        'Traffic_30 (% of km of roadway)',
        'Street_Km_30',
        'Parking_density (stands/1K pop)',
        'Score Political Commitment',
        'Score Urban Planning'
    ]
    
    outcomes = [
        'Modal_share_2024_% \n(or nearest post-Covid)',
        'Bike_trips_women_%',
        'Cyclist_deaths',
        'Safety_rate (rate/100K pop)',
        'Modal_delta (percentage points)',
        'Index Score'
    ]
    
    # Ensure columns exist in df
    interventions = [c for c in interventions if c in df.columns]
    outcomes = [c for c in outcomes if c in df.columns]
    
    st.markdown("### 1. Impact (Scatter Plot)")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("Select Intervention (X-Axis)", interventions)
    with col_y:
        y_axis = st.selectbox("Select Desired Outcome (Y-Axis)", outcomes)
        
    # Scatter Plot
    try:
        fig_corr = px.scatter(
            df_filtered, x=x_axis, y=y_axis, color='Continent', hover_name='City',
            trendline='ols', title=f"Impact of {x_axis.split('(')[0].strip()} on {y_axis.split('(')[0].strip()}",
            template='plotly_white'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # --- Analysis of correlations ---
        # Calculate Pearson correlation coefficient dropping empty rows
        clean_df = df_filtered[[x_axis, y_axis]].dropna()
        if len(clean_df) > 5:
            corr_val = clean_df[x_axis].corr(clean_df[y_axis])
            
            # Smart text generation based on the correlation value
            st.markdown("#### 💡 Insight:")
            
            # Handle reverse logic for "Deaths" (where Negative is Good)
            is_negative_good = "death" in y_axis.lower() or "safety_rate" in y_axis.lower()
            
            if corr_val > 0.6:
                if is_negative_good:
                    st.error(f"**Warning (Correlation: {corr_val:.2f}):** There is a strong, alarming relationship here. As the intervention increases, fatalities/danger actually increase. This requires immediate auditing.")
                else:
                    st.success(f"**Strong Positive Impact (Correlation: {corr_val:.2f}):** The data strongly supports this intervention. Cities that invest heavily in this metric see a reliable, significant increase in the desired outcome.")
            elif 0.3 < corr_val <= 0.6:
                if is_negative_good:
                    st.warning(f"**Concerning Trend (Correlation: {corr_val:.2f}):** There is a moderate relationship showing danger increasing with this metric. Proceed with caution.")
                else:
                    st.info(f"**Moderate Positive Impact (Correlation: {corr_val:.2f}):** This intervention contributes to the outcome. But it must be paired with other policies to guarantee results.")
            elif -0.3 <= corr_val <= 0.3:
                st.markdown(f"**No Significant Relationship (Correlation: {corr_val:.2f}):** The data shows a random scattering. Changing this input alone does not reliably impacts the chosen outcome.")
            elif -0.6 <= corr_val < -0.3:
                if is_negative_good:
                    st.info(f"**Moderate Safety Benefit (Correlation: {corr_val:.2f}):** There is a moderate relationship showing that this intervention helps reduce fatalities/danger.")
                else:
                    st.warning(f"**Negative Impact (Correlation: {corr_val:.2f}):** Surprisingly, as this intervention increases, the outcome tends to drop. Further urban context is required to understand why.")
            else: # < -0.6
                if is_negative_good:
                    st.success(f"**Strong Safety Benefit (Correlation: {corr_val:.2f}):** Excellent policy indicator. Cities that invest in this intervention see a dramatic drop in fatalities and danger.")
                else:
                    st.error(f"**Strong Negative Impact (Correlation: {corr_val:.2f}):** There is a stark inverse relationship. This intervention is heavily correlated with a decline in the desired outcome.")
        else:
            st.caption("Not enough data points to calculate a reliable insight.")
            
    except Exception as e:
        st.warning("Could not calculate trendline or insights due to insufficient data variance.")

    st.markdown("---")
    
    # --- Part 2: INTERACTIVE HEATMAP ---
    st.markdown("### 2. Global Correlation Matrix (Heatmap)")
    st.markdown("A macro-view of how metrics move together. Values closer to **1** (dark red) mean strong positive correlation. Values closer to **-1** (dark blue) mean strong negative correlation.")
    
    col_method, col_vars = st.columns([1, 2])
    
    with col_method:
        corr_method = st.radio("Select Correlation Method:", ["pearson", "spearman"], 
            help="Pearson measures linear relationships. Spearman measures monotonic (ranked) relationships.")
        
    with col_vars:
        # Pre-select highly relevant metrics for a planner's baseline view
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        default_heatmap_cols = ['Index Score', 'Infra_density (km of bicycle infra/100 km of roadway)', 'Modal_share_2024_% \n(or nearest post-Covid)', 'Cyclist_deaths', 'Spending_per_capita (€/capita/year)']
        
        selected_heatmap_cols = st.multiselect(
            "Select Metrics for the Heatmap:",
            options=[c for c in numeric_cols if c not in ['Rank', 'Population']],
            default=[c for c in default_heatmap_cols if c in df.columns]
        )
        
    if len(selected_heatmap_cols) > 1:
        corr_matrix = df_filtered[selected_heatmap_cols].corr(method=corr_method)
        fig_heat = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template='plotly_white'
        )
        fig_heat.update_layout(height=600, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")
    else:
        st.warning("Please select at least 2 metrics to generate the correlation heatmap.")

# --- TAB 4: CITY COMPARISON ---
with tab4:
    st.subheader("⚖️ Advanced Benchmarking")
    st.markdown("Compare up to 5 cities or regional averages across all 13 Index indicators simultaneously.")
    
    cities_list = sorted(df['City'].unique().tolist())
    regions_list = sorted(df['Continent'].dropna().unique().tolist())
    
    average_options = [f"Average: {r}" for r in regions_list] + ["Average: Global Top 10", "Average: Global Top 30"]
    all_options = cities_list + average_options
    
    selected_targets = st.multiselect(
        "Select up to 5 targets to compare:",
        options=all_options,
        default=["Copenhagen", "Bogotá", "Average: Global Top 10"],
        max_selections=5
    )

    if len(selected_targets) == 0:
        st.warning("Please select at least one city or benchmark to compare.")
    else:
        entity_data = {}
        for target in selected_targets:
            if "Average:" in target:
                if "Top 10" in target:
                    entity_data[target] = df.nsmallest(10, 'Rank').mean(numeric_only=True)
                elif "Top 30" in target:
                    entity_data[target] = df.nsmallest(30, 'Rank').mean(numeric_only=True)
                else:
                    region_name = target.split("Average: ")[1]
                    entity_data[target] = df[df['Continent'] == region_name].mean(numeric_only=True)
            else:
                entity_data[target] = df[df['City'] == target].iloc[0]

        st.markdown("---")
        st.markdown("### 🎯 Indicator Radar")
        
        score_cols = [c for c in df.columns if 'Score ' in c and c not in ['Index Score', 'Score per Pillar']]
        radar_labels = [c.replace('Score ', '') for c in score_cols]
        
        # ---> To close the circular loop
        closed_theta = radar_labels + [radar_labels[0]]
        
        fig_radar = go.Figure()
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        
        for idx, target in enumerate(selected_targets):
            
            # ---> Extract the data values, then append the first value to the end
            r_vals = [entity_data[target].get(c, 0) for c in score_cols]
            closed_r = r_vals + [r_vals[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=closed_r,          # Use the closed data list
                theta=closed_theta,  # Use the closed label list
                fill='toself' if len(selected_targets) <= 2 else 'none',
                name=target,
                line_color=colors[idx]
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=650,
            template='plotly_white',
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

       
        st.markdown("### 📊 Diagnostic table")
        st.markdown("**Raw Data** (actual counts) ➡️ **Correlated Data** (normalized per capita/roadway) ➡️ **Final Score** (0-100).")
        
        #  Define the pipeline for the major indicators that have all 3 tiers
        diagnostic_pipeline = {
            "Bicycle Infrastructure": {
                "Raw": "Protected_km",
                "Correlated": "Infra_density (km of bicycle infra/100 km of roadway)",
                "Score": "Score Bicycle Infrastructure"
            },
            "Political Commitment": {
                "Raw": "Bicycle_budget_5yr",
                "Correlated": "Spending_per_capita (€/capita/year)",
                "Score": "Score Political Commitment"
            },
            "Safety": {
                "Raw": "Cyclist_deaths",
                "Correlated": "Safety_rate (rate/100K pop)",
                "Score": "Score Safety" # Note: In safety, lower raw/correlated is better!
            },
            "Traffic Calming": {
                "Raw": "Street_Km_30",
                "Correlated": "Traffic_30 (% of km of roadway)",
                "Score": "Traffic Calming" 
            },
            "Bike Share": {
                "Raw": "Bike_share_fleet",
                "Correlated": "Bike_share_cov_density (bikes/1K pop)",
                "Score": "Score Bike Share"
            }
        }

        # Build the dynamic dataframe
        matrix_rows = []
        
        for category, metrics in diagnostic_pipeline.items():
            for data_type, col_name in metrics.items():
                if col_name in df.columns:
                    # ---> Clean up the metric name for the display table
                    display_metric_name = col_name.replace('_', ' ')
                    
                    # ---> Added "Metric" to the row structure
                    row_data = {
                        "Indicator": category, 
                        "Data Tier": data_type,
                        "Metric": display_metric_name
                    }
                    
                    # Get the baseline value (the first target selected)
                    baseline_val = entity_data[selected_targets[0]].get(col_name, np.nan)
                    
                    for idx, target in enumerate(selected_targets):
                        val = entity_data[target].get(col_name, np.nan)
                        
                        # Formatting logic
                        if pd.isna(val):
                            display_val = "N/A"
                        else:
                            # Add comparison arrows if there are exactly 2 targets
                            if len(selected_targets) == 2 and idx == 1 and not pd.isna(baseline_val):
                                # Determine if higher is better (Safety is reversed)
                                higher_is_better = False if category == "Safety" and data_type != "Score" else True
                                
                                if val > baseline_val:
                                    arrow = "🟢" if not higher_is_better else "🔴" 
                                    display_val = f"{val:,.2f} {arrow}"
                                elif val < baseline_val:
                                    arrow = "🔴" if not higher_is_better else "🟢"
                                    display_val = f"{val:,.2f} {arrow}"
                                else:
                                    display_val = f"{val:,.2f} ⚪"
                            else:
                                display_val = f"{val:,.2f}"
                                
                        row_data[target] = display_val
                    
                    matrix_rows.append(row_data)

        # Convert to DataFrame
        matrix_df = pd.DataFrame(matrix_rows)
        
        # Table for Streamlit
        st.dataframe(
            matrix_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Indicator": st.column_config.TextColumn("Category", width="medium"),
                "Data Tier": st.column_config.TextColumn("Data Tier", width="small"),
                "Metric": st.column_config.TextColumn("Exact Metric", width="large"), 
            }
        )
        
        st.caption("🟢 Indicates the benchmark is outperforming the target city. 🔴 Indicates the benchmark is underperforming. (Note: For Safety Raw/Correlated metrics, lower numbers are better).")


# --- TAB 5: INDICATOR METRICS (MIN/MAX/AVG/MEDIAN) ---
with tab5:
    st.subheader("📏 Indicator Metrics & Distributions")
    st.markdown("Select a category below to see regional distributions, minimums, maximums, and averages.")
    
    # Category Selector
    selected_category = st.selectbox("📂 Select Indicator Category:", list(indicator_categories.keys()))
    
    metrics_to_show = indicator_categories[selected_category]
    
    # Loop through the specific metrics for that category
    for metric in metrics_to_show:
        if metric in df_filtered.columns:
            st.markdown(f"### 🔹 {metric.replace('_', ' ')}")
            
            # Check if the metric is Binary (Yes/No, 0/1) or Continuous (Km, %, etc.)
            unique_vals = df[metric].dropna().unique()
            is_binary = set(unique_vals).issubset({0, 1})
            
            if is_binary:
                # For binary/Yes-No policies, we show a percentage Bar Chart
                st.markdown("*Percentage of cities that have implemented this policy.*")
                agg_df = df_filtered.groupby('Continent')[metric].mean().reset_index()
                agg_df[metric] = agg_df[metric] * 100 # Convert to percentage
                
                fig_dist = px.bar(
                    agg_df, x='Continent', y=metric, color='Continent',
                    labels={metric: '% of Cities (Yes)'},
                    template='plotly_white'
                )
                fig_dist.update_yaxes(range=[0, 100])
                
                # KEY TO PREVENT DUPLICATE PLOTY CHART
                st.plotly_chart(fig_dist, use_container_width=True, key=f"bar_{metric}")
                
            else:
                # For continuous numbers, we use Box Plot
                fig_dist = px.box(
                    df_filtered, x='Continent', y=metric, color='Continent', 
                    points="all", # Shows the individual cities as dots next to the box
                    hover_name="City",
                    template='plotly_white'
                )
                
                # KEY TO PREVENT DUPLICATE PLOTY CHART
                st.plotly_chart(fig_dist, use_container_width=True, key=f"box_{metric}")
            
            # Generate the pandas describe() summary table
            st.markdown(f"**Statistical Summary (Grouped by Region):**")
            
            # Calculate the describe stats and format 
            summary_stats = df_filtered.groupby('Continent')[metric].describe()
            
            # Rename columns to be more readable 
            summary_stats = summary_stats.rename(columns={
                'count': 'Data Points',
                'mean': 'Average',
                'std': 'Std Dev',
                'min': 'Minimum',
                '25%': '25th Pct',
                '50%': 'Median',
                '75%': '75th Pct',
                'max': 'Maximum'
            })
            
            # Format Data Points remain whole numbers
            def formatter(val, col):
                if col == 'Data Points':
                    return f"{val:.0f}"
                else:
                    return f"{val:.2f}"

            st.dataframe(
                summary_stats.style.format({col: (lambda x: f"{x:.0f}" if col == 'Data Points' else f"{x:.2f}") for col in summary_stats.columns}),
                use_container_width=True
            )
            st.markdown("---")