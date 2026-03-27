import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import cast, Literal
from fpdf import FPDF
import base64
import scipy.stats as stats
import tempfile
import os

# 1. Page Configuration
st.set_page_config(page_title="Copenhagenize Index 2025 Dashboard", page_icon="🚲", layout="wide")
# --- CIZE FORMAT ---
st.markdown("""
    <style>
    /* Apply TT Norms font (will use system fallbacks if not locally installed) */
    html, body, [class*="css"] {
        font-family: 'TT Norms', 'TT Norms Regular', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("master_copenhagenize_data.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("master_copenhagenize_data.csv", encoding='cp1252')
    return df

df = load_data()

# --- PDF REPORT CONFIGURATION  ---
def generate_pdf_report(city_data, sorted_scores, missing_policies):
    pdf = FPDF()
    pdf.add_page()
    
    # Insert Logo 
    try:
        pdf.image("logo.png", x=165, y=8, w=35)
    except FileNotFoundError:
        pass 
    
    # Header
    pdf.set_text_color(0, 0, 0) # Black for the title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 30, "CITY PROFILE CARD", ln=True, align='C')
    pdf.set_text_color(0, 0, 0) # Black for body
    pdf.set_font("Arial", 'I', 10)
        
    # City Title & KPIs
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, f"{city_data['City']}, {city_data['Country']}", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f"Global Rank: #{city_data['Rank']}   |   Overall Index Score: {city_data['Index Score']:.1f} / 100", ln=True)
    pdf.cell(0, 8, f"Population: {city_data['Population']:,}", ln=True)
    pdf.ln(10)
    
    # The 3 Core Pillars Header
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0) # Black for the title
    pdf.cell(0, 10, "THE 3 CORE PILLARS", ln=True)
    pdf.set_text_color(0, 0, 0) # Black for body text
    
    try:
        # Extract scores
        safe_score = city_data['Safe and Connected Infrastructure']
        usage_score = city_data['Usage and Reach']
        policy_score = city_data['Policy and Support']
        
        # Combine the pillar names with their exact scores for the chart labels
        radar_categories = [
            f"Safe & Connected<br>Infrastructure<br>({safe_score:.1f}/100)", 
            f"Usage & Reach<br>({usage_score:.1f}/100)", 
            f"Policy & Support<br>({policy_score:.1f}/100)"
        ]
        
        # Close the loop to draw a full triangle
        r_vals = [safe_score, usage_score, policy_score]
        closed_r = r_vals + [r_vals[0]]
        closed_theta = radar_categories + [radar_categories[0]]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=closed_r,
            theta=closed_theta,
            fill='toself', 
            name=city_data['City'], 
            line=dict(color='#1BBBEC', width=3),
            marker=dict(color='#192f51', size=8) # Adds visible dots to the corners of the triangle
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=13, color="black"))
            ),
            showlegend=False, 
            height=450, # Taller to fit labels
            width=650,  # Wider to prevent horizontal cut-offs
            margin=dict(t=50, b=50, l=100, r=100) # Extra wide margins so the text isn't cropped by Kaleido
        )
        
        # Save to temp file and insert
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig_radar.write_image(tmpfile.name)
            tmp_path = tmpfile.name
        
        # Insert image into PDF, centered 
        pdf.image(tmp_path, x=15, w=180)
        os.remove(tmp_path)
        
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        # This will print the actual system error directly onto the PDF so we can read it!
        pdf.multi_cell(0, 5, f"(Error generating Pillar Radar figure: {str(e)})")

        pdf.ln(10)
    
    # --- 2 COLUMN STRENGTHS & IMPROVEMENTS  ---
    
   
    start_y = pdf.get_y()
    
    # LEFT COLUMN (Strengths)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0) 
    pdf.cell(95, 6, "Top 3 Strengths", ln=True)
    
    pdf.set_text_color(0, 0, 0) # Black
    pdf.set_font("Arial", '', 12)
    for metric, score in sorted_scores[:3]:
        pdf.cell(95, 6, f"- {metric}: {score:.1f}/100", ln=True)
        
    # RIGHT COLUMN (Improvements)
    pdf.set_y(start_y)
    pdf.set_x(105)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0) 
    pdf.cell(95, 6, "Top 3 Areas for Improvement", ln=True)
    
    pdf.set_text_color(0, 0, 0) # Black
    pdf.set_font("Arial", '', 12)
    for metric, score in reversed(sorted_scores[-3:]):
        pdf.set_x(105) # Must reset X after a line break in the second column
        pdf.cell(95, 6, f"- {metric}: {score:.1f}/100", ln=True)
        
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    return pdf_bytes

# --- INDICATOR DICTIONARY ---
indicator_categories = {
    "Infrastructure": ["Protected_km", "Infra_density (km of bicycle infra/100 km of roadway)"],
    "Parking": ["Public_spaces", "Enclosed_spaces", "Parking_density (stands/1K pop)"],
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
try:
    st.sidebar.image("logo.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.title("🚲 Copenhagenize") 

st.sidebar.markdown("**Dashboard Analytics 2025**")
st.sidebar.divider()


# --- Region Selector ---
regions = ["All Regions", "🌟 Global Top 10", "🌟 Global Top 30"] + sorted(df['Continent'].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("🌍 Select Region or Tier", regions)

# --- Population Range Selector (Multi-Select) ---
pop_options = ["< 500,000", "500,000 - 1.5 million", "1.5 - 3 million", "> 3 million"]
selected_pops = st.sidebar.multiselect(
    "👥 Select Population Sizes", 
    options=pop_options,
    default=pop_options # Defaults to showing all sizes so the dashboard isn't blank
)

# Apply Region Filter
if selected_region == "All Regions":
    df_filtered = df.copy()
elif selected_region == "🌟 Global Top 10":
    df_filtered = df.nsmallest(10, 'Rank')
elif selected_region == "🌟 Global Top 30":
    df_filtered = df.nsmallest(30, 'Rank')
else:
    df_filtered = df[df['Continent'] == selected_region]

# Apply Population Filter (Multi-select logic)
if len(selected_pops) == 0:
    # If the user clears the box, show an empty dataframe
    df_filtered = df_filtered.iloc[0:0] 
else:
    # Create an empty filter mask
    pop_mask = pd.Series(False, index=df_filtered.index)
    
    # Add an OR (|) condition for every option the user selected
    if "< 500,000" in selected_pops:
        pop_mask |= (df_filtered['Population'] < 500000)
    if "500,000 - 1.5 million" in selected_pops:
        pop_mask |= ((df_filtered['Population'] >= 500000) & (df_filtered['Population'] <= 1500000))
    if "1.5 - 3 million" in selected_pops:
        pop_mask |= ((df_filtered['Population'] > 1500000) & (df_filtered['Population'] <= 3000000))
    if "> 3 million" in selected_pops:
        pop_mask |= (df_filtered['Population'] > 3000000)
        
    # Apply the combined mask to the dataframe
    df_filtered = df_filtered[pop_mask]

st.sidebar.divider()

st.sidebar.info("Navigate tabs to explore rankings, correlations, and comparisons.")

st.sidebar.divider()

# --- Export Settings (SVG/PNG) ---
st.sidebar.markdown("**⚙️ Export Settings**")
download_format = st.sidebar.radio("Default Chart Download Format", ["png", "svg"], horizontal=True)

export_config = {
    'toImageButtonOptions': {
        'format': download_format,               # Dynamically updates to SVG or PNG
        'filename': 'copenhagenize_chart', 
        'height': 800,              
        'width': 1200,              
        'scale': 2 if download_format == 'png' else 1  # High-res for PNG, 1 for vector SVG
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['v1hovermode', 'togglespikes']
}



# 4. Main App Layout
st.title("🚲 Copenhagenize Index 2025")
st.markdown("Explore infrastructure, policies, and cycling usage across the globe's top 100 cities.")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Regional Overview", 
    "🏙️ City Profile",      
    "📈 Correlation Explorer", 
    "⚖️ City Comparison", 
    "📏 Indicator Metrics",
    "🏛️ 3 Core Pillars"
    ])

# --- TAB 1: REGIONAL OVERVIEW ---
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
        fig_bubble = px.scatter(
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
        st.plotly_chart(fig_bubble, use_container_width=True, config=export_config)
        
    # ---> MAP SECTION <---
    st.markdown("### 🗺️ Geographic Viewer")
    if 'Lat' in df_filtered.columns and 'Lon' in df_filtered.columns:
        fig_map = px.scatter_geo(
            df_filtered,
            lat='Lat',
            lon='Lon',
            color='Index Score', 
            size='Population',   
            hover_name='City',
            hover_data={
                'Lat': False, 
                'Lon': False, 
                'Country': True, 
                'Rank': True, 
                'Index Score': ':.1f',
                'Population': ':,' 
            },
            projection="natural earth", 
            color_continuous_scale="Viridis",
            title="City Locations & Performance"
        )
        
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            geo=dict(showland=True, landcolor="lightgray", showcoastlines=True, coastlinecolor="white", showcountries=True, countrycolor="white")
        )
        st.plotly_chart(fig_map, use_container_width=True, config=export_config)
    else:
        st.info("Map data is updating. Please ensure you ran the coordinate fetching script.")
    
    st.markdown("---")
    st.markdown("### 📋 Data Viewer")
    
    # Hide Lat/Lon from the table, and format Population 
    display_df = df_filtered.drop(columns=['Lat', 'Lon'], errors='ignore')
    st.dataframe(
        display_df.style.format({"Population": "{:,.0f}"}), 
        use_container_width=True
    )
    
# --- TAB 2: CITY PROFILE ---

with tab2:
    st.subheader("🏙️ City Profile")
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
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#192F51"},
                    'bar': {'color': "#1BBBEC"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#ffcccb'},   # Red for poor
                        {'range': [40, 70], 'color': '#ffffcc'},  # Yellow for average
                        {'range': [70, 100], 'color': '#ccffcc'}  # Green for excellent
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True, config=export_config)
            
        with col_stats:
            st.markdown(f"### Global Rank: **#{city_data['Rank']}**")
            st.markdown(f"**Country:** {city_data['Country']} | **Population:** {city_data['Population']:,}")
            st.markdown("#### The 3 Core Pillars")

            #  Streamlit's progress bars
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
            # Top 3 scores
            for metric, score in sorted_scores[:3]:
                st.markdown(f"**{metric}:** {score:.1f} / 100")
                
        with col_weak:
            st.error("#### ⚠️ Top 3 Areas for Improvement")
            # Bottom 3 scores
            for metric, score in reversed(sorted_scores[-3:]):
                st.markdown(f"**{metric}:** {score:.1f} / 100")

             
        # 2. Quick Wins 
        quick_wins_map = {
            "Cycling_masterplan_yes_no": "Draft and formally adopt a dedicated Cycling Masterplan or Sustainable Urban Mobility Plan.",
            "Cycling_unit_yes_no": "Establish a dedicated 'Cycling Unit' within the city administration.",
            "Cycling_standards_yes_no": "Adopt official, modern Bicycle Infrastructure Design Standards.",
            "Cycling_monitoring_yes_no": "Launch a systematic Cycling Data Monitoring & Counting program.",
            "Household_purchase_subsidy_yes_no": "Introduce a household purchase subsidy program for everyday bicycles and e-bikes.",
            "Logistics_business_subsidy_yes_no": "Create financial incentives for local businesses adopting cargo bikes.",
            "PT_integration_yes_no": "Improve physical and fare integration between cycling and Public Transit hubs."
        }
        missing_policies = [advice for col, advice in quick_wins_map.items() if col in city_data and city_data[col] == 0]

        st.markdown("---")
        st.markdown("### 📥 Benchmark Card")
        st.markdown("Export PDF detailing the city's performance, strengths, and areas of improvement.")
        
        # Generate the PDF report card
        pdf_data = generate_pdf_report(city_data, sorted_scores, missing_policies)
        
        # Streamlit Download Button
        st.download_button(
            label=f"📄 Download {selected_city} Benchmark Card (PDF)",
            data=pdf_data,
            file_name=f"{selected_city}_Copenhagenize_Card.pdf",
            mime="application/pdf",
            type="primary"
        )


# --- TAB 3: CORRELATION EXPLORER ---
with tab3:
    st.subheader("📈 Correlation & Policy Impact Explorer")
    st.markdown("Select a City Input (X) and observe its relationship with a City Outcome (Y).")
    
    # 1. Curated list of Correlated, Normalized, and Continuous Metrics
    # (Excludes raw counts and binary Yes/No policies to ensure clean scatter plots)
    base_correlated_metrics = [
        # --- Core Pillars & Overall ---
        'Index Score',
        'Safe and Connected Infrastructure',
        'Usage and Reach',
        'Policy and Support',
        
        # --- Infrastructure & Urban Planning ---
        'Infra_density (km of bicycle infra/100 km of roadway)',
        'Infra_increase (km of bicycle infra/100 km of roadway)',
        'Parking_density (stands/1K pop)',
        'Traffic_30 (% of km of roadway)',
        
        # --- Financial & Bike Share ---
        'Spending_per_capita (€/capita/year)',
        'Bike_share_cov_density (bikes/1K pop)',
        'Bike_share_usage (trips/bike/day)',
        
        # --- Usage, Demographics & Safety ---
        'Modal_share_2024_% \n(or nearest post-Covid)',
        'Modal_share_2019_% \n(or nearest pre-Covid)',
        'Modal_delta (percentage points)',
        'Bike_trips_women_%',
        'Safety_rate (rate/100K pop)'
    ]
    
    # Automatically grab all the 0-100 sub-indicator scores as well
    score_cols = [c for c in df.columns if 'Score ' in c and c not in ['Index Score', 'Score per Pillar']]
    
    # Combine the lists and ensure they actually exist in the loaded dataframe
    all_correlated_metrics = base_correlated_metrics + score_cols
    valid_metrics = [c for c in all_correlated_metrics if c in df.columns]
    
    st.markdown("### 1. Impact (Scatter Plot)")
    
    col_x, col_y = st.columns(2)
    with col_x:
        # Set a logical default for the X-axis (an intervention)
        default_x = 'Infra_density (km of bicycle infra/100 km of roadway)'
        x_idx = valid_metrics.index(default_x) if default_x in valid_metrics else 0
        x_axis = st.selectbox("Select X-Axis Metric", valid_metrics, index=x_idx)
        
    with col_y:
        # Set a logical default for the Y-axis (an outcome)
        default_y = 'Modal_share_2024_% \n(or nearest post-Covid)'
        y_idx = valid_metrics.index(default_y) if default_y in valid_metrics else 1
        y_axis = st.selectbox("Select Y-Axis Metric", valid_metrics, index=y_idx)
        
    # Scatter Plot
    try:
        fig_corr = px.scatter(
            df_filtered, x=x_axis, y=y_axis, color='Continent', hover_name='City',
            trendline='ols', 
            trendline_scope='overall',         # Draws ONE line for all data
            trendline_color_override='black',  # Makes the global line stand out
            title=f"Impact of {x_axis.split('(')[0].strip()} on {y_axis.split('(')[0].strip()}",
            template='plotly_white'
        )
        st.plotly_chart(fig_corr, use_container_width=True, config=export_config)
        
        # --- Analysis of correlations ---
        # Calculate regression and correlation coefficients dropping empty rows
        clean_df = df_filtered[[x_axis, y_axis]].dropna()
        n = len(clean_df)
        
        if n > 5:
            # Using scipy.stats to get all the advanced metrics instantly
            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_df[x_axis], clean_df[y_axis])
            r_squared = r_value ** 2
            
            # Calculate 95% Confidence Interval for the slope
            # We use a t-distribution because our sample size (n) might be small
            t_crit = stats.t.ppf(0.975, n - 2)
            ci_lower = slope - (t_crit * std_err)
            ci_upper = slope + (t_crit * std_err)
            
            # Smart text generation based on the correlation value
            st.markdown("#### 💡 Insight:")
            
            # Handle reverse logic for "Deaths" (where Negative is Good)
            is_negative_good = "death" in y_axis.lower() or "safety_rate" in y_axis.lower()
            
            if r_value > 0.6:
                if is_negative_good:
                    st.error(f"**Warning (Correlation: {r_value:.2f}):** There is a strong, alarming relationship here. As the intervention increases, fatalities/danger actually increase. This requires immediate auditing.")
                else:
                    st.success(f"**Strong Positive Impact (Correlation: {r_value:.2f}):** The data strongly supports this intervention. Cities that invest heavily in this metric see a reliable, significant increase in the desired outcome.")
            elif 0.3 < r_value <= 0.6:
                if is_negative_good:
                    st.warning(f"**Concerning Trend (Correlation: {r_value:.2f}):** There is a moderate relationship showing danger increasing with this metric. Proceed with caution.")
                else:
                    st.info(f"**Moderate Positive Impact (Correlation: {r_value:.2f}):** This intervention contributes to the outcome. But it must be paired with other policies to guarantee results.")
            elif -0.3 <= r_value <= 0.3:
                st.markdown(f"**No Significant Relationship (Correlation: {r_value:.2f}):** The data shows a random scattering. Changing this input alone does not reliably impacts the chosen outcome.")
            elif -0.6 <= r_value < -0.3:
                if is_negative_good:
                    st.info(f"**Moderate Safety Benefit (Correlation: {r_value:.2f}):** There is a moderate relationship showing that this intervention helps reduce fatalities/danger.")
                else:
                    st.warning(f"**Negative Impact (Correlation: {r_value:.2f}):** Surprisingly, as this intervention increases, the outcome tends to drop. Further urban context is required to understand why.")
            else: # < -0.6
                if is_negative_good:
                    st.success(f"**Strong Safety Benefit (Correlation: {r_value:.2f}):** Excellent policy indicator. Cities that invest in this intervention see a dramatic drop in fatalities and danger.")
                else:
                    st.error(f"**Strong Negative Impact (Correlation: {r_value:.2f}):** There is a stark inverse relationship. This intervention is heavily correlated with a decline in the desired outcome.")
                    
            # ---> NEW: DEEPER STATISTICAL INTERPRETATION <---
            with st.expander("📊 View Deeper Statistical Interpretation"):
                
                # Format p-value cleanly
                p_text = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                sig_text = "✅ Statistically Significant" if p_value < 0.05 else "⚠️ Not Statistically Significant"
                
                # Format slope and CI using general format (.4g) in case of very small/large numbers
                st.markdown(f"**1. Pearson Correlation (r):** `{r_value:.2f}`")
                st.markdown("*Measures the strength and direction of the linear relationship (1.0 is perfect positive, -1.0 is perfect negative).*")
                
                st.markdown(f"**2. Explained Variance (R²):** `{r_squared:.2f}`")
                st.markdown(f"*Mathematically, **{r_squared*100:.1f}%** of the variation in `{y_axis.split('(')[0].strip()}` is directly explained by changes in `{x_axis.split('(')[0].strip()}`.*")
                
                st.markdown(f"**3. Sample Size (n):** `{n}` cities")
                st.markdown("*The number of cities with valid data points for both metrics. Larger samples yield more reliable statistics.*")
                
                st.markdown(f"**4. P-Value:** `{p_text}` ({sig_text})")
                st.markdown("*Measures the probability that this relationship occurred by random chance. A standard threshold for validity is p < 0.05.*")
                
                st.markdown(f"**5. Regression Slope (β):** `{slope:.4g}`")
                st.markdown(f"*For every 1-unit increase in the X-axis, the Y-axis changes by exactly {slope:.4g} units.*")
                
                st.markdown(f"**6. 95% Confidence Interval (Slope):** `[{ci_lower:.4g}, {ci_upper:.4g}]`")
                st.markdown("*We are 95% confident the true impact (slope) falls within this range. If this range crosses zero (e.g., goes from negative to positive), the true effect is uncertain.*")
                
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
        corr_method = st.radio("Select Correlation Method:", ["pearson", "spearman"])
    with col_vars:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        default_heatmap_cols = ['Index Score', 'Infra_density (km of bicycle infra/100 km of roadway)', 'Modal_share_2024_% \n(or nearest post-Covid)', 'Cyclist_deaths', 'Spending_per_capita (€/capita/year)']
        selected_heatmap_cols = st.multiselect("Select Metrics for the Heatmap:", options=[c for c in numeric_cols if c not in ['Rank', 'Population']], default=[c for c in default_heatmap_cols if c in df.columns])
        
    if len(selected_heatmap_cols) > 1 and not df_filtered.empty:
        corr_method_literal = cast(Literal["pearson", "kendall", "spearman"], corr_method if corr_method in ["pearson", "kendall", "spearman"] else "pearson")
        corr_matrix = df_filtered[selected_heatmap_cols].corr(method=corr_method_literal)
        
        # Draw the Heatmap
        fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template='plotly_white')
        fig_heat.update_layout(height=600, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_heat, use_container_width=True, key="heatmap", config=export_config)
        
        # ---> DYNAMIC HEATMAP INTERPRETATION EXPANDER <---
        with st.expander("📊 View Heatmap Interpretation & Automated Insights"):
            st.markdown("**How to read this matrix:**")
            st.markdown("- **Dark Red (Closer to 1.0):** Strong positive correlation. As one metric increases, the other reliably increases.")
            st.markdown("- **Dark Blue (Closer to -1.0):** Strong negative/inverse correlation. As one metric increases, the other reliably decreases.")
            st.markdown("- **White/Light Colors (Closer to 0):** No mathematical relationship. The metrics move independently of each other.")
            
            st.markdown(f"**Current Method Context: {corr_method.capitalize()}**")
            if corr_method == 'pearson':
                st.markdown("*Pearson measures **linear** relationships (do they move together at a consistent, straight-line rate?). It is best for comparing hard, continuous physical metrics.*")
            else:
                st.markdown("*Spearman measures **rank-based** relationships (if City A outranks City B in Metric 1, does it also outrank it in Metric 2?). It is best when comparing abstract scores or indexes.*")
            
            # ---  Matrix Interpretation ---
            st.markdown("**Matrix Insights:**")
            
            # Mask the upper triangle and diagonal to avoid duplicate pairings and self-correlations (1.0)
            mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
            lower_tri = corr_matrix.where(mask)
            
            # Extract Max and Min correlations
            max_val = lower_tri.max().max()
            min_val = lower_tri.min().min()
            
            if pd.notna(max_val):
                max_idx = lower_tri.stack().idxmax()
                st.markdown(f"- **Strongest Positive Pairing:** `{max_idx[0].split('(')[0].strip()}` 🤝 `{max_idx[1].split('(')[0].strip()}` (r = **{max_val:.2f}**)")
                
            if pd.notna(min_val):
                min_idx = lower_tri.stack().idxmin()
                st.markdown(f"- **Strongest Inverse Pairing:** `{min_idx[0].split('(')[0].strip()}` ⚖️ `{min_idx[1].split('(')[0].strip()}` (r = **{min_val:.2f}**)")
                
            st.caption("*Note: The matrix automatically filters out self-correlations (1.0) when finding the strongest pairs.*")

# --- TAB 4: CITY COMPARISON ---

with tab4:
    st.subheader("⚖️ City Benchmarking & Peer Analysis")
    st.markdown("Compare a city against its most relevant peers. The tool automatically suggests comparable cities based on geographic region and population size.")
    
    # 1. Build dynamic lists for the options
    cities_list = sorted(df['City'].unique().tolist())
    regions_list = sorted(df['Continent'].dropna().unique().tolist())
    average_options = [f"Average: {r}" for r in regions_list] + ["Average: Global Top 10", "Average: Global Top 30"]
    all_options = cities_list + average_options
    
    # 2. UI Layout
    col_target, col_peers = st.columns([1, 2])
    
    with col_target:
        # Step A: User selects their primary city
        target_city = st.selectbox(
            " 1. Select Target City:", 
            cities_list, 
            index=cities_list.index('Paris') if 'Paris' in cities_list else 0
        )
        
    # Step B: Peers comparison algorithm by population and continent
    target_data = df[df['City'] == target_city].iloc[0]
    target_pop = target_data['Population']
    target_continent = target_data['Continent']
    
    # Filter for cities in the same continent (excluding the target city) 
    peers_df = df[(df['Continent'] == target_continent) & (df['City'] != target_city)].copy()
    
    # Calculate which cities have the closest population scale
    peers_df['Pop_Diff'] = abs(peers_df['Population'] - target_pop)
    
    # Grab the top 3 closest cities mathematically
    top_3_peers = peers_df.sort_values('Pop_Diff').head(3)['City'].tolist()
    
    # Create Benchmark layout
    regional_avg = f"Average: {target_continent}"
    smart_defaults = [target_city, regional_avg] + top_3_peers
    
    # Ensure our defaults don't crash if data is missing, and limit to 5
    valid_defaults = [x for x in smart_defaults if x in all_options][:5]
    
    with col_peers:
        # Step C: The Multiselect automatically populates with the algorithm's results
        selected_targets = st.multiselect(
            " 2. Benchmark Peers:",
            options=all_options,
            default=valid_defaults,
            max_selections=5
        )

    if len(selected_targets) == 0:
        st.warning("Please select at least one city or benchmark to compare.")
    else:
        # Extract and calculate data dynamically for selected city
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
        
        # label to the end to close the circular loop
        closed_theta = radar_labels + [radar_labels[0]]
        
        fig_radar = go.Figure()
        
        # Color palette
        colors = ['#1BBBEC', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        
        for idx, target in enumerate(selected_targets):
            r_vals = [entity_data[target].get(c, 0) for c in score_cols]
            closed_r = r_vals + [r_vals[0]]
            
            # --- HIGHLIGHT LOGIC ---
            is_target = (idx == 0) # The first item in the list is the Target City
            is_average = "Average:" in target
            
            fig_radar.add_trace(go.Scatterpolar(
                r=closed_r,
                theta=closed_theta,
                # The target is filled. Peers are only filled if it's a 1-on-1 comparison.
                fill='toself' if is_target or len(selected_targets) <= 2 else 'none',
                name=f"🎯 {target}" if is_target else target,
                line=dict(
                    color=colors[idx],
                    width=4.5 if is_target else 2, # Thicker target line
                    dash='dash' if is_average and not is_target else 'solid' # Averages dashed lines
                ),
                opacity=1.0 if is_target else 0.65 # Peers are faded into the background
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=650,
            template='plotly_white',
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config=export_config)
       
        st.markdown("### 📊 Diagnostic table")
        st.markdown("**Raw Data** ➡️ **Correlated Data** (normalized) ➡️ **Final Score** (0-100).")
        
        #  Define the pipeline for all quantitative indicators
        diagnostic_pipeline = {
            "Bicycle Infrastructure": {
                "Raw": "Protected_km",
                "Correlated": "Infra_density (km of bicycle infra/100 km of roadway)",
                "Score": "Score Bicycle Infrastructure"
            },
            "Parking": {
                "Raw": ["Public_spaces", "Enclosed_spaces"],
                "Correlated": "Parking_density (stands/1K pop)",
                "Score": "Score Parking"
            },
            "Traffic Calming": {
                "Raw": ["Street_Km_30", "Street_km_total"],
                "Correlated": "Traffic_30 (% of km of roadway)",
                "Score": "Score Traffic Calming" 
            },
            "Women Modal Share": {
                "Raw": "Bike_trips_women_%",
                "Score": "Score Women Modal Share"
            },
            "Modal Share": {
                "Raw": "Modal_share_2024_% \n(or nearest post-Covid)",
                "Score": "Score Modal Share"
            },    
             "Modal Share increase": {
                "Correlated": "Modal_delta (percentage points)", # Fixed capitalization
                "Score": "Score Modal Share"
            },
            "Safety": {
                "Raw": "Cyclist_deaths",
                "Correlated": "Safety_rate (rate/100K pop)",
                "Score": "Score Safety" # Note: In safety, lower raw/correlated is better
            },
            "Political Commitment": {
                "Raw": "Bicycle_budget_5yr",
                "Correlated": "Spending_per_capita (€/capita/year)",
                "Score": "Score Political Commitment"
            },
            "Bike Share": {
                "Raw": ["Bike_share_fleet", "Bike_share_trips"],
                "Correlated": "Bike_share_cov_density (bikes/1K pop)",
                "Score": "Score Bike Share"
            },
            "Urban Planning": {
                "Raw": "3yr_new_lanes_km",
                "Correlated": "Infra_increase (km of bicycle infra/100 km of roadway)",
                "Score": "Score Urban Planning"
            }
        }

        # Build the dynamic dataframe
        matrix_rows = []
        
        for category, metrics in diagnostic_pipeline.items():
            for data_type, cols in metrics.items():
                
                # ---> FIX: Convert to list if it's a single string so we can handle both smoothly
                col_list = cols if isinstance(cols, list) else [cols]
                
                for col_name in col_list:
                    if col_name in df.columns:
                        # Clean up the metric name for the display table
                        display_metric_name = col_name.replace('_', ' ')
                        
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
        
        st.caption(" For 2-city comparison: 🟢 Indicates the benchmark is outperforming the target city. 🔴 Indicates the benchmark is underperforming. (Note: For Safety Raw/Correlated metrics, lower numbers are better).")

# --- TAB 5: INDICATOR METRICS (MIN/MAX/AVG/MEDIAN) ---
with tab5:
    st.subheader("📏 Indicator Metrics & Distributions")
    st.markdown("Select multiple indicators from across the index to view regional distributions, and statistical summaries.")
    
     # 1. Flatten the dictionary so we can search/select any indicator
    all_indicators_flat = []
    for cat, metrics in indicator_categories.items():
        for m in metrics:
            if m not in all_indicators_flat:
                all_indicators_flat.append(m)

    # 2. Multi-Selector allows choosing multiple indicators to compare directly
    selected_metrics = st.multiselect(
        "📂 Select Indicators to Analyze:",
        options=all_indicators_flat,
        default=indicator_categories["Infrastructure"]
    )
    
    # Loop through the multiple selected metrics
    if not df_filtered.empty:
        for metric in selected_metrics:
            if metric in df_filtered.columns:
                st.markdown(f"### 🔹 {metric.replace('_', ' ')}")
                
                unique_vals = df[metric].dropna().unique()
                is_binary = set(unique_vals).issubset({0, 1})
                
                if is_binary:
                    st.markdown("*Geographic distribution of cities that have implemented this policy.*")
                    if 'Lat' in df_filtered.columns and 'Lon' in df_filtered.columns:
                        map_df = df_filtered.copy()
                        map_df['Status'] = map_df[metric].map({1: 'Yes', 0: 'No', 1.0: 'Yes', 0.0: 'No'})
                        map_df = map_df.dropna(subset=['Status', 'Lat', 'Lon'])
                        
                        fig_map = px.scatter_geo(
                            map_df, lat='Lat', lon='Lon', color='Status',
                            color_discrete_map={'Yes': '#1BBBEC', 'No': '#D53C4C'},
                            hover_name='City', hover_data={'Lat': False, 'Lon': False, 'Country': True, 'Status': True, metric: False},
                            projection="natural earth", title=f"Global Adoption: {metric.replace('_', ' ')}"
                        )
                        fig_map.update_traces(marker=dict(size=9, line=dict(width=1, color='white')))
                        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), geo=dict(showland=True, landcolor="#f4f6f9", showcoastlines=True, coastlinecolor="white", showcountries=True, countrycolor="white"))
                        st.plotly_chart(fig_map, use_container_width=True, config=export_config, key=f"map_{metric}")
                        
                        with st.expander("📊 View Regional Percentage Summary"):
                            agg_df = df_filtered.groupby('Continent')[metric].mean().reset_index()
                            agg_df[metric] = agg_df[metric] * 100 
                            fig_bar = px.bar(agg_df, x='Continent', y=metric, color='Continent', template='plotly_white')
                            fig_bar.update_yaxes(range=[0, 100])
                            st.plotly_chart(fig_bar, use_container_width=True, config=export_config, key=f"bar_{metric}")
                else:
                    st.markdown("*Regional distribution and outliers for this metric.*")
                    fig_box = px.box(
                        df_filtered, x='Continent', y=metric, color='Continent', 
                        points="all", hover_name="City", template='plotly_white'
                    )
                    st.plotly_chart(fig_box, use_container_width=True, key=f"box_{metric}", config=export_config)
                
                # Statistical Summary
                st.markdown("**Statistical Summary (Grouped by Region):**")
                summary_stats = df_filtered.groupby('Continent')[metric].describe()
                
                top10_stats = df.nsmallest(10, 'Rank')[metric].describe().to_frame().T
                top10_stats.index = ['🌟 Global Top 10']
                top30_stats = df.nsmallest(30, 'Rank')[metric].describe().to_frame().T
                top30_stats.index = ['🌟 Global Top 30']
                global100_stats = df[metric].describe().to_frame().T
                global100_stats.index = ['🌟 Global 100']
                
                summary_stats = pd.concat([summary_stats, top10_stats, top30_stats, global100_stats])
                summary_stats = summary_stats.rename(columns={'count': 'Data Points', 'mean': 'Average', 'std': 'Std Dev', 'min': 'Minimum', '25%': '25th Pct', '50%': 'Median', '75%': '75th Pct', 'max': 'Maximum'})
                
                st.dataframe(summary_stats.style.format({col: (lambda x: f"{x:.0f}" if col == 'Data Points' else f"{x:.2f}") for col in summary_stats.columns}), use_container_width=True)
                
                with st.expander(f"🏅 View City Rankings: {metric.replace('_', ' ')}"):
                    is_lower_better = "death" in metric.lower() or "safety_rate" in metric.lower()
                    ranking_df = df_filtered[['City', 'Country', 'Continent', metric]].dropna(subset=[metric])
                    ranking_df = ranking_df.sort_values(by=metric, ascending=is_lower_better).reset_index(drop=True)
                    ranking_df.index = ranking_df.index + 1 
                    if is_binary:
                        ranking_df[metric] = ranking_df[metric].map({1: 'Yes', 0: 'No', 1.0: 'Yes', 0.0: 'No'})
                        st.dataframe(ranking_df, use_container_width=True)
                    else:
                        st.dataframe(ranking_df.style.format({metric: "{:.2f}"}), use_container_width=True)
                st.markdown("---")

# --- TAB 6: 3 CORE PILLARS ANALYSIS ---
with tab6:
    st.subheader("🏛️ Pillar Analysis")
    st.markdown("Analyze the Overall Index Score to see which cities lead in each pillar, and how they balance these three foundations.")
    
    pillars = [
        'Safe and Connected Infrastructure', 
        'Usage and Reach', 
        'Policy and Support'
    ]
    
    # Check if the pillars exist in the dataset to prevent errors
    if all(p in df_filtered.columns for p in pillars):
        
        # --- 1. Top 10 Leaderboards ---
        st.markdown("### 🏆 Top 10 Performers by Pillar")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        cols_list = [col_p1, col_p2, col_p3]
        
        for i, pillar in enumerate(pillars):
            # Get top 10 for this specific pillar
            top_10_df = df_filtered.nlargest(10, pillar).sort_values(by=pillar, ascending=True)
            
            fig_bar = px.bar(
                top_10_df, 
                x=pillar, 
                y='City', 
                orientation='h',
                title=f"Best in: {pillar}", 
                text_auto='.1f',
                color=pillar,
                color_continuous_scale="Teal" if i==0 else ("Purp" if i==1 else "Blues")
            )
            fig_bar.update_layout(
                xaxis_title="Score (0-100)", 
                yaxis_title="", 
                showlegend=False, 
                coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=40, b=0),
                height=400
            )
            cols_list[i].plotly_chart(fig_bar, use_container_width=True, config=export_config)

        st.markdown("---")
        
        # --- 2. Ternary Plot (Balance) ---
        st.markdown("### 🔺 The Balance Triangle (Ternary Plot)")
        st.markdown("This plot shows the *proportion* of a city's strengths. Cities closer to the center are perfectly balanced. " \
            "Cities pulled toward a corner rely heavily on that specific pillar.")
            
        fig_ternary = px.scatter_ternary(
                df_filtered, 
                a='Safe and Connected Infrastructure', 
                b='Usage and Reach', 
                c='Policy and Support', 
                hover_name='City',
                color='Continent',
                size='Index Score', # Bigger bubbles = higher overall score
                template='plotly_white'
            )
        fig_ternary.update_layout(
                ternary=dict(
                    sum=100,
                    aaxis_title="Infrastructure",
                    baxis_title="Usage",
                    caxis_title="Policy"
                ),
                height=500,
                margin=dict(l=20, r=20, t=40, b=30)
            )
        st.plotly_chart(fig_ternary, use_container_width=True, config=export_config)

        st.markdown("---")
        
        # --- 3. Searchable Data Table ---
        st.markdown("### 📋 Pillar Data Explorer")
        st.markdown("Sort and search through the exact scores for all cities currently in view.")
        
        clean_pillar_df = df_filtered[['Rank', 'City', 'Country', 'Index Score'] + pillars].sort_values('Rank')
        st.dataframe(
            clean_pillar_df.style.format({
                'Index Score': "{:.1f}",
                'Safe and Connected Infrastructure': "{:.1f}",
                'Usage and Reach': "{:.1f}",
                'Policy and Support': "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.warning("The 3 Core Pillars data columns were not found in the dataset. Please ensure they are named exactly: 'Safe and Connected Infrastructure', 'Usage and Reach', and 'Policy and Support'.")