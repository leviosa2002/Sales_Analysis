import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Superstore Sales Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved models
@st.cache_resource
def load_models():
    sales_model = joblib.load('best_sales_model.pkl')
    profit_model = joblib.load('best_profit_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return sales_model, profit_model, scaler, label_encoders

sales_model, profit_model, scaler, label_encoders = load_models()

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv", encoding='ISO-8859-1')
    
    # Data preprocessing
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Create additional date features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Day_of_Week'] = df['Order Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Month_Name'] = df['Order Date'].dt.month_name()
    df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Create profit margin
    df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
    
    # Add season
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Day of week mapping
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Day_Name'] = df['Day_of_Week'].map(dict(enumerate(dow_names)))
    
    return df

df = load_data()

# Feature columns for prediction
feature_columns = [
    'Quantity', 'Discount', 'Year', 'Month', 'Quarter', 'Day_of_Week', 'Shipping_Days',
    'Ship Mode_encoded', 'Segment_encoded', 'Region_encoded',
    'Category_encoded', 'Sub-Category_encoded'
]

# Sidebar navigation
st.sidebar.title("📊 Superstore Analytics")
page = st.sidebar.selectbox(
    "Choose a Page",
    ["Dashboard", "Sales Analysis", "Customer Analysis", "Product Analysis", "Predictions"]
)

# Function to create KPI metrics
def create_kpi_metric(title, value, prefix="", suffix="", delta=None):
    if delta:
        st.metric(title, f"{prefix}{value:,.2f}{suffix}", delta=delta)
    else:
        st.metric(title, f"{prefix}{value:,.2f}{suffix}")

# Dashboard page
if page == "Dashboard":
    st.title("📈 Superstore Sales Dashboard")
    st.subheader("Key Performance Indicators")
    
    # Calculate KPIs
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_order_value = df['Sales'].mean()
    profit_margin = (total_profit / total_sales) * 100
    
    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_kpi_metric("Total Sales", total_sales, prefix="$")
    with col2:
        create_kpi_metric("Total Profit", total_profit, prefix="$")
    with col3:
        create_kpi_metric("Avg Order Value", avg_order_value, prefix="$")
    with col4:
        create_kpi_metric("Profit Margin", profit_margin, suffix="%")
    
    # Sales trend over time
    st.subheader("Sales Trend Over Time")
    
    # Prepare data for time trend
    monthly_sales = df.groupby(['Year', 'Month']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
    }).reset_index()
    
    monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
    
    # Create plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sales line
    fig.add_trace(
        go.Scatter(
            x=monthly_sales['Date'], 
            y=monthly_sales['Sales'],
            name="Sales",
            line=dict(color='#1f77b4', width=2)
        ),
        secondary_y=False,
    )
    
    # Add profit line
    fig.add_trace(
        go.Scatter(
            x=monthly_sales['Date'], 
            y=monthly_sales['Profit'],
            name="Profit",
            line=dict(color='#2ca02c', width=2)
        ),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title='Monthly Sales and Profit Trends',
        xaxis_title='Date',
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional and Category Performance
    st.subheader("Performance by Region and Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Region Analysis
        region_sales = df.groupby('Region')['Sales'].sum().reset_index()
        fig = px.pie(
            region_sales, 
            values='Sales', 
            names='Region', 
            title='Sales by Region',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category Analysis
        category_sales = df.groupby('Category')['Sales'].sum().reset_index()
        fig = px.pie(
            category_sales, 
            values='Sales', 
            names='Category', 
            title='Sales by Category',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Subcategories by Sales
    st.subheader("Top Subcategories by Sales")
    
    top_subcategories = df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)
    fig = px.bar(
        top_subcategories, 
        x='Sales', 
        y='Sub-Category',
        orientation='h',
        title='Top 10 Sub-Categories by Sales',
        labels={'Sales': 'Total Sales ($)', 'Sub-Category': ''},
        color='Sales',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Executive Summary
    st.subheader("💡 Key Insights")
    with open('executive_summary.txt', 'r') as file:
        summary = file.read()
    
    # Extract strategic recommendations section
    start_marker = "💡 STRATEGIC RECOMMENDATIONS"
    end_marker = "📈 GROWTH OPPORTUNITIES"
    start_idx = summary.find(start_marker)
    end_idx = summary.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        recommendations = summary[start_idx:end_idx].strip()
        st.write(recommendations)
    else:
        st.write("Strategic recommendations not found in the summary.")

# Sales Analysis page
elif page == "Sales Analysis":
    st.title("🔍 Sales Analysis")
    
    # Time period filter
    st.sidebar.subheader("Filter Data")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df['Order Date'].min(), df['Order Date'].max()],
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        filtered_df = df[(df['Order Date'] >= pd.Timestamp(date_range[0])) & 
                          (df['Order Date'] <= pd.Timestamp(date_range[1]))]
    else:
        filtered_df = df
    
    # KPIs after filtering
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        create_kpi_metric("Total Sales", total_sales, prefix="$")
    with col2:
        create_kpi_metric("Total Profit", total_profit, prefix="$")
    with col3:
        create_kpi_metric("Profit Margin", profit_margin, suffix="%")
    
    # Sales by time period
    st.subheader("Sales by Time Period")
    time_period = st.selectbox("Select Time Period", ["Month", "Quarter", "Year", "Day of Week", "Season"])
    
    if time_period == "Month":
        period_sales = filtered_df.groupby('Month_Name')['Sales'].sum().reindex(
            ["January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"]
        ).reset_index()
        x_title = "Month"
    elif time_period == "Quarter":
        period_sales = filtered_df.groupby('Quarter')['Sales'].sum().reset_index()
        x_title = "Quarter"
    elif time_period == "Year":
        period_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        x_title = "Year"
    elif time_period == "Day of Week":
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        period_sales = filtered_df.groupby('Day_Name')['Sales'].sum().reindex(dow_order).reset_index()
        x_title = "Day of Week"
    else:  # Season
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        period_sales = filtered_df.groupby('Season')['Sales'].sum().reindex(season_order).reset_index()
        x_title = "Season"
    
    fig = px.bar(
        period_sales,
        x=time_period if time_period != "Month" else "Month_Name" if time_period != "Day of Week" else "Day_Name",
        y="Sales",
        title=f"Sales by {time_period}",
        labels={"Sales": "Total Sales ($)", time_period if time_period != "Month" else "Month_Name": x_title}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional Performance
    st.subheader("Regional Performance")
    
    region_metrics = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).rename(columns={'Order ID': 'Orders'}).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Region Bar Chart
        fig = px.bar(
            region_metrics,
            x="Region",
            y="Sales",
            title="Sales by Region",
            labels={"Sales": "Total Sales ($)"},
            color="Region",
            text="Sales"
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit by Region Bar Chart
        fig = px.bar(
            region_metrics,
            x="Region",
            y="Profit",
            title="Profit by Region",
            labels={"Profit": "Total Profit ($)"},
            color="Region",
            text="Profit"
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    
    # Calculate correlation
    correlation_data = filtered_df[['Sales', 'Quantity', 'Discount', 'Profit', 'Profit_Margin']].corr()
    
    # Create heatmap
    fig = px.imshow(
        correlation_data,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Discount Effect Analysis
    st.subheader("Discount Effect Analysis")
    
    discount_analysis = filtered_df.copy()
    discount_analysis['Discount_Bin'] = pd.cut(
        discount_analysis['Discount'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-100%']
    )
    
    discount_perf = discount_analysis.groupby('Discount_Bin').agg({
        'Sales': 'mean',
        'Profit': 'mean',
        'Profit_Margin': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=discount_perf['Discount_Bin'],
        y=discount_perf['Sales'],
        name='Avg Sales',
        marker_color='skyblue'
    ))
    fig.add_trace(go.Bar(
        x=discount_perf['Discount_Bin'],
        y=discount_perf['Profit'],
        name='Avg Profit',
        marker_color='lightgreen'
    ))
    fig.add_trace(go.Scatter(
        x=discount_perf['Discount_Bin'],
        y=discount_perf['Profit_Margin'],
        name='Profit Margin %',
        marker_color='red',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    # Set layout with dual y-axis
    fig.update_layout(
        title='Discount Effect on Sales and Profit',
        yaxis=dict(
            title='Amount ($)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Profit Margin (%)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(title='Discount Range'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Customer Analysis page
elif page == "Customer Analysis":
    st.title("👥 Customer Analysis")
    
    # Customer Segments
    st.subheader("Customer Segments Performance")
    
    segment_metrics = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Customer ID': 'nunique'
    }).rename(columns={'Customer ID': 'Unique Customers'}).reset_index()
    
    # Add average order value
    segment_metrics['Avg Order Value'] = segment_metrics['Sales'] / segment_metrics['Unique Customers']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment Sales
        fig = px.bar(
            segment_metrics,
            x="Segment",
            y="Sales",
            title="Sales by Customer Segment",
            color="Segment",
            text="Sales"
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment Profit
        fig = px.bar(
            segment_metrics,
            x="Segment",
            y="Profit",
            title="Profit by Customer Segment",
            color="Segment",
            text="Profit"
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment Metrics Table
    st.subheader("Customer Segment Metrics")
    
    segment_display = segment_metrics.copy()
    segment_display['Sales'] = segment_display['Sales'].apply(lambda x: f"${x:,.2f}")
    segment_display['Profit'] = segment_display['Profit'].apply(lambda x: f"${x:,.2f}")
    segment_display['Avg Order Value'] = segment_display['Avg Order Value'].apply(lambda x: f"${x:,.2f}")
    
    st.table(segment_display)
    
    # Top Customers
    st.subheader("Top Customers Analysis")
    
    top_n = st.slider("Number of Top Customers to Show", min_value=5, max_value=20, value=10)
    
    top_customers = df.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).rename(columns={'Order ID': 'Orders'})
    
    top_customers['Avg Order Value'] = top_customers['Sales'] / top_customers['Orders']
    top_customers = top_customers.sort_values('Sales', ascending=False).head(top_n)
    
    fig = px.bar(
        top_customers.reset_index(),
        x="Sales",
        y="Customer Name",
        title=f"Top {top_n} Customers by Sales",
        color="Profit",
        orientation='h',
        color_continuous_scale='Viridis',
        labels={"Sales": "Total Sales ($)", "Customer Name": ""},
        hover_data=["Orders", "Avg Order Value"]
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Region Distribution
    st.subheader("Customer Distribution by Region")
    
    customer_region = df.groupby(['Region', 'Segment']).agg({
        'Customer ID': 'nunique'
    }).rename(columns={'Customer ID': 'Count'}).reset_index()
    
    fig = px.bar(
        customer_region,
        x="Region",
        y="Count",
        color="Segment",
        title="Customer Distribution by Region and Segment",
        barmode="group",
        labels={"Count": "Number of Customers", "Region": "Region", "Segment": "Customer Segment"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # RFM Analysis teaser
    st.subheader("Customer Segmentation (RFM Analysis)")
    
    st.info("""
        RFM Analysis would provide advanced customer segmentation based on:
        - Recency: How recently a customer has made a purchase
        - Frequency: How often they purchase
        - Monetary Value: How much they spend
        
        This feature would require additional customer purchase history data.
    """)

# Product Analysis page
elif page == "Product Analysis":
    st.title("🛒 Product Analysis")
    
    # Category filters
    st.sidebar.subheader("Filter Products")
    
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=sorted(df['Category'].unique()),
        default=list(df['Category'].unique())
    )
    
    if not selected_categories:
        filtered_df = df
    else:
        filtered_df = df[df['Category'].isin(selected_categories)]
    
    # Category Performance
    st.subheader("Category Performance")
    
    category_metrics = filtered_df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean',
        'Order ID': 'count'
    }).rename(columns={'Order ID': 'Orders'}).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category Sales
        fig = px.bar(
            category_metrics,
            x="Category",
            y="Sales",
            title="Sales by Category",
            color="Category",
            text="Sales"
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category Profit
        fig = px.bar(
            category_metrics,
            x="Category",
            y="Profit",
            title="Profit by Category",
            color="Category",
            text="Profit"
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Subcategory Analysis
    st.subheader("Sub-Category Analysis")
    
    selected_subcategory_view = st.radio(
        "Select View",
        ["Top Selling Sub-Categories", "All Sub-Categories by Category"]
    )
    
    if selected_subcategory_view == "Top Selling Sub-Categories":
        top_n = st.slider("Number of Sub-Categories to Show", min_value=5, max_value=20, value=10)
        
        subcategory_metrics = filtered_df.groupby('Sub-Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Profit_Margin': 'mean'
        }).sort_values('Sales', ascending=False).head(top_n).reset_index()
        
        fig = px.bar(
            subcategory_metrics,
            x="Sales",
            y="Sub-Category",
            orientation='h',
            title=f"Top {top_n} Sub-Categories by Sales",
            color="Profit",
            color_continuous_scale='RdYlGn',
            text="Sales",
            labels={"Sales": "Total Sales ($)", "Sub-Category": ""}
        )
        fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Grouped subcategories by category
        subcategory_by_cat = filtered_df.groupby(['Category', 'Sub-Category']).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        fig = px.bar(
            subcategory_by_cat,
            x="Sub-Category",
            y="Sales",
            color="Category",
            title="Sub-Category Performance by Category",
            labels={"Sales": "Total Sales ($)"},
            barmode="group"
        )
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Product Profitability Analysis
    st.subheader("Product Profitability Analysis")
    
    profitability_df = filtered_df.groupby('Sub-Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    profitability_df['Profit_Margin'] = (profitability_df['Profit'] / profitability_df['Sales']) * 100
    
    fig = px.scatter(
        profitability_df,
        x="Sales",
        y="Profit",
        size="Sales",
        color="Profit_Margin",
        hover_name="Sub-Category",
        color_continuous_scale="RdYlGn",
        title="Sub-Category Profitability Analysis",
        labels={"Sales": "Total Sales ($)", "Profit": "Total Profit ($)", "Profit_Margin": "Profit Margin (%)"},
        size_max=50
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=profitability_df['Sales'].median(), line_dash="dash", line_color="gray")
    
    # Add annotations for quadrants
    quadrant_annotations = [
        dict(
            x=profitability_df['Sales'].max() * 0.9,
            y=profitability_df['Profit'].max() * 0.9,
            text="High Sales, High Profit",
            showarrow=False,
            font=dict(size=12)
        ),
        dict(
            x=profitability_df['Sales'].min() * 1.5,
            y=profitability_df['Profit'].max() * 0.9,
            text="Low Sales, High Profit",
            showarrow=False,
            font=dict(size=12)
        ),
        dict(
            x=profitability_df['Sales'].max() * 0.9,
            y=profitability_df['Profit'].min() * 1.5,
            text="High Sales, Low Profit",
            showarrow=False,
            font=dict(size=12)
        ),
        dict(
            x=profitability_df['Sales'].min() * 1.5,
            y=profitability_df['Profit'].min() * 1.5,
            text="Low Sales, Low Profit",
            showarrow=False,
            font=dict(size=12)
        )
    ]
    
    fig.update_layout(annotations=quadrant_annotations)
    st.plotly_chart(fig, use_container_width=True)

# Predictions page
elif page == "Predictions":
    st.title("🔮 Sales & Profit Predictions")
    
    st.write("""
    Use this tool to predict sales and profit based on various factors. 
    Adjust the parameters on the left to get predictions.
    """)
    
    # Input parameters
    st.sidebar.subheader("Prediction Parameters")
    
    quantity = st.sidebar.slider(
        "Quantity", 
        min_value=1, 
        max_value=14, 
        value=3
    )
    
    discount = st.sidebar.slider(
        "Discount", 
        min_value=0.0, 
        max_value=0.8, 
        value=0.0,
        step=0.05,
        format="%.2f"
    )
    
    region = st.sidebar.selectbox(
        "Region",
        sorted(df['Region'].unique())
    )
    
    category = st.sidebar.selectbox(
        "Category",
        sorted(df['Category'].unique())
    )
    
    # Dynamically update subcategory options based on category
    subcategory_options = sorted(df[df['Category'] == category]['Sub-Category'].unique())
    subcategory = st.sidebar.selectbox(
        "Sub-Category",
        subcategory_options
    )
    
    segment = st.sidebar.selectbox(
        "Customer Segment",
        sorted(df['Segment'].unique())
    )
    
    ship_mode = st.sidebar.selectbox(
        "Shipping Mode",
        sorted(df['Ship Mode'].unique())
    )
    
    month = st.sidebar.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda x: ["January", "February", "March", "April", "May", "June", 
                               "July", "August", "September", "October", "November", "December"][x-1]
    )
    
    if st.sidebar.button("Generate Prediction"):
        # Prepare the input data
        input_data = pd.DataFrame({
            'Quantity': [quantity],
            'Discount': [discount],
            'Year': [2024],  # Default to current year
            'Month': [month],
            'Quarter': [((month-1)//3) + 1],
            'Day_of_Week': [1],  # Default to Monday
            'Shipping_Days': [4],  # Default shipping days
        })
        
        # Add encoded categorical variables
        for col in ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']:
            if col in label_encoders:
                if col == 'Ship Mode':
                    encoded_val = label_encoders[col].transform([ship_mode])[0]
                elif col == 'Segment':
                    encoded_val = label_encoders[col].transform([segment])[0]
                elif col == 'Region':
                    encoded_val = label_encoders[col].transform([region])[0]
                elif col == 'Category':
                    encoded_val = label_encoders[col].transform([category])[0]
                elif col == 'Sub-Category':
                    encoded_val = label_encoders[col].transform([subcategory])[0]
                
                input_data[col + '_encoded'] = [encoded_val]
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = [0]  # Default value
        
        # Reorder columns to match training data
        input_data = input_data[feature_columns]
        
        # Make predictions
        sales_pred = sales_model.predict(input_data)[0]
        profit_pred = profit_model.predict(input_data)[0]
        profit_margin = (profit_pred / sales_pred) * 100 if sales_pred > 0 else 0
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_kpi_metric("Predicted Sales", sales_pred, prefix="$")
        with col2:
            create_kpi_metric("Predicted Profit", profit_pred, prefix="$")
        with col3:
            create_kpi_metric("Profit Margin", profit_margin, suffix="%")
        with col4:
            create_kpi_metric("Revenue per Unit", sales_pred/quantity, prefix="$")
        
        # Compare to category averages
        st.subheader("Comparison to Category Averages")
        
        # Get category averages
        category_avg = df[df['Category'] == category].agg({
            'Sales': 'mean',
            'Profit': 'mean',
            'Profit_Margin': 'mean'
        })
        
        comparison_data = pd.DataFrame({
            'Metric': ['Sales', 'Profit', 'Profit Margin'],
            'Predicted': [sales_pred, profit_pred, profit_margin],
            'Category Average': [
                category_avg['Sales'], 
                category_avg['Profit'],
                category_avg['Profit_Margin']
            ]
        })
        
        # Create comparison chart
        for i, metric in enumerate(['Sales', 'Profit']):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Predicted', 'Category Average'],
                y=[comparison_data.iloc[i, 1], comparison_data.iloc[i, 2]],
                text=[f'${comparison_data.iloc[i, 1]:.2f}', f'${comparison_data.iloc[i, 2]:.2f}'],
                textposition='auto',
                marker_color=['royalblue', 'lightblue']
            ))
            fig.update_layout(
                title=f'Predicted {metric} vs Category Average',
                yaxis=dict(title=f'{metric} ($)'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Business insights based on prediction
        st.subheader("Business Insights")
        
        insight_text = ""
        
        if profit_margin > category_avg['Profit_Margin']:
            insight_text += "✅ This combination has an **above average** profit margin compared to the category average.\n\n"
        else:
            insight_text += "⚠️ This combination has a **below average** profit margin compared to the category average.\n\n"
            
        if discount > 0.2 and profit_margin < 15:
            insight_text += "⚠️ **High discount rate** may be negatively affecting the profit margin. Consider reducing discounts.\n\n"
        
        if profit_pred < 0:
            insight_text += "🚨 This combination is projected to result in a **loss**. Reconsider this business decision.\n\n"
        
        if sales_pred > category_avg['Sales'] * 1.5:
            insight_text += "🌟 This combination has **exceptionally high** projected sales compared to the category average.\n\n"
            
        st.markdown(insight_text)
        
        # What-if scenario
        st.subheader("What-if Scenarios")
        
        # Show how changing discount would affect results
        discount_range = np.linspace(0, 0.5, 6)
        scenario_results = []
        
        for disc in discount_range:
            temp_input = input_data.copy()
            temp_input['Discount'] = disc
            
            temp_sales = sales_model.predict(temp_input)[0]
            temp_profit = profit_model.predict(temp_input)[0]
            temp_margin = (temp_profit / temp_sales) * 100 if temp_sales > 0 else 0
            
            scenario_results.append({
                'Discount': f"{disc:.0%}",
                'Sales': temp_sales,
                'Profit': temp_profit,
                'Profit_Margin': temp_margin
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bars for sales and profit
        fig.add_trace(
            go.Bar(
                x=scenario_df['Discount'],
                y=scenario_df['Sales'],
                name="Sales",
                marker_color='royalblue'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=scenario_df['Discount'],
                y=scenario_df['Profit'],
                name="Profit",
                marker_color='lightgreen'
            ),
            secondary_y=False
        )
        
        # Add line for profit margin
        fig.add_trace(
            go.Scatter(
                x=scenario_df['Discount'],
                y=scenario_df['Profit_Margin'],
                name="Profit Margin %",
                mode='lines+markers',
                marker_color='red',
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        # Set titles
        fig.update_layout(
            title='Effect of Discount on Sales, Profit, and Margin',
            xaxis_title='Discount Rate',
            barmode='group',
            height=500
        )
        
        fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
        fig.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    📊 Superstore Sales Analysis Dashboard | Built with Streamlit | Data Science Project
</div>
""", unsafe_allow_html=True) 