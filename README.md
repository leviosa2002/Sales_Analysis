# Superstore Sales Analysis Dashboard

This interactive Streamlit dashboard provides comprehensive analysis and predictions for the Superstore dataset.

## Features

- **Dashboard Overview**: Key performance indicators and summary metrics
- **Sales Analysis**: Detailed breakdown of sales by time period, region, and more
- **Customer Analysis**: Customer segmentation and top customer insights
- **Product Analysis**: Category and sub-category performance metrics
- **Sales & Profit Predictions**: ML-powered prediction tool for sales forecasting

## Setup Instructions

1. Clone this repository
2. Create and activate a virtual environment (recommended):
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
   
If you encounter a "streamlit command not found" error, ensure that:
- You've installed the requirements correctly using `pip install -r requirements.txt`
- You're using the environment where streamlit was installed
- Try using the full path: `python -m streamlit run app.py`

## Data

The application uses the Superstore dataset (`superstore.csv`), which contains retail sales data including:
- Orders, customers, and products information
- Sales and profit metrics
- Geographic data
- Shipping information

## Machine Learning Models

The application uses pre-trained machine learning models:
- `best_sales_model.pkl`: Gradient Boosting model for sales prediction
- `best_profit_model.pkl`: Gradient Boosting model for profit prediction
- `feature_scaler.pkl`: StandardScaler for feature normalization
- `label_encoders.pkl`: Label encoders for categorical variables

## Usage

The sidebar navigation menu allows you to switch between different analysis pages:

1. **Dashboard**: View overall KPIs and business performance metrics
2. **Sales Analysis**: Explore sales trends, regional performance, and discount effects
3. **Customer Analysis**: Analyze customer segments and top customers
4. **Product Analysis**: Investigate product category performance and profitability
5. **Predictions**: Make sales and profit predictions based on different scenarios

## Screenshots

*Dashboard Overview:*
![Dashboard](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

*Prediction Interface:*
![Predictions](https://via.placeholder.com/800x400?text=Predictions+Screenshot)

## Development

This project was developed using:
- Python 3.8+
- Streamlit for web interface
- Pandas and NumPy for data processing
- Plotly and Matplotlib for visualization
- Scikit-learn for machine learning models 