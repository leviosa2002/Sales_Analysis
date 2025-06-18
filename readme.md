# 🛍️ Store Sales Analysis | Python, Pandas, ML Models

This project provides an end-to-end sales analysis for a retail store using Python and Jupyter Notebook. It includes visualizations, KPIs, and trained machine learning models to predict and optimize sales and profit strategies.


---

## 📊 Dataset Source
The sales data used in this project is sourced from the Superstore Dataset on Kaggle, which contains historical sales records of a global retail store, including details like order dates, regions, product categories, and financial metrics.
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

---

## 📂 Project Structure

```
📦 Store_Sales_Analysis/
├── Store_Sales_Analysis.ipynb
├── models/
│   ├── best_sales_model.pkl
│   ├── best_profit_model.pkl
│   ├── feature_scaler.pkl
│   └── label_encoders.pkl
├── data/
│   └── superstore.csv
├── README.md
└── requirements.txt
```
---

## 🤖 PREDICTIVE MODEL PERFORMANCE
• Sales Prediction Model: Gradient Boosting (R² = 0.193)
• Profit Prediction Model: Gradient Boosting (R² = 0.191)
• Model Accuracy: Fair

 More analysis is mentioned in ./executive_summary
 
---

## 📊Some Graphs from the project
![image](https://github.com/user-attachments/assets/21c55760-5ad1-42c4-a885-55cd6a49e3ee)


## 🚀 Features

- 📊 Exploratory Data Analysis with trends & KPIs
- 🧠 Machine learning models for:
  - Predicting **top sales months/products**
  - Identifying **most profitable strategies**
- 🔍 Data preprocessing with Scikit-learn encoders and scalers
- 📈 Visual insights using Seaborn, Plotly, and Matplotlib

---

## 🛠️ Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/store-sales-analysis.git
cd store-sales-analysis
```

2. Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook:
```bash
jupyter notebook Store_Sales_Analysis.ipynb
```
---

## 📬 Contact

For questions or collaborations, reach out via GitHub Issues or email.

---

## 📘 License

MIT License
