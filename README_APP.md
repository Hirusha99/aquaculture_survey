# 🐟 Fish Weight Predictor App

Predict fish final weight based on farming parameters using Machine Learning.

## 📦 Installation

```bash
pip install -r requirements_app.txt
```

## 🚀 Running the Apps

### Option 1: Streamlit App (Recommended)
```bash
streamlit run weight_predictor_app.py
```
Opens at: http://localhost:8501

**Features:**
- Beautiful interactive UI
- Organized tabs for inputs
- Real-time metrics
- Download prediction reports
- Responsive design

### Option 2: Flask Web App
```bash
python weight_predictor_flask.py
```
Opens at: http://localhost:5000

**Features:**
- Simple web interface
- REST API included
- Standalone HTML page
- JSON API responses

## 📊 Input Parameters

The app requires 11 features:

### Farm Characteristics
- **Area (Ha)**: Farm area in hectares (0.1 - 10.0)
- **Pens**: Number of fish pens (1 - 10)
- **Distance (m)**: Distance from water source (0 - 5000)
- **Duration (months)**: Cycle duration (1 - 24)

### Fish Parameters
- **Initial Weight (kg)**: Average initial weight (0.01 - 1.0)
- **Density (fish/Ha)**: Stocking density (1000 - 100000)
- **Harvest Number**: Fish to harvest (100 - 100000)

### Production Metrics
- **Production (kg)**: Total production (100 - 50000)
- **Production/Ha (kg)**: Per hectare production (500 - 30000)

### Environment
- **Vegetation (%)**: Coverage percentage (0 - 100)
- **Depth (feet)**: Average water depth (1 - 20)

## 🎯 Output

The app predicts:
- **Final Fish Weight** (kg and grams)
- **Weight Gain** from initial weight
- **Total Harvest Weight**
- **Growth Rate** (g/month)
- **Feed Conversion Ratio** estimate

## 📥 Features

### Streamlit App
✅ Tabbed input interface
✅ Real-time metrics calculation
✅ Interactive sliders and inputs
✅ Prediction confidence
✅ Download CSV reports
✅ Model information sidebar

### Flask App
✅ Single-page web interface
✅ REST API endpoint
✅ JSON response format
✅ Auto model training
✅ CORS enabled

## 🔧 API Usage (Flask)

### Predict Weight
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "Area (Ha 1)": 0.8,
  "INITIAL WEIGHT/kg": 0.10,
  "DENSITY 1 (total stock/Ha)": 10000,
  "HARVEST NO. 1": 8000,
  "PRODUCTION 1": 3600,
  "PRODUCTION 1/Ha": 4500,
  "DISTANCE/m": 300,
  "VEGETATION%": 60,
  "AVERAGE_DEPTH/ft": 3,
  "PENS": 2,
  "DURATIONMONTHS": 8.0
}
```

### Response
```json
{
  "success": true,
  "predicted_weight_kg": 0.456,
  "predicted_weight_grams": 456,
  "weight_gain_grams": 356
}
```

## 📁 Files

- `weight_predictor_app.py` - Streamlit application
- `weight_predictor_flask.py` - Flask web app with API
- `requirements_app.txt` - Python dependencies
- `survey-data.csv` - Training data (required)
- `weight_model.pkl` - Trained model (auto-generated)
- `weight_scaler.pkl` - Feature scaler (auto-generated)

## 🤖 Model

- **Algorithm**: Linear Regression
- **Features**: 11 parameters
- **Accuracy**: ~52% R² score
- **Target**: Fish final weight (AVG WEIGHT 1)

## 🎨 Screenshots

### Streamlit App
- Clean, modern interface
- Organized input tabs
- Real-time metrics
- Beautiful predictions display

### Flask App
- Single-page design
- Gradient UI
- Responsive layout
- Loading animations

## 💡 Tips

1. **First Run**: App auto-trains model from survey-data.csv
2. **Model Saved**: Subsequent runs load faster
3. **Validation**: Input ranges are validated
4. **Reports**: Download predictions as CSV
5. **API**: Flask app provides REST API

## 🐛 Troubleshooting

**Model not loading?**
- Ensure `survey-data.csv` is in same directory
- Check file permissions

**Port already in use?**
- Streamlit: `streamlit run weight_predictor_app.py --server.port 8502`
- Flask: Change port in code: `app.run(port=5001)`

**Dependencies error?**
- Run: `pip install -r requirements_app.txt`

## 📚 Documentation

For more details, see the main analysis files:
- `ml_analysis_with_pdf.py` - Complete ML analysis
- `reports/pdf/` - Generated PDF reports

---

**Created with Python, Streamlit, Flask, and scikit-learn**
