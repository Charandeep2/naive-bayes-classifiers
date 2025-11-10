# Naive Bayes Classifier Web Application

A modern, interactive web application for training and using Naive Bayes classifiers with beautiful visualizations.

## Features

- 🎨 **Modern UI**: Beautiful, responsive interface with gradient designs
- 📊 **Interactive Visualizations**: Powered by Plotly.js for dynamic charts
- 📁 **CSV Upload**: Easy dataset upload via CSV files
- ⌨️ **Manual Input**: Enter data manually through a user-friendly interface
- 🔮 **Real-time Predictions**: Make predictions with detailed probability breakdowns
- 📈 **Comprehensive Analytics**: View class distributions, feature analysis, and performance metrics

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5000`

## Usage

### Training a Model

#### Option 1: CSV Upload
1. Click on the "Upload CSV" tab
2. Select your CSV file
3. (Optional) Specify the target column name (defaults to last column)
4. Click "Upload & Train"

#### Option 2: Manual Input
1. Click on the "Manual Input" tab
2. Enter the number of features and feature names
3. Enter your data in the format: `feature1,feature2,...,featureN,label`
4. Click "Train Model"

### Making Predictions

1. After training, navigate to the "Make Predictions" section
2. Enter test data values for each feature
3. Click "Add Sample" to add more test cases
4. Click "Make Predictions" to see results
5. View detailed probability breakdowns for each class

### Visualizations

The application automatically generates:
- Class distribution pie charts
- Feature distribution histograms
- Categorical probability bars
- Confusion matrices
- Performance metrics

## Project Structure

```
.
├── app.py              # Flask application
├── model.py            # Naive Bayes implementation
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css  # Custom styles
│   └── js/
│       └── main.js    # Frontend JavaScript
└── uploads/           # CSV upload directory
```

## Technologies

- **Backend**: Flask (Python)
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Visualizations**: Plotly.js
- **ML**: Custom Naive Bayes implementation (Gaussian + Multinomial)

## CSV Format

Your CSV file should have:
- Feature columns
- One target/label column (can be specified or defaults to last column)

Example:
```csv
Age,Income,Education,Outcome
25,50000,Bachelor,Success
30,60000,Masters,Success
22,30000,High School,Failure
```

## License

MIT License - Feel free to use and modify!

