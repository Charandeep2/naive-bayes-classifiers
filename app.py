"""
Flask Web Application for Naive Bayes Classifier
Modern UI with interactive visualizations
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import json
import os
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.utils
from model import NaiveBayes
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model storage
trained_model = None
training_data = None
feature_names = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Handle CSV file upload"""
    global trained_model, training_data, feature_names
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Get target column
            if not target_column or target_column not in df.columns:
                target_column = df.columns[-1]
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col != target_column]
            X = df[feature_cols].values
            y = df[target_column].values
            
            # Train model
            nb = NaiveBayes()
            nb.fit(X, y, feature_names=feature_cols)
            
            trained_model = nb
            training_data = {'X': X.tolist(), 'y': y.tolist()}
            feature_names = feature_cols
            
            return jsonify({
                'success': True,
                'message': 'Dataset uploaded and model trained successfully',
                'data': {
                    'n_samples': len(X),
                    'n_features': len(feature_cols),
                    'feature_names': feature_cols,
                    'classes': nb.classes.tolist(),
                    'feature_types': nb.feature_types
                }
            })
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual-data', methods=['POST'])
def manual_data():
    """Handle manual data input"""
    global trained_model, training_data, feature_names
    
    try:
        data = request.json
        rows = data.get('rows', [])
        feature_names_input = data.get('feature_names', [])
        
        if not rows:
            return jsonify({'error': 'No data provided'}), 400
        
        # Parse data
        X_list = []
        y_list = []
        
        for row in rows:
            if len(row) < 2:
                continue
            features = row[:-1]
            label = row[-1]
            X_list.append(features)
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Use provided feature names or generate defaults
        if not feature_names_input or len(feature_names_input) != X.shape[1]:
            feature_names_input = [f'Feature_{i+1}' for i in range(X.shape[1])]
        
        # Train model
        nb = NaiveBayes()
        nb.fit(X, y, feature_names=feature_names_input)
        
        trained_model = nb
        training_data = {'X': X.tolist(), 'y': y.tolist()}
        feature_names = feature_names_input
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'data': {
                'n_samples': len(X),
                'n_features': len(feature_names_input),
                'feature_names': feature_names_input,
                'classes': nb.classes.tolist(),
                'feature_types': nb.feature_types
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on test data"""
    global trained_model
    
    if trained_model is None:
        return jsonify({'error': 'No model trained yet. Please upload or enter training data first.'}), 400
    
    try:
        data = request.json
        test_rows = data.get('test_data', [])
        
        if not test_rows:
            return jsonify({'error': 'No test data provided'}), 400
        
        # Convert to numpy array
        test_X = np.array(test_rows)
        
        # Make predictions
        predictions = trained_model.predict(test_X)
        probabilities = trained_model.predict_proba(test_X)
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                'sample': i + 1,
                'prediction': str(pred),
                'probabilities': {str(k): float(v) for k, v in probs.items()},
                'confidence': float(max(probs.values()))
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and statistics"""
    global trained_model
    
    if trained_model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    try:
        info = {
            'classes': trained_model.classes.tolist(),
            'class_probs': {str(k): float(v) for k, v in trained_model.class_probs.items()},
            'feature_names': trained_model.feature_names,
            'feature_types': trained_model.feature_types,
            'n_features': trained_model.n_features,
            'feature_params': {}
        }
        
        # Add feature parameters
        for cls in trained_model.classes:
            info['feature_params'][str(cls)] = {}
            for feat_idx in range(trained_model.n_features):
                params = trained_model.feature_params[cls][feat_idx]
                if params['type'] == 'numerical':
                    info['feature_params'][str(cls)][feat_idx] = {
                        'type': 'numerical',
                        'mean': float(params['mean']),
                        'std': float(params['std'])
                    }
                else:
                    info['feature_params'][str(cls)][feat_idx] = {
                        'type': 'categorical',
                        'prob_dist': {k: float(v) for k, v in params['prob_dist'].items()}
                    }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """Generate visualization data"""
    global trained_model, training_data
    
    if trained_model is None or training_data is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    try:
        X = np.array(training_data['X'])
        y = np.array(training_data['y'])
        classes = trained_model.classes
        
        # Generate visualization data
        viz_data = {}
        
        # 1. Class distribution (pie chart)
        class_counts = {str(cls): int(np.sum(y == cls)) for cls in classes}
        viz_data['class_distribution'] = class_counts
        
        # 2. Feature distributions for numerical features
        numerical_features = [i for i, ft in enumerate(trained_model.feature_types) if ft == 'numerical']
        viz_data['numerical_features'] = {}
        
        for feat_idx in numerical_features:
            feat_name = trained_model.feature_names[feat_idx]
            viz_data['numerical_features'][feat_idx] = {
                'name': feat_name,
                'classes': {}
            }
            
            for cls in classes:
                cls_mask = y == cls
                cls_feature_values = pd.to_numeric(X[cls_mask, feat_idx], errors='coerce')
                cls_feature_values = cls_feature_values[~np.isnan(cls_feature_values)]
                
                params = trained_model.feature_params[cls][feat_idx]
                viz_data['numerical_features'][feat_idx]['classes'][str(cls)] = {
                    'data': cls_feature_values.tolist(),
                    'mean': float(params['mean']),
                    'std': float(params['std'])
                }
        
        # 3. Categorical features
        categorical_features = [i for i, ft in enumerate(trained_model.feature_types) if ft == 'categorical']
        viz_data['categorical_features'] = {}
        
        for feat_idx in categorical_features:
            feat_name = trained_model.feature_names[feat_idx]
            viz_data['categorical_features'][feat_idx] = {
                'name': feat_name,
                'classes': {}
            }
            
            for cls in classes:
                params = trained_model.feature_params[cls][feat_idx]
                viz_data['categorical_features'][feat_idx]['classes'][str(cls)] = {
                    'prob_dist': {k: float(v) for k, v in params['prob_dist'].items()}
                }
        
        # 4. Model performance (if we have predictions)
        try:
            predictions = trained_model.predict(X)
            accuracy = float(trained_model.score(X, y))
            
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y, predictions)
            viz_data['performance'] = {
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'class_labels': [str(c) for c in classes]
            }
        except:
            pass
        
        return jsonify(viz_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-datasets', methods=['GET'])
def get_sample_datasets():
    """Get list of available sample datasets"""
    return jsonify({
        'datasets': [
            {
                'id': 'weather',
                'name': 'Weather Classification',
                'description': 'Predict weather condition based on outlook, temperature, humidity, and wind',
                'feature_names': ['Outlook', 'Temperature', 'Humidity', 'Wind'],
                'target': 'PlayTennis',
                'data': [
                    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
                    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
                    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
                    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
                    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
                    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
                    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
                    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
                    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
                    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
                    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
                    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
                    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
                    ['Rain', 'Mild', 'High', 'Strong', 'No']
                ]
            },
            {
                'id': 'email',
                'name': 'Email Spam Detection',
                'description': 'Classify emails as spam or ham based on word features',
                'feature_names': ['Free', 'Money', 'Urgent', 'Meeting'],
                'target': 'Class',
                'data': [
                    ['Yes', 'Yes', 'No', 'No', 'Spam'],
                    ['Yes', 'No', 'No', 'Yes', 'Spam'],
                    ['No', 'No', 'Yes', 'Yes', 'Spam'],
                    ['No', 'No', 'Yes', 'No', 'Spam'],
                    ['Yes', 'Yes', 'No', 'Yes', 'Spam'],
                    ['No', 'Yes', 'Yes', 'Yes', 'Spam'],
                    ['No', 'No', 'No', 'No', 'Ham'],
                    ['No', 'No', 'No', 'Yes', 'Ham'],
                    ['No', 'No', 'Yes', 'Yes', 'Ham'],
                    ['No', 'No', 'No', 'No', 'Ham'],
                    ['Yes', 'No', 'No', 'No', 'Ham'],
                    ['No', 'No', 'Yes', 'No', 'Ham'],
                    ['Yes', 'No', 'Yes', 'Yes', 'Spam'],
                    ['No', 'Yes', 'No', 'Yes', 'Spam'],
                    ['Yes', 'Yes', 'Yes', 'No', 'Spam'],
                    ['No', 'No', 'No', 'Yes', 'Ham'],
                    ['Yes', 'No', 'No', 'Yes', 'Ham'],
                    ['No', 'Yes', 'Yes', 'No', 'Ham'],
                    ['No', 'No', 'Yes', 'Yes', 'Ham']
                ]
            },
            {
                'id': 'customer',
                'name': 'Customer Purchase Prediction',
                'description': 'Predict if customer will purchase based on demographics and behavior',
                'feature_names': ['Age_Group', 'Gender', 'Income_Level', 'Previous_Purchase'],
                'target': 'Purchase',
                'data': [
                    ['Young', 'Male', 'Low', 'No', 'No'],
                    ['Young', 'Male', 'Low', 'Yes', 'No'],
                    ['Young', 'Female', 'Low', 'No', 'Yes'],
                    ['Young', 'Female', 'Low', 'Yes', 'Yes'],
                    ['Young', 'Male', 'Medium', 'No', 'No'],
                    ['Young', 'Male', 'Medium', 'Yes', 'Yes'],
                    ['Young', 'Female', 'Medium', 'No', 'Yes'],
                    ['Young', 'Female', 'Medium', 'Yes', 'Yes'],
                    ['Middle', 'Male', 'Low', 'No', 'No'],
                    ['Middle', 'Male', 'Low', 'Yes', 'Yes'],
                    ['Middle', 'Female', 'Low', 'No', 'Yes'],
                    ['Middle', 'Female', 'Low', 'Yes', 'Yes'],
                    ['Middle', 'Male', 'Medium', 'No', 'Yes'],
                    ['Middle', 'Male', 'Medium', 'Yes', 'Yes'],
                    ['Middle', 'Female', 'Medium', 'No', 'Yes'],
                    ['Middle', 'Female', 'Medium', 'Yes', 'Yes'],
                    ['Middle', 'Male', 'High', 'No', 'Yes'],
                    ['Middle', 'Male', 'High', 'Yes', 'Yes'],
                    ['Middle', 'Female', 'High', 'No', 'Yes'],
                    ['Middle', 'Female', 'High', 'Yes', 'Yes'],
                    ['Old', 'Male', 'Medium', 'No', 'No'],
                    ['Old', 'Male', 'Medium', 'Yes', 'Yes'],
                    ['Old', 'Female', 'Medium', 'No', 'Yes'],
                    ['Old', 'Female', 'Medium', 'Yes', 'Yes'],
                    ['Old', 'Male', 'High', 'No', 'Yes'],
                    ['Old', 'Male', 'High', 'Yes', 'Yes'],
                    ['Old', 'Female', 'High', 'No', 'Yes'],
                    ['Old', 'Female', 'High', 'Yes', 'Yes']
                ]
            }
        ]
    })

@app.route('/api/load-sample', methods=['POST'])
def load_sample_dataset():
    """Load and train a sample dataset"""
    global trained_model, training_data, feature_names
    
    try:
        data = request.json
        dataset_id = data.get('dataset_id')
        
        # Get sample datasets
        datasets_response = get_sample_datasets()
        datasets = datasets_response.get_json()['datasets']
        
        # Find the requested dataset
        dataset = next((d for d in datasets if d['id'] == dataset_id), None)
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Extract data
        rows = dataset['data']
        X_list = []
        y_list = []
        
        for row in rows:
            if len(row) < len(dataset['feature_names']) + 1:
                continue
            features = row[:-1]
            label = row[-1]
            X_list.append(features)
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        feature_names_list = dataset['feature_names']
        
        # Train model - This is the key training step
        nb = NaiveBayes()
        print(f"\n[TRAINING] Training Naive Bayes model on '{dataset['name']}' dataset...")
        print(f"  - Samples: {len(X)}")
        print(f"  - Features: {len(feature_names_list)}")
        print(f"  - Classes: {np.unique(y)}")
        
        # Fit/train the model
        nb.fit(X, y, feature_names=feature_names_list)
        
        # Calculate training accuracy
        training_accuracy = nb.score(X, y)
        print(f"  - Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
        print(f"[TRAINING] Model trained successfully!\n")
        
        trained_model = nb
        training_data = {'X': X.tolist(), 'y': y.tolist()}
        feature_names = feature_names_list
        
        return jsonify({
            'success': True,
            'message': f"Sample dataset '{dataset['name']}' loaded and trained successfully",
            'data': {
                'n_samples': len(X),
                'n_features': len(feature_names_list),
                'feature_names': feature_names_list,
                'classes': nb.classes.tolist(),
                'feature_types': nb.feature_types,
                'dataset_name': dataset['name'],
                'description': dataset['description'],
                'dataset_rows': rows,  # Include the actual dataset
                'target_column': dataset['target'],
                'training_accuracy': float(training_accuracy)  # Include training accuracy
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample-data')
def sample_data():
    """Serve sample CSV file"""
    return send_from_directory('.', 'sample_data.csv')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Naive Bayes Classifier Web Application")
    print("="*70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

