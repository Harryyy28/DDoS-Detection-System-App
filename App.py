import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import concurrent.futures
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super_secret_key' # Change in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MODEL_FOLDER'] = 'model'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def predict_svm_chunk(model, chunk):
    return model.predict(chunk)

def predict_svm_parallel(model, data, n_jobs=10):
    """Predicts using SVM in parallel using ThreadPoolExecutor."""
    chunks = np.array_split(data, n_jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda c: predict_svm_chunk(model, c), chunks))
    return np.concatenate(results)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle File Upload
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process Data
                df = pd.read_csv(filepath)
                
                # Attempt to find true label column for accuracy metrics
                label_col = next((col for col in df.columns if 'Label' in col or 'label' in col), None)
                
                true_labels = None
                if label_col:
                    true_labels = df[label_col].apply(lambda x: 1 if 'DDoS' in str(x) else 0)
                    df_features = df.drop(columns=[label_col])
                else:
                    df_features = df.copy()

                # Extract exactly 51 network flow features (numeric)
                numeric_df = df_features.select_dtypes(include=[np.number])
                if numeric_df.shape[1] >= 51:
                    features = numeric_df.iloc[:, :51]
                else:
                    flash('CSV does not contain enough numeric features (51 required).', 'danger')
                    return redirect(request.url)
                
                # Clean Data
                features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Scale Data
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(features)
                
                # Load Models
                svm_path = os.path.join(app.config['MODEL_FOLDER'], 'svm_model.pkl')
                rf_path = os.path.join(app.config['MODEL_FOLDER'], 'random_forest_model.pkl')
                
                if not os.path.exists(svm_path) or not os.path.exists(rf_path):
                    flash('Pre-trained models not found! Please place svm_model.pkl and random_forest_model.pkl in the model/ directory.', 'warning')
                    return redirect(request.url)
                    
                svm_model = joblib.load(svm_path)
                rf_model = joblib.load(rf_path)
                
                # Predict
                rf_pred = rf_model.predict(scaled_features)
                svm_pred = predict_svm_parallel(svm_model, scaled_features, n_jobs=10)
                
                # Combine predictions: if (rf_pred + svm_pred) >= 1, label it 'DDoS', else 'BENIGN'
                # Assuming models output 1 for DDoS and 0 for BENIGN
                final_pred_numeric = np.where((np.array(rf_pred) + np.array(svm_pred)) >= 1, 1, 0)
                final_labels = ['DDoS' if p == 1 else 'BENIGN' for p in final_pred_numeric]
                
                df['Prediction'] = final_labels
                
                # Calculate Metrics
                accuracy = None
                report = None
                if true_labels is not None:
                    accuracy = accuracy_score(true_labels, final_pred_numeric)
                    report = classification_report(true_labels, final_pred_numeric, target_names=['BENIGN', 'DDoS'], zero_division=0)
                else:
                    accuracy = "N/A (No true labels found in uploaded CSV)"
                    report = "N/A"
                
                # Visualizations
                benign_count = final_labels.count('BENIGN')
                ddos_count = final_labels.count('DDoS')
                
                # Pie Chart
                plt.figure(figsize=(6, 6))
                plt.pie([benign_count, ddos_count], labels=['BENIGN', 'DDoS'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
                plt.title('Prediction Distribution')
                pie_path = os.path.join(app.config['STATIC_FOLDER'], 'prediction_pie_chart.png')
                plt.savefig(pie_path)
                plt.close()
                
                # Bar Chart
                plt.figure(figsize=(6, 6))
                plt.bar(['BENIGN', 'DDoS'], [benign_count, ddos_count], color=['#4CAF50', '#F44336'])
                plt.title('Prediction Counts')
                plt.ylabel('Count')
                bar_path = os.path.join(app.config['STATIC_FOLDER'], 'prediction_bar_chart.png')
                plt.savefig(bar_path)
                plt.close()
                
                # Split CSVs and save
                benign_df = df[df['Prediction'] == 'BENIGN']
                ddos_df = df[df['Prediction'] == 'DDoS']
                
                benign_csv_path = os.path.join(app.config['STATIC_FOLDER'], 'benign_results.csv')
                ddos_csv_path = os.path.join(app.config['STATIC_FOLDER'], 'ddos_results.csv')
                
                benign_df.to_csv(benign_csv_path, index=False)
                ddos_df.to_csv(ddos_csv_path, index=False)
                
                return render_template('results.html', accuracy=accuracy, report=report)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload a CSV.', 'danger')
            return redirect(request.url)
            
    return render_template('ddos.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)