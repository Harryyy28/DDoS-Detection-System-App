# 🛡️ DDoS Detection System

A Full-Stack Python web application that uses Machine Learning (Random Forest & SVM) to analyze network traffic and predict whether the traffic is `BENIGN` or a `DDoS` attack.

## 🌟 Features
* **Machine Learning Models:** Powered by highly accurate Random Forest and Support Vector Machine algorithms.
* **Beautiful Web Interface:** Built with Flask, HTML, CSS, and Bootstrap for a modern, easy-to-use upload screen.
* **Data Visualization:** Automatically generates Pie Charts and Bar Charts to visually break down the predictions.
* **Downloadable Results:** Instantly splits the results and allows you to download clean CSV files for both BENIGN and DDoS traffic.

## 📁 Project Structure
* `App.py`: The main Flask Web Server and backend logic.
* `model_train.py`: The script used to train the AI models on the dataset.
* `requirements.txt`: The list of Python libraries needed to run the app.
* `templates/`: Contains the frontend HTML files (`ddos.html` and `results.html`).

---

## 🚀 How to Run the Project on Your Computer

### Step 1: Extract the Dataset
I have included the massive network traffic dataset as a compressed ZIP file in this repository to save space. Before doing anything else, you must **extract/unzip** the `Friday-WorkingHours...zip` file so the original `.csv` file is sitting in your main folder.

### Step 2: Install Dependencies
Open your terminal and install the required Python libraries:
```bash
pip install -r requirements.txt
