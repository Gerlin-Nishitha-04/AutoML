import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, r2_score, mean_absolute_error, mean_squared_error,
                             confusion_matrix, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, silhouette_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os
import sys
import io
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

def run_automl(file, target_column, vis_choices):
    df = pd.read_csv(file)
    print("\n✅ File loaded successfully!")
    print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\n\U0001F4C4 Dataset Preview:")
    print(df.head())

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    metrics = {}
    is_supervised = target_column != ""

    if is_supervised:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        task_type = "CLASSIFICATION" if y.nunique() < 20 and y.dtype in [np.int64, np.int32, np.object_] else "REGRESSION"
        print(f"\n🔍 Detected Task Type: {task_type}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        best_model = None
        best_score = -np.inf

        if task_type == "CLASSIFICATION":
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                print(f"{name} Accuracy: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    metrics = {
                        "Accuracy": score,
                        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
                        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
                        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=1)
                    }
                    best_cm = confusion_matrix(y_test, y_pred)

        elif task_type == "REGRESSION":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                print(f"{name} R² Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    metrics = {
                        "R² Score": score,
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                    }

        if best_model:
            print(f"\nSaving the best model ({type(best_model).__name__})...")
            joblib.dump(best_model, "best_model.pkl")
            print("Best model saved as 'best_model.pkl'")

    else:
        task_type = "CLUSTERING"
        print(f"\n🔍 Detected Task Type: {task_type}")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        df['Cluster'] = labels
        silhouette = silhouette_score(scaled_data, labels)
        metrics["Silhouette Score"] = silhouette
        print(f"\n✅ Clustering complete! Silhouette Score: {silhouette:.4f}")
        print("\n\U0001F4C4 Clustered Data Preview:")
        print(df.head())
        best_model = kmeans

    vis_images = []
    if vis_choices:
        if '1' in vis_choices:
            print("Generating Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            heatmap_path = "heatmap.png"
            plt.savefig(heatmap_path)
            plt.close()
            vis_images.append(("Correlation Heatmap", heatmap_path))
            print(f"Saved {heatmap_path}")

        if '2' in vis_choices and task_type == "CLASSIFICATION":
            print("Generating Confusion Matrix")
            disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=np.unique(y))
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            vis_images.append(("Confusion Matrix", cm_path))
            print(f"Saved {cm_path}")

        if '2' in vis_choices and task_type == "REGRESSION":
            print("Generating Actual vs Predicted")
            y_pred = best_model.predict(X_test)
            plt.scatter(y_test, y_pred)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            reg_path = "actual_vs_predicted.png"
            plt.savefig(reg_path)
            plt.close()
            vis_images.append(("Actual vs Predicted", reg_path))
            print(f"Saved {reg_path}")

        if '2' in vis_choices and task_type == "CLUSTERING":
            print("Generating Cluster Visualization")
            reduced = PCA(n_components=2).fit_transform(scaled_data)
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis')
            plt.title("Cluster Visualization")
            cluster_path = "cluster.png"
            plt.savefig(cluster_path)
            plt.close()
            vis_images.append(("Cluster Visualization", cluster_path))
            print(f"Saved {cluster_path}")

        if '3' in vis_choices and task_type in ["CLASSIFICATION", "REGRESSION"]:
            print("Generating Pairplot")
            pairplot_path = "pairplot.png"
            sns.pairplot(df, hue=target_column if target_column in df.columns else None)
            plt.savefig(pairplot_path)
            plt.close()
            vis_images.append(("Pairplot", pairplot_path))
            print(f"Saved {pairplot_path}")

    styles = getSampleStyleSheet()
    elements = []
    report = SimpleDocTemplate("AutoML_Report.pdf", pagesize=letter)
    elements.append(Paragraph("<b>Summary Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Task Type:</b> {task_type}", styles['Heading2']))
    elements.append(Paragraph(f"<b>Best Model:</b> {type(best_model).__name__}" if best_model else "Best Model: None", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Dataset Overview:</b>", styles['Heading2']))

    data = [df.columns.tolist()] + df.head().values.tolist()
    table = Table(data)
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Evaluation Metrics:</b>", styles['Heading2']))
    for k, v in metrics.items():
        elements.append(Paragraph(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A", styles['Normal']))
    elements.append(Spacer(1, 12))

    if vis_images:
        elements.append(Paragraph("<b>Visualizations:</b>", styles['Heading2']))
        for title, img_path in vis_images:
            elements.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            elements.append(Image(img_path, width=400, height=300))
            elements.append(Spacer(1, 12))
        print(f"Added {len(vis_images)} visualizations to PDF")
    else:
        print("No visualizations to add to PDF")

    report.build(elements)
    print("\n✅ Report generated successfully as 'AutoML_Report.pdf'")
    return "Report generated successfully as 'AutoML_Report.pdf'"

@app.route('/api/detect-task', methods=['POST'])
def detect_task():
    file = request.files.get('file')
    target_column = request.form.get('targetColumn', '')

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        if target_column != "":
            y = df[target_column]
            task_type = "CLASSIFICATION" if y.nunique() < 20 and y.dtype in [np.int64, np.int32, np.object_] else "REGRESSION"
        else:
            task_type = "CLUSTERING"

        return jsonify({"task_type": task_type}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-automl', methods=['POST'])
def run_automl_endpoint():
    file = request.files.get('file')
    target_column = request.form.get('targetColumn', '')
    vis_choices = json.loads(request.form.get('visualizations', '[]'))
    print(f"Received vis_choices: {vis_choices}")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        result = run_automl(file, target_column, vis_choices)
        return jsonify({"message": result, "report": "AutoML_Report.pdf"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-report', methods=['GET'])
def download_report():
    print(f"Checking for AutoML_Report.pdf in {os.getcwd()}")
    if not os.path.exists("AutoML_Report.pdf"):
        print("File not found!")
        return jsonify({"error": "Report not found"}), 404
    try:
        return send_file("AutoML_Report.pdf", as_attachment=True)
    except Exception as e:
        print(f"Error sending file: {e}")
        return jsonify({"error": "Report not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)