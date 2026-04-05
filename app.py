from flask import Flask, render_template, request,make_response
import pandas as pd
import os
from werkzeug.utils import secure_filename
import time
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from flask import send_file
import pdfkit
import plotly.io as pio

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_FOLDER = "models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

df = None

# ===============================
# DATA CLEANING
# ===============================

def clean_data(df):

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].astype(str)

            df[col] = df[col].str.replace(",", "", regex=True)
            df[col] = df[col].str.replace("kms", "", regex=True)
            df[col] = df[col].str.replace("km", "", regex=True)
            df[col] = df[col].str.replace("₹", "", regex=True)
            df[col] = df[col].str.strip()

            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    # Fill missing values
    for col in df.columns:

        if df[col].dtype == "object":

            df[col].fillna(df[col].mode()[0], inplace=True)

        else:

            df[col].fillna(df[col].median(), inplace=True)

    return df


# ===============================
# EDA GRAPHS
# ===============================

def generate_graphs(df):

    graphs = []
    df = df.sample(min(len(df), 5000))
    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    heatmap_path = "static/heatmap.png"
    plt.title("Heat Map")
    plt.savefig(heatmap_path)
    plt.close()
    graphs.append(heatmap_path)

    # Plotly histogram (interactive)
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] > 0:

        fig = px.histogram(numeric_df)

        plot_path = "static/histogram.html"

        fig.write_html(plot_path)

        graphs.append(plot_path)

    # Missing value plot
    # plt.figure(figsize=(8,5))
    # df.isnull().sum().plot(kind='bar')
    # plt.title("Missing Values")
    # missing_path = "static/missing.png"
    # plt.savefig(missing_path)
    # plt.close()

    # graphs.append(missing_path)

    return graphs

# ===============================
# DATA SUMMARY
# ===============================

def generate_eda(df):

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

    return summary


# ===============================
# HOME PAGE
# ===============================

@app.route("/")
def home():

    return render_template("index.html")


# ===============================
# DATASET UPLOAD
# ===============================

@app.route("/upload", methods=["POST"])
def upload():

    global df

    file = request.files["dataset"]

    if file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(path)

    df = pd.read_csv(path)

    summary = generate_eda(df)

    graphs = generate_graphs(df)

    return render_template(
        "eda.html",
        tables=[df.head().to_html(classes="table-auto w-full border border-gray-300 text-sm text-black-500", border=0)],
        summary=summary,
        graphs=graphs
    )


# ===============================
# MODEL TRAINING
# ===============================
summary = None
global_results = None
global_best_model = None
global_cm = None
global_roc = None
global_residual = None
importance_path = None
global_chart_path=None
@app.route("/train", methods=["POST"])
def train():

    global df

    if df is None:
        return "No dataset uploaded"

    target = request.form["target"]

    if target not in df.columns:
        return "Invalid target column"

    df_clean = clean_data(df.copy())
    X = df_clean.drop(target, axis=1)
    y = df_clean[target]

    X = pd.get_dummies(X, drop_first=True)
    columns = df_clean.columns.tolist()

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y   
    )

    # ===============================
    # TRAIN MODELS
    # ===============================

    results = {}
    models = {}

    if y.nunique() < 20:

        task = "Classification"

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=50),
            "LogisticRegression": LogisticRegression(max_iter=500),
            "DecisionTree": DecisionTreeClassifier()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results[name] = round(acc, 3)

    else:

        task = "Regression"

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=50),
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            results[name] = round(score, 3)

    # ===============================
    # BEST MODEL
    # ===============================

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")

    preds = best_model.predict(X_test)

    # ===============================
    # EVALUATION
    # ===============================

    cm_path = None
    roc_path = None
    residual_path = None

    if task == "Classification":

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")

        cm_path = "static/confusion.png"
        plt.savefig(cm_path)
        plt.close()

        # ROC CURVE (ONLY BINARY)
        from sklearn.preprocessing import label_binarize

        classes = np.unique(y_test)

        if hasattr(best_model, "predict_proba") and len(classes) > 1:

            probs = best_model.predict_proba(X_test)

            plt.figure()

            # 🔥 CASE 1: BINARY (SPECIAL FIX)
            if len(classes) == 2:

                fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                plt.plot(fpr, tpr, label="ROC Curve")

            # 🔥 CASE 2: MULTICLASS
            else:

                y_test_bin = label_binarize(y_test, classes=classes)

                for i in range(len(classes)):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                    plt.plot(fpr, tpr, label=f"Class {classes[i]}")

            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve")
            plt.legend()

            roc_path = "static/roc.png"
            plt.savefig(roc_path)
            plt.close()

        else:
            print("Skipping ROC: invalid condition")
    else:

        # RESIDUAL PLOT
        residuals = y_test - preds

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=preds, y=residuals)

        plt.axhline(0, color='red', linestyle='--')

        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")

        residual_path = "static/residual.png"
        plt.savefig(residual_path)
        plt.close()

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================

    importance_path=None

    if hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_
        features = X.columns

        plt.figure(figsize=(8, 4))
        sns.barplot(x=importance, y=features, palette="viridis")

        plt.title("Feature Importance")

        importance_path = "static/importance.png"
        plt.savefig(importance_path)
        plt.close()

    elif hasattr(best_model, "coef_"):

        importance = abs(best_model.coef_[0])
        features = X.columns

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance, y=features)

        plt.title("Feature Importance")

        importance_path = "static/importance.png"
        plt.savefig(importance_path)
        plt.close()

    # ===============================
    # MODEL COMPARISON CHART
    # ===============================

    model_names = list(results.keys())
    scores = list(results.values())

    fig = px.bar(
        x=model_names,
        y=scores,
        text=scores,
        color=scores,
        color_continuous_scale="viridis",
        title="Model Performance Comparison"
    )

    fig.update_traces(textposition='outside')


    chart_path = "static/model_comparison.png"
    pio.write_image(fig, chart_path)
    # ===============================
    # RETURN RESULTS
    # ===============================
    global summary, global_results, global_best_model, global_cm, global_roc, global_residual,global_importance,global_chart_path


    summary = generate_eda(df)
    global_results = results
    global_best_model = best_model_name
    global_cm = cm_path
    global_roc = roc_path
    global_residual = residual_path
    global_importance=importance_path
    global_chart_path=chart_path

    print("Importance path:", global_importance)

    return render_template(
        "results.html",
        results=results,
        best_model=best_model_name,
        importance=importance_path,
        chart=chart_path,
        cm=cm_path,
        roc=roc_path,
        residual=residual_path,
        task=task,
        columns=columns
    )

@app.route("/download_model")
def download_model():

    return send_file(
        "models/best_model.pkl",
        as_attachment=True
    )


@app.route("/predict")
def predict_page():

    features = joblib.load("models/features.pkl")

    return render_template("predict.html", features=features)


@app.route("/predict_result", methods=["POST"])
def predict_result():

    model = joblib.load("models/best_model.pkl")
    features = joblib.load("models/features.pkl")

    input_data = []

    for feature in features:
        val = request.form.get(feature)

        if val is None or val == "":
            val = 0

        input_data.append(float(val))

    prediction = model.predict([input_data])[0]

    return render_template(
        "predict.html",
        prediction=prediction,
        show_prediction=True,
        features=features
    )

@app.route("/explore")
def explore():

    global df

    if df is None:
        return "Upload dataset first"

    query = request.args.get("q")

    filtered_df = df.copy()

    if query:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: row.astype(str).str.contains(query, case=False).any(),
                axis=1
            )
        ]

    return render_template(
        "explore.html",
        tables=[filtered_df.head(50).to_html(classes="table-auto w-full bg-black-100 overflow-x-auto border rounded-lg text-black-500", border=0)],
        query=query
    )




import os

@app.route("/generate_report")
def generate_report():

    global summary, global_results, global_best_model
    global global_cm, global_roc, global_residual,global_importance

    if summary is None or global_results is None:
        return "Please train model first"

    insight = f"""
    Best model is {global_best_model} with score {max(global_results.values())}.
    """

    # Convert to absolute paths (VERY IMPORTANT)
    cm_path = os.path.abspath(global_cm) if global_cm else None
    roc_path = os.path.abspath(global_roc) if global_roc else None
    residual_path = os.path.abspath(global_residual) if global_residual else None
    importance_path = os.path.abspath(global_importance) if global_importance else None
    chart_path = os.path.abspath(global_chart_path) if global_chart_path else None

    html = render_template(
        "reports.html",
        summary=summary,
        results=global_results,
        best_model=global_best_model,
        insight=insight,
        cm=cm_path,
        roc=roc_path,
        residual=residual_path,
        importance=importance_path,
        chart=chart_path,
    )

    config = pdfkit.configuration(
        wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    )

    pdf = pdfkit.from_string(
        html,
        False,
        configuration=config,
        options={"enable-local-file-access": ""}
    )

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=report.pdf"

    return response

@app.route("/generate_plot", methods=["POST"])
def generate_plot():

    global df

    data = request.get_json()

    graph = data.get("graph")
    x = data.get("x")
    y = data.get("y")

    df_clean = clean_data(df.copy())

    import plotly.express as px

    try:
        # ✅ Validate columns
        if x not in df_clean.columns:
            return "<p style='color:red'>Invalid X column</p>"

        if graph != "histogram" and y not in df_clean.columns:
            return "<p style='color:red'>Invalid Y column</p>"

        # ✅ Generate plots
        elif graph == "scatter":
            fig = px.scatter(df_clean, x=x, y=y)

        elif graph == "bar":
            fig = px.bar(df_clean, x=x, y=y)

        elif graph == "line":
            fig = px.line(df_clean, x=x, y=y)

        elif graph == "histogram":
            fig = px.histogram(df_clean, x=x, nbins=30)

        else:
            return "<p style='color:red'>Invalid graph type</p>"

        # ✅ Styling (match your dark UI 🔥)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        return f"<p style='color:red'>Error: {str(e)}</p>"
# ===============================
# RUN APP
# ===============================

if __name__ == "__main__":

    app.run(debug=True)
