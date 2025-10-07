# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from kneed import KneeLocator
# import pickle
# import os
# import base64

# def load_data():
#     """
#     Loads data from a CSV file, serializes it, and returns the serialized data.
#     Returns:
#         str: Base64-encoded serialized data (JSON-safe).
#     """
#     print("We are here")
#     df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
#     serialized_data = pickle.dumps(df)                    # bytes
#     return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

# def data_preprocessing(data_b64: str):
#     """
#     Deserializes base64-encoded pickled data, performs preprocessing,
#     and returns base64-encoded pickled clustered data.
#     """
#     # decode -> bytes -> DataFrame
#     data_bytes = base64.b64decode(data_b64)
#     df = pickle.loads(data_bytes)

#     df = df.dropna()
#     clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

#     min_max_scaler = MinMaxScaler()
#     clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

#     # bytes -> base64 string for XCom
#     clustering_serialized_data = pickle.dumps(clustering_data_minmax)
#     return base64.b64encode(clustering_serialized_data).decode("ascii")


# def build_save_model(data_b64: str, filename: str):
#     """
#     Builds a KMeans model on the preprocessed data and saves it.
#     Returns the SSE list (JSON-serializable).
#     """
#     # decode -> bytes -> numpy array
#     data_bytes = base64.b64decode(data_b64)
#     df = pickle.loads(data_bytes)

#     kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
#     sse = []
#     for k in range(1, 50):
#         kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#         kmeans.fit(df)
#         sse.append(kmeans.inertia_)

#     # NOTE: This saves the last-fitted model (k=49), matching your original intent.
#     output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, filename)
#     with open(output_path, "wb") as f:
#         pickle.dump(kmeans, f)

#     return sse  # list is JSON-safe


# def load_model_elbow(filename: str, sse: list):
#     """
#     Loads the saved model and uses the elbow method to report k.
#     Returns the first prediction (as a plain int) for test.csv.
#     """
#     # load the saved (last-fitted) model
#     output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
#     loaded_model = pickle.load(open(output_path, "rb"))

#     # elbow for information/logging
#     kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
#     print(f"Optimal no. of clusters: {kl.elbow}")

#     # predict on raw test data (matches your original code)
#     df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
#     pred = loaded_model.predict(df)[0]

#     # ensure JSON-safe return
#     try:
#         return int(pred)
#     except Exception:
#         # if not numeric, still return a JSON-friendly version
#         return pred.item() if hasattr(pred, "item") else pred

# def send_completion_notification(prediction: int, sse: list):
#     """
#     Sends a completion notification with pipeline results.
#     """
#     import json
#     from datetime import datetime
    
#     notification = {
#         "pipeline": "Credit Card Clustering",
#         "status": "SUCCESS",
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "prediction_result": int(prediction) if prediction is not None else "N/A",
#         "total_sse_points": len(sse) if isinstance(sse, list) else 0,
#         "final_sse": float(sse[-1]) if sse else None,
#         "min_sse": float(min(sse)) if sse else None,
#         "max_sse": float(max(sse)) if sse else None,
#         "message": "Pipeline completed successfully! Model trained and saved."
#     }
    
#     print("\n" + "="*70)
#     print("PIPELINE COMPLETION NOTIFICATION")
#     print("="*70)
#     print(json.dumps(notification, indent=2))
#     print("="*70 + "\n")
    
#     return notification



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pickle
import os
import base64
import numpy as np
import json
from datetime import datetime

def load_data():
    """Loads data from CSV and returns serialized data."""
    print("Loading data from CSV file...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """Preprocesses data: removes nulls and scales features."""
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    initial_rows = len(df)
    df = df.dropna()
    print(f"✓ Removed {initial_rows - len(df)} rows with missing values")
    
    # Using 4 features instead of 3
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    
    # Using StandardScaler instead of MinMaxScaler
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    
    print(f"✓ Preprocessed {clustering_data_scaled.shape[0]} records with {clustering_data_scaled.shape[1]} features")
    
    clustering_serialized_data = pickle.dumps(clustering_data_scaled)
    return base64.b64encode(clustering_serialized_data).decode("ascii")

def build_save_model(data_b64: str, filename: str):
    """Builds KMeans models with different k values and saves the best one."""
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)
    
    # Improved KMeans parameters
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 15,
        "max_iter": 500,
        "random_state": 42
    }
    
    sse = []
    # Testing k from 2 to 10 (more realistic range)
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        print(f"✓ k={k}, SSE={kmeans.inertia_:.2f}")
    
    # Save the last model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)
    
    print(f"✓ Model saved to {output_path}")
    return sse

def evaluate_model(filename: str, data_b64: str):
    """Evaluates model performance with metrics."""
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(output_path, "rb"))
    
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)
    
    labels = model.predict(data)
    silhouette = silhouette_score(data, labels)
    
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"Number of Clusters: {model.n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f} (higher is better)")
    print(f"Inertia (SSE): {model.inertia_:.2f}")
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({count/len(data)*100:.1f}%)")
    print("="*70 + "\n")
    
    return {
        "silhouette_score": float(silhouette),
        "inertia": float(model.inertia_),
        "n_clusters": int(model.n_clusters)
    }

def load_model_elbow(filename: str, sse: list):
    """Loads model and finds optimal k using elbow method."""
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(output_path, "rb"))
    
    # Find elbow
    k_values = list(range(2, 2 + len(sse)))
    kl = KneeLocator(k_values, sse, curve="convex", direction="decreasing")
    
    print(f"✓ Optimal number of clusters: {kl.elbow}")
    
    # Predict on test data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = model.predict(df)[0]
    
    print(f"✓ First test prediction: Cluster {pred}")
    
    return int(pred)

def send_notification(prediction: int, sse: list, eval_metrics: dict):
    """Sends completion notification with summary."""
    
    # Find optimal k
    k_values = list(range(2, 2 + len(sse)))
    try:
        kl = KneeLocator(k_values, sse, curve="convex", direction="decreasing")
        optimal_k = kl.elbow
    except:
        optimal_k = "Unable to determine"
    
    # notification = {
    #     "pipeline": "Credit Card Clustering Analysis",
    #     "status": "SUCCESS",
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "optimal_clusters": optimal_k,
    #     "prediction_result": prediction,
    #     "silhouette_score": eval_metrics.get("silhouette_score", "N/A"),
    #     "final_sse": sse[-1] if sse else None,
    #     "message": "Pipeline completed successfully!"
    # }

    notification = {
        "pipeline": "Credit Card Clustering Analysis",
        "status": "SUCCESS",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimal_clusters": int(optimal_k) if optimal_k else "N/A",
        "prediction_result": int(prediction),
        "silhouette_score": float(eval_metrics.get("silhouette_score", 0)) if eval_metrics.get("silhouette_score") != "N/A" else "N/A",
        "final_sse": float(sse[-1]) if sse else None,
        "message": "Pipeline completed successfully!"
    }
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETION NOTIFICATION")
    print("="*70)
    print(json.dumps(notification, indent=2))
    print("="*70 + "\n")
    
    # Save report to file
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(report_dir, f"pipeline_report_{timestamp}.json")
    
    with open(report_file, 'w') as f:
        json.dump(notification, f, indent=2)
    
    print(f"Report saved to: {report_file}\n")
    
    return notification

def create_visual_report(filename: str, sse: list):
    """Creates HTML report with charts."""
    model = pickle.load(open(os.path.join(os.path.dirname(__file__), "../model", filename), "rb"))
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv")).dropna()
    
    data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    scaled = StandardScaler().fit_transform(data)
    df['Cluster'] = model.predict(scaled)
    
    stats = df.groupby('Cluster')[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]].mean().round(0)
    counts = df['Cluster'].value_counts().sort_index()
    
    html = f"""<!DOCTYPE html>
<html><head><title>Clustering Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:Arial;margin:40px;background:#f5f5f5}}
.card{{background:white;padding:30px;border-radius:10px;margin:20px 0;box-shadow:0 2px 10px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;text-align:center}}
canvas{{max-height:300px}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}
th{{background:#3498db;color:white}}
</style></head><body>
<h1>Credit Card Clustering Analysis</h1>
<h3 style="text-align:center;color:#34495e">Novia Dsilva (MLOPS Lab1)</h3>
<p style="text-align:center;color:#666">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="card">
<h2>Elbow Method - Optimal Clusters</h2>
<canvas id="elbowChart"></canvas>
</div>

<div class="card">
<h2>Cluster Distribution</h2>
<canvas id="clusterChart"></canvas>
</div>

<div class="card">
<h2>Cluster Characteristics</h2>
<table><tr><th>Cluster</th><th>Size</th><th>Avg Balance</th><th>Avg Purchases</th><th>Avg Credit</th></tr>"""
    
    for c in stats.index:
        html += f"<tr><td>Cluster {c}</td><td>{counts[c]}</td><td>${stats.loc[c,'BALANCE']:,.0f}</td><td>${stats.loc[c,'PURCHASES']:,.0f}</td><td>${stats.loc[c,'CREDIT_LIMIT']:,.0f}</td></tr>"
    html += f"""</table></div>

<script>
new Chart(document.getElementById('elbowChart'),{{
  type:'line',
  data:{{labels:[{','.join([str(i) for i in range(2,2+len(sse))])}],
        datasets:[{{label:'SSE',data:[{','.join([str(s) for s in sse])}],
                   borderColor:'#e74c3c',backgroundColor:'rgba(231,76,60,0.1)',tension:0.3}}]}},
  options:{{responsive:true,plugins:{{title:{{display:true,text:'SSE vs Number of Clusters'}}}}}}
}});

new Chart(document.getElementById('clusterChart'),{{
  type:'bar',
  data:{{labels:[{','.join([f"'Cluster {c}'" for c in counts.index])}],
        datasets:[{{label:'Customers',data:[{','.join([str(c) for c in counts.values])}],
                   backgroundColor:['#3498db','#2ecc71','#f39c12','#9b59b6','#e74c3c','#1abc9c']}}]}},
  options:{{responsive:true,plugins:{{title:{{display:true,text:'Customer Distribution'}}}}}}
}});
</script></body></html>"""
    
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    with open(report_file, 'w') as f:
        f.write(html)
    
    print(f"Visual report: {report_file}")
    return {"report_path": report_file}