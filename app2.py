import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, Birch, OPTICS
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Constants for clustering and visualization
CLUSTER_ALGORITHMS = {
    "KMeans": KMeans,
    "AgglomerativeClustering": AgglomerativeClustering,
    "DBSCAN": DBSCAN,
    "MeanShift": MeanShift,
    "SpectralClustering": SpectralClustering,
    "Birch": Birch,
    "OPTICS": OPTICS
}
DIMENSION_REDUCTION_ALGOS = ["PCA", "t-SNE"]

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Customer Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_dimensionality_reduction(data, method="PCA", n_components=2):
    if method == "PCA":
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    elif method == "t-SNE":
        tsne = TSNE(n_components=n_components)
        return tsne.fit_transform(data)
    return data

def load_data():
    try:
        df = pd.read_csv('D:\Visual Studio\Dataset Tugas MBC\Week 5\cluster\dataset\data_cluster.csv')  # Replace with your actual dataset path
        st.success("Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        return None
    return df

def visualize_clusters(data, labels):
    # Compact Interactive Cluster Distribution with Plotly
    st.subheader("Interactive Cluster Distribution")
    fig = px.histogram(data, x=labels, color=labels, labels={'color': 'Cluster'}, title="Cluster Distribution")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Interactive 3D Scatter Plot with compact layout
    if 'PCA1' in data.columns and 'PCA2' in data.columns and 'PCA3' in data.columns:
        st.subheader("Interactive 3D Cluster Visualization")
        fig = px.scatter_3d(
            data, x='PCA1', y='PCA2', z='PCA3', color=labels.astype(str),
            title="3D Scatter Plot of Clusters",
            labels={"color": "Cluster"}
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=400)
        st.plotly_chart(fig, use_container_width=True)

def interactive_pair_plot(data, cluster_labels):
    # Allow user to select features for pairplot
    st.subheader("Interactive Pair Plot")
    features = st.multiselect("Select features for pairplot", options=data.columns.tolist(), default=data.columns.tolist()[:3])
    if len(features) > 1:
        fig = sns.pairplot(data[features + ['Cluster']], hue='Cluster', palette='coolwarm')
        st.pyplot(fig)

def elbow_method(features):
    # Determine the optimal number of clusters using the Elbow method for KMeans
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal Clusters')
    plt.tight_layout()
    st.pyplot(fig)

def display_descriptive_stats(df):
    # Display descriptive statistics for each cluster
    st.subheader("Descriptive Statistics by Cluster")
    cluster_stats = df.groupby('Cluster').describe()
    st.write(cluster_stats)

def feature_importance_visualization(df, cluster_labels):
    # Interactive Feature Analysis using Plotly
    st.subheader("Interactive Feature Analysis by Cluster")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Create a grid layout for boxplots
    cols = st.columns(2)  # Display two plots side by side
    for i, feature in enumerate(numeric_features):
        with cols[i % 2]:  # Alternate between columns
            fig = go.Figure()
            for cluster in np.unique(cluster_labels):
                cluster_data = df[df['Cluster'] == cluster]
                fig.add_trace(go.Box(y=cluster_data[feature], name=f'Cluster {cluster}', boxmean=True))

            fig.update_layout(
                title_text=f'{feature} by Cluster', 
                xaxis_title='Cluster', 
                yaxis_title=feature, 
                margin=dict(l=20, r=20, t=30, b=20), 
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

def calculate_clustering_metrics(features, cluster_labels):
    # Calculate silhouette, Davies-Bouldin, and Calinski-Harabasz scores
    if len(set(cluster_labels)) > 1:  # Ensure there are at least two clusters
        silhouette_avg = silhouette_score(features, cluster_labels)
        davies_bouldin = davies_bouldin_score(features, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
        
        # Display metrics
        st.subheader("Clustering Quality Metrics")
        st.write(f"**Silhouette Score**: {silhouette_avg:.2f} (closer to 1 indicates better clustering)")
        st.write(f"**Davies-Bouldin Index**: {davies_bouldin:.2f} (lower values indicate better clustering)")
        st.write(f"**Calinski-Harabasz Index**: {calinski_harabasz:.2f} (higher values indicate better clustering)")

def main():
    st.title("Enhanced Customer Clustering Dashboard")
    st.write("Analyze customer data using different clustering algorithms and visualize the results interactively.")

    # Load data
    df = load_data()
    if df is None:
        return  # Exit if the dataset could not be loaded

    st.write("### Dataset Overview", df.head())

    # Sidebar configuration for clustering
    st.sidebar.header("Clustering Options")
    selected_algo = st.sidebar.selectbox("Select Clustering Algorithm", list(CLUSTER_ALGORITHMS.keys()))
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

    # Dimensionality reduction options
    st.sidebar.subheader("Dimensionality Reduction")
    dim_red_algo = st.sidebar.selectbox("Select Dimensionality Reduction Algorithm", DIMENSION_REDUCTION_ALGOS)
    use_standardization = st.sidebar.checkbox("Standardize Data", value=True)

    # Prepare data for clustering
    features = df.select_dtypes(include=[np.number])
    if use_standardization:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    # Apply dimensionality reduction if necessary
    features_reduced = apply_dimensionality_reduction(features, method=dim_red_algo, n_components=3)
    df['PCA1'], df['PCA2'], df['PCA3'] = features_reduced[:, 0], features_reduced[:, 1], features_reduced[:, 2]

    # Cluster data
    cluster_algo_class = CLUSTER_ALGORITHMS[selected_algo]
    cluster_algo = cluster_algo_class(n_clusters=n_clusters) if selected_algo not in ["DBSCAN", "MeanShift", "OPTICS"] else cluster_algo_class()
    cluster_labels = cluster_algo.fit_predict(features_reduced)

    # Display clustering results
    st.write("### Clustering Results")
    df['Cluster'] = cluster_labels
    visualize_clusters(df, df['Cluster'])

    # Additional Interactive Visualizations and Analysis
    st.sidebar.subheader("Analysis Tools")
    if st.sidebar.button("Show Elbow Method"):
        elbow_method(features)

    # Add button to display clustering metrics
    if st.sidebar.button("Show Clustering Metrics"):
        calculate_clustering_metrics(features, cluster_labels)

    display_descriptive_stats(df)
    feature_importance_visualization(df, cluster_labels)
    interactive_pair_plot(df, cluster_labels)

if __name__ == "__main__":
    main()
