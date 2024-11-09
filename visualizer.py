import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

class ClusterVisualizer:
    def plot_clusters(self, vectors, cluster_labels, doc_names):
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors.toarray())
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': [f'Cluster {label}' for label in cluster_labels],
            'document': doc_names
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['document'],
            title='Document Clusters Visualization',
            labels={'x': 'First Principal Component', 
                   'y': 'Second Principal Component'}
        )
        
        return fig
