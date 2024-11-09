from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DocumentClustering:
    def __init__(self, max_clusters=3):
        self.max_clusters = max_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.kmeans = None  # Will be initialized in cluster_documents
        
    def cluster_documents(self, processed_texts):
        # Vectorize the texts
        vectors = self.vectorizer.fit_transform(processed_texts)
        
        # Dynamically set number of clusters based on input size
        n_samples = len(processed_texts)
        n_clusters = min(self.max_clusters, n_samples)
        
        # Initialize KMeans with dynamic number of clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(vectors)
        
        # Get cluster centers
        cluster_centers = self.kmeans.cluster_centers_
        
        return vectors, cluster_labels, cluster_centers
    
    def get_cluster_statistics(self, cluster_labels):
        unique_labels = np.unique(cluster_labels)
        stats = {}
        
        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            stats[f"Cluster {label}"] = {
                "Number of documents": count,
                "Percentage": f"{(count/len(cluster_labels))*100:.2f}%"
            }
            
        return pd.DataFrame.from_dict(stats, orient='index')
    
    def calculate_jaccard_similarity(self, doc1, doc2):
        # Calculate Jaccard similarity between two documents with error handling
        try:
            set1 = set(doc1.split())
            set2 = set(doc2.split())
            
            # Handle empty sets
            if not set1 or not set2:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            # Handle zero division
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0

    def calculate_centroid_similarity(self, vector, cluster_label):
        # Calculate similarity between a document and its cluster centroid with error handling
        try:
            # Check if vector or centroid is zero
            if vector.getnnz() == 0 or not hasattr(self.kmeans, 'cluster_centers_'):
                return 0.0
                
            centroid = self.kmeans.cluster_centers_[cluster_label]
            
            # Check if centroid is zero vector
            if np.all(np.abs(centroid) < 1e-10):
                return 0.0
                
            similarity = cosine_similarity(vector, centroid.reshape(1, -1))[0][0]
            
            # Handle NaN values
            return float(similarity) if not np.isnan(similarity) else 0.0
        except Exception:
            return 0.0

    def get_similarity_metrics(self, target_doc, doc_names, vectors, cluster_labels, processed_texts):
        # Calculate multiple similarity metrics for documents with proper error handling
        # Define expected columns
        columns = ['Document', 'Cosine Similarity', 'Euclidean Similarity', 
                  'Jaccard Similarity', 'Centroid Similarity', 'Average Similarity']
        
        # Initialize empty DataFrame with zeros
        empty_df = pd.DataFrame(columns=columns)
        empty_df['Document'] = []
        empty_df[columns[1:]] = 0.0
        
        # Return empty DataFrame if target_doc not found
        if target_doc not in doc_names:
            return empty_df
            
        target_idx = doc_names.index(target_doc)
        target_cluster = cluster_labels[target_idx]
        target_vector = vectors[target_idx]
        
        # Calculate similarities for documents in the same cluster
        same_cluster_indices = [i for i in range(len(doc_names)) 
                              if cluster_labels[i] == target_cluster and i != target_idx]
        
        # Return empty DataFrame if no documents in same cluster
        if not same_cluster_indices:
            return empty_df
        
        similarity_metrics = []
        
        for idx in same_cluster_indices:
            try:
                # Initialize metrics with zeros
                metrics_dict = {
                    'Document': doc_names[idx],
                    'Cosine Similarity': 0.0,
                    'Euclidean Similarity': 0.0,
                    'Jaccard Similarity': 0.0,
                    'Centroid Similarity': 0.0,
                    'Average Similarity': 0.0
                }
                
                # Cosine similarity with error handling
                try:
                    if target_vector.getnnz() > 0 and vectors[idx].getnnz() > 0:
                        cosine_sim = cosine_similarity(target_vector, vectors[idx])[0][0]
                        metrics_dict['Cosine Similarity'] = round(float(cosine_sim), 4)
                except Exception:
                    pass
                
                # Euclidean similarity with error handling
                try:
                    eucl_dist = euclidean_distances(target_vector, vectors[idx])[0][0]
                    if eucl_dist != 0:
                        eucl_sim = 1 / (1 + eucl_dist)
                        metrics_dict['Euclidean Similarity'] = round(float(eucl_sim), 4)
                except Exception:
                    pass
                
                # Jaccard similarity with error handling
                try:
                    jaccard_sim = self.calculate_jaccard_similarity(
                        processed_texts[target_idx],
                        processed_texts[idx]
                    )
                    metrics_dict['Jaccard Similarity'] = round(float(jaccard_sim), 4)
                except Exception:
                    pass
                
                # Centroid similarity with error handling
                try:
                    centroid_sim = self.calculate_centroid_similarity(vectors[idx], cluster_labels[idx])
                    metrics_dict['Centroid Similarity'] = round(float(centroid_sim), 4)
                except Exception:
                    pass
                
                # Calculate average similarity excluding zeros
                valid_metrics = [v for k, v in metrics_dict.items() 
                               if k != 'Document' and v > 0]
                if valid_metrics:
                    metrics_dict['Average Similarity'] = round(sum(valid_metrics) / len(valid_metrics), 4)
                
                similarity_metrics.append(metrics_dict)
            except Exception as e:
                print(f"Error calculating similarity metrics for document {doc_names[idx]}: {str(e)}")
                continue
        
        # If no valid metrics were calculated, return empty DataFrame
        if not similarity_metrics:
            return empty_df
        
        # Create DataFrame and sort by Average Similarity
        df = pd.DataFrame(similarity_metrics)
        return df.sort_values('Average Similarity', ascending=False)
