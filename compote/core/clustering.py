"""
Algorithm 3: Context Clustering

Purpose: Group similar messages into context pools using DBSCAN.
Complexity: O(n·log n) on average, dominated by DBSCAN's spatial indexing.

Uses DBSCAN clustering on feature vectors with Euclidean distance to create
"message pools" representing specific consensus contexts.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
from ..core.types import MessageFeatures, MessageCluster


class ContextClustering:
    """
    Implements context clustering using DBSCAN algorithm.
    
    Groups similar consensus messages into context pools based on their
    feature similarity. Each cluster represents a specific consensus context
    (e.g., normal operation, view changes, leader failures).
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 3, 
                 metric: str = 'euclidean', scale_features: bool = True):
        """
        Initialize context clustering.
        
        Args:
            eps: Maximum distance between samples for them to be in same cluster
            min_samples: Minimum number of samples in a cluster
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            scale_features: Whether to scale features before clustering
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.scale_features = scale_features
        
        # Clustering components
        self.dbscan = None
        self.scaler = StandardScaler() if scale_features else None
        
        # Clustering results
        self.clusters = {}
        self.cluster_labels = None
        self.feature_matrix = None
        self.message_id_mapping = {}
        
        # Statistics
        self.silhouette_score_ = None
        self.n_clusters_ = 0
        self.n_noise_points_ = 0
        
        self.logger = logging.getLogger(__name__)
    
    def fit_predict(self, message_features: List[MessageFeatures]) -> Dict[int, MessageCluster]:
        """
        Perform clustering on message features and return clusters.
        
        Args:
            message_features: List of MessageFeatures to cluster
            
        Returns:
            Dictionary mapping cluster_id to MessageCluster objects
        """
        if not message_features:
            self.logger.warning("No message features provided for clustering")
            return {}
        
        # Step 1: Prepare feature matrix
        self._prepare_feature_matrix(message_features)
        
        # Step 2: Scale features if required
        if self.scale_features and self.scaler is not None:
            scaled_features = self.scaler.fit_transform(self.feature_matrix)
        else:
            scaled_features = self.feature_matrix
        
        # Step 3: Apply DBSCAN clustering
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, 
                           metric=self.metric)
        self.cluster_labels = self.dbscan.fit_predict(scaled_features)
        
        # Step 4: Create cluster objects
        clusters = self._create_clusters(message_features)
        
        # Step 5: Calculate clustering statistics
        self._calculate_statistics(scaled_features)
        
        self.clusters = clusters
        return clusters
    
    def _prepare_feature_matrix(self, message_features: List[MessageFeatures]):
        """Prepare feature matrix and message ID mapping"""
        n_features = len(message_features[0].features)
        n_messages = len(message_features)
        
        self.feature_matrix = np.zeros((n_messages, n_features))
        self.message_id_mapping = {}
        
        for i, msg_features in enumerate(message_features):
            self.feature_matrix[i] = msg_features.features
            self.message_id_mapping[i] = msg_features.message_id
    
    def _create_clusters(self, message_features: List[MessageFeatures]) -> Dict[int, MessageCluster]:
        """Create MessageCluster objects from clustering results"""
        clusters = {}
        
        # Get unique cluster labels (excluding noise points labeled as -1)
        unique_labels = set(self.cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label
        
        self.n_clusters_ = len(unique_labels)
        self.n_noise_points_ = sum(1 for label in self.cluster_labels if label == -1)
        
        # Create clusters
        for cluster_id in unique_labels:
            # Get indices of messages in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # Get message IDs in this cluster
            message_ids = [self.message_id_mapping[idx] for idx in cluster_indices]
            
            # Calculate centroid
            cluster_features = self.feature_matrix[cluster_indices]
            centroid = np.mean(cluster_features, axis=0)
            
            # Get feature names from first message
            feature_names = message_features[0].feature_names
            
            # Create cluster object
            cluster = MessageCluster(
                cluster_id=cluster_id,
                message_ids=message_ids,
                centroid=centroid,
                feature_names=feature_names
            )
            
            clusters[cluster_id] = cluster
        
        # Handle noise points as individual clusters if needed
        noise_indices = np.where(self.cluster_labels == -1)[0]
        for idx in noise_indices:
            noise_cluster_id = -(idx + 1)  # Negative IDs for noise clusters
            message_id = self.message_id_mapping[idx]
            
            cluster = MessageCluster(
                cluster_id=noise_cluster_id,
                message_ids=[message_id],
                centroid=self.feature_matrix[idx],
                feature_names=message_features[0].feature_names
            )
            
            clusters[noise_cluster_id] = cluster
        
        return clusters
    
    def _calculate_statistics(self, scaled_features: np.ndarray):
        """Calculate clustering quality statistics"""
        if self.n_clusters_ > 1:
            try:
                # Calculate silhouette score (only if we have multiple clusters)
                valid_labels = self.cluster_labels[self.cluster_labels != -1]
                valid_features = scaled_features[self.cluster_labels != -1]
                
                if len(set(valid_labels)) > 1:
                    self.silhouette_score_ = silhouette_score(valid_features, valid_labels)
                else:
                    self.silhouette_score_ = 0.0
            except Exception as e:
                self.logger.warning(f"Could not calculate silhouette score: {e}")
                self.silhouette_score_ = 0.0
        else:
            self.silhouette_score_ = 0.0
    
    def predict_cluster(self, message_features: MessageFeatures) -> int:
        """
        Predict which cluster a new message belongs to.
        
        Args:
            message_features: Features of the new message
            
        Returns:
            Cluster ID (or -1 if noise/outlier)
        """
        if not self.clusters or self.dbscan is None:
            raise ValueError("Must fit clustering model first")
        
        # Find the closest cluster centroid
        feature_vector = message_features.features.reshape(1, -1)
        
        if self.scale_features and self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        
        min_distance = float('inf')
        closest_cluster = -1
        
        for cluster_id, cluster in self.clusters.items():
            if cluster_id < 0:  # Skip noise clusters
                continue
            
            # Calculate distance to centroid
            if self.scale_features and self.scaler is not None:
                centroid = cluster.centroid.reshape(1, -1)
                centroid = self.scaler.transform(centroid).flatten()
            else:
                centroid = cluster.centroid
            
            distance = np.linalg.norm(feature_vector.flatten() - centroid)
            
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # Check if distance is within eps threshold
        if min_distance <= self.eps:
            return closest_cluster
        else:
            return -1  # Noise point
    
    def get_cluster_summary(self) -> Dict[str, any]:
        """Get summary of clustering results"""
        summary = {
            'total_clusters': self.n_clusters_,
            'noise_points': self.n_noise_points_,
            'silhouette_score': self.silhouette_score_,
            'cluster_sizes': {},
            'cluster_stats': {}
        }
        
        if not self.clusters:
            return summary
        
        # Calculate cluster sizes and statistics
        for cluster_id, cluster in self.clusters.items():
            if cluster_id >= 0:  # Only count real clusters, not noise
                summary['cluster_sizes'][cluster_id] = len(cluster.message_ids)
                
                # Calculate cluster statistics
                cluster_indices = [i for i, cid in enumerate(self.cluster_labels) if cid == cluster_id]
                if cluster_indices:
                    cluster_features = self.feature_matrix[cluster_indices]
                    
                    summary['cluster_stats'][cluster_id] = {
                        'size': len(cluster_indices),
                        'feature_mean': np.mean(cluster_features, axis=0).tolist(),
                        'feature_std': np.std(cluster_features, axis=0).tolist(),
                        'intra_cluster_distance': self._calculate_intra_cluster_distance(cluster_features)
                    }
        
        return summary
    
    def _calculate_intra_cluster_distance(self, cluster_features: np.ndarray) -> float:
        """Calculate average intra-cluster distance"""
        if len(cluster_features) <= 1:
            return 0.0
        
        distances = []
        for i in range(len(cluster_features)):
            for j in range(i + 1, len(cluster_features)):
                distance = np.linalg.norm(cluster_features[i] - cluster_features[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize_parameters(self, message_features: List[MessageFeatures], 
                          eps_range: Tuple[float, float] = (0.1, 2.0),
                          min_samples_range: Tuple[int, int] = (2, 10)) -> Dict[str, float]:
        """
        Optimize DBSCAN parameters using silhouette score.
        
        Args:
            message_features: Features to cluster
            eps_range: Range of eps values to try
            min_samples_range: Range of min_samples values to try
            
        Returns:
            Dictionary with optimal parameters and score
        """
        if not message_features:
            return {'eps': self.eps, 'min_samples': self.min_samples, 'score': 0.0}
        
        best_score = -1
        best_params = {'eps': self.eps, 'min_samples': self.min_samples}
        
        # Prepare feature matrix
        self._prepare_feature_matrix(message_features)
        
        if self.scale_features and self.scaler is not None:
            scaled_features = self.scaler.fit_transform(self.feature_matrix)
        else:
            scaled_features = self.feature_matrix
        
        # Grid search over parameter space
        eps_values = np.linspace(eps_range[0], eps_range[1], 10)
        min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)
        
        for eps_val in eps_values:
            for min_samples_val in min_samples_values:
                try:
                    # Test clustering with these parameters
                    dbscan_test = DBSCAN(eps=eps_val, min_samples=min_samples_val, 
                                       metric=self.metric)
                    labels = dbscan_test.fit_predict(scaled_features)
                    
                    # Calculate score
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters > 1:
                        valid_labels = labels[labels != -1]
                        valid_features = scaled_features[labels != -1]
                        
                        if len(set(valid_labels)) > 1:
                            score = silhouette_score(valid_features, valid_labels)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {'eps': eps_val, 'min_samples': min_samples_val}
                
                except Exception as e:
                    self.logger.debug(f"Parameter optimization failed for eps={eps_val}, "
                                    f"min_samples={min_samples_val}: {e}")
                    continue
        
        # Update parameters if better ones were found
        if best_score > -1:
            self.eps = best_params['eps']
            self.min_samples = best_params['min_samples']
        
        return {**best_params, 'score': best_score}
    
    def visualize_clusters(self, message_features: List[MessageFeatures], 
                         save_path: Optional[str] = None) -> Optional[str]:
        """
        Create 2D visualization of clusters using PCA.
        
        Args:
            message_features: Features to visualize
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            if not self.clusters or self.feature_matrix is None:
                self.logger.warning("No clusters to visualize")
                return None
            
            # Reduce to 2D using PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(self.feature_matrix)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Plot each cluster with different color
            unique_labels = set(self.cluster_labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points in black
                    color = 'black'
                    label_name = 'Noise'
                else:
                    label_name = f'Cluster {label}'
                
                mask = self.cluster_labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[color], label=label_name, alpha=0.7)
            
            plt.title(f'COMPOTE Context Clusters (n_clusters={self.n_clusters_})')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except ImportError:
            self.logger.warning("matplotlib not available for visualization")
            return None
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return None
    
    def get_clustering_stats(self) -> Dict[str, any]:
        """Get detailed clustering statistics"""
        return {
            'parameters': {
                'eps': self.eps,
                'min_samples': self.min_samples,
                'metric': self.metric,
                'scale_features': self.scale_features
            },
            'results': {
                'n_clusters': self.n_clusters_,
                'n_noise_points': self.n_noise_points_,
                'silhouette_score': self.silhouette_score_,
                'total_messages': len(self.cluster_labels) if self.cluster_labels is not None else 0
            },
            'cluster_summary': self.get_cluster_summary()
        }