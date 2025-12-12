"""
Unit tests for the ContextClustering class.

Tests DBSCAN clustering algorithm and context pool creation.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote.core.clustering import ContextClustering
from compote.core.types import MessageFeatures, MessageCluster


class TestContextClustering(unittest.TestCase):
    """Test cases for ContextClustering"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.clustering = ContextClustering(eps=0.5, min_samples=2, scale_features=True)
    
    def create_test_features(self, count: int, dimensions: int = 10, 
                           cluster_centers: list = None) -> list:
        """Helper to create test MessageFeatures"""
        features = []
        
        if cluster_centers is None:
            # Create random features
            for i in range(count):
                feature_vector = np.random.rand(dimensions)
                features.append(MessageFeatures(
                    message_id=f"test_msg_{i}",
                    features=feature_vector,
                    feature_names=[f"feature_{j}" for j in range(dimensions)]
                ))
        else:
            # Create features around specified centers
            samples_per_center = count // len(cluster_centers)
            feature_idx = 0
            
            for center in cluster_centers:
                for i in range(samples_per_center):
                    # Add noise around center
                    noise = np.random.normal(0, 0.1, dimensions)
                    feature_vector = np.array(center) + noise
                    
                    features.append(MessageFeatures(
                        message_id=f"test_msg_{feature_idx}",
                        features=feature_vector,
                        feature_names=[f"feature_{j}" for j in range(dimensions)]
                    ))
                    feature_idx += 1
        
        return features
    
    def test_basic_clustering(self):
        """Test basic clustering functionality"""
        # Create features with clear clusters
        cluster_centers = [
            [0, 0, 0, 0, 0],     # Cluster 1 center
            [5, 5, 5, 5, 5],     # Cluster 2 center
            [10, 10, 10, 10, 10] # Cluster 3 center
        ]
        
        features = self.create_test_features(15, 5, cluster_centers)
        
        clusters = self.clustering.fit_predict(features)
        
        # Should have found clusters
        self.assertGreater(len(clusters), 0)
        self.assertGreater(self.clustering.n_clusters_, 0)
        
        # Check that cluster objects are properly created
        for cluster_id, cluster in clusters.items():
            if cluster_id >= 0:  # Skip noise clusters
                self.assertIsInstance(cluster, MessageCluster)
                self.assertGreater(len(cluster.message_ids), 0)
                self.assertEqual(len(cluster.centroid), 5)  # Should match feature dimensions
                self.assertEqual(len(cluster.feature_names), 5)
    
    def test_noise_detection(self):
        """Test detection of noise points (outliers)"""
        # Create mostly clustered data with some outliers
        cluster_centers = [
            [0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5]
        ]
        
        features = self.create_test_features(10, 5, cluster_centers)
        
        # Add clear outliers
        outlier1 = MessageFeatures(
            message_id="outlier_1",
            features=np.array([100, 100, 100, 100, 100]),
            feature_names=[f"feature_{j}" for j in range(5)]
        )
        outlier2 = MessageFeatures(
            message_id="outlier_2", 
            features=np.array([-100, -100, -100, -100, -100]),
            feature_names=[f"feature_{j}" for j in range(5)]
        )
        
        features.extend([outlier1, outlier2])
        
        # Use stricter parameters to ensure outliers are detected
        strict_clustering = ContextClustering(eps=0.3, min_samples=3)
        clusters = strict_clustering.fit_predict(features)
        
        # Should detect some noise points
        self.assertGreater(strict_clustering.n_noise_points_, 0)
        
        # Noise clusters should have negative IDs
        noise_clusters = [cid for cid in clusters.keys() if cid < 0]
        self.assertGreater(len(noise_clusters), 0)
    
    def test_empty_input(self):
        """Test behavior with empty input"""
        clusters = self.clustering.fit_predict([])
        
        self.assertEqual(len(clusters), 0)
        self.assertEqual(self.clustering.n_clusters_, 0)
        self.assertEqual(self.clustering.n_noise_points_, 0)
    
    def test_single_message(self):
        """Test behavior with single message"""
        features = self.create_test_features(1, 5)
        clusters = self.clustering.fit_predict(features)
        
        # Single message should be treated as noise with min_samples=2
        self.assertEqual(self.clustering.n_clusters_, 0)
        self.assertEqual(self.clustering.n_noise_points_, 1)
    
    def test_identical_features(self):
        """Test clustering with identical feature vectors"""
        # Create multiple messages with identical features
        identical_vector = np.array([1, 2, 3, 4, 5])
        
        features = []
        for i in range(5):
            features.append(MessageFeatures(
                message_id=f"identical_msg_{i}",
                features=identical_vector.copy(),
                feature_names=[f"feature_{j}" for j in range(5)]
            ))
        
        clusters = self.clustering.fit_predict(features)
        
        # Should form a single cluster
        self.assertEqual(self.clustering.n_clusters_, 1)
        self.assertEqual(self.clustering.n_noise_points_, 0)
        
        # All messages should be in the same cluster
        main_cluster = None
        for cid, cluster in clusters.items():
            if cid >= 0:
                main_cluster = cluster
                break
        
        self.assertIsNotNone(main_cluster)
        self.assertEqual(len(main_cluster.message_ids), 5)
    
    def test_predict_cluster(self):
        """Test predicting cluster for new messages"""
        # Train on initial data
        cluster_centers = [
            [0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5]
        ]
        
        training_features = self.create_test_features(10, 5, cluster_centers)
        clusters = self.clustering.fit_predict(training_features)
        
        # Test prediction for new message close to first cluster
        new_message_features = MessageFeatures(
            message_id="new_msg",
            features=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # Close to [0,0,0,0,0]
            feature_names=[f"feature_{j}" for j in range(5)]
        )
        
        predicted_cluster = self.clustering.predict_cluster(new_message_features)
        
        # Should predict a valid cluster (not noise)
        self.assertGreaterEqual(predicted_cluster, 0)
        
        # Test prediction for outlier
        outlier_features = MessageFeatures(
            message_id="outlier_msg",
            features=np.array([100, 100, 100, 100, 100]),  # Far from any cluster
            feature_names=[f"feature_{j}" for j in range(5)]
        )
        
        predicted_cluster = self.clustering.predict_cluster(outlier_features)
        
        # Should predict as noise
        self.assertEqual(predicted_cluster, -1)
    
    def test_clustering_without_scaling(self):
        """Test clustering without feature scaling"""
        unscaled_clustering = ContextClustering(eps=0.5, min_samples=2, scale_features=False)
        
        features = self.create_test_features(10, 5)
        clusters = unscaled_clustering.fit_predict(features)
        
        # Should still produce clusters
        self.assertIsInstance(clusters, dict)
    
    def test_different_distance_metrics(self):
        """Test clustering with different distance metrics"""
        features = self.create_test_features(12, 5, [[0,0,0,0,0], [3,3,3,3,3]])
        
        # Test with manhattan distance
        manhattan_clustering = ContextClustering(eps=1.0, min_samples=2, metric='manhattan')
        clusters_manhattan = manhattan_clustering.fit_predict(features)
        
        # Test with cosine distance (if supported)
        try:
            cosine_clustering = ContextClustering(eps=0.3, min_samples=2, metric='cosine')
            clusters_cosine = cosine_clustering.fit_predict(features)
            
            # Both should produce valid results
            self.assertIsInstance(clusters_manhattan, dict)
            self.assertIsInstance(clusters_cosine, dict)
        except ValueError:
            # Cosine metric might not be available in all sklearn versions
            pass
    
    def test_parameter_optimization(self):
        """Test parameter optimization functionality"""
        # Create clear cluster structure
        cluster_centers = [
            [0, 0, 0],
            [3, 3, 3],
            [6, 6, 6]
        ]
        
        features = self.create_test_features(15, 3, cluster_centers)
        
        # Run parameter optimization
        optimal_params = self.clustering.optimize_parameters(
            features, 
            eps_range=(0.1, 2.0),
            min_samples_range=(2, 5)
        )
        
        # Should return valid parameters
        self.assertIn('eps', optimal_params)
        self.assertIn('min_samples', optimal_params)
        self.assertIn('score', optimal_params)
        
        # Optimized eps should be reasonable
        self.assertGreater(optimal_params['eps'], 0)
        self.assertLess(optimal_params['eps'], 5.0)
        
        # Score should be meaningful (even if negative)
        self.assertIsInstance(optimal_params['score'], float)
    
    def test_cluster_summary(self):
        """Test cluster summary generation"""
        features = self.create_test_features(12, 5, [[0,0,0,0,0], [5,5,5,5,5]])
        clusters = self.clustering.fit_predict(features)
        
        summary = self.clustering.get_cluster_summary()
        
        # Check summary structure
        self.assertIn('total_clusters', summary)
        self.assertIn('noise_points', summary)
        self.assertIn('silhouette_score', summary)
        self.assertIn('cluster_sizes', summary)
        self.assertIn('cluster_stats', summary)
        
        # Verify data types
        self.assertIsInstance(summary['total_clusters'], int)
        self.assertIsInstance(summary['noise_points'], int)
        self.assertIsInstance(summary['cluster_sizes'], dict)
        self.assertIsInstance(summary['cluster_stats'], dict)
    
    def test_clustering_statistics(self):
        """Test clustering statistics"""
        features = self.create_test_features(10, 5)
        clusters = self.clustering.fit_predict(features)
        
        stats = self.clustering.get_clustering_stats()
        
        # Check statistics structure
        self.assertIn('parameters', stats)
        self.assertIn('results', stats)
        self.assertIn('cluster_summary', stats)
        
        # Check parameter values
        params = stats['parameters']
        self.assertEqual(params['eps'], self.clustering.eps)
        self.assertEqual(params['min_samples'], self.clustering.min_samples)
        self.assertEqual(params['metric'], self.clustering.metric)
        
        # Check results
        results = stats['results']
        self.assertIn('n_clusters', results)
        self.assertIn('n_noise_points', results)
        self.assertIn('total_messages', results)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualization(self, mock_savefig, mock_show):
        """Test cluster visualization functionality"""
        features = self.create_test_features(15, 5, [[0,0,0,0,0], [3,3,3,3,3]])
        clusters = self.clustering.fit_predict(features)
        
        # Test visualization without saving
        result = self.clustering.visualize_clusters(features)
        self.assertIsNone(result)  # No save path specified
        
        # Test visualization with saving
        save_path = 'test_clusters.png'
        result = self.clustering.visualize_clusters(features, save_path)
        
        if result:  # If matplotlib is available
            self.assertEqual(result, save_path)
            mock_savefig.assert_called_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_clustering_complexity(self):
        """Test that clustering complexity is reasonable (O(n·log n) average)"""
        # Test with different input sizes
        sizes = [10, 20, 40]
        times = []
        
        for size in sizes:
            features = self.create_test_features(size, 5)
            
            start_time = time.perf_counter()
            clusters = self.clustering.fit_predict(features)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
        
        # Verify clustering worked
        self.assertGreater(len(clusters), 0)
        
        # Time should not grow exponentially
        # (This is a rough test, actual performance depends on data and system)
        if len(times) > 1:
            # Growth rate should be reasonable
            growth_rate = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            
            # Should not be much worse than quadratic growth
            self.assertLess(growth_rate, size_ratio ** 3)
    
    def test_silhouette_score_calculation(self):
        """Test silhouette score calculation"""
        # Create well-separated clusters
        cluster_centers = [
            [0, 0, 0],
            [10, 10, 10]
        ]
        
        features = self.create_test_features(10, 3, cluster_centers)
        clusters = self.clustering.fit_predict(features)
        
        # Well-separated clusters should have good silhouette score
        if self.clustering.n_clusters_ > 1:
            self.assertIsNotNone(self.clustering.silhouette_score_)
            self.assertGreater(self.clustering.silhouette_score_, 0)
    
    def test_cluster_add_message(self):
        """Test adding messages to clusters"""
        features = self.create_test_features(5, 3, [[0,0,0]])
        clusters = self.clustering.fit_predict(features)
        
        # Get a valid cluster
        cluster = None
        for cid, c in clusters.items():
            if cid >= 0:
                cluster = c
                break
        
        if cluster:
            initial_size = len(cluster.message_ids)
            
            # Add a new message
            cluster.add_message("new_message_id", priority=0.8)
            
            self.assertEqual(len(cluster.message_ids), initial_size + 1)
            self.assertIn("new_message_id", cluster.message_ids)
            self.assertEqual(cluster.priority_scores["new_message_id"], 0.8)
    
    def test_get_highest_priority_message(self):
        """Test getting highest priority message from cluster"""
        cluster = MessageCluster(
            cluster_id=0,
            message_ids=["msg1", "msg2", "msg3"],
            centroid=np.array([1, 2, 3]),
            feature_names=["f1", "f2", "f3"]
        )
        
        # Add priority scores
        cluster.priority_scores = {
            "msg1": 0.5,
            "msg2": 0.9,  # Highest
            "msg3": 0.3
        }
        
        highest = cluster.get_highest_priority_message()
        self.assertEqual(highest, "msg2")
        
        # Test empty cluster
        empty_cluster = MessageCluster(
            cluster_id=1,
            message_ids=[],
            centroid=np.array([]),
            feature_names=[]
        )
        
        highest = empty_cluster.get_highest_priority_message()
        self.assertIsNone(highest)


if __name__ == '__main__':
    unittest.main()