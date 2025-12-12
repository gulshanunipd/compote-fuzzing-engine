"""
Unit tests for the FeatureExtractor class.

Tests feature extraction algorithms and numerical vector generation.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote.core.feature_extractor import FeatureExtractor
from compote.core.types import ParsedMessage, MessageFeatures, MessageType, NodeRole


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor(normalize=True)
        self.test_timestamp = time.time()
    
    def create_test_message(self, message_type=MessageType.PROPOSE, round_num=1, 
                          view_num=0, height=1, sender_id='test_node', 
                          role=NodeRole.VALIDATOR, **kwargs) -> ParsedMessage:
        """Helper to create test ParsedMessage"""
        return ParsedMessage(
            message_id=f"test_msg_{int(time.time_ns())}",
            message_type=message_type,
            round_number=round_num,
            view_number=view_num,
            block_height=height,
            sender_id=sender_id,
            sender_role=role,
            timestamp=self.test_timestamp,
            payload_hash="test_hash_123",
            signature="test_signature",
            additional_fields=kwargs
        )
    
    def test_feature_names_consistency(self):
        """Test that feature names are consistent and complete"""
        feature_names = self.extractor.feature_names
        
        # Check expected categories of features
        msg_type_features = [f for f in feature_names if f.startswith('msg_type_')]
        role_features = [f for f in feature_names if f.startswith('role_')]
        
        # Should have all message types
        expected_msg_types = [
            'msg_type_propose', 'msg_type_prevote', 'msg_type_precommit',
            'msg_type_commit', 'msg_type_round_change', 'msg_type_new_view',
            'msg_type_prepare', 'msg_type_view_change', 'msg_type_other'
        ]
        
        for expected in expected_msg_types:
            self.assertIn(expected, feature_names)
        
        # Should have all role types
        expected_roles = [
            'role_leader', 'role_validator', 'role_observer', 
            'role_proposer', 'role_unknown'
        ]
        
        for expected in expected_roles:
            self.assertIn(expected, feature_names)
        
        # Check for other expected features
        expected_other = [
            'round_number', 'view_number', 'block_height',
            'timestamp_normalized', 'message_latency', 'round_deviation',
            'payload_hash_entropy', 'signature_present', 'additional_fields_count',
            'sender_role_weight', 'message_complexity', 'temporal_distance'
        ]
        
        for expected in expected_other:
            self.assertIn(expected, feature_names)
    
    def test_single_message_feature_extraction(self):
        """Test feature extraction for a single message"""
        test_message = self.create_test_message(
            message_type=MessageType.PROPOSE,
            round_num=5,
            view_num=1,
            sender_id='leader_node',
            role=NodeRole.LEADER,
            extra_field='extra_value'
        )
        
        features = self.extractor.extract_features([test_message])
        
        self.assertEqual(len(features), 1)
        
        feature = features[0]
        self.assertIsInstance(feature, MessageFeatures)
        self.assertEqual(feature.message_id, test_message.message_id)
        self.assertEqual(len(feature.features), len(self.extractor.feature_names))
        
        # Check one-hot encoding for message type
        propose_idx = self.extractor.feature_names.index('msg_type_propose')
        self.assertEqual(feature.features[propose_idx], 1.0)
        
        # Check one-hot encoding for role
        leader_idx = self.extractor.feature_names.index('role_leader')
        self.assertEqual(feature.features[leader_idx], 1.0)
        
        # Check numerical features
        round_idx = self.extractor.feature_names.index('round_number')
        self.assertEqual(feature.features[round_idx], 5)
        
        view_idx = self.extractor.feature_names.index('view_number')
        self.assertEqual(feature.features[view_idx], 1)
    
    def test_multiple_message_feature_extraction(self):
        """Test feature extraction for multiple messages"""
        messages = [
            self.create_test_message(MessageType.PROPOSE, 1, 0, role=NodeRole.LEADER),
            self.create_test_message(MessageType.PREVOTE, 1, 0, role=NodeRole.VALIDATOR),
            self.create_test_message(MessageType.PRECOMMIT, 1, 0, role=NodeRole.VALIDATOR),
            self.create_test_message(MessageType.COMMIT, 1, 0, role=NodeRole.VALIDATOR)
        ]
        
        features = self.extractor.extract_features(messages)
        
        self.assertEqual(len(features), 4)
        
        # Check that each message has correct type encoding
        type_indices = [
            self.extractor.feature_names.index('msg_type_propose'),
            self.extractor.feature_names.index('msg_type_prevote'),
            self.extractor.feature_names.index('msg_type_precommit'),
            self.extractor.feature_names.index('msg_type_commit')
        ]
        
        for i, feature in enumerate(features):
            # Only the corresponding type should be 1.0
            for j, type_idx in enumerate(type_indices):
                if i == j:
                    self.assertEqual(feature.features[type_idx], 1.0)
                else:
                    self.assertEqual(feature.features[type_idx], 0.0)
    
    def test_latency_calculation(self):
        """Test message latency calculation"""
        base_time = time.time()
        
        messages = [
            self.create_test_message(round_num=1),  # Will have timestamp from setUp
            self.create_test_message(round_num=1),  # Same round, later timestamp
        ]
        
        # Modify timestamps to test latency
        messages[0].timestamp = base_time
        messages[1].timestamp = base_time + 5.0  # 5 seconds later
        
        features = self.extractor.extract_features(messages)
        
        latency_idx = self.extractor.feature_names.index('message_latency')
        
        # First message should have 0 latency (earliest in round)
        self.assertEqual(features[0].features[latency_idx], 0.0)
        
        # Second message should have 5.0 latency
        self.assertEqual(features[1].features[latency_idx], 5.0)
    
    def test_round_deviation_calculation(self):
        """Test round deviation calculation"""
        messages = [
            self.create_test_message(round_num=1),
            self.create_test_message(round_num=5),
            self.create_test_message(round_num=10)
        ]
        
        features = self.extractor.extract_features(messages)
        
        # Mean round should be (1 + 5 + 10) / 3 = 5.33
        expected_mean = 16 / 3
        
        deviation_idx = self.extractor.feature_names.index('round_deviation')
        
        # Check deviations
        expected_deviations = [
            abs(1 - expected_mean),
            abs(5 - expected_mean),
            abs(10 - expected_mean)
        ]
        
        for i, expected_dev in enumerate(expected_deviations):
            actual_dev = features[i].features[deviation_idx]
            self.assertAlmostEqual(actual_dev, expected_dev, places=2)
    
    def test_hash_entropy_calculation(self):
        """Test payload hash entropy calculation"""
        # Test with different hash patterns
        test_cases = [
            ("0000000000000000", 0.0),  # No entropy (all zeros)
            ("abcdefghijklmnop", 4.0),  # High entropy (all different)
            ("aaaaaaaaaaaaaaaa", 0.0),  # No entropy (all same)
        ]
        
        entropy_idx = self.extractor.feature_names.index('payload_hash_entropy')
        
        for hash_val, expected_entropy in test_cases:
            with self.subTest(hash_value=hash_val):
                message = self.create_test_message()
                message.payload_hash = hash_val
                
                features = self.extractor.extract_features([message])
                actual_entropy = features[0].features[entropy_idx]
                
                self.assertAlmostEqual(actual_entropy, expected_entropy, places=1)
    
    def test_signature_present_feature(self):
        """Test signature presence detection"""
        sig_idx = self.extractor.feature_names.index('signature_present')
        
        # Message with signature
        msg_with_sig = self.create_test_message()
        msg_with_sig.signature = "test_signature"
        
        features = self.extractor.extract_features([msg_with_sig])
        self.assertEqual(features[0].features[sig_idx], 1.0)
        
        # Message without signature
        msg_without_sig = self.create_test_message()
        msg_without_sig.signature = None
        
        features = self.extractor.extract_features([msg_without_sig])
        self.assertEqual(features[0].features[sig_idx], 0.0)
    
    def test_role_weight_calculation(self):
        """Test sender role weight calculation"""
        role_weight_idx = self.extractor.feature_names.index('sender_role_weight')
        
        test_cases = [
            (NodeRole.LEADER, 1.0),
            (NodeRole.PROPOSER, 0.9),
            (NodeRole.VALIDATOR, 0.7),
            (NodeRole.OBSERVER, 0.3),
            (NodeRole.UNKNOWN, 0.1)
        ]
        
        for role, expected_weight in test_cases:
            with self.subTest(role=role):
                message = self.create_test_message(role=role)
                features = self.extractor.extract_features([message])
                
                actual_weight = features[0].features[role_weight_idx]
                self.assertEqual(actual_weight, expected_weight)
    
    def test_message_complexity_calculation(self):
        """Test message complexity score calculation"""
        complexity_idx = self.extractor.feature_names.index('message_complexity')
        
        # Simple message
        simple_msg = self.create_test_message(message_type=MessageType.PREVOTE)
        simple_msg.signature = None
        simple_msg.additional_fields = {}
        
        # Complex message
        complex_msg = self.create_test_message(message_type=MessageType.PROPOSE)
        complex_msg.signature = "signature"
        complex_msg.additional_fields = {f'field_{i}': f'value_{i}' for i in range(5)}
        
        features = self.extractor.extract_features([simple_msg, complex_msg])
        
        simple_complexity = features[0].features[complexity_idx]
        complex_complexity = features[1].features[complexity_idx]
        
        # Complex message should have higher complexity
        self.assertGreater(complex_complexity, simple_complexity)
    
    def test_normalization(self):
        """Test feature normalization"""
        # Test with normalization enabled
        normalizing_extractor = FeatureExtractor(normalize=True)
        
        # Test with normalization disabled
        non_normalizing_extractor = FeatureExtractor(normalize=False)
        
        messages = [
            self.create_test_message(round_num=1, view_num=0, height=100),
            self.create_test_message(round_num=100, view_num=10, height=1000)
        ]
        
        normalized_features = normalizing_extractor.extract_features(messages)
        raw_features = non_normalizing_extractor.extract_features(messages)
        
        # Check that normalized features are different from raw features
        # for numerical features like round_number
        round_idx = normalizing_extractor.feature_names.index('round_number')
        
        norm_round_1 = normalized_features[0].features[round_idx]
        norm_round_2 = normalized_features[1].features[round_idx]
        raw_round_1 = raw_features[0].features[round_idx]
        raw_round_2 = raw_features[1].features[round_idx]
        
        # Raw features should be the actual values
        self.assertEqual(raw_round_1, 1)
        self.assertEqual(raw_round_2, 100)
        
        # Normalized features should be z-score normalized
        self.assertNotEqual(norm_round_1, raw_round_1)
        self.assertNotEqual(norm_round_2, raw_round_2)
    
    def test_similarity_calculation(self):
        """Test cosine similarity calculation between features"""
        # Create two identical messages
        msg1 = self.create_test_message(message_type=MessageType.PROPOSE, round_num=1)
        msg2 = self.create_test_message(message_type=MessageType.PROPOSE, round_num=1)
        
        # Create a different message
        msg3 = self.create_test_message(message_type=MessageType.COMMIT, round_num=10)
        
        features = self.extractor.extract_features([msg1, msg2, msg3])
        
        # Similarity between identical messages should be 1.0
        similarity_12 = self.extractor.compute_similarity(features[0], features[1])
        self.assertAlmostEqual(similarity_12, 1.0, places=3)
        
        # Similarity between different messages should be less than 1.0
        similarity_13 = self.extractor.compute_similarity(features[0], features[2])
        self.assertLess(similarity_13, 1.0)
        self.assertGreaterEqual(similarity_13, 0.0)
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        importance = self.extractor.get_feature_importance()
        
        expected_categories = ['categorical', 'numerical', 'temporal', 'structural', 'derived']
        
        for category in expected_categories:
            self.assertIn(category, importance)
            self.assertGreater(importance[category], 0)
            self.assertLessEqual(importance[category], 1)
        
        # All importance values should sum to 1
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=3)
    
    def test_empty_message_list(self):
        """Test behavior with empty message list"""
        features = self.extractor.extract_features([])
        self.assertEqual(len(features), 0)
    
    def test_complexity_O_M_f(self):
        """Test that complexity is O(M·f) for M messages and f features"""
        # Test with different message counts
        message_counts = [10, 20, 50]
        times = []
        
        for count in message_counts:
            messages = [self.create_test_message(round_num=i) for i in range(count)]
            
            start_time = time.perf_counter()
            features = self.extractor.extract_features(messages)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            
            # Verify we got the expected number of features
            self.assertEqual(len(features), count)
        
        # Time should roughly scale linearly with message count
        # (This is a rough test - actual scaling depends on system performance)
        # We just verify that it doesn't scale exponentially
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            expected_ratio = message_counts[i] / message_counts[0]
            
            # Allow for some variance but should be roughly linear
            self.assertLess(ratio, expected_ratio * 2)  # Not more than 2x expected
    
    def test_get_stats(self):
        """Test statistics generation"""
        stats = self.extractor.get_stats()
        
        expected_keys = [
            'total_features', 'feature_names', 'normalization_enabled',
            'global_stats', 'feature_importance'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['total_features'], len(self.extractor.feature_names))
        self.assertEqual(len(stats['feature_names']), len(self.extractor.feature_names))
        self.assertTrue(stats['normalization_enabled'])


if __name__ == '__main__':
    unittest.main()