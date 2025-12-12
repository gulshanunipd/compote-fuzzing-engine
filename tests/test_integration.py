"""
Integration tests for the complete COMPOTE fuzzing engine.

Tests the full workflow from message loading to fuzzing execution.
"""

import unittest
import json
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote import CompoteFuzzer
from compote.core.types import RawMessage, MessageType, NodeRole


class TestCompoteFuzzerIntegration(unittest.TestCase):
    """Integration test cases for CompoteFuzzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'simulation_mode': True,
            'max_iterations': 10,
            'max_workers': 1,
            'normalize_features': True,
            'clustering_eps': 0.5,
            'clustering_min_samples': 2
        }
    
    def create_test_messages(self, count: int = 20) -> list:
        """Create test consensus messages"""
        messages = []
        
        for i in range(count):
            msg_data = {
                'message_id': f'test_msg_{i}',
                'message_type': ['propose', 'prevote', 'precommit', 'commit'][i % 4],
                'round_number': i // 4 + 1,
                'view_number': i // 16,
                'block_height': i // 4 + 1,
                'sender_id': f'node_{i % 5}',
                'role': 'validator',
                'timestamp': time.time() + i,
                'block_hash': f'hash_{i // 4}',
                'signature': f'sig_{i}'
            }
            
            # Add type-specific fields
            if msg_data['message_type'] == 'propose':
                msg_data['proposal_hash'] = f'proposal_{i}'
                msg_data['block_data'] = f'data_{i}'
            
            raw_data = json.dumps(msg_data).encode('utf-8')
            messages.append(RawMessage(
                data=raw_data,
                timestamp=msg_data['timestamp'],
                source_id=msg_data['sender_id']
            ))
        
        return messages
    
    def test_complete_fuzzing_workflow(self):
        """Test the complete fuzzing workflow"""
        with CompoteFuzzer(self.test_config) as fuzzer:
            # Step 1: Load messages
            test_messages = self.create_test_messages(15)
            loaded_count = fuzzer.load_messages(test_messages)
            
            self.assertEqual(loaded_count, 15)
            self.assertEqual(len(fuzzer.raw_messages), 15)
            
            # Step 2: Initialize seed pool
            success = fuzzer.initialize_seed_pool()
            self.assertTrue(success)
            
            # Verify initialization
            self.assertGreater(len(fuzzer.parsed_messages), 0)
            self.assertGreater(len(fuzzer.message_features), 0)
            self.assertGreater(len(fuzzer.current_clusters), 0)
            
            # Step 3: Run fuzzing
            fuzzing_success = fuzzer.start_fuzzing(max_iterations=5)
            self.assertTrue(fuzzing_success)
            
            # Step 4: Verify results
            report = fuzzer.get_comprehensive_report()
            
            self.assertGreater(report['summary']['total_iterations'], 0)
            self.assertEqual(report['summary']['messages_processed'], 15)
            self.assertGreater(report['summary']['clusters_created'], 0)
    
    def test_load_messages_different_formats(self):
        """Test loading messages in different formats"""
        fuzzer = CompoteFuzzer(self.test_config)
        
        # Test with list of dictionaries
        dict_messages = [
            {
                'message_type': 'propose',
                'round_number': 1,
                'sender_id': 'node_1',
                'timestamp': time.time()
            }
        ]
        
        loaded_count = fuzzer.load_messages(dict_messages)
        self.assertEqual(loaded_count, 1)
        
        # Test with RawMessage objects
        raw_messages = self.create_test_messages(3)
        loaded_count = fuzzer.load_messages(raw_messages)
        self.assertEqual(loaded_count, 3)
        
        # Total should be 4 messages
        self.assertEqual(len(fuzzer.raw_messages), 4)
    
    def test_empty_message_handling(self):
        """Test handling of empty message lists"""
        fuzzer = CompoteFuzzer(self.test_config)
        
        # Load empty list
        loaded_count = fuzzer.load_messages([])
        self.assertEqual(loaded_count, 0)
        
        # Try to initialize with no messages
        success = fuzzer.initialize_seed_pool()
        self.assertFalse(success)
    
    def test_malformed_message_handling(self):
        """Test handling of malformed messages"""
        fuzzer = CompoteFuzzer(self.test_config)
        
        # Mix of good and bad messages
        mixed_messages = [
            {'message_type': 'propose', 'round_number': 1, 'sender_id': 'node_1'},  # Good
            {'invalid': 'message'},  # Missing required fields
            None,  # Invalid message
            {'message_type': 'prevote', 'round_number': 2, 'sender_id': 'node_2'}  # Good
        ]
        
        loaded_count = fuzzer.load_messages(mixed_messages)
        
        # Should load at least the good messages
        self.assertGreater(loaded_count, 0)
        self.assertLessEqual(loaded_count, 4)
    
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test with invalid configuration
        invalid_config = {
            'clustering_eps': -1,  # Invalid: must be positive
            'max_iterations': -5,  # Invalid: must be positive
            'max_workers': 0       # Invalid: must be positive
        }
        
        # Should still create fuzzer with defaults
        fuzzer = CompoteFuzzer(invalid_config)
        self.assertIsNotNone(fuzzer)
    
    def test_state_save_and_load(self):
        """Test saving and loading fuzzer state"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with CompoteFuzzer(self.test_config) as fuzzer:
                # Setup fuzzer
                test_messages = self.create_test_messages(10)
                fuzzer.load_messages(test_messages)
                fuzzer.initialize_seed_pool()
                
                # Save state
                save_success = fuzzer.save_state(temp_path)
                self.assertTrue(save_success)
                self.assertTrue(os.path.exists(temp_path))
                
                # Verify saved file is valid JSON
                with open(temp_path, 'r') as f:
                    saved_data = json.load(f)
                
                self.assertIn('config', saved_data)
                self.assertIn('session_stats', saved_data)
                self.assertIn('cluster_count', saved_data)
                
                # Load state in new fuzzer
                new_fuzzer = CompoteFuzzer()
                load_success = new_fuzzer.load_state(temp_path)
                self.assertTrue(load_success)
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_callback_functionality(self):
        """Test callback function execution"""
        progress_calls = []
        result_calls = []
        fault_calls = []
        
        def progress_callback(iteration, max_iterations, result):
            progress_calls.append((iteration, max_iterations, result.success))
        
        def result_callback(result):
            result_calls.append(result.message_id)
        
        def fault_callback(result):
            fault_calls.append(result.message_id)
        
        with CompoteFuzzer(self.test_config) as fuzzer:
            # Set callbacks
            fuzzer.set_progress_callback(progress_callback)
            fuzzer.set_result_callback(result_callback)
            fuzzer.set_fault_callback(fault_callback)
            
            # Setup and run fuzzing
            test_messages = self.create_test_messages(8)
            fuzzer.load_messages(test_messages)
            fuzzer.initialize_seed_pool()
            fuzzer.start_fuzzing(max_iterations=3)
            
            # Verify callbacks were called
            self.assertGreater(len(result_calls), 0)
    
    def test_fuzzing_with_no_clusters(self):
        """Test fuzzing behavior when no clusters are formed"""
        # Create messages that are too dissimilar to cluster
        dissimilar_config = self.test_config.copy()
        dissimilar_config['clustering_eps'] = 0.01  # Very strict clustering
        dissimilar_config['clustering_min_samples'] = 10  # High minimum
        
        fuzzer = CompoteFuzzer(dissimilar_config)
        
        # Load few diverse messages
        test_messages = self.create_test_messages(3)
        fuzzer.load_messages(test_messages)
        
        # May succeed in initialization but with mostly noise clusters
        init_success = fuzzer.initialize_seed_pool()
        
        if init_success:
            # Should handle fuzzing gracefully even with poor clustering
            fuzzer.start_fuzzing(max_iterations=2)
            
            report = fuzzer.get_comprehensive_report()
            self.assertGreaterEqual(report['summary']['total_iterations'], 0)
    
    def test_concurrent_execution(self):
        """Test concurrent execution with multiple workers"""
        concurrent_config = self.test_config.copy()
        concurrent_config['max_workers'] = 2
        concurrent_config['max_iterations'] = 8
        
        with CompoteFuzzer(concurrent_config) as fuzzer:
            test_messages = self.create_test_messages(12)
            fuzzer.load_messages(test_messages)
            fuzzer.initialize_seed_pool()
            
            start_time = time.time()
            success = fuzzer.start_fuzzing()
            end_time = time.time()
            
            self.assertTrue(success)
            
            # Verify execution completed in reasonable time
            execution_time = end_time - start_time
            self.assertLess(execution_time, 30)  # Should complete within 30 seconds
    
    def test_performance_with_large_message_set(self):
        """Test performance with larger message sets"""
        large_config = self.test_config.copy()
        large_config['max_iterations'] = 20
        
        with CompoteFuzzer(large_config) as fuzzer:
            # Create larger message set
            large_message_set = self.create_test_messages(50)
            
            start_time = time.time()
            loaded_count = fuzzer.load_messages(large_message_set)
            load_time = time.time() - start_time
            
            self.assertEqual(loaded_count, 50)
            self.assertLess(load_time, 5.0)  # Should load within 5 seconds
            
            start_time = time.time()
            init_success = fuzzer.initialize_seed_pool()
            init_time = time.time() - start_time
            
            self.assertTrue(init_success)
            self.assertLess(init_time, 10.0)  # Should initialize within 10 seconds
            
            # Check that clusters were formed
            self.assertGreater(len(fuzzer.current_clusters), 0)
    
    def test_fuzzing_interruption(self):
        """Test graceful handling of fuzzing interruption"""
        with CompoteFuzzer(self.test_config) as fuzzer:
            test_messages = self.create_test_messages(10)
            fuzzer.load_messages(test_messages)
            fuzzer.initialize_seed_pool()
            
            # Start fuzzing in simulation mode
            fuzzer.start_fuzzing(max_iterations=100)  # Long running
            
            # Interrupt fuzzing
            fuzzer.stop_fuzzing()
            
            # Should stop gracefully
            self.assertFalse(fuzzer.is_running)
    
    def test_report_generation(self):
        """Test comprehensive report generation"""
        with CompoteFuzzer(self.test_config) as fuzzer:
            test_messages = self.create_test_messages(15)
            fuzzer.load_messages(test_messages)
            fuzzer.initialize_seed_pool()
            fuzzer.start_fuzzing(max_iterations=5)
            
            report = fuzzer.get_comprehensive_report()
            
            # Check report structure
            expected_sections = [
                'summary', 'algorithm_stats', 'performance', 'current_state'
            ]
            
            for section in expected_sections:
                self.assertIn(section, report)
            
            # Check summary content
            summary = report['summary']
            self.assertIn('total_runtime', summary)
            self.assertIn('total_iterations', summary)
            self.assertIn('messages_processed', summary)
            self.assertIn('clusters_created', summary)
            self.assertIn('faults_discovered', summary)
            
            # Check algorithm stats
            algo_stats = report['algorithm_stats']
            self.assertIn('parsing', algo_stats)
            self.assertIn('feature_extraction', algo_stats)
            self.assertIn('clustering', algo_stats)
            self.assertIn('priority_calculation', algo_stats)
            self.assertIn('state_analysis', algo_stats)
            
            # Check performance metrics
            performance = report['performance']
            self.assertIn('iterations_per_second', performance)
            self.assertIn('fault_discovery_rate', performance)
            self.assertIn('success_rate', performance)
    
    def test_edge_cases(self):
        """Test various edge cases"""
        fuzzer = CompoteFuzzer(self.test_config)
        
        # Test with single message
        single_message = self.create_test_messages(1)
        loaded_count = fuzzer.load_messages(single_message)
        self.assertEqual(loaded_count, 1)
        
        # Try to initialize with single message
        init_success = fuzzer.initialize_seed_pool()
        # May or may not succeed depending on clustering parameters
        
        # Test loading additional messages after initialization
        additional_messages = self.create_test_messages(5)
        additional_count = fuzzer.load_messages(additional_messages)
        self.assertEqual(additional_count, 5)
        self.assertEqual(len(fuzzer.raw_messages), 6)  # 1 + 5


class TestCompoteFuzzerErrorHandling(unittest.TestCase):
    """Test error handling in COMPOTE fuzzer"""
    
    def test_initialization_without_messages(self):
        """Test initialization without loading messages"""
        fuzzer = CompoteFuzzer({'simulation_mode': True})
        
        # Should fail gracefully
        success = fuzzer.initialize_seed_pool()
        self.assertFalse(success)
    
    def test_fuzzing_without_initialization(self):
        """Test starting fuzzing without initialization"""
        fuzzer = CompoteFuzzer({'simulation_mode': True})
        
        # Should fail gracefully
        success = fuzzer.start_fuzzing()
        self.assertFalse(success)
    
    def test_invalid_file_paths(self):
        """Test handling of invalid file paths"""
        fuzzer = CompoteFuzzer({'simulation_mode': True})
        
        # Test saving to invalid path
        save_success = fuzzer.save_state('/invalid/path/state.json')
        self.assertFalse(save_success)
        
        # Test loading from non-existent file
        load_success = fuzzer.load_state('/non/existent/file.json')
        self.assertFalse(load_success)
    
    @patch('compote.core.parser.MessageParser.parse')
    def test_parsing_failures(self, mock_parse):
        """Test handling of parsing failures"""
        # Make parser always fail
        mock_parse.side_effect = ValueError("Parsing failed")
        
        fuzzer = CompoteFuzzer({'simulation_mode': True})
        
        test_messages = [
            RawMessage(b'{"test": "data"}', time.time(), 'test_node')
        ]
        
        loaded_count = fuzzer.load_messages(test_messages)
        self.assertEqual(loaded_count, 1)  # Still loads raw message
        
        # Should fail during initialization due to parsing failures
        success = fuzzer.initialize_seed_pool()
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()