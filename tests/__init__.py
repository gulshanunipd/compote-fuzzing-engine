"""
COMPOTE Test Suite

Comprehensive unit and integration tests for the COMPOTE fuzzing engine.

Test Categories:
- Unit Tests: Individual algorithm components
- Integration Tests: Complete fuzzing workflows  
- Performance Tests: Scalability and efficiency
- Error Handling Tests: Edge cases and failures

Usage:
    python tests/run_tests.py                # Run all tests
    python tests/run_tests.py --coverage     # Run with coverage
    python tests/run_tests.py --performance  # Performance tests only
    python tests/run_tests.py --pattern "test_parser*"  # Specific tests
"""

# Test configuration
TEST_CONFIG = {
    'timeout': 30.0,          # Test timeout in seconds
    'max_test_messages': 100, # Maximum messages for performance tests
    'simulation_mode': True,  # Always use simulation in tests
    'temp_dir': '/tmp/compote_tests',
    'log_level': 'WARNING'    # Reduce logging noise in tests
}

# Test data templates
TEST_MESSAGE_TEMPLATES = {
    'propose': {
        'message_type': 'propose',
        'round_number': 1,
        'view_number': 0,
        'block_height': 1,
        'sender_id': 'leader_node',
        'role': 'leader',
        'proposal_hash': 'test_proposal',
        'block_data': 'test_block_data'
    },
    'prevote': {
        'message_type': 'prevote',
        'round_number': 1,
        'view_number': 0,
        'block_height': 1,
        'sender_id': 'validator_node',
        'role': 'validator',
        'block_hash': 'test_block_hash',
        'vote_type': 'yes'
    },
    'precommit': {
        'message_type': 'precommit',
        'round_number': 1,
        'view_number': 0,
        'block_height': 1,
        'sender_id': 'validator_node',
        'role': 'validator',
        'block_hash': 'test_block_hash'
    },
    'commit': {
        'message_type': 'commit',
        'round_number': 1,
        'view_number': 0,
        'block_height': 1,
        'sender_id': 'validator_node',
        'role': 'validator',
        'block_hash': 'test_block_hash',
        'commit_signatures': ['sig1', 'sig2', 'sig3']
    },
    'round_change': {
        'message_type': 'round_change',
        'round_number': 2,
        'view_number': 1,
        'block_height': 1,
        'sender_id': 'timeout_node',
        'role': 'validator',
        'new_round': 3,
        'justification': 'timeout_detected'
    }
}

# Performance test benchmarks
PERFORMANCE_BENCHMARKS = {
    'parsing_throughput_min': 100,      # messages/second
    'feature_extraction_min': 50,       # messages/second  
    'clustering_max_time': 5.0,         # seconds for 100 messages
    'priority_calculation_max': 2.0,    # seconds for 100 messages
    'end_to_end_max_time': 30.0        # seconds for full workflow
}