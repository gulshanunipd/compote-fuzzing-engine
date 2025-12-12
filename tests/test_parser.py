"""
Unit tests for the MessageParser class.

Tests the selective and conditional parsing algorithms for various message formats.
"""

import unittest
import json
import time
import struct
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compote.core.parser import MessageParser
from compote.core.types import RawMessage, ParsedMessage, MessageType, NodeRole


class TestMessageParser(unittest.TestCase):
    """Test cases for MessageParser"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = MessageParser()
        self.test_timestamp = time.time()
    
    def create_raw_message(self, data: dict, message_id: str = "test_msg") -> RawMessage:
        """Helper to create RawMessage from dict"""
        raw_data = json.dumps(data).encode('utf-8')
        return RawMessage(
            data=raw_data,
            timestamp=self.test_timestamp,
            source_id=data.get('sender_id', 'test_node'),
            message_id=message_id
        )
    
    def test_parse_propose_message(self):
        """Test parsing of propose messages"""
        propose_data = {
            'message_type': 'propose',
            'round_number': 5,
            'view_number': 1,
            'block_height': 100,
            'sender_id': 'leader_node',
            'role': 'leader',
            'timestamp': self.test_timestamp,
            'signature': 'propose_signature',
            'proposal_hash': 'prop_hash_123',
            'block_data': 'block_payload_data'
        }
        
        raw_msg = self.create_raw_message(propose_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        # Verify basic fields
        self.assertEqual(parsed_msg.message_type, MessageType.PROPOSE)
        self.assertEqual(parsed_msg.round_number, 5)
        self.assertEqual(parsed_msg.view_number, 1)
        self.assertEqual(parsed_msg.block_height, 100)
        self.assertEqual(parsed_msg.sender_id, 'leader_node')
        self.assertEqual(parsed_msg.sender_role, NodeRole.LEADER)
        self.assertEqual(parsed_msg.signature, 'propose_signature')
        
        # Verify type-specific fields
        self.assertIn('proposal_hash', parsed_msg.additional_fields)
        self.assertIn('block_data', parsed_msg.additional_fields)
        self.assertIn('proposal_size', parsed_msg.additional_fields)
        self.assertEqual(parsed_msg.additional_fields['proposal_hash'], 'prop_hash_123')
    
    def test_parse_prevote_message(self):
        """Test parsing of prevote messages"""
        prevote_data = {
            'message_type': 'prevote',
            'round_number': 3,
            'view_number': 0,
            'block_height': 50,
            'sender_id': 'validator_1',
            'role': 'validator',
            'timestamp': self.test_timestamp,
            'block_hash': 'voted_block_hash',
            'vote_type': 'yes'
        }
        
        raw_msg = self.create_raw_message(prevote_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        self.assertEqual(parsed_msg.message_type, MessageType.PREVOTE)
        self.assertEqual(parsed_msg.sender_role, NodeRole.VALIDATOR)
        self.assertIn('block_hash', parsed_msg.additional_fields)
        self.assertIn('vote_value', parsed_msg.additional_fields)
        self.assertTrue(parsed_msg.additional_fields['vote_value'])  # Non-empty hash = True
    
    def test_parse_commit_message(self):
        """Test parsing of commit messages"""
        commit_data = {
            'message_type': 'commit',
            'round_number': 10,
            'view_number': 2,
            'block_height': 200,
            'sender_id': 'aggregator_node',
            'role': 'validator',
            'timestamp': self.test_timestamp,
            'block_hash': 'committed_block',
            'commit_signatures': ['sig1', 'sig2', 'sig3', 'sig4']
        }
        
        raw_msg = self.create_raw_message(commit_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        self.assertEqual(parsed_msg.message_type, MessageType.COMMIT)
        self.assertIn('commit_signatures', parsed_msg.additional_fields)
        self.assertIn('signature_count', parsed_msg.additional_fields)
        self.assertEqual(parsed_msg.additional_fields['signature_count'], 4)
    
    def test_parse_round_change_message(self):
        """Test parsing of round change messages"""
        round_change_data = {
            'message_type': 'round_change',
            'round_number': 7,
            'view_number': 1,
            'block_height': 150,
            'sender_id': 'timeout_node',
            'role': 'validator',
            'timestamp': self.test_timestamp,
            'new_round': 8,
            'justification': 'timeout_detected'
        }
        
        raw_msg = self.create_raw_message(round_change_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        self.assertEqual(parsed_msg.message_type, MessageType.ROUND_CHANGE)
        self.assertIn('new_round', parsed_msg.additional_fields)
        self.assertIn('justification', parsed_msg.additional_fields)
        self.assertEqual(parsed_msg.additional_fields['new_round'], 8)
    
    def test_parse_binary_format(self):
        """Test parsing of binary format messages"""
        # Create binary message: [type][round][view][payload]
        msg_type = 0  # propose
        round_num = 42
        view_num = 3
        payload = b"binary_payload_data"
        
        binary_data = struct.pack('!III', msg_type, round_num, view_num) + payload
        
        raw_msg = RawMessage(
            data=binary_data,
            timestamp=self.test_timestamp,
            source_id='binary_node'
        )
        
        parsed_msg = self.parser.parse(raw_msg, format_type='binary')
        
        self.assertEqual(parsed_msg.message_type, MessageType.PROPOSE)
        self.assertEqual(parsed_msg.round_number, 42)
        self.assertEqual(parsed_msg.view_number, 3)
        self.assertIn('payload', parsed_msg.additional_fields)
    
    def test_parse_custom_format(self):
        """Test parsing of custom format messages"""
        custom_data = "message_type=prevote\nround_number=15\nsender_id=custom_node\nblock_hash=custom_hash"
        
        raw_msg = RawMessage(
            data=custom_data.encode('utf-8'),
            timestamp=self.test_timestamp,
            source_id='custom_node'
        )
        
        parsed_msg = self.parser.parse(raw_msg, format_type='custom')
        
        self.assertEqual(parsed_msg.message_type, MessageType.PREVOTE)
        self.assertEqual(parsed_msg.round_number, 15)
        self.assertEqual(parsed_msg.sender_id, 'custom_node')
    
    def test_selective_parsing_performance(self):
        """Test that selective parsing is O(1) per message"""
        # This is more of a conceptual test - we verify the parser 
        # processes common fields efficiently
        
        message_data = {
            'message_type': 'propose',
            'round_number': 1,
            'view_number': 0,
            'block_height': 1,
            'sender_id': 'test_node',
            'timestamp': self.test_timestamp,
            'signature': 'test_sig',
            # Add many additional fields that should not slow down common field extraction
            **{f'extra_field_{i}': f'value_{i}' for i in range(100)}
        }
        
        raw_msg = self.create_raw_message(message_data)
        
        # Time the parsing - should be fast regardless of additional fields
        start_time = time.perf_counter()
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        end_time = time.perf_counter()
        
        parsing_time = end_time - start_time
        
        # Verify parsing worked correctly
        self.assertEqual(parsed_msg.message_type, MessageType.PROPOSE)
        self.assertEqual(parsed_msg.round_number, 1)
        
        # Parsing should be very fast (less than 1ms for this simple case)
        self.assertLess(parsing_time, 0.001)
    
    def test_conditional_parsing_efficiency(self):
        """Test that conditional parsing only processes relevant fields"""
        # Test that propose-specific fields are only processed for propose messages
        
        propose_data = {
            'message_type': 'propose',
            'round_number': 1,
            'view_number': 0,
            'block_height': 1,
            'sender_id': 'test_node',
            'proposal_hash': 'test_proposal',
            'block_data': 'test_block',
            # Fields that shouldn't be processed for propose messages
            'vote_type': 'should_be_ignored',
            'commit_signatures': ['should', 'be', 'ignored']
        }
        
        raw_msg = self.create_raw_message(propose_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        # Should have propose-specific fields
        self.assertIn('proposal_hash', parsed_msg.additional_fields)
        self.assertIn('block_data', parsed_msg.additional_fields)
        self.assertIn('proposal_size', parsed_msg.additional_fields)
        
        # Should NOT have vote-specific or commit-specific fields
        self.assertNotIn('vote_value', parsed_msg.additional_fields)
        self.assertNotIn('signature_count', parsed_msg.additional_fields)
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON messages"""
        malformed_data = b'{"invalid": json, "missing_quote: true}'
        
        raw_msg = RawMessage(
            data=malformed_data,
            timestamp=self.test_timestamp,
            source_id='test_node'
        )
        
        with self.assertRaises(ValueError):
            self.parser.parse(raw_msg, format_type='json')
    
    def test_missing_required_fields(self):
        """Test parsing with missing required fields"""
        minimal_data = {
            'message_type': 'propose'
            # Missing most required fields
        }
        
        raw_msg = self.create_raw_message(minimal_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        # Should still parse with default values
        self.assertEqual(parsed_msg.message_type, MessageType.PROPOSE)
        self.assertEqual(parsed_msg.round_number, 0)  # Default
        self.assertEqual(parsed_msg.view_number, 0)   # Default
        self.assertEqual(parsed_msg.sender_id, 'unknown')  # Default
    
    def test_alternative_field_names(self):
        """Test parsing with alternative field names"""
        alt_data = {
            'type': 'prevote',  # Alternative to 'message_type'
            'round': 5,         # Alternative to 'round_number'
            'view': 2,          # Alternative to 'view_number'
            'height': 100,      # Alternative to 'block_height'
            'from': 'alt_node', # Alternative to 'sender_id'
            'sig': 'alt_sig'    # Alternative to 'signature'
        }
        
        raw_msg = self.create_raw_message(alt_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        self.assertEqual(parsed_msg.message_type, MessageType.PREVOTE)
        self.assertEqual(parsed_msg.round_number, 5)
        self.assertEqual(parsed_msg.view_number, 2)
        self.assertEqual(parsed_msg.block_height, 100)
        self.assertEqual(parsed_msg.sender_id, 'alt_node')
        self.assertEqual(parsed_msg.signature, 'alt_sig')
    
    def test_unsupported_format_type(self):
        """Test error handling for unsupported format types"""
        raw_msg = self.create_raw_message({'message_type': 'test'})
        
        with self.assertRaises(ValueError):
            self.parser.parse(raw_msg, format_type='unsupported_format')
    
    def test_payload_hash_generation(self):
        """Test that payload hash is generated correctly"""
        test_data = {
            'message_type': 'propose',
            'round_number': 1,
            'view_number': 0,
            'block_height': 1,
            'sender_id': 'test_node'
        }
        
        raw_msg = self.create_raw_message(test_data)
        parsed_msg = self.parser.parse(raw_msg, format_type='json')
        
        # Payload hash should be generated
        self.assertIsNotNone(parsed_msg.payload_hash)
        self.assertIsInstance(parsed_msg.payload_hash, str)
        self.assertEqual(len(parsed_msg.payload_hash), 64)  # SHA256 hex length
    
    def test_parser_statistics(self):
        """Test parser statistics functionality"""
        stats = self.parser.get_parsing_stats()
        
        self.assertIn('common_fields_count', stats)
        self.assertIn('supported_message_types', stats)
        self.assertIn('supported_formats', stats)
        
        self.assertGreater(stats['common_fields_count'], 0)
        self.assertGreater(stats['supported_message_types'], 0)
        self.assertGreater(stats['supported_formats'], 0)
    
    def test_message_type_determination(self):
        """Test message type determination with various formats"""
        test_cases = [
            ('propose', MessageType.PROPOSE),
            ('proposal', MessageType.PROPOSE),
            ('prevote', MessageType.PREVOTE),
            ('pre-vote', MessageType.PREVOTE),
            ('precommit', MessageType.PRECOMMIT),
            ('pre-commit', MessageType.PRECOMMIT),
            ('commit', MessageType.COMMIT),
            ('round_change', MessageType.ROUND_CHANGE),
            ('roundchange', MessageType.ROUND_CHANGE),
            ('new_view', MessageType.NEW_VIEW),
            ('newview', MessageType.NEW_VIEW),
            ('unknown_type', MessageType.OTHER)
        ]
        
        for type_str, expected_type in test_cases:
            with self.subTest(message_type=type_str):
                data = {
                    'message_type': type_str,
                    'round_number': 1,
                    'sender_id': 'test_node'
                }
                
                raw_msg = self.create_raw_message(data)
                parsed_msg = self.parser.parse(raw_msg, format_type='json')
                
                self.assertEqual(parsed_msg.message_type, expected_type)
    
    def test_node_role_determination(self):
        """Test node role determination"""
        test_cases = [
            ('leader', NodeRole.LEADER),
            ('validator', NodeRole.VALIDATOR),
            ('observer', NodeRole.OBSERVER),
            ('proposer', NodeRole.PROPOSER),
            ('unknown_role', NodeRole.UNKNOWN)
        ]
        
        for role_str, expected_role in test_cases:
            with self.subTest(role=role_str):
                data = {
                    'message_type': 'propose',
                    'role': role_str,
                    'sender_id': 'test_node'
                }
                
                raw_msg = self.create_raw_message(data)
                parsed_msg = self.parser.parse(raw_msg, format_type='json')
                
                self.assertEqual(parsed_msg.sender_role, expected_role)


if __name__ == '__main__':
    unittest.main()