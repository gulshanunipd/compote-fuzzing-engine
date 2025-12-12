"""
Algorithm 1: Selective & Conditional Parsing

Purpose: Efficiently extract only critical fields from consensus messages.
Complexity: O(1) per message through selective and conditional parsing.
"""

import json
import struct
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from ..core.types import RawMessage, ParsedMessage, MessageType, NodeRole


class MessageParser:
    """
    Implements selective and conditional parsing for consensus messages.
    
    The parser uses a two-stage approach:
    1. Selective parsing: Extract fields common to all message types (O(1))
    2. Conditional parsing: Extract fields specific to message type (O(1))
    """
    
    def __init__(self):
        # Common fields present in all message types (selective parsing)
        self.common_fields = {
            'message_type', 'timestamp', 'sender_id', 'round_number', 
            'view_number', 'block_height', 'signature'
        }
        
        # Message-type specific fields (conditional parsing)
        self.type_specific_fields = {
            MessageType.PROPOSE: {'proposal_hash', 'block_data', 'proof'},
            MessageType.PREVOTE: {'block_hash', 'vote_type'},
            MessageType.PRECOMMIT: {'block_hash', 'commit_proof'},
            MessageType.COMMIT: {'block_hash', 'commit_signatures'},
            MessageType.ROUND_CHANGE: {'new_round', 'justification'},
            MessageType.NEW_VIEW: {'view_data', 'prepare_messages'},
            MessageType.PREPARE: {'sequence_number', 'request_digest'},
            MessageType.VIEW_CHANGE: {'new_view_number', 'checkpoint_proof'}
        }
        
        # Parser functions for different message formats
        self.parsers = {
            'json': self._parse_json,
            'binary': self._parse_binary, 
            'protobuf': self._parse_protobuf,
            'custom': self._parse_custom
        }
    
    def parse(self, raw_message: RawMessage, format_type: str = 'json') -> ParsedMessage:
        """
        Main parsing function implementing selective and conditional parsing.
        
        Args:
            raw_message: Raw consensus message
            format_type: Format of the message ('json', 'binary', 'protobuf', 'custom')
            
        Returns:
            ParsedMessage with extracted fields
        """
        if format_type not in self.parsers:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        # Stage 1: Selective parsing - extract common fields (O(1))
        parsed_data = self.parsers[format_type](raw_message.data)
        common_data = self._selective_parse(parsed_data)
        
        # Determine message type for conditional parsing
        msg_type = self._determine_message_type(common_data)
        
        # Stage 2: Conditional parsing - extract type-specific fields (O(1))
        specific_data = self._conditional_parse(parsed_data, msg_type)
        
        # Combine and create ParsedMessage
        all_data = {**common_data, **specific_data}
        
        return self._create_parsed_message(all_data, raw_message, msg_type)
    
    def _selective_parse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selective parsing: Extract only common fields (O(1) operation).
        
        This function efficiently extracts fields that are present in all
        consensus message types, avoiding unnecessary processing.
        """
        common_data = {}
        
        # Extract common fields with default values
        common_data['sender_id'] = data.get('sender_id', data.get('from', 'unknown'))
        common_data['timestamp'] = data.get('timestamp', time.time())
        common_data['round_number'] = int(data.get('round', data.get('round_number', 0)))
        common_data['view_number'] = int(data.get('view', data.get('view_number', 0)))
        common_data['block_height'] = int(data.get('height', data.get('block_height', 0)))
        common_data['signature'] = data.get('signature', data.get('sig'))
        
        return common_data
    
    def _conditional_parse(self, data: Dict[str, Any], msg_type: MessageType) -> Dict[str, Any]:
        """
        Conditional parsing: Extract fields specific to message type (O(1) operation).
        
        Only processes fields relevant to the specific message type,
        maintaining constant time complexity.
        """
        specific_data = {}
        
        if msg_type not in self.type_specific_fields:
            return specific_data
        
        required_fields = self.type_specific_fields[msg_type]
        
        # Extract type-specific fields
        for field in required_fields:
            if field in data:
                specific_data[field] = data[field]
        
        # Add computed fields based on message type
        if msg_type == MessageType.PROPOSE:
            specific_data['proposal_size'] = len(str(data.get('block_data', '')))
            
        elif msg_type in [MessageType.PREVOTE, MessageType.PRECOMMIT]:
            specific_data['vote_value'] = data.get('block_hash', '') != ''
            
        elif msg_type == MessageType.COMMIT:
            commit_sigs = data.get('commit_signatures', [])
            specific_data['signature_count'] = len(commit_sigs) if isinstance(commit_sigs, list) else 0
        
        return specific_data
    
    def _determine_message_type(self, data: Dict[str, Any]) -> MessageType:
        """Determine message type from parsed data"""
        msg_type_str = data.get('message_type', data.get('type', '')).lower()
        
        # Map string representations to MessageType enum
        type_mapping = {
            'propose': MessageType.PROPOSE,
            'proposal': MessageType.PROPOSE,
            'prevote': MessageType.PREVOTE,
            'pre-vote': MessageType.PREVOTE,
            'precommit': MessageType.PRECOMMIT, 
            'pre-commit': MessageType.PRECOMMIT,
            'commit': MessageType.COMMIT,
            'round_change': MessageType.ROUND_CHANGE,
            'roundchange': MessageType.ROUND_CHANGE,
            'new_view': MessageType.NEW_VIEW,
            'newview': MessageType.NEW_VIEW,
            'prepare': MessageType.PREPARE,
            'view_change': MessageType.VIEW_CHANGE,
            'viewchange': MessageType.VIEW_CHANGE
        }
        
        return type_mapping.get(msg_type_str, MessageType.OTHER)
    
    def _determine_node_role(self, data: Dict[str, Any]) -> NodeRole:
        """Determine node role from parsed data"""
        role_str = data.get('role', data.get('node_role', '')).lower()
        
        role_mapping = {
            'leader': NodeRole.LEADER,
            'validator': NodeRole.VALIDATOR,
            'observer': NodeRole.OBSERVER,
            'proposer': NodeRole.PROPOSER
        }
        
        return role_mapping.get(role_str, NodeRole.UNKNOWN)
    
    def _create_parsed_message(self, data: Dict[str, Any], raw_msg: RawMessage, 
                             msg_type: MessageType) -> ParsedMessage:
        """Create ParsedMessage from extracted data"""
        
        # Generate payload hash
        payload_data = json.dumps(data, sort_keys=True).encode()
        payload_hash = hashlib.sha256(payload_data).hexdigest()
        
        # Determine sender role
        sender_role = self._determine_node_role(data)
        
        # Create ParsedMessage
        parsed_msg = ParsedMessage(
            message_id=raw_msg.message_id,
            message_type=msg_type,
            round_number=data['round_number'],
            view_number=data['view_number'],
            block_height=data['block_height'],
            sender_id=data['sender_id'],
            sender_role=sender_role,
            timestamp=data['timestamp'],
            payload_hash=payload_hash,
            signature=data.get('signature'),
            additional_fields={k: v for k, v in data.items() 
                             if k not in {'round_number', 'view_number', 'block_height', 
                                        'sender_id', 'timestamp', 'signature'}}
        )
        
        return parsed_msg
    
    # Format-specific parsers
    
    def _parse_json(self, data: bytes) -> Dict[str, Any]:
        """Parse JSON format messages"""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to parse JSON message: {e}")
    
    def _parse_binary(self, data: bytes) -> Dict[str, Any]:
        """Parse binary format messages (custom binary protocol)"""
        try:
            # Example binary format: 
            # [4 bytes: message_type][4 bytes: round][4 bytes: view][remaining: payload]
            if len(data) < 12:
                raise ValueError("Binary message too short")
            
            msg_type_int = struct.unpack('!I', data[:4])[0]
            round_num = struct.unpack('!I', data[4:8])[0] 
            view_num = struct.unpack('!I', data[8:12])[0]
            payload = data[12:]
            
            # Map integer to message type
            type_map = {0: 'propose', 1: 'prevote', 2: 'precommit', 3: 'commit'}
            
            return {
                'message_type': type_map.get(msg_type_int, 'other'),
                'round_number': round_num,
                'view_number': view_num,
                'payload': payload.hex(),
                'sender_id': f"node_{hash(data) % 1000}",  # Derive sender from hash
                'timestamp': time.time()
            }
            
        except struct.error as e:
            raise ValueError(f"Failed to parse binary message: {e}")
    
    def _parse_protobuf(self, data: bytes) -> Dict[str, Any]:
        """Parse Protocol Buffer format messages"""
        # Placeholder for protobuf parsing
        # In real implementation, would use generated protobuf classes
        try:
            # For demo, assume simple key-value format
            decoded = data.decode('utf-8', errors='ignore')
            # Simple parsing logic - in reality would use protobuf library
            return {'raw_protobuf': decoded, 'message_type': 'propose', 
                   'round_number': 1, 'view_number': 1, 'sender_id': 'proto_node'}
        except Exception as e:
            raise ValueError(f"Failed to parse protobuf message: {e}")
    
    def _parse_custom(self, data: bytes) -> Dict[str, Any]:
        """Parse custom format messages"""
        # Placeholder for custom protocol parsing
        # Can be extended for specific consensus protocols
        try:
            # Example: assume newline-delimited key=value format
            decoded = data.decode('utf-8')
            result = {}
            for line in decoded.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    result[key.strip()] = value.strip()
            
            # Ensure required fields exist
            if 'message_type' not in result:
                result['message_type'] = 'other'
            if 'round_number' not in result:
                result['round_number'] = '0'
            if 'sender_id' not in result:
                result['sender_id'] = 'unknown'
                
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to parse custom message: {e}")
    
    def get_parsing_stats(self) -> Dict[str, int]:
        """Get statistics about parsing operations"""
        return {
            'common_fields_count': len(self.common_fields),
            'supported_message_types': len(self.type_specific_fields),
            'supported_formats': len(self.parsers)
        }