"""Utility functions for the A2A agent"""

import re
from typing import Dict, Any, List, Union


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def convert_keys_to_camel(data: Union[Dict[str, Any], List[Any], Any]) -> Union[Dict[str, Any], List[Any], Any]:
    """Recursively convert all dictionary keys from snake_case to camelCase"""
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Special case for some known fields that should remain unchanged
            if key in ['jsonrpc', 'id']:
                new_key = key
            else:
                new_key = snake_to_camel(key)
            
            # Recursively convert nested structures
            new_dict[new_key] = convert_keys_to_camel(value)
        
        return new_dict
    elif isinstance(data, list):
        return [convert_keys_to_camel(item) for item in data]
    else:
        return data


def fix_a2a_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix A2A response format issues including snake_case to camelCase conversion"""
    # Convert snake_case to camelCase
    fixed_data = convert_keys_to_camel(response_data)
    
    # Fix specific known issues
    if 'result' in fixed_data and isinstance(fixed_data['result'], dict):
        result = fixed_data['result']
        
        # Fix artifact-update to have correct structure
        if result.get('kind') == 'artifact-update':
            # Ensure artifact has required fields
            if 'artifact' in result and isinstance(result['artifact'], dict):
                artifact = result['artifact']
                # Add artifactId if missing
                if 'artifactId' not in artifact and 'artifact_id' not in artifact:
                    # Generate a unique artifact ID
                    import uuid
                    artifact['artifactId'] = f'artifact_{uuid.uuid4().hex[:8]}'
                
                # Ensure parts is a list
                if 'parts' in artifact and not isinstance(artifact['parts'], list):
                    artifact['parts'] = [artifact['parts']]
                
                # Ensure each part has required fields
                if 'parts' in artifact:
                    for i, part in enumerate(artifact['parts']):
                        if isinstance(part, dict):
                            # Ensure part has kind field
                            if 'kind' not in part:
                                # Detect kind based on content
                                if 'text' in part:
                                    part['kind'] = 'text'
                                elif 'fileName' in part:
                                    part['kind'] = 'file'
                                else:
                                    part['kind'] = 'text'
                                    part['text'] = ''
    
    return fixed_data


def preprocess_chunk_data(chunk_data: Any) -> Any:
    """Preprocess chunk data to fix format issues before parsing"""
    if hasattr(chunk_data, '__dict__'):
        # Convert object to dict
        data = chunk_data.__dict__
    elif isinstance(chunk_data, dict):
        data = chunk_data
    else:
        return chunk_data
    
    # Apply fixes
    return fix_a2a_response(data)


def validate_and_fix_artifact_update(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix artifact update messages"""
    if not isinstance(data, dict):
        return data
    
    # Check if this is an artifact update
    result = data.get('result', {})
    if result.get('kind') == 'artifact-update':
        artifact = result.get('artifact', {})
        
        # Ensure artifact has artifactId
        if 'artifactId' not in artifact:
            import uuid
            artifact['artifactId'] = f'artifact_{uuid.uuid4().hex[:8]}'
        
        # Ensure artifact has parts
        if 'parts' not in artifact:
            artifact['parts'] = []
        
        # Validate parts
        valid_parts = []
        for part in artifact.get('parts', []):
            if isinstance(part, dict):
                # Ensure part has kind
                if 'kind' not in part:
                    if 'text' in part:
                        part['kind'] = 'text'
                    else:
                        part['kind'] = 'text'
                        part['text'] = ''
                valid_parts.append(part)
        
        artifact['parts'] = valid_parts
        result['artifact'] = artifact
        data['result'] = result
    
    return data