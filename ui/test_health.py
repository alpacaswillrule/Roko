import requests
import json
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

def get_health_from_api():
    """Get health data directly from health.petals.dev"""
    try:
        # Try the API endpoint instead of the web interface
        response = requests.get('https://health.petals.dev/api/v1/status')
        print(f"API Response Status Code: {response.status_code}")
        print(f"API Response Content: {response.text[:500]}...")  # Print first 500 chars of response
        
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError as je:
                return f"Error parsing JSON: {str(je)}\nRaw response: {response.text[:500]}..."
        else:
            return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"Error fetching from API: {str(e)}"

def get_health_from_model():
    """Get health data using the current implementation with AutoDistributedModelForCausalLM"""
    print("\nInitializing model...")
    try:
        model_name = "bigscience/bloom-560m"
        print(f"Loading model: {model_name}")
        model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
        print("Model initialized successfully")
        
        # Print model attributes for debugging
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):
                continue
            try:
                val = getattr(model, attr)
                print(f"{attr}: {type(val)}")
            except:
                pass
        
        # Group servers by model
        model_groups = {}
        print("\nChecking for active servers...")
        
        # Try different attributes that might contain server info
        servers = []
        if hasattr(model, '_remote_forward_backward'):
            print("Found _remote_forward_backward attribute")
            if hasattr(model._remote_forward_backward, 'active_servers'):
                servers = model._remote_forward_backward.active_servers
                print(f"Found {len(servers)} active servers")
            else:
                print("No active_servers attribute in _remote_forward_backward")
        elif hasattr(model, 'active_servers'):
            print("Found active_servers attribute directly on model")
            servers = model.active_servers
            print(f"Found {len(servers)} active servers")
        else:
            print("No server attributes found")
            servers = []
        
        # Process each server
        for i, server in enumerate(servers):
            try:
                # Get server metrics from the server's handler
                handler = server.handler if hasattr(server, 'handler') else None
                
                # Extract metrics with safe fallbacks
                throughput = getattr(handler, 'throughput', 0) if handler else 0
                cache_info = getattr(handler, 'cache_info', None) if handler else None
                cache_tokens = cache_info.available if cache_info else 0
                max_cache = cache_info.total if cache_info else 100
                
                # Get server details
                server_info = {
                    'id': getattr(server, 'peer_id', f"Server {i}")[:8],
                    'full_id': getattr(server, 'peer_id', None),
                    'state': get_server_state(server),
                    'version': getattr(server, 'version', 'Unknown'),
                    'throughput': throughput,
                    'precision': getattr(handler, 'precision', 'Unknown') if handler else 'Unknown',
                    'cache_tokens_left': cache_tokens,
                    'max_cache_tokens': max_cache
                }
                
                # Group by model
                model_type = getattr(server, 'model_name', 'Unknown Model')
                if model_type not in model_groups:
                    model_groups[model_type] = []
                model_groups[model_type].append(server_info)
                
            except Exception as e:
                print(f"Error getting server {i} data: {str(e)}")
                server_info = {
                    'id': f"Server {i}",
                    'state': 'ERROR',
                    'version': 'Unknown',
                    'throughput': 0,
                    'precision': 'Unknown',
                    'cache_tokens_left': 0,
                    'max_cache_tokens': 100
                }
                
                if 'Error' not in model_groups:
                    model_groups['Error'] = []
                model_groups['Error'].append(server_info)
        
        return model_groups
    except Exception as e:
        return f"Error getting model health data: {str(e)}"

def get_server_state(server):
    """Determine server state based on various metrics"""
    try:
        if not hasattr(server, 'is_active'):
            return 'UNREACHABLE'
        if not server.is_active:
            return 'OFFLINE'
        if hasattr(server, 'is_joining') and server.is_joining:
            return 'JOINING'
        return 'ONLINE'
    except:
        return 'UNREACHABLE'

def main():
    print("\n=== Testing Petals Network Health ===\n")
    
    print("1. Getting health data from health.petals.dev:")
    api_data = get_health_from_api()
    print(json.dumps(api_data, indent=2))
    
    print("\n2. Getting health data from model implementation:")
    model_data = get_health_from_model()
    print(json.dumps(model_data, indent=2))
    
    print("\nComparison complete. Check the outputs above to compare the data from both sources.")

if __name__ == "__main__":
    main()
