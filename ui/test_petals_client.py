import requests
import json
from datetime import datetime

def get_network_health():
    """Get health data from Petals health monitor API"""
    try:
        # Try the main state endpoint
        print("Fetching network state from health monitor API...")
        response = requests.get('https://health.petals.dev/api/v1/state')
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content: {response.text[:500]}...")  # Print first 500 chars
        
        if response.status_code == 200:
            try:
                data = response.json()
                return data
            except json.JSONDecodeError as je:
                print(f"JSON decode error: {str(je)}")
                # Try parsing the response as text
                return {"error": "Failed to parse JSON", "raw_response": response.text[:1000]}
        else:
            # Try the root endpoint to see what's available
            print("\nTrying root endpoint...")
            root_response = requests.get('https://health.petals.dev/')
            print(f"Root endpoint status: {root_response.status_code}")
            print(f"Root endpoint content: {root_response.text[:500]}...")
            
            return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"Error fetching from API: {str(e)}"

def format_model_info(model):
    """Format model information for display"""
    return f"""
    Name: {model.get('name', 'Unknown')}
    State: {model.get('state', 'Unknown')}
    Number of Blocks: {model.get('num_blocks', 'Unknown')}
    DHT Prefix: {model.get('dht_prefix', 'Unknown')}
    Official: {model.get('official', False)}
    Limited: {model.get('limited', False)}
    
    Servers:
    {format_server_rows(model.get('server_rows', []))}
    """

def format_server_rows(servers):
    """Format server rows information"""
    if not servers:
        return "    No active servers"
    
    server_info = []
    for server in servers:
        info = f"""    - ID: {server.get('id', 'Unknown')}
      State: {server.get('state', 'Unknown')}
      Version: {server.get('version', 'Unknown')}
      Blocks: {', '.join(str(b) for b in server.get('blocks', []))}"""
        server_info.append(info)
    
    return "\n".join(server_info)

if __name__ == "__main__":
    print("\n=== Testing Petals Network Health ===\n")
    result = get_network_health()
    
    print("\nFinal Results:")
    if isinstance(result, dict):
        if "error" in result:
            print(f"Error: {result['error']}")
            print("Raw response excerpt:")
            print(result.get('raw_response', 'No raw response available'))
        else:
            print("Network Status:")
            print(f"Bootstrap States: {result.get('bootstrap_states', [])}")
            print(f"\nModel Reports ({len(result.get('model_reports', []))} models):")
            for model in result.get('model_reports', []):
                print(format_model_info(model))
    else:
        print(result)
