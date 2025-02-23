import sys
import queue
import logging
from io import StringIO

# Set up logging and stdout capture first
log_queue = queue.Queue()

class PrintCapture:
    def __init__(self, queue, stream_name='stdout'):
        self.queue = queue
        self.stream_name = stream_name
        self.original_stream = sys.stdout if stream_name == 'stdout' else sys.stderr

    def write(self, text):
        if text.strip():  # Only process non-empty lines
            # Clean up the message
            msg = text.strip()
            # Extract timestamp if present in common formats
            timestamp = time.strftime('%H:%M:%S')
            if '[DEBUG]' in msg or '[INFO]' in msg or '[ERROR]' in msg or '[WARNING]' in msg:
                msg_type = 'debug' if '[DEBUG]' in msg else 'info' if '[INFO]' in msg else 'error' if '[ERROR]' in msg else 'warning'
            else:
                msg_type = 'info'
            
            self.queue.put({
                'type': msg_type,
                'message': msg,
                'timestamp': timestamp
            })
        self.original_stream.write(text)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        # Delegate isatty to the original stream if it has the method
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()

    def fileno(self):
        # Delegate fileno to the original stream
        return self.original_stream.fileno()

    # Add other file-like object methods that might be needed
    def readable(self): return True
    def writable(self): return True
    def seekable(self): return False

# Replace both stdout and stderr before any other imports
sys.stdout = PrintCapture(log_queue, 'stdout')
sys.stderr = PrintCapture(log_queue, 'stderr')

# Now import everything else
import json
import threading
import time
from flask import Flask, request, render_template, jsonify
from flask_sock import Sock
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
from .gpu_monitor import compute_resource_usage
from .train_distributed import main as train_model


class QueueHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.queue.put({
                'type': record.levelname.lower(),
                'message': msg,
                'timestamp': time.strftime('%H:%M:%S')
            })
        except Exception:
            self.handleError(record)

# Configure logging and stdout capture
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[
        QueueHandler(log_queue),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Replace stdout with our print capture
sys.stdout = PrintCapture(log_queue)

# Global model instance for health monitoring
model = None

app = Flask(__name__)
sock = Sock(app)

# Global variables for progress tracking
current_progress = 0
progress_lock = threading.Lock()

def update_progress(progress):
    global current_progress
    with progress_lock:
        current_progress = progress

@app.route("/")
def main():
    return render_template('main.html', response=None)


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

def get_petals_health():
    """Get health data from Petals network"""
    global model
    try:
        if model is None:
            model_name = "bigscience/bloom-560m"
            model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
        
        # Group servers by model
        model_groups = {}
        
        # Get active servers from the model's routing info
        if hasattr(model, '_remote_forward_backward'):
            servers = model._remote_forward_backward.active_servers
        else:
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
        
        # If no servers were found, return a message in the Unknown Model group
        if not model_groups:
            return {
                'Unknown Model': [{
                    'id': 'No Servers',
                    'state': 'OFFLINE',
                    'version': 'Unknown',
                    'throughput': 0,
                    'precision': 'Unknown',
                    'cache_tokens_left': 0,
                    'max_cache_tokens': 100
                }]
            }
        
        return model_groups
    except Exception as e:
        print(f"Error getting health data: {str(e)}")
        # Return error group if we can't get any data
        return {
            'Error': [{
                'id': 'Error',
                'state': 'ERROR',
                'version': 'Unknown',
                'throughput': 0,
                'precision': 'Unknown',
                'cache_tokens_left': 0,
                'max_cache_tokens': 100
            }]
        }

@sock.route('/ws')
def websocket(ws):
    """WebSocket endpoint for real-time updates"""
    try:
        while True:
            # Get compute usage
            cpu_usage, gpu_usage, compute_usage = compute_resource_usage()
            
            # Send compute usage data
            ws.send(json.dumps({
                'type': 'compute_usage',
                'timestamp': time.strftime("%H:%M:%S"),
                'usage': compute_usage
            }))
            
            # Get and send health data
            model_groups = get_petals_health()
            ws.send(json.dumps({
                'type': 'health_update',
                'model_groups': model_groups
            }))
            
            # Send current progress if processing
            if current_progress > 0:
                ws.send(json.dumps({
                    'type': 'progress',
                    'progress': current_progress
                }))
            
            # Send any new log messages
            while not log_queue.empty():
                log_entry = log_queue.get_nowait()
                ws.send(json.dumps({
                    'type': 'log_update',
                    'log': log_entry
                }))
            
            time.sleep(1)  # Update every second
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

@app.route('/get_response', methods=['GET', 'POST'])
def get_response():
    if request.method == 'POST':
        query = request.form.get('query')
        max_tokens = request.form.get('max_tokens', type=int)

        try:
            response_string = getResponse(query, max_tokens)
            return jsonify({'response': response_string})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to handle model training"""
    try:
        # Start training in a separate thread
        thread = threading.Thread(target=train_model)
        thread.start()
        return jsonify({'message': 'Training started successfully'})
    except Exception as e:
        return jsonify({'message': f'Error starting training: {str(e)}'}), 500

def getResponse(query, max_tokens):
    try:
        update_progress(10)  # Starting
        logger.info("Starting response generation")
        
        # Initialize the model and tokenizer
        model_name = "bigscience/bloom-560m"
        logger.debug(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        update_progress(30)  # Tokenizer loaded
        logger.info("Tokenizer loaded successfully")
        
        logger.debug(f"Loading model {model_name}")
        model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
        update_progress(50)  # Model loaded
        logger.info("Model loaded successfully")
        
        # Prepare input text
        logger.debug("Preparing input text")
        inputs = tokenizer(query, return_tensors="pt", max_length=1024, padding=True)["input_ids"]
        update_progress(70)  # Input prepared
        logger.info("Input prepared successfully")
        
        # Generate response
        logger.debug("Generating response")
        outputs = model.generate(inputs, max_new_tokens=max_tokens)
        update_progress(90)  # Generation complete
        logger.info("Response generated successfully")
        
        # Decode response
        logger.debug("Decoding response")
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        update_progress(100)  # Finished
        logger.info("Response decoded successfully")
        
        # Reset progress after completion
        update_progress(0)
        
        return decoded_text
    except Exception as e:
        update_progress(0)  # Reset progress on error
        logger.error(f"Error generating response: {str(e)}")
        raise e

if __name__ == '__main__':
    app.run(debug=True, port=9000)
