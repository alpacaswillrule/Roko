from flask import Flask, request, render_template, jsonify
from flask_sock import Sock
import json
import threading
import time
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
from .gpu_monitor import compute_resource_usage
from .train_distributed import main as train_model

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
            
            # Send current progress if processing
            if current_progress > 0:
                ws.send(json.dumps({
                    'type': 'progress',
                    'progress': current_progress
                }))
            
            time.sleep(1)  # Update every second
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

@app.route('/get_response', methods=['GET', 'POST'])
def get_response():
    if request.method == 'POST':
        query = request.form.get('query')
        max_tokens = request.form.get('max_tokens', type=int)

        try:
            response_string = getResponse(query, max_tokens)
            return render_template('main.html', response=response_string)
        except Exception as e:
            return render_template('main.html', response=f"Error: {str(e)}")

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
        
        # Initialize the model and tokenizer
        model_name = "bigscience/bloom-560m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        update_progress(30)  # Tokenizer loaded
        
        model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
        update_progress(50)  # Model loaded
        
        # Prepare input text
        inputs = tokenizer(query, return_tensors="pt", max_length=1024, padding=True)["input_ids"]
        update_progress(70)  # Input prepared
        
        # Generate response
        outputs = model.generate(inputs, max_new_tokens=max_tokens)
        update_progress(90)  # Generation complete
        
        # Decode response
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        update_progress(100)  # Finished
        
        # Reset progress after completion
        update_progress(0)
        
        return decoded_text
    except Exception as e:
        update_progress(0)  # Reset progress on error
        raise e

if __name__ == '__main__':
    app.run(debug=True)
