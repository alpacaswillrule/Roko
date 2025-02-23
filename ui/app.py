from flask import Flask, request, render_template

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')

@app.route('/get_response', methods=['GET', 'POST'])
def get_response():
    if request.method == 'POST':
        query = request.form.get('query')
        max_tokens = request.form.get('max_tokens', type=int)

        response_string = getResponse(query, max_tokens)
   
        return render_template('index.html', response=response_string)
    

def getResponse(query, max_tokens):

    # Initialize the model and tokenizer
    print("check1")
    model_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("check2")
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
    print("check3")
    # Prepare input text
    # text = "Write a short story about a robot learning to paint:"
    inputs = tokenizer(query, return_tensors="pt", max_length=1024, padding=True)["input_ids"]
    print("check4")
    # inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
    print("check5")
    outputs = model.generate(inputs, max_new_tokens=max_tokens)
    print("check6")
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_text)

    return decoded_text


if __name__ == '__main__':
    app.run(debug=True)