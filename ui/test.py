from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

def execute(input_text, starter_sent):
    model_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
    text = input_text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, padding=True)["input_ids"]
    inputs = tokenizer(starter_sent, return_tensors="pt")["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=60)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_text)
    return decoded_text


if __name__ == "__main__":
    # Initialize the model and tokenizer
    model_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
    print("check3")
    text = "Write a short story about a cat sitting on the floor:"
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, padding=True)["input_ids"]
    inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=60)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_text)
