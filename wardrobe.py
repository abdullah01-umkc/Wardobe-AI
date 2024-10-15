from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load T5 model and tokenizer
model_name = "t5-base"  # You can use other T5 variants as needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def home():
    return "Welcome to the Wardrobe AI!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    clothing_items = data.get('clothingItems', '')
    occasion = data.get('occasion', '')

    # Prepare the input for the T5 model
    input_text = f"Suggest an outfit for {occasion} using {clothing_items}."
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate a response from the model
    output_ids = model.generate(input_ids)
    suggestion = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Return the suggestion as JSON
    return jsonify({"suggestion": suggestion})

if __name__ == '__main__':
    app.run(debug=True)
