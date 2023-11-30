from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from flask import Flask


app = Flask(__name__)


model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        

        sequences = pipeline(
            user_message,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = sequences[0]['generated_text']
        return jsonify({'result': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
