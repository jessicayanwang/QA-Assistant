from flask import Flask, request, jsonify, render_template
from generate_response import *

app = Flask(__name__)

# Initial text
current_text = "You can answer the question as following:"
current_confidence = 'high'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_target_text', methods=['GET'])
def get_target_text():
    global current_text
    global current_confidence
    return jsonify({"target_text": current_text, 
                    "confidence": current_confidence})

@app.route('/set_target_text', methods=['POST'])
def set_target_text():
    global current_text
    global current_confidence
    question = request.json.get('target_text')
    if question is not None:
        current_text = ''
        for chunk, confidence in generate_response(question):
            current_text += chunk
            current_confidence = confidence
        return jsonify({"message": "Target text updated successfully"})
    else:
        return jsonify({"error": "Invalid request format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
