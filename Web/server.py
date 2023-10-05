from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Initial text
current_text = "Initial Text"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_target_text', methods=['GET'])
def get_target_text():
    global current_text
    return jsonify({"target_text": current_text})

@app.route('/set_target_text', methods=['POST'])
def set_target_text():
    global current_text
    new_text = request.json.get('target_text')
    if new_text is not None:
        current_text = new_text
        return jsonify({"message": "Target text updated successfully"})
    else:
        return jsonify({"error": "Invalid request format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
