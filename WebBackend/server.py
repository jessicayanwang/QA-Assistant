from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample initial text
current_text = "red"

# Route for handshake (accepts handshakes)
@app.route('/get_text', methods=['GET'])
def get_text():
    return jsonify({"text": current_text})

# Route for handling target text change
@app.route('/set_text', methods=['POST'])
def set_text():
    global current_text
    data = request.get_json()
    current_text = data["text"]
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
