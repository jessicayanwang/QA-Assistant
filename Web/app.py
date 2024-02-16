from flask import Flask, request, jsonify, render_template
import asyncio
import threading
from Web.generate_response import *

app = Flask(__name__)

# Initial text
current_text = "You can answer the question as following:"
current_confidence = 'high'

@app.route('/')
def index():
    return render_template('index.html')

def start_async_task(question):
    async def async_task():
        global current_text
        global current_confidence
        current_text=''
        # This will be used to send data to the client
        async for chunk, confidence in main(question):
            current_text += chunk
            current_confidence = confidence
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_task())
    loop.close()

@app.route('/get_target_text/', methods=['GET'])
def get_target_text():
    global current_text
    global current_confidence
    return jsonify({"target_text": current_text, 
                    "confidence": current_confidence})

@app.route('/set_target_text/', methods=['POST'])
def set_target_text():
    question = request.json.get('target_text')
    if question is not None:
        thread = threading.Thread(target=start_async_task, args=(question,))
        thread.start()
        return jsonify({"message": "Target text updated successfully"})
    else:
        return jsonify({"error": "Invalid request format"}), 400
    
if __name__ == "__main__":
    app.run(debug=False)
