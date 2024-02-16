from flask import Flask, request, jsonify, render_template
import asyncio
import threading
from flask_socketio import SocketIO
from Web.generate_response import *

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')


def start_async_task(question):
    async def async_task():
        # This will be used to send data to the client
        async for chunk, confidence in main(question):
            socketio.emit('response_chunk', {'chunk': chunk, 'confidence': confidence})
        # Once done, notify the client
        socketio.emit('response_complete', {'message': 'Response generation complete'})

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_task())
    loop.close()

@socketio.on('generate_response')
def handle_generate_response(json):
    question = json.get('question')
    if question:
        # Use the existing start_async_task function to handle question processing
        thread = threading.Thread(target=start_async_task, args=(question,))
        thread.start()
    
if __name__ == "__main__":
    app.run(debug=False)
