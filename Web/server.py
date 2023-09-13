from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Initial text
current_text = "red"

@app.route('/', methods=['GET', 'POST']) 
def contact():
    global current_text 

    if request.method == 'POST':
        if 'blue' in request.form:
            current_text = 'blue'  # change title to 'blue'
        elif 'green' in request.form:
            current_text = 'green'  # change title to 'green'
    return render_template('index.html', text=current_text)

if __name__ == "__main__":
    app.run(debug=True)
