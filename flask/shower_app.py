from flask import Flask, abort, request, render_template, jsonify
from api import predict_st

app = Flask('ShowerthoughtsApp')

@app.route('/predict', methods=['POST'])
def do_prediction():
    if not request.json:
        abort(400)
    data = request.json

    response = predict_st(data)

    return jsonify(response), 201

@app.route('/')
def index():
    return render_template('bootstrap_st.html')

if __name__ == '__main__':
    app.run(debug=True)