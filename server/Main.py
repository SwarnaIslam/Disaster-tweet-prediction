from flask import Flask, request, jsonify
from flask_cors import CORS
from Predict import DisaterPredictionSystem

app = Flask(__name__)
CORS(app)

@app.route('/predictDisaster', methods=['POST'])
def predictDisaster():
    # print(request.json['description'])
    print('Model is predicting')
    disasterPredictionSystem = DisaterPredictionSystem()
    answer, accuracy = disasterPredictionSystem.predict(request.json['description'])
    print(answer, accuracy*100)
    return jsonify({'answer': answer[0], 'accuracy': accuracy*100})
if __name__ == '__main__':
    app.run(host='0.0.0.0')