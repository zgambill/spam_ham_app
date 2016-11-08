from flask import Flask, request, jsonify
from sklearn.externals import joblib


app = Flask(__name__)

model = joblib.load("models/spam_ham.pkl")

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        print(request.form)
        text = request.form["text"]
        print(text)
        prediction = model.predict([text])[0]
        data = {
            "result": prediction
        }
        return jsonify(data)
    else:
        return "App is live!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=9000)
