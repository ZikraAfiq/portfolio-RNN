from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your RNN model
model = tf.keras.models.load_model("timeseries_rnn.keras")

# API endpoint
@app.route("/predict_sequence", methods=["POST"])
def predict_sequence():
    try:
        data = request.get_json()
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' in request"}), 400

        seq = data["sequence"]
        seq_array = np.array(seq).reshape(1, len(seq), 1)
        prediction = model.predict(seq_array, verbose=0)
        next_number = float(prediction[0][0])
        return jsonify({"next_number": next_number})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# HTML file
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)