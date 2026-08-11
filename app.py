import os
import tempfile
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from leaf_disease_model import get_remedy_tip, predict_disease

app = Flask(__name__)
app.secret_key = "leaf-disease-demo"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if not uploaded_file or not uploaded_file.filename:
            flash("Please choose an image before predicting.")
            return redirect(url_for("index"))

        if not uploaded_file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            flash("Please upload a JPG, JPEG, PNG, or WebP image.")
            return redirect(url_for("index"))

        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.filename).suffix, delete=False) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            predicted_label, confidence = predict_disease(temp_path)
            result = {
                "predicted_label": predicted_label.replace("___", " → ").replace("_", " "),
                "confidence": round(confidence * 100, 1),
                "remedy_tip": get_remedy_tip(predicted_label),
            }
        except Exception as exc:
            flash(f"Prediction failed: {exc}")
            result = None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
