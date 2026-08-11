# Leaf Disease Detector

A simple Flask web app that lets a user upload a leaf image and receive a disease prediction plus a basic remedy tip.

## Setup

```bash
cd /home/dhiraj/Documents/LeafDiseaseApp
. venv/bin/activate
pip install -r requirements.txt
python -m py_compile app.py leaf_disease_model.py
python - <<'PY'
from leaf_disease_model import train_model
train_model()
PY
python app.py
```

Then open http://127.0.0.1:5000/ in a browser.
