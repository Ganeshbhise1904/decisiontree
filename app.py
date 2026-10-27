from flask import Flask, request
import pickle
import os
import html

app = Flask(__name__)

# Load the trained model saved beside this file.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Use saved feature names when the model has them.
FEATURES = list(getattr(model, "feature_names_in_", []))
N_FEATURES = int(getattr(model, "n_features_in_", len(FEATURES) or 2))

if not FEATURES:
    if N_FEATURES == 3:
        FEATURES = ["Gender", "Age", "EstimatedSalary"]
    elif N_FEATURES == 2:
        FEATURES = ["Age", "EstimatedSalary"]
    else:
        FEATURES = [f"Feature {i + 1}" for i in range(N_FEATURES)]


def parse_value(feature, raw_value):
    """Convert form values to the type expected by common Social Network Ads models."""
    name = feature.lower().replace("_", "").replace(" ", "")

    if name == "gender":
        value = raw_value.strip().lower()
        if value in ("male", "m"):
            return 1
        if value in ("female", "f"):
            return 0
        # If the model was trained with text categories, retain the text.
        return raw_value.strip()

    # Most Social Network Ads model inputs are numeric.
    try:
        return float(raw_value)
    except ValueError:
        return raw_value.strip()


def page(message=""):
    fields = []
    for feature in FEATURES:
        safe_name = html.escape(str(feature), quote=True)
        low = str(feature).lower()
        if low == "gender":
            control = (
                f'<select name="{safe_name}" required>'
                '<option value="">Select gender</option>'
                '<option value="Female">Female</option>'
                '<option value="Male">Male</option>'
                '</select>'
            )
        else:
            control = (
                f'<input name="{safe_name}" type="number" step="any" '
                f'placeholder="Enter {safe_name}" required>'
            )
        fields.append(f'<label>{safe_name}</label>{control}')

    safe_message = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decision Tree Prediction</title>
<style>
body {{ font-family: Arial, sans-serif; background:#f1f5f9; margin:0; padding:24px; }}
.card {{ max-width:440px; margin:30px auto; background:white; padding:26px;
         border-radius:12px; box-shadow:0 4px 18px #0001; }}
h1 {{ font-size:24px; text-align:center; color:#1e293b; }}
label {{ display:block; margin-top:16px; font-weight:600; color:#334155; }}
input, select {{ box-sizing:border-box; width:100%; margin-top:7px; padding:11px;
                 border:1px solid #cbd5e1; border-radius:6px; background:white; }}
button {{ width:100%; margin-top:22px; padding:12px; border:0; border-radius:6px;
          background:#2563eb; color:white; font-size:16px; cursor:pointer; }}
.result {{ margin-top:18px; padding:12px; border-radius:6px; background:#eff6ff;
           color:#1e3a8a; overflow-wrap:anywhere; }}
small {{ color:#64748b; }}
</style>
</head>
<body><main class="card">
<h1>Decision Tree Prediction</h1>
<p><small>Enter the values used by your trained model.</small></p>
<form method="post">
{''.join(fields)}
<button type="submit">Predict</button>
</form>
{f'<div class="result"><strong>{safe_message}</strong></div>' if message else ''}
</main></body></html>"""


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return page()

    try:
        values = []
        for feature in FEATURES:
            raw = request.form.get(str(feature), "").strip()
            if not raw:
                return page(f"Please enter a value for {feature}.")
            values.append(parse_value(feature, raw))

        # Most scikit-learn models expect a 2D row of features.
        prediction = model.predict([values])[0]
        result_text = f"Prediction result: {prediction}"
        return page(result_text)

    except Exception as exc:
        # Show a readable error on the page; details remain in the terminal.
        app.logger.exception("Prediction failed")
        return page(f"Prediction failed: {exc}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
