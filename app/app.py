import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# ------------------------
# PATH SETUP (NO ERROR)
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DENOM_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "denomination_model.keras")
FAKE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fake_500.keras")

# ------------------------
# LOAD MODELS
# ------------------------
denom_model = load_model(DENOM_MODEL_PATH)
fake_model = load_model(FAKE_MODEL_PATH)

# ------------------------
# FLASK APP
# ------------------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# NOTE / NOT NOTE CHECK
# ------------------------
def is_currency_note(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for consistency
    gray = cv2.resize(gray, (224,224))

    # Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # Texture check using variance
    texture_var = np.var(gray)

    # DEBUG PRINT (IMPORTANT)
    print("EDGE:", edge_ratio, "TEXTURE:", texture_var)

    # FINAL RULE (TUNED)
    if edge_ratio < 0.025:
        return False
    if texture_var < 500:
        return False

    return True



# ------------------------
# IMAGE PREPROCESS
# ------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img_norm = img / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    return img, img_input

# ------------------------
# ROUTES
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    confidence = ""
    note_type = ""

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            raw_img, img_input = preprocess_image(img_path)

            # STEP 1: NOTE CHECK
            if not is_currency_note(raw_img):
                return render_template(
                    "index.html",
                    result="❌ Not a currency note",
                    confidence="N/A",
                    note_type="N/A",
                    image=file.filename
                )

            # STEP 2: DENOMINATION
            denom_pred = denom_model.predict(img_input)
            denom_class = np.argmax(denom_pred)

            denominations = [10, 20, 50, 100, 200, 500]
            note_type = f"₹{denominations[denom_class]}"

            # STEP 3: FAKE / REAL
            fake_pred = fake_model.predict(img_input)[0][0]


            if fake_pred > 0.5:
                result = "❌ FAKE NOTE"
                confidence = f"{fake_pred*100:.2f}%"
            else:
                result = "✅ REAL NOTE"
                confidence = f"{(1-fake_pred)*100:.2f}%"
    

            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                note_type=note_type,
                image=file.filename
            )

    return render_template("index.html")

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)

