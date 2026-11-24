import os, json, uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageStat, ImageFilter
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "fashion_improved_model.keras"
LABEL_PATH = MODEL_DIR / "improved_class_names.json"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

CATEGORIES = [
    "T-shirt", "Shirt", "Hoodie/Sweatshirt", "Jeans", "Trousers",
    "Dress", "Footwear", "Bags", "Accessories"
]
_model = None
_model_labels = None
_model_error = None

def load_model():
    global _model, _model_labels, _model_error
    if _model is not None:
        return _model, _model_labels
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        _model_error = "Trained model files are not installed. Choose the clothing category manually, or add your Colab-trained model to the models folder."
        return None, None
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        _model_labels = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        return _model, _model_labels
    except Exception as exc:
        _model_error = f"Model could not be loaded: {exc}"
        return None, None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def rgb_to_name(rgb):
    r, g, b = [int(x) for x in rgb]
    mx, mn = max(r,g,b), min(r,g,b)
    if mx < 55: return "Black"
    if mx - mn < 22:
        if mx > 220: return "White"
        if mx > 150: return "Light Gray"
        return "Gray"
    if r > 1.25*g and r > 1.25*b: return "Red"
    if g > 1.18*r and g > 1.12*b: return "Green"
    if b > 1.2*r and b > 1.12*g: return "Blue"
    if r > 1.12*b and g > 1.08*b: return "Yellow"
    if r > 1.18*g and b > 1.12*g: return "Purple"
    if r > 1.15*g and g > 1.12*b: return "Orange"
    return "Mixed"

def extract_attributes(image):
    im = ImageOps.exif_transpose(image).convert("RGB")
    small = im.resize((100, 100))
    # Quantized colour histogram; fast and dependency-light.
    arr = np.asarray(small, dtype=np.uint8).reshape(-1, 3)
    quant = (arr // 32) * 32 + 16
    unique, counts = np.unique(quant, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:3]
    palette = []
    total = int(counts.sum())
    for idx in order:
        rgb = unique[idx].astype(int).tolist()
        palette.append({
            "name": rgb_to_name(rgb),
            "rgb": rgb,
            "percentage": round(float(counts[idx] / total * 100), 1)
        })
    gray = im.convert("L").resize((128,128))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_mean = float(ImageStat.Stat(edges).mean[0])
    texture = "Low visual texture" if edge_mean < 18 else ("High visual texture" if edge_mean > 42 else "Moderate visual texture")
    return {"dominant_colour": palette[0]["name"], "palette": palette, "texture": texture}

def predict_category(image):
    model, labels = load_model()
    if model is None:
        return None, None
    try:
        im = ImageOps.exif_transpose(image).convert("RGB").resize((160,160))
        x = np.asarray(im, dtype=np.float32)[None, ...]
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return str(labels[idx]), round(float(probs[idx])*100, 1)
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return None, None

def styling_recommendations(category, colour, occasion, season, style, preferences):
    c = category.lower()
    colour = colour.lower()
    palette = {
        "black": ["white", "beige", "grey", "denim blue"],
        "white": ["black", "navy", "olive", "light blue"],
        "blue": ["white", "cream", "grey", "tan"],
        "red": ["black", "white", "navy", "beige"],
        "green": ["cream", "black", "brown", "white"],
        "yellow": ["white", "navy", "denim blue", "brown"],
        "purple": ["white", "grey", "black", "cream"],
        "pink": ["white", "grey", "navy", "beige"],
        "grey": ["black", "white", "burgundy", "navy"],
        "orange": ["white", "navy", "cream", "brown"],
    }
    matches = palette.get(colour, ["white", "black", "navy", "beige"])
    if "hoodie" in c or "sweatshirt" in c:
        pieces = ["Straight-fit jeans or relaxed cargo trousers", "Clean sneakers", "A simple watch or cap"]
    elif "shirt" in c or "t-shirt" in c:
        pieces = ["Dark or neutral jeans / chinos", "Minimal sneakers for casual wear or loafers for smart casual", "A watch and a simple belt"]
    elif "jeans" in c or "trouser" in c:
        pieces = ["A plain T-shirt, shirt or knit top", "Sneakers, loafers or boots depending on the occasion", "A coordinated belt"]
    elif "dress" in c:
        pieces = ["A light jacket or cardigan when layering is needed", "Flats for daytime or heels for an evening look", "Small earrings and a compact bag"]
    elif "footwear" in c:
        pieces = ["Pair with a simple outfit in one or two coordinating colours", "Keep accessories minimal so the footwear stands out"]
    elif "bag" in c:
        pieces = ["Coordinate the bag with your shoes, belt or one outfit accent", "Keep the remaining accessories understated"]
    else:
        pieces = ["Use the item as an accent with a neutral outfit", "Repeat one colour from the item elsewhere in the look"]
    if season == "Summer":
        season_tip = "Prefer breathable fabrics, lighter layers and comfortable footwear."
    elif season == "Winter":
        season_tip = "Layer with a coat, knitwear or a warm outer layer; choose season-appropriate footwear."
    elif season == "Rainy":
        season_tip = "Choose quick-drying layers and practical, water-resistant footwear."
    else:
        season_tip = "Use light layering so the outfit can adapt to changing temperatures."
    if occasion == "Office / Formal":
        occasion_tip = "Keep the silhouette neat, colours coordinated and accessories understated."
    elif occasion == "Party / Evening":
        occasion_tip = "Add one statement element, such as sharper footwear, jewellery or a structured bag."
    elif occasion == "College / Everyday":
        occasion_tip = "Prioritize comfort, easy movement and versatile pieces."
    elif occasion == "Festive":
        occasion_tip = "Add a richer accent colour or a refined accessory while keeping the outfit balanced."
    else:
        occasion_tip = "Aim for a comfortable, relaxed combination with coordinated colours."
    pref_tip = f"Preference considered: {preferences}." if preferences else "No personal style preference supplied; suggestions use general styling rules."
    return {
        "matching_colours": matches[:4],
        "outfit_ideas": pieces,
        "occasion_tip": occasion_tip,
        "season_tip": season_tip,
        "style_tip": f"For a {style.lower()} look, keep the outfit cohesive and avoid too many competing statement pieces.",
        "personalization": pref_tip
    }

def generate_description(category, attrs):
    return (f"This item is identified as {category.lower()} with {attrs['dominant_colour'].lower()} as its dominant detected colour. "
            f"The image shows {attrs['texture'].lower()}. These are visual estimates from the uploaded photograph.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("photo")
        if not file or not file.filename:
            flash("Please choose a clothing image first.", "error")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Upload a JPG, JPEG, PNG or WEBP image.", "error")
            return redirect(url_for("index"))
        try:
            image = Image.open(file.stream)
            image.load()
            attrs = extract_attributes(image)
            predicted, confidence = predict_category(image)
            manual = request.form.get("manual_category", "").strip()
            category = predicted or (manual if manual in CATEGORIES else "Unclassified")
            occasion = request.form.get("occasion", "Casual")
            season = request.form.get("season", "All-season")
            style = request.form.get("style", "Casual")
            preferences = request.form.get("preferences", "").strip()[:300]
            recommendations = styling_recommendations(category, attrs["dominant_colour"], occasion, season, style, preferences)
            description = generate_description(category, attrs)
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = UPLOAD_DIR / filename
            image.convert("RGB").save(save_path, format="JPEG", quality=90)
            return render_template(
                "result.html", image_url=url_for("static", filename=f"uploads/{filename}"),
                category=category, confidence=confidence, model_available=predicted is not None,
                attrs=attrs, recommendations=recommendations, description=description,
                occasion=occasion, season=season, style=style
            )
        except Exception:
            app.logger.exception("Image processing failed")
            flash("This image could not be processed. Please try another JPG or PNG image.", "error")
            return redirect(url_for("index"))
    model, _ = load_model()
    return render_template("index.html", categories=CATEGORIES, model_available=model is not None, model_error=_model_error)

@app.errorhandler(413)
def too_large(_):
    flash("Image is too large. Maximum upload size is 8 MB.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
