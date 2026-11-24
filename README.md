# FashionVision AI Website

A responsive Flask website for fashion image analysis and personalized styling suggestions.

## Run on Windows

1. Extract the ZIP.
2. Install Python 3.10 or newer.
3. Double-click `start.bat`, or run:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

4. Open http://127.0.0.1:5000

## Connect your trained Colab model

Copy these two files from Google Drive into the website's `models` folder:

- `fashion_improved_model.keras`
- `improved_class_names.json`

Install the ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Restart the website. The model must output the exact class order stored in `improved_class_names.json` and must use 160x160 RGB input with the same preprocessing as training (pixel values 0–255; model includes rescaling to -1..1).

If the model files are absent, the website still runs: select a category manually. Colour analysis and rule-based personalized styling remain available.

## Features

- Image upload with JPG/JPEG/PNG/WEBP validation and 8 MB limit
- CNN/MobileNetV2 prediction when your trained model is connected
- Dominant colour palette estimation
- Basic visual texture indicator
- Occasion, season and style preferences
- Outfit, footwear, accessory and colour-combination suggestions
- Responsive desktop/mobile GUI

## Important notes

- This version uses rule-based styling recommendations. It does not claim to call a Generative AI service.
- Colour and texture outputs are visual estimates, not guaranteed garment attributes.
- Do not expose the Flask development server directly to the public internet. For deployment, use a production WSGI server and HTTPS.
