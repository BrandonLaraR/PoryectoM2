from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os


ALLOWED_ORIGIN = "https://formulario-precios-auto.onrender.com"

MODEL_PATH = "modelo_regresion_completo.pkl"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

# ─────────────────── Carga del modelo ───────────────────
try:
    model = joblib.load(MODEL_PATH)
    print("✅  Modelo cargado:", MODEL_PATH)
except Exception as e:
    model = None
    print("❌  Error al cargar modelo:", e)

# ─────────── Función auxiliar: nombres de columnas finales ───────────
def get_feature_names():
    """
    Devuelve la lista de features que el preprocesador deja al estimador.
    Funciona con ColumnTransformer + OneHotEncoder/Scaler.
    """
    if model is None:
        return []
    try:
        pre = model.named_steps["preprocessing"]
        names = []
        for _, transformer, cols in pre.transformers_:
            if transformer is None or transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    new_cols = transformer.get_feature_names_out(cols)
                except Exception:
                    new_cols = cols
            else:
                new_cols = cols
            names.extend(new_cols)
        return list(names)
    except Exception:
        return []

# ───────────────────── Endpoints ─────────────────────
@app.route("/")
def home():
    return {
        "service": "API Predicción Precio Auto",
        "endpoints": ["/predict (POST)", "/model-info (GET)", "/health (GET)"]
    }

@app.route("/health")
def health():
    if model is None:
        return jsonify(status="error", message="Modelo no cargado"), 500
    return jsonify(status="ok")

@app.route("/model-info")
def model_info():
    if model is None:
        return jsonify(error="Modelo no cargado"), 500
    features = get_feature_names()
    return jsonify(selected_features=features, total=len(features))

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="Modelo no cargado"), 500

    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify(error="JSON vacío o sin campo 'features'"), 400

    features = data["features"]
    if not isinstance(features, list):
        return jsonify(error="'features' debe ser lista"), 400

    expected_len = len(get_feature_names())
    if len(features) != expected_len:
        return jsonify(
            error=f"Se requieren {expected_len} características",
            expected_features=expected_len,
            received=len(features)
        ), 400

    try:
        X = np.array(features).reshape(1, -1)
        pred = float(model.predict(X)[0])
        return jsonify(predicted_price=round(pred, 2))
    except Exception as e:
        return jsonify(error=f"Error durante la predicción: {str(e)}"), 500

# ──────────────────── Arranque ────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
