import joblib
import numpy as np

FEATURE_DEFAULTS = {
    "speed": 0.0,
    "rpm": 900.0,
    "engine_load": 0.0,
    "throttle": 0.0,
    "timing_advance": 0.0,
    "accel_magnitude": 0.0,
}

def safe_float(value, default=0.0):
    """Convert value to float, handling None, '' and bad types."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def calculate_radar_area(data):
    # Normaliza o RPM
    rpm = data['rpm'] / 100
    speed = data['speed']
    throttle = data['throttle']
    engine = data['engine_load']

    values = [rpm, speed, throttle, engine]

    # Fórmula da área do polígono
    angle = 2 * np.pi / len(values)
    area = 0.5 * np.abs(np.dot(values, np.roll(values, 1)) * np.sin(angle))
    
    return area

def predict_fuel_type(dados):
    if "fuel_type" in dados:
        prob = 1.0
        return dados["fuel_type"], prob
    elif "ethanol_percentage" in dados:
        model = joblib.load("./models/ethanol_model_rf.pkl")
        X = [float(dados.get("ethanol_percentage",0.0)),
             dados["speed"],
             dados["rpm"],
             dados["engine_load"],
             dados["throttle"],
             float(dados.get("timing_advance", 0.0))]
        X = np.array(X).reshape(1, -1)
        fuel_type = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if fuel_type == 1:
            fuel_type_str = "Gasoline"
        elif fuel_type == 0:
            fuel_type_str = "Ethanol"

        return fuel_type_str, prob
    # else:
    #     pass
    return "Gasoline", 1.0
        
def predict_city_highway(dados):
    model = joblib.load("./models/city_highway_rf.pkl")

    def f(key):
        return safe_float(dados.get(key, FEATURE_DEFAULTS[key]), FEATURE_DEFAULTS[key])

    X = [
        f("speed"),
        f("rpm"),
        f("engine_load"),
        f("throttle"),
        f("timing_advance"),
        f("accel_magnitude"),
    ]

    X = np.array(X, dtype=float).reshape(1, -1)

    city_highway = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return city_highway, prob