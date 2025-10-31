# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import json
from io import BytesIO, StringIO

st.set_page_config(layout="wide", page_title="Pick the Safer Road — Road Safety Game")

st.title("🏁 Pick the Safer Road — test your intuition about road safety")
st.markdown(
    """
Try to pick which road is safer using your intuition. Upload your XGBoost model (JSON or pickle) or use the demo random model.
- Upload **model.json** (XGBoost native) or **model.pkl** (sklearn wrapper or any pickled model that supports predict_proba).
- Provide a comma-separated **feature order** (feature names expected by your model) if the app can't infer them.
"""
)

# -------------------------
# Utility functions
# -------------------------
def load_model_from_pickle(pkl_bytes):
    pkl_bytes.seek(0)
    obj = pickle.load(pkl_bytes)
    return obj

def load_xgb_json(json_bytes):
    booster = xgb.Booster()
    # xgboost Booster.load_model accepts filename or raw string bytes depending on version.
    # Write to BytesIO temporary file then load.
    tmp = BytesIO(json_bytes.read())
    tmp.seek(0)
    # save tmp to disk-like object required; Booster.load_model accepts file-like in newer versions,
    # but to be robust, write to a temp file
    try:
        booster.load_model(tmp)
    except Exception:
        # fallback: write to a real temp file
        import tempfile
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmpfile.write(tmp.read())
        tmpfile.flush()
        booster.load_model(tmpfile.name)
        tmpfile.close()
    return booster

def ensure_feature_array(df_row, feature_order):
    # df_row: pandas Series or 1D array; feature_order: list of feature names
    if isinstance(df_row, pd.Series):
        arr = df_row.reindex(feature_order).values.reshape(1, -1)
    else:
        # assume already array-like in correct order
        arr = np.asarray(df_row).reshape(1, -1)
    return arr

def model_predict_proba(model_obj, X_df, feature_order=None):
    """
    Returns the probability of the positive class (safety) for each row in X_df.
    Works with:
    - sklearn-like estimators with predict_proba
    - xgboost.Booster (native)
    """
    # If sklearn-like
    if hasattr(model_obj, "predict_proba"):
        # ensure order if feature names supplied
        if feature_order is not None:
            X = X_df[feature_order]
        else:
            X = X_df
        probs = model_obj.predict_proba(X)
        # take class 1 probability
        if probs.shape[1] == 1:
            # some models return single column
            return probs.ravel()
        return probs[:, 1]
    # If XGBoost native Booster
    elif isinstance(model_obj, xgb.Booster):
        # need DMatrix
        if feature_order is not None:
            dmat = xgb.DMatrix(X_df[feature_order])
        else:
            dmat = xgb.DMatrix(X_df)
        # predict returns probability for binary objective
        preds = model_obj.predict(dmat)
        # if preds shape is (n, 2) maybe it's softmax; handle both
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[1] == 2:
            return preds[:, 1]
        else:
            return preds.ravel()
    else:
        raise ValueError("Unsupported model object type. Provide sklearn-like model or xgboost.Booster.")

# -------------------------
# Model upload / load
# -------------------------
st.sidebar.header("Model configuration")

uploaded_pkl = st.sidebar.file_uploader("Upload model.pkl (pickle)", type=["pkl"], key="pkl")
uploaded_json = st.sidebar.file_uploader("Upload xgboost model.json", type=["json"], key="json")
use_demo = st.sidebar.checkbox("Use demo fake model (no upload)", value=False)

model = None
inferred_feature_names = None

if uploaded_pkl is not None:
    try:
        uploaded_pkl.seek(0)
        model = load_model_from_pickle(uploaded_pkl)
        st.sidebar.success("Loaded pickle model.")
        # try infer feature names
        if hasattr(model, "feature_names_in_"):
            inferred_feature_names = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                inferred_feature_names = booster.feature_names
            except Exception:
                inferred_feature_names = None
        else:
            inferred_feature_names = None
    except Exception as e:
        st.sidebar.error(f"Failed to load pickle: {e}")

elif uploaded_json is not None:
    try:
        uploaded_json.seek(0)
        booster = load_xgb_json(uploaded_json)
        model = booster
        st.sidebar.success("Loaded XGBoost JSON Booster.")
        # try infer features
        try:
            inferred_feature_names = booster.feature_names
        except Exception:
            inferred_feature_names = None
    except Exception as e:
        st.sidebar.error(f"Failed to load JSON booster: {e}")
elif use_demo:
    # Build a tiny demo sklearn logistic model trained on simulated features
    from sklearn.linear_model import LogisticRegression
    # we will simulate feature names
    inferred_feature_names = ["lanes", "curvature", "speed_limit", "lighting", "road_type_rural", "road_type_highway"]
    # create a toy model trained quickly
    rng = np.random.RandomState(0)
    X_demo = rng.normal(size=(200, len(inferred_feature_names)))
    # craft labels such that low curvature, more lanes, good lighting are safer
    weights = np.array([0.8, -1.5, 0.3, 1.0, -0.5, 0.6])
    y_demo = (1 / (1 + np.exp(-(X_demo @ weights))) > 0.5).astype(int)
    clf = LogisticRegression()
    try:
        clf.fit(X_demo, y_demo)
        model = clf
        st.sidebar.info("Using embedded demo logistic model (approximate).")
    except Exception:
        model = None
        st.sidebar.error("Failed to create demo model.")
else:
    st.sidebar.info("Upload a model (pkl or json) or check 'Use demo' to try the app without a model.")

# If we still don't know feature names, let user supply them
feature_order_input = st.sidebar.text_input(
    "Feature order (comma-separated). Provide the exact order your model expects.",
    value=",".join(inferred_feature_names) if inferred_feature_names else ""
)

feature_order = [f.strip() for f in feature_order_input.split(",") if f.strip()] if feature_order_input else None

# Provide simple guidance about features
st.sidebar.markdown("**Feature guidance:** Provide features typically used by your model, e.g.:")
st.sidebar.markdown("- lanes (int), curvature (0-1), speed_limit (km/h), lighting (0=night,1=day), road_type_* (one-hot)")

# -------------------------
# Road feature schema
# -------------------------
st.header("Road feature editor & pair generator")

# If feature order available, build UI controls from it. Otherwise, provide common defaults.
if feature_order:
    features = feature_order
else:
    features = ["lanes", "curvature", "speed_limit", "lighting", "road_type_rural", "road_type_highway"]

st.write("Features used by the app (you can edit values for each road):")
st.write(", ".join(features))

# Helper to create a road interactive form given a prefix
def road_form(prefix, defaults=None):
    if defaults is None:
        defaults = {}
    cols = st.columns([1,1])
    with cols[0]:
        lanes = st.number_input(f"{prefix} - Lanes", min_value=1, max_value=10, value=int(defaults.get("lanes", 2)), step=1, key=f"{prefix}_lanes")
        curvature = st.slider(f"{prefix} - Curvature (0 straight, 1 very curvy)", 0.0, 1.0, float(defaults.get("curvature", 0.5)), key=f"{prefix}_curv")
        speed_limit = st.number_input(f"{prefix} - Speed limit (km/h)", min_value=20, max_value=140, value=int(defaults.get("speed_limit", 80)), step=5, key=f"{prefix}_spd")
    with cols[1]:
        lighting = st.selectbox(f"{prefix} - Lighting", options=["daylight","night"], index=0 if defaults.get("lighting","daylight")=="daylight" else 1, key=f"{prefix}_light")
        road_type = st.selectbox(f"{prefix} - Road type", options=["highway","rural","urban"], index=0 if defaults.get("road_type","rural")=="highway" else (1 if defaults.get("road_type","rural")=="rural" else 2), key=f"{prefix}_type")
        weather = st.selectbox(f"{prefix} - Weather", options=["clear","rain","fog"], index=0, key=f"{prefix}_weather")
    # convert to feature vector / dict compatible with feature list
    feat = {}
    for f in features:
        if f == "lanes":
            feat[f] = lanes
        elif f == "curvature":
            feat[f] = curvature
        elif f == "speed_limit":
            feat[f] = speed_limit
        elif f == "lighting":
            feat[f] = 1 if lighting == "daylight" else 0
        elif f.startswith("road_type_"):
            # one-hot road type
            typ = f.split("road_type_")[-1]
            feat[f] = 1 if road_type == typ else 0
        else:
            # unknown additional features: try defaults or zero
            feat[f] = defaults.get(f, 0)
    return feat

# Buttons to randomize pairs
col_rand = st.columns(3)
if col_rand[0].button("Randomize roads"):
    st.session_state["randomize"] = np.random.randint(1e9)

if "randomize" not in st.session_state:
    st.session_state["randomize"] = 0

rng = np.random.RandomState(st.session_state["randomize"])

# Create two roads (defaults from RNG)
def random_defaults(rng):
    return {
        "lanes": int(rng.choice([1,2,2,3,3,4])),
        "curvature": float(rng.beta(2,5)),  # more small curvature
        "speed_limit": int(rng.choice([50,60,80,90,100,120])),
        "lighting": rng.choice(["daylight","night"], p=[0.7,0.3]),
        "road_type": rng.choice(["highway","rural","urban"], p=[0.25,0.4,0.35]),
    }

d1 = random_defaults(rng)
d2 = random_defaults(rng)

st.subheader("Road A")
roadA = road_form("A", defaults=d1)
st.subheader("Road B")
roadB = road_form("B", defaults=d2)

# Convert to DataFrame respecting feature order
dfA = pd.DataFrame([roadA])
dfB = pd.DataFrame([roadB])

# If model exists, run predict_proba
st.header("Make your guess")
cols = st.columns(3)
with cols[0]:
    choice = st.radio("Which road do you think is safer?", ("A", "B"))
with cols[1]:
    submit_guess = st.button("Submit guess")

# Show model prediction live
model_available = model is not None and feature_order is not None and len(feature_order) > 0
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.write("### Road A")
    st.table(dfA.T.rename(columns={0:"value"}))
with col2:
    st.write("### Road B")
    st.table(dfB.T.rename(columns={0:"value"}))

st.markdown("---")

if model is None:
    st.warning("No model loaded. Upload a pickle or XGBoost JSON model in the sidebar, or enable demo mode. The app still lets you play by guessing but cannot show model predictions.")
else:
    # Ensure DataFrame contains all expected features
    # Add missing features with zeros
    for f in features:
        if f not in dfA.columns:
            dfA[f] = 0
        if f not in dfB.columns:
            dfB[f] = 0

    try:
        probsA = model_predict_proba(model, dfA, feature_order=features)
        probsB = model_predict_proba(model, dfB, feature_order=features)
        probA = float(probsA[0])
        probB = float(probsB[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        probA = probB = None

    # Display predictions
    pred_col1, pred_col2 = st.columns(2)
    if probA is not None:
        with pred_col1:
            st.metric("Model's predicted safety (Road A)", f"{probA:.2%}")
        with pred_col2:
            st.metric("Model's predicted safety (Road B)", f"{probB:.2%}")

    # Show which road model thinks safer
    if probA is not None:
        if probA > probB:
            model_choice = "A"
        elif probB > probA:
            model_choice = "B"
        else:
            model_choice = "Tie"
        st.info(f"Model predicts road **{model_choice}** is safer.")

# Handle the user's guess submission and basic scoring
if submit_guess:
    st.session_state.setdefault("plays", 0)
    st.session_state.setdefault("correct", 0)
    st.session_state["plays"] += 1

    if model is None or probA is None:
        # No model to compare correctness to; just record guess
        st.success(f"You chose **Road {choice}**. (No model available to compare.)")
    else:
        # Consider model's safer choice as "ground truth" for the game
        if probA > probB:
            safer = "A"
        elif probB > probA:
            safer = "B"
        else:
            safer = choice  # tie - accept user's choice as correct
        if choice == safer:
            st.success(f"✅ You matched the model's safer pick (Road {safer}).")
            st.session_state["correct"] += 1
        else:
            st.error(f"❌ You picked Road {choice}, but the model thinks Road {safer} is safer.")
        # show probabilities
        st.write(f"Model probs — Road A: **{probA:.2%}**, Road B: **{probB:.2%}**")

    st.write(f"Plays: {st.session_state['plays']}, Matches with model: {st.session_state['correct']}")

# Optional: show a small explainability section if SHAP is available
st.markdown("---")
st.subheader("Explainability (optional)")

try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

if shap_available and model is not None and hasattr(model, "predict_proba"):
    st.markdown("SHAP explanations for the model's prediction (approximate).")
    # compute SHAP values for small background
    try:
        explainer = shap.Explainer(model.predict_proba, pd.DataFrame(np.zeros((1, len(features))), columns=features))
        shap_vals_A = explainer(pd.DataFrame(dfA[features]))
        shap_vals_B = explainer(pd.DataFrame(dfB[features]))
        st.write("Road A SHAP (positive contributes to safety):")
        st.pyplot(shap.plots.waterfall(shap_vals_A[0], show=False))
        st.write("Road B SHAP (positive contributes to safety):")
        st.pyplot(shap.plots.waterfall(shap_vals_B[0], show=False))
    except Exception as e:
        st.info("SHAP failed to run for this kind of model: " + str(e))
else:
    st.info("Install `shap` and use a sklearn-compatible model with `predict_proba` to enable SHAP visuals.")

# Footer: tips and saving dataset
st.markdown("---")
st.write(
    """
**Tips**
- If your model expects scaled inputs (standardization / one-hot encoding), provide the features in the exact format/order the model expects (use the Feature order box).
- For better play/testing, upload a small CSV with example rows and a header to help you understand feature order.
- You can extend the app: store user choices, add leaderboard, or seed with real road records.
"""
)
