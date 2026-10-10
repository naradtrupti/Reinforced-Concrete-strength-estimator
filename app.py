import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Concrete Strength Estimator", page_icon="🏗️")

@st.cache_resource
def load_files():
    return joblib.load("concrete_final_model.pkl"), joblib.load("concrete_final_meta.pkl")

model, meta = load_files()
features = meta["features"]

labels = {
    "silicafumecontent": "Silica fume (kg/m³)",
    "additivecontent": "Additive (admixture) (kg/m³)",
    "water": "Water (kg/m³)",
    "cementecontent": "Cement (kg/m³)",
    "flyashcontent": "Fly ash (kg/m³)",
    "fibercontent": "Fiber content (kg/m³)",
}

st.title("🏗️ Fiber-Reinforced Concrete Strength Estimator")
st.caption("Predicts compressive strength (MPa) from mix proportions. "
           "Random Forest trained on 521 literature mixes.")

st.subheader("Mix proportions")
inputs = {}
cols = st.columns(2)
for i, f in enumerate(features):
    with cols[i % 2]:
        inputs[f] = st.number_input(
            labels.get(f, f),
            min_value=0.0,
            value=float(meta["median"][f]),
            step=1.0,
            help=f"Training range: {meta['min'][f]:.1f} to {meta['max'][f]:.1f}",
        )

if st.button("Predict strength", type="primary"):
    row = pd.DataFrame([inputs])[features]

    outside = [labels.get(c, c) for c in features
               if not (meta["min"][c] <= inputs[c] <= meta["max"][c])]
    if outside:
        st.warning("Outside the training range for: " + ", ".join(outside)
                   + ". The prediction may be unreliable.")

    per_tree = np.array([t.predict(row.values)[0] for t in model.estimators_])
    mean_pred, spread = float(per_tree.mean()), float(per_tree.std())

    st.metric("Estimated compressive strength", f"{mean_pred:.1f} MPa")
    st.write(f"Tree disagreement: ±{spread:.1f} MPa")
    st.write("Typical error on new, unseen mixes is about **±8 MPa**, "
             f"so a realistic range is roughly **{mean_pred - 8:.0f} to {mean_pred + 8:.0f} MPa**.")

st.info("This is an estimate from 521 literature mixes and does not include curing age. "
        "Verify with lab testing. Not for structural design use.")
