import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Load model (cached so it only reads from disk once) ────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("california_knn_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            "⚠️ Model file **california_knn_pipeline.pkl** not found. "
            "Make sure it is in the same directory as this script."
        )
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

model = load_model()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown(
        """
        **Model:** KNN Regression Pipeline  
        **Dataset:** California Housing (sklearn)  
        **Target:** Median house value  
        **Unit:** Predicted value × $100,000
        """
    )
    st.divider()
    st.markdown(
        """
        **Feature guide**
        - *MedInc*: Median block income (tens of $1,000s)
        - *HouseAge*: Median house age in block (years)
        - *AveRooms*: Avg rooms per household
        - *AveBedrms*: Avg bedrooms per household
        - *Population*: Block population
        - *AveOccup*: Avg household occupancy
        - *Latitude / Longitude*: Block centroid (California only)
        """
    )

# ── Title ───────────────────────────────────────────────────────────────────────
st.title("🏘️ California Housing Price Prediction")
st.write("Enter the block-group characteristics below, then click **Predict** to estimate the median house value.")

# ── Input form ──────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Housing Features")

    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input(
            "Median Income (tens of $1,000s)",
            min_value=0.0, max_value=20.0, value=3.0, step=0.1,
            help="Median income for households in the block, measured in tens of thousands of USD."
        )
        HouseAge = st.number_input(
            "House Age (years)",
            min_value=0.0, max_value=100.0, value=20.0, step=1.0,
            help="Median age of houses in the block."
        )
        AveRooms = st.number_input(
            "Average Rooms per Household",
            min_value=1.0, max_value=50.0, value=5.0, step=0.5,
            help="Average number of rooms per household in the block."
        )
        AveBedrms = st.number_input(
            "Average Bedrooms per Household",
            min_value=0.5, max_value=20.0, value=1.0, step=0.5,
            help="Average number of bedrooms per household. Must be ≤ average rooms."
        )

    with col2:
        Population = st.number_input(
            "Block Population",
            min_value=1.0, max_value=40000.0, value=1000.0, step=10.0,
            help="Total population living in the block group."
        )
        AveOccup = st.number_input(
            "Average Occupancy",
            min_value=1.0, max_value=20.0, value=3.0, step=0.1,
            help="Average number of household members."
        )
        Latitude = st.number_input(
            "Latitude",
            min_value=32.5, max_value=42.0, value=34.0, step=0.01,
            help="Block centroid latitude. California ranges from ~32.5°N to 42°N."
        )
        Longitude = st.number_input(
            "Longitude",
            min_value=-124.5, max_value=-114.1, value=-118.0, step=0.01,
            help="Block centroid longitude. California ranges from ~-124.5° to -114.1°."
        )

    submitted = st.form_submit_button("🔍 Predict House Price", use_container_width=True)

# ── Validation & prediction ─────────────────────────────────────────────────────
if submitted:
    # --- Input validation ---
    errors = []

    if AveBedrms > AveRooms:
        errors.append("Average bedrooms cannot exceed average rooms.")
    if AveOccup > Population:
        errors.append("Average occupancy cannot exceed block population.")

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
        st.stop()

    # --- Build input DataFrame ---
    input_data = pd.DataFrame({
        "MedInc":     [MedInc],
        "HouseAge":   [HouseAge],
        "AveRooms":   [AveRooms],
        "AveBedrms":  [AveBedrms],
        "Population": [Population],
        "AveOccup":   [AveOccup],
        "Latitude":   [Latitude],
        "Longitude":  [Longitude],
    })

    # --- Predict ---
    try:
        prediction = model.predict(input_data)
        predicted_price = prediction[0] * 100_000

        # --- Confidence range via KNN neighbor variance ---
        try:
            steps = list(model.named_steps.keys())
            # Transform input through all steps except the final estimator
            X_transformed = input_data.copy()
            for step_name in steps[:-1]:
                X_transformed = model.named_steps[step_name].transform(X_transformed)

            estimator = model.named_steps[steps[-1]]
            distances, indices = estimator.kneighbors(X_transformed)
            neighbor_preds = estimator._y[indices[0]] * 100_000
            std = neighbor_preds.std()
            low  = max(0, predicted_price - std)
            high = predicted_price + std
            show_range = True
        except Exception:
            show_range = False

        # --- Display results ---
        st.divider()
        st.subheader("Prediction Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Price", f"${predicted_price:,.0f}")
        if show_range:
            m2.metric("Lower Estimate",  f"${low:,.0f}")
            m3.metric("Upper Estimate",  f"${high:,.0f}")

        if show_range:
            st.info(
                f"The model's {len(indices[0])} nearest neighbours suggest a price range of "
                f"**${low:,.0f} – ${high:,.0f}**."
            )

        # --- Show submitted inputs ---
        with st.expander("📋 View submitted inputs"):
            display_df = input_data.T.rename(columns={0: "Value"})
            st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

