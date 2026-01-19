# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from imblearn.over_sampling import SMOTE

import shap
import matplotlib.pyplot as plt

import dice_ml


st.set_page_config(page_title="Diabetes XAI Dashboard", layout="wide")


@st.cache_resource
def load_and_train(path: str = "./archive/diabetes.csv"):
    df = pd.read_csv(path)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # quick test metrics (for display only)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }

    # SHAP explainer (TreeExplainer works for GBDT)
    explainer = shap.TreeExplainer(model)

    # DiCE data/model objects (use train split, not SMOTE-resampled, for realism)
    dice_data = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=list(X.columns),
        outcome_name="Outcome",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")

    return X_train, X_test, y_train, y_test, model, explainer, dice_exp, metrics


def make_input_df(vals: dict) -> pd.DataFrame:
    # Keep feature order consistent with training columns
    cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    return pd.DataFrame([[vals[c] for c in cols]], columns=cols)


# ---------------- UI ----------------
st.title("🩺 Diabetes Prediction + Explainability (SHAP + DiCE)")
st.caption("Single-page mock dashboard: input 8 clinical features → prediction + explanations.")

X_train, X_test, y_train, y_test, model, explainer, dice_exp, metrics = load_and_train()

with st.expander("Model quick metrics (held-out test split)", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("F1", f"{metrics['f1']:.3f}")
    c3.metric("RMSE", f"{metrics['rmse']:.3f}")

# --- User Role Selection ---
role = st.radio(
    "Select User Role:",
    [
        "Domain Specialists",
        "Regulators & Governance Bodies",
        "End Users (Patients)",
        "Data Scientists & AI Developers",
    ],
    horizontal=True
)

st.divider()

left, right = st.columns([0.9, 1.6], gap="large")

with left:
    st.subheader("1) Patient inputs (8 features)")
    st.write("Enter the values below, then click **Predict**.")

    # Inputs (sane ranges; adjust freely)
    pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=2, step=1)
    glucose = st.number_input("Plasma glucose concentration", min_value=0, max_value=300, value=120, step=1)
    bloodpressure = st.number_input("Diastolic blood pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)
    skinthickness = st.number_input("Triceps skinfold thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    insulin = st.number_input("Serum insulin level (mu U/ml)", min_value=0, max_value=1000, value=80, step=1)
    bmi = st.number_input("Body mass index (BMI)", min_value=0.0, max_value=80.0, value=28.5, step=0.1)
    dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.40, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, value=33, step=1)

    user_vals = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    if st.button("🔎 Predict", type="primary", use_container_width=True):
        st.session_state["predict_clicked"] = True

with right:
    st.subheader("2) Results & explanations")

    if not st.session_state.get("predict_clicked", False):
        st.info("Fill in the inputs on the left, then click **Predict**.")
    else:
        query_df = make_input_df(user_vals)

        # Prediction
        pred = int(model.predict(query_df)[0])
        proba = float(model.predict_proba(query_df)[0, 1])  # probability of class 1

        top1, top2, top3 = st.columns([1, 1, 1])
        top1.metric("Predicted Outcome", "Diabetes (1)" if pred == 1 else "No Diabetes (0)")
        top2.metric("P(Outcome=1)", f"{proba:.3f}")
        top3.metric("Decision", "High risk" if proba >= 0.5 else "Lower risk")

        # st.write("**Input snapshot**")
        # st.dataframe(query_df, use_container_width=True)

        st.divider()

        # -------- SHAP local: waterfall --------
        if role in [
            "Domain Specialists",
            "End Users (Patients)",
            "Data Scientists & AI Developers",
        ]:
            st.markdown("### Local explanation (SHAP waterfall)")
            shap_values = explainer(query_df)  # Explanation object (n=1)

            # fig1 = plt.figure()
            # shap.plots.waterfall(shap_values[0], show=False)
            # st.pyplot(fig1, clear_figure=True)

            plot_col, _ = st.columns([1, 1])  # left half only

            with plot_col:
                fig1 = plt.figure(figsize=(3.2, 2.4))  # FIXED SIZE
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig1, clear_figure=True)

        # -------- SHAP global: summary (sampled) --------
        if role in [
            "Domain Specialists",
            "Regulators & Governance Bodies",
            "Data Scientists & AI Developers",
        ]:
            st.markdown("### Global explanation (SHAP summary, sampled)")
            # st.caption("This is computed on a sample to keep the dashboard responsive.")
            # sample_n = min(300, len(X_test))
            # X_samp = X_test.sample(sample_n, random_state=42)
            X_samp = X_test.copy()
            sv_samp = explainer(X_samp)

            # fig2 = plt.figure()
            # shap.summary_plot(sv_samp.values, X_samp, show=False)
            # st.pyplot(fig2, clear_figure=True)

            plot_col, _ = st.columns([1, 1])  # left half only

            with plot_col:
                fig2 = plt.figure(figsize=(3.2, 2.4))  # FIXED SIZE
                shap.summary_plot(sv_samp.values, X_samp, show=False)
                st.pyplot(fig2, clear_figure=True)

        # Optional: dependence plot for Glucose
        # st.markdown("### Feature dependence (Glucose)")
        # fig3 = plt.figure()
        # shap.dependence_plot("Glucose", sv_samp.values, X_samp, show=False)
        # st.pyplot(fig3, clear_figure=True)

        st.divider()

        # -------- DiCE counterfactuals --------
        if role in ["End Users (Patients)", "Data Scientists & AI Developers"]:
            st.markdown("### Counterfactuals (DiCE)")
            st.caption("Suggested minimal changes to flip the prediction (method=random).")

            try:
                cf = dice_exp.generate_counterfactuals(
                    query_df,
                    total_CFs=3,
                    desired_class="opposite",
                )
                cf_df = cf.cf_examples_list[0].final_cfs_df
                st.dataframe(cf_df, use_container_width=True)
            except Exception as e:
                st.warning("Counterfactual generation failed on this run.")
                st.exception(e)

st.caption("Note: This is a mock dashboard for demonstration. Do not use as medical advice.")
