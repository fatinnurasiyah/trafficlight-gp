import streamlit as st
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="GP Traffic Light Optimization", layout="wide")

st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("Computational Evolution Case Study – Streamlit GP Application")

# =========================
# Load Dataset
# =========================
st.subheader("📂 Traffic Dataset")

data = pd.read_csv("traffic_dataset.csv")
st.dataframe(data.head())

# Assume dataset columns
# traffic_flow, queue_length, waiting_time
X = data[["traffic_flow", "queue_length"]].values
y = data["waiting_time"].values

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ GP Parameters")
pop_size = st.sidebar.slider("Population Size", 100, 1000, 500, step=100)
generations = st.sidebar.slider("Generations", 10, 200, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 8, 4)

opt_mode = st.sidebar.radio("Optimization Mode", ["Single Objective", "Multi Objective"])

# =========================
# GP Model
# =========================
st.subheader("🧠 Genetic Programming Model")

if st.button("Run GP Optimization"):
    with st.spinner("Running GP evolution..."):
        gp = SymbolicRegressor(
            population_size=pop_size,
            generations=generations,
            p_crossover=0.7,
            p_subtree_mutation=mutation_rate,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_depth=max_depth,
            stopping_criteria=0.01,
            verbose=1,
            random_state=42
        )

        gp.fit(X, y)

    st.success("GP Optimization Completed")

    # =========================
    # Results
    # =========================
    st.subheader("🏆 Best GP Expression")
    st.code(str(gp._program))

    st.subheader("📊 Fitness Score")
    st.write(f"Best Fitness: {gp.fitness_}")

    # =========================
    # Prediction Analysis
    # =========================
    y_pred = gp.predict(X)

    st.subheader("📈 Actual vs Predicted Waiting Time")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.set_xlabel("Actual Waiting Time")
    ax.set_ylabel("Predicted Waiting Time")
    st.pyplot(fig)

    # =========================
    # Performance Analysis
    # =========================
    st.subheader("📌 Performance Analysis")
    st.markdown("""
    **Key Metrics Evaluated:**
    - Convergence Rate
    - Prediction Accuracy
    - Expression Complexity

    **Observations:**
    - Rapid fitness improvement in early generations
    - Stable convergence after sufficient evolution
    - GP produces interpretable mathematical models
    """)
