import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# =========================
# Page Config
# =========================
st.set_page_config(page_title="GP Traffic Light Optimization", layout="wide")
st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("JIE 42903 - Evolutionary Computing (Lab Report and Project)")

# =========================
# Load Dataset
# =========================
st.subheader("Traffic Dataset")
data = pd.read_csv("traffic_dataset.csv")
feature_names = list(data.drop(columns=["vehicle_count"]).columns)

# Encode categorical features if needed
for col in feature_names:
    if data[col].dtype == object:
        data[col] = data[col].astype('category').cat.codes

st.markdown("Encoded Dataset Preview")
st.dataframe(data.head())

# =========================
# Features & Target
# =========================
X = data.drop(columns=["vehicle_count"]).astype(float).values
y = data["vehicle_count"].astype(float).values

# =========================
# Sidebar Parameters
# =========================
population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
coef_range = st.sidebar.slider("Coefficient Range (±)", 0.5, 5.0, 2.0)
bias_range = st.sidebar.slider("Bias Range (±)", 1.0, 10.0, 5.0)

optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

if optimization_mode == "Multi Objective":
    complexity_weight = st.sidebar.slider(
        "Complexity Weight",
        0.0, 1.0, 0.2,
        help="Trade-off between accuracy and simplicity"
    )
else:
    complexity_weight = 0.0

# =========================
# GP Helper Functions
# =========================
def random_expression(features):
    feature_idx = random.randint(0, len(features) - 1)
    coef = random.uniform(-coef_range, coef_range)
    bias = random.uniform(-bias_range, bias_range)
    return (coef, feature_idx, bias)

def predict(expr, X):
    coef, feature_idx, bias = expr
    return coef * X[:, feature_idx] + bias

def fitness(expr, X, y):
    y_pred = predict(expr, X)
    mse = np.mean((y - y_pred) ** 2)
    if optimization_mode == "Single Objective":
        return mse
    else:
        # Penalize complex coefficients for interpretability
        return mse + complexity_weight * abs(expr[0])

def mutate(expr):
    coef, feature_idx, bias = expr
    coef += random.uniform(-0.2*coef_range, 0.2*coef_range)
    bias += random.uniform(-0.2*bias_range, 0.2*bias_range)
    return (coef, feature_idx, bias)

# =========================
# Run GP Optimization
# =========================
st.subheader("Genetic Programming Optimization Results")

if st.button("Run Genetic Programming (GP)"):
    start_time = time.time()
    with st.spinner("Running GP evolution..."):

        # Initialize population
        population = [random_expression(feature_names) for _ in range(population_size)]
        fitness_history = []

        for gen in range(generations):
            # Evaluate fitness
            scored = [(expr, fitness(expr, X, y)) for expr in population]
            scored.sort(key=lambda x: x[1])
            fitness_history.append(scored[0][1])

            # Selection: keep top 50%
            population = [expr for expr, _ in scored[:population_size//2]]

            # Reproduction & Mutation
            while len(population) < population_size:
                parent = random.choice(population)
                if random.random() < mutation_rate:
                    population.append(mutate(parent))
                else:
                    population.append(parent)

        # Extract best individual
        best_expr = min(population, key=lambda e: fitness(e, X, y))
        best_fitness = fitness(best_expr, X, y)
        y_pred = predict(best_expr, X)

    exec_time = time.time() - start_time

    # =========================
    # Status
    # =========================
    st.success("Genetic Programming Optimization Completed")
    st.info(f"Mode: **{optimization_mode}**")
    st.metric("Execution Time (s)", f"{exec_time:.4f}")

    # =========================
    # Best Mathematical Model
    # =========================
    coef, feature_idx, bias = best_expr
    feature_name = feature_names[feature_idx]
    st.subheader("Best Interpretable Mathematical Model")
    st.code(f"vehicle_count = {coef:.3f} × {feature_name} + {bias:.3f}")
    st.write(f"📉 Best Fitness (MSE): {best_fitness:.4f}")
    st.write(f"⏱ Execution Time: {exec_time:.4f} seconds")

    # =========================
    # Visualizations
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Convergence Curve**")
        st.line_chart(pd.DataFrame({"Best Fitness (MSE)": fitness_history}), height=280)
    with col2:
        st.markdown("**Actual vs Predicted Vehicle Count**")
        st.scatter_chart(pd.DataFrame({"Actual": y, "Predicted": y_pred}), height=280)

    # =========================
    # Performance Analysis
    # =========================
    st.subheader("Performance Analysis")
    st.markdown(
        "- **Convergence Rate:** Shows fitness improvement over generations\n"
        "- **Accuracy:** Evaluated via Mean Squared Error\n"
        "- **Efficiency:** Execution time shows computational cost\n"
        "- **Interpretability:** Simple linear model provides clarity for traffic decisions\n"
        "- Multi-objective optimization balances accuracy and simplicity"
    )

    # =========================
    # Conclusion
    # =========================
    st.subheader("Conclusion")
    st.markdown(
        "This GP system automatically generates interpretable models for predicting vehicle counts. "
        "By evolving solutions over generations, it identifies the feature most affecting traffic and "
        "produces a simple, effective formula suitable for traffic light optimization."
    )

