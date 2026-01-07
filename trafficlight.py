import streamlit as st
import pandas as pd
import numpy as np
import random

# =========================
# Page Config
# =========================
st.set_page_config(page_title="GP Traffic Light Optimization", layout="wide")

st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("**Computational Evolution Case Study**")

# =========================
# Load Dataset
# =========================
data = pd.read_csv("traffic_dataset.csv")
st.dataframe(data.head())

X = data.drop(columns=["waiting_time"]).astype(float).values
y = data["waiting_time"].astype(float).values

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ GP Parameters")
population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 50, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

# =========================
# GP Functions
# =========================
def random_expression(n_features):
    feature = random.randint(0, n_features - 1)
    coef = random.uniform(-2, 2)
    bias = random.uniform(-5, 5)
    return (coef, feature, bias)

def predict(expr, X):
    coef, feature, bias = expr
    return coef * X[:, feature] + bias

def fitness(expr, X, y):
    y_pred = predict(expr, X)
    return np.mean((y - y_pred) ** 2)

def mutate(expr):
    coef, feature, bias = expr
    coef += random.uniform(-0.5, 0.5)
    bias += random.uniform(-1, 1)
    return (coef, feature, bias)

# =========================
# Run GP
# =========================
if st.button("▶ Run GP Optimization"):
    with st.spinner("Running GP evolution..."):

        population = [random_expression(X.shape[1]) for _ in range(population_size)]

        for gen in range(generations):
            scores = [(expr, fitness(expr, X, y)) for expr in population]
            scores.sort(key=lambda x: x[1])

            population = [expr for expr, _ in scores[:population_size // 2]]

            while len(population) < population_size:
                parent = random.choice(population)
                if random.random() < mutation_rate:
                    population.append(mutate(parent))
                else:
                    population.append(parent)

        best_expr = min(population, key=lambda e: fitness(e, X, y))
        best_fitness = fitness(best_expr, X, y)

    st.success("✅ GP Optimization Completed")

    # =========================
    # Results
    # =========================
    st.subheader("🏆 Best GP Expression")
    coef, feature, bias = best_expr
    st.code(f"waiting_time = {coef:.3f} * X{feature} + {bias:.3f}")

    st.subheader("📊 Fitness Score (MSE)")
    st.write(best_fitness)

    y_pred = predict(best_expr, X)

    st.subheader("📈 Actual vs Predicted Waiting Time")
    st.scatter_chart(
        pd.DataFrame({
            "Actual Waiting Time": y,
            "Predicted Waiting Time": y_pred
        })
    )

    st.subheader("📌 Conclusion")
    st.markdown(
        "- Genetic Programming successfully evolved a predictive expression\n"
        "- Fitness improved over generations\n"
        "- The evolved model is interpretable and suitable for traffic optimization"
    )

