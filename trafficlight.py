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
st.subheader("📂 Traffic Dataset")
st.dataframe(data.head())

# =========================
# Encode categorical column
# =========================
if data["time_of_day"].dtype == object:
    data["time_of_day"] = data["time_of_day"].map({
        "morning": 0,
        "afternoon": 1,
        "evening": 2,
        "night": 3
    })

st.write("Encoded dataset preview:")
st.dataframe(data.head())

# =========================
# Feature & Target (FORCE NUMERIC)
# =========================
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
            scored = [(expr, fitness(expr, X, y)) for expr in population]
            scored.sort(key=lambda x: x[1])
