import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error

st.title("🚦 Traffic Light Optimization using Genetic Programming")

# ==============================
# Genetic Programming Components
# ==============================

functions = ['+', '-', '*', '/']
terminals = ['vehicle_count', 'average_speed',
             'lane_occupancy', 'flow_rate', 'time_of_day']


def safe_div(a, b):
    return a / b if b != 0 else 1


def generate_expression(depth=3):
    if depth == 0 or random.random() < 0.3:
        return random.choice(terminals)
    return (
        random.choice(functions),
        generate_expression(depth - 1),
        generate_expression(depth - 1)
    )


def evaluate(expr, row):
    if isinstance(expr, str):
        return row[expr]
    op, left, right = expr
    a = evaluate(left, row)
    b = evaluate(right, row)
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/': return safe_div(a, b)


def fitness(expr, X, y):
    preds = []
    for _, row in X.iterrows():
        preds.append(evaluate(expr, row))
    return mean_squared_error(y, preds)


def crossover(e1, e2):
    if not isinstance(e1, tuple) or not isinstance(e2, tuple):
        return e1
    return (e1[0], e1[1], e2[2])


def mutate(expr):
    if random.random() < 0.2:
        return generate_expression()
    return expr


# ==============================
# Streamlit UI
# ==============================

uploaded_file = st.file_uploader("Upload Traffic Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Dataset Preview", df.head())

    X = df[terminals]
    y = df['waiting_time']

    if st.button("Run Genetic Programming"):
        population = [generate_expression() for _ in range(30)]

        for gen in range(10):
            scores = [(fitness(expr, X, y), expr) for expr in population]
            scores.sort(key=lambda x: x[0])
            population = [expr for _, expr in scores[:10]]

            while len(population) < 30:
                p1, p2 = random.sample(population[:10], 2)
                child = crossover(p1, p2)
                child = mutate(child)
                population.append(child)

        best_expr = population[0]
        best_mse = fitness(best_expr, X, y)

        st.success("✅ Genetic Programming Completed")

        st.subheader("📐 Best Mathematical Model")
        st.code(best_expr)

        st.subheader("📉 Fitness (MSE)")
        st.write(best_mse)
