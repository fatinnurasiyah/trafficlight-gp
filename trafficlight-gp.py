import streamlit as st
import matplotlib.pyplot as plt

st.title("Traffic Prediction using Genetic Programming")

st.write("Best GP Model:")
st.code(str(best_model))

st.metric("Test MSE", round(test_mse, 4))

st.subheader("Convergence Curve")
fig, ax = plt.subplots()
ax.plot(gen, min_fitness)
ax.set_xlabel("Generation")
ax.set_ylabel("MSE")
st.pyplot(fig)
