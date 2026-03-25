import matplotlib.pyplot as plt

print("=== GENERATING CLEAN TENSOR DIAGRAM ===")

fig, ax = plt.subplots(figsize=(12, 4))

# -----------------------------
# BOX POSITIONS
# -----------------------------
input_x = 0.1
lstm_x = 0.5
output_x = 0.85
y = 0.5

# -----------------------------
# INPUT BOX
# -----------------------------
ax.text(
    input_x, y,
    "Input Tensor\n(312165 samples)\n(10 timesteps × 84 features)",
    fontsize=11,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#cfe8f3", edgecolor="black")
)

# -----------------------------
# LSTM BOX
# -----------------------------
ax.text(
    lstm_x, y,
    "LSTM Model\nTemporal Learning",
    fontsize=11,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#b7e4c7", edgecolor="black")
)

# -----------------------------
# OUTPUT BOX
# -----------------------------
ax.text(
    output_x, y,
    "Prediction\n(Attack / Normal)",
    fontsize=11,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4a261", edgecolor="black")
)

# -----------------------------
# ARROWS (FIXED)
# -----------------------------
ax.annotate(
    "",
    xy=(lstm_x - 0.08, y),
    xytext=(input_x + 0.12, y),
    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black")
)

ax.annotate(
    "",
    xy=(output_x - 0.08, y),
    xytext=(lstm_x + 0.12, y),
    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black")
)

# -----------------------------
# TITLE
# -----------------------------
ax.set_title("LSTM Input-Output Flow (Temporal Modeling)", fontsize=14)

ax.axis("off")

plt.tight_layout()
plt.savefig("docs/tensor_flow_diagram.png", dpi=300)
plt.show()

print("[SUCCESS] Saved → docs/tensor_flow_diagram.png")