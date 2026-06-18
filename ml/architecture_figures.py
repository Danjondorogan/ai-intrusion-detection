from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS = Path("results/architecture")
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (14,8)
plt.rcParams["font.size"] = 11

# =====================================================
# Helper
# =====================================================

def add_box(ax,x,y,w,h,text,color="#E8EEF7"):
    box = FancyBboxPatch(
        (x,y),
        w,
        h,
        boxstyle="round,pad=0.03",
        linewidth=2
    )

    ax.add_patch(box)

    ax.text(
        x+w/2,
        y+h/2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold"
    )

# =====================================================
# SYSTEM ARCHITECTURE
# =====================================================

fig, ax = plt.subplots(figsize=(16,8))

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.axis("off")

steps = [
    "Network Traffic",
    "Feature Extraction",
    "StandardScaler",
    "Temporal Buffer\n10 Timesteps",
    "LSTM IDS Model",
    "Threat Engine",
    "React Dashboard"
]

x = 5

for s in steps:

    add_box(ax,x,40,12,15,s)

    if x < 80:
        ax.arrow(
            x+12,
            47,
            5,
            0,
            head_width=2,
            length_includes_head=True
        )

    x += 17

plt.title(
    "System Architecture of AI-Based Intrusion Detection System",
    fontsize=18,
    weight="bold",
    pad=20
)

plt.savefig(
    RESULTS/"system_architecture.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# LSTM ARCHITECTURE
# =====================================================

fig, ax = plt.subplots(figsize=(16,8))

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.axis("off")

layers = [
    "Input\n10 x 84",
    "LSTM\n128 Units",
    "BatchNorm",
    "Dropout",
    "LSTM\n64 Units",
    "BatchNorm",
    "Dropout",
    "Dense\n64",
    "Dense\n1 Sigmoid"
]

x = 3

for l in layers:

    add_box(ax,x,40,10,15,l)

    if x < 90:
        ax.arrow(
            x+10,
            47,
            3,
            0,
            head_width=2,
            length_includes_head=True
        )

    x += 11

plt.title(
    "Deep LSTM Architecture Used for DoS Detection",
    fontsize=18,
    weight="bold",
    pad=20
)

plt.savefig(
    RESULTS/"lstm_architecture.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# TEMPORAL BUFFER
# =====================================================

fig, ax = plt.subplots(figsize=(14,6))

ax.set_xlim(0,100)
ax.set_ylim(0,50)
ax.axis("off")

for i in range(10):

    add_box(
        ax,
        5+i*8,
        20,
        6,
        10,
        f"T{i}"
    )

ax.arrow(
    87,
    25,
    8,
    0,
    head_width=2,
    length_includes_head=True
)

ax.text(
    92,
    32,
    "Prediction",
    ha="center",
    weight="bold"
)

plt.title(
    "Temporal Buffer Construction (10 Timesteps)",
    fontsize=18,
    weight="bold",
    pad=20
)

plt.savefig(
    RESULTS/"temporal_buffer.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# REAL TIME PIPELINE
# =====================================================

fig, ax = plt.subplots(figsize=(16,8))

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.axis("off")

pipeline = [
    "Incoming Traffic",
    "API Request",
    "Feature Scaling",
    "Buffer Update",
    "LSTM Prediction",
    "Probability",
    "Severity Class",
    "Dashboard"
]

x = 2

for p in pipeline:

    add_box(
        ax,
        x,
        40,
        10,
        15,
        p
    )

    if x < 90:
        ax.arrow(
            x+10,
            47,
            2,
            0,
            head_width=2,
            length_includes_head=True
        )

    x += 12

plt.title(
    "Real-Time Intrusion Detection Pipeline",
    fontsize=18,
    weight="bold",
    pad=20
)

plt.savefig(
    RESULTS/"inference_pipeline.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nGenerated:")
print("system_architecture.png")
print("lstm_architecture.png")
print("temporal_buffer.png")
print("inference_pipeline.png")