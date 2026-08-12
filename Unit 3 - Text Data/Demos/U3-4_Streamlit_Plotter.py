import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Page setup
# ============================================================

# Configure the Streamlit page.
# page_title appears in the browser tab.
# layout="wide" allows more horizontal space for plotting.
st.set_page_config(
    page_title="Interactive Function Plotter",
    layout="wide"
)

# Main title shown at the top of the page
st.title("Interactive Function Plotter")

# Short descriptive text under the title
st.caption("Choose a function and customize how it is plotted.")

# ============================================================
# Sidebar controls
# ============================================================

# Everything inside st.sidebar appears in the left panel.
# This is typically where user controls go.

st.sidebar.header("Function Settings")

# Dropdown menu allowing the user to choose a function.
# The selected value is stored in function_name.
function_name = st.sidebar.selectbox(
    "Choose a function",
    [
        "sin(x)",
        "cos(x)",
        "tan(x)",
        "exp(x)",
        "log(x)",
        "x^2",
        "x^3",
        "sqrt(|x|)",
    ]
)

st.sidebar.header("Axis Limits")

# number_input allows numeric typing or arrow-click changes.
# step controls how much the value changes each click.

x_min = st.sidebar.number_input(
    "x min",
    value=-5.0,
    step=0.5
)

x_max = st.sidebar.number_input(
    "x max",
    value=5.0,
    step=0.5
)

y_min = st.sidebar.number_input(
    "y min",
    value=-5.0,
    step=0.5
)

y_max = st.sidebar.number_input(
    "y max",
    value=5.0,
    step=0.5
)

st.sidebar.header("Plot Style")

# Slider for controlling line thickness
line_width = st.sidebar.slider(
    "Line width",
    min_value=1,
    max_value=10,
    value=2
)

# Dropdown for selecting matplotlib plotting styles
# Different styles change colors, grid appearance, etc.
plt_style = st.sidebar.selectbox(
    "Matplotlib style",
    [
        "default",
        "dark_background",
        "ggplot",
        "seaborn-v0_8",
        "bmh",
        "fivethirtyeight"
    ]
)

# Controls how many x values are generated.
# More points → smoother curve but slightly slower plotting.
num_points = st.sidebar.slider(
    "Number of points",
    min_value=10,
    max_value=100,
    value=50
)

# ============================================================
# Generate x values
# ============================================================

# Create evenly spaced x values between x_min and x_max.
# linspace is commonly used in plotting continuous functions.
x = np.linspace(x_min, x_max, num_points)

# ============================================================
# Define functions
# ============================================================

# This function maps the selected function name
# to the corresponding mathematical operation.
#
# Students can extend this by adding more cases.

def compute_function(name, x):

    # Trigonometric functions
    if name == "sin(x)":
        return np.sin(x)

    elif name == "cos(x)":
        return np.cos(x)

    elif name == "tan(x)":
        return np.tan(x)

    # Exponential function
    elif name == "exp(x)":
        return np.exp(x)

    # Logarithm (requires positive x values)
    elif name == "log(x)":
        # Replace invalid x values with NaN
        # so matplotlib does not plot them.
        x_safe = np.where(x <= 0, np.nan, x)
        return np.log(x_safe)

    # Polynomial functions
    elif name == "x^2":
        return x**2

    elif name == "x^3":
        return x**3

    # Square root of absolute value
    elif name == "sqrt(|x|)":
        return np.sqrt(np.abs(x))

    # Fallback case (should not normally occur)
    else:
        return np.zeros_like(x)

# Compute y values from selected function
y = compute_function(function_name, x)

# ============================================================
# Plotting
# ============================================================

# Apply selected matplotlib style
plt.style.use(plt_style)

# Create figure and axes
# figsize controls width and height in inches.
fig, ax = plt.subplots(figsize=(10, 5))

# Plot x vs y
ax.plot(
    x,
    y,
    linewidth=line_width
)

# Set axis limits based on user inputs
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# ============================================================
# Center axis spines at zero
# ============================================================

# Move axes so that x=0 and y=0 intersect in the middle.
# This makes the plot look like a standard math coordinate plane.

ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")

# Hide the outer box edges
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# Keep tick marks only on visible axes
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

# Labels and title
ax.set_title(f"Plot of {function_name}")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add grid lines for readability
ax.grid(True, alpha=0.4)

# ============================================================
# Layout with columns
# ============================================================

# Create two columns.
# The first column is wider (3 units vs 1).
col1, col2 = st.columns([3, 1])

# Left column: main plot
with col1:

    # Display matplotlib figure in Streamlit
    st.pyplot(fig)

# Right column: summary information
with col2:

    st.subheader("Plot Summary")

    # st.metric displays labeled values prominently
    st.metric(
        "Function",
        function_name
    )

    st.metric(
        "Points",
        num_points
    )

    st.write("Current Axis Limits:")

    st.write(
        f"x: [{x_min}, {x_max}]"
    )

    st.write(
        f"y: [{y_min}, {y_max}]"
    )