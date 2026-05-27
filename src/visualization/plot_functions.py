# plot the functions from basictest.py functions over interval [-10, 10]
import numpy as np
import matplotlib.pyplot as plt
from src.data.generators import (
    piecewise_linear_function,
    multiple_steps_function,
    sine_with_discontinuity,
    linear_with_varying_slopes
)


def main():
    # Define the interval
    x = np.linspace(-10, 10, 1000)

    # List of functions to plot (name, function_object)
    # The simple step_function is excluded as requested
    functions_to_plot = [
        ("Piecewise Linear", piecewise_linear_function),
        ("Multiple Steps", multiple_steps_function),
        ("Sine with Discontinuity", sine_with_discontinuity),
        ("Linear Varying Slopes", linear_with_varying_slopes)
    ]

    # Create a figure with subplots side by side (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (name, func) in zip(axes, functions_to_plot):
        # Calculate y values
        y = func(x)

        # Handle discontinuities: insert NaNs where jumps occur to prevent plotting connecting lines
        # The resolution is high enough (dx approx 0.02) that continuous slopes (max ~2)
        # produce small deltas (< 0.1), while jumps in these functions are >= 2.0.
        threshold = 1.0
        jump_indices = np.where(np.abs(np.diff(y)) > threshold)[0] + 1

        if len(jump_indices) > 0:
            x_plot = np.insert(x, jump_indices, np.nan)
            y_plot = np.insert(y, jump_indices, np.nan)
        else:
            x_plot = x
            y_plot = y

        # Plot
        ax.plot(x_plot, y_plot, linewidth=2, color='navy')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add some margin to y-axis using original y values (ignoring the NaNs implicitly)
        y_range = y.max() - y.min()
        if y_range == 0:
            y_range = 1.0
        ax.set_ylim(y.min() - 0.1 * y_range, y.max() + 0.1 * y_range)

    plt.tight_layout()

    # Save the plot
    output_file = 'basic_test_functions_side_by_side.png'
    plt.savefig(output_file)
    print(f"Plot saved to '{output_file}'")


if __name__ == "__main__":
    main()