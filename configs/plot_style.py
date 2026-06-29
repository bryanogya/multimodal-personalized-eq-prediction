import matplotlib.pyplot as plt

# Font
FONT_FAMILY = "Times New Roman"

TITLE_SIZE = 14
SUPTITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11
TEXT_SIZE = 12


# Figure
FIG_DEFAULT = (8, 5)
FIG_WIDE = (10, 5)
FIG_SQUARE = (6, 6)
FIG_LARGE = (10, 6)
FIG_EXTRA_LARGE = (15, 10)

DPI = 300

# Line
LINE_WIDTH = 2.0
MARKER_SIZE = 6

# Grid
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"

# Save Fig
SAVE_BBOX = "tight"

# Color
TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#ff7f0e"

TARGET_COLOR = "#2ca02c"
PREDICT_COLOR = "#d62728"
DEVICE_COLOR = "#9467bd"

PROPOSED_BAR_COLOR = "#55A868"
BAR_COLOR =  "#4C72B0"

# Apply Style
def apply_plot_style():

    plt.rcParams.update({

        # Font
        "font.family": FONT_FAMILY,

        # Font size
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,

        # Figure
        "figure.dpi": DPI,
        "savefig.dpi": DPI,

        # Grid
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "grid.linestyle": GRID_LINESTYLE,

        # Legend
        "legend.frameon": True,

        # Save
        "savefig.bbox": SAVE_BBOX,
    })
