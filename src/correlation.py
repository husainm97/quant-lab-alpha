# src/correlation.py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_correlation_heatmap(parent, portfolio_obj, method="pearson", is_dark = False,):

    if is_dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    returns_df = portfolio_obj.get_common_monthly_returns()
    corr = returns_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", norm=Normalize(-1, 1))

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(f"Asset Return Correlation ({method.capitalize()}, Monthly)")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    # --- embed in Tk ---
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    return fig