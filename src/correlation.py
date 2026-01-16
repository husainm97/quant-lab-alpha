import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk

def plot_correlation_heatmap(parent, portfolio_obj, method="pearson", is_dark=False):
    """
    Interactive correlation heatmap with clickable cells and detailed pairwise plots.
    """
    # Set plot style globally for this session
    if is_dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    returns_df = portfolio_obj.get_common_monthly_returns()
    corr = returns_df.corr(method=method)
    
    # Container to hold current view (switches between heatmap and pairwise)
    container = ttk.Frame(parent)
    container.pack(fill="both", expand=True)

    def show_heatmap():
        """Display the correlation heatmap with interaction hints."""
        
        # 1. Define internal handlers first to avoid UnboundLocalError
        def on_click(event):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                col_idx = int(round(event.xdata))
                row_idx = int(round(event.ydata))
                
                if 0 <= col_idx < len(corr.columns) and 0 <= row_idx < len(corr.index):
                    asset1 = corr.index[row_idx]
                    asset2 = corr.columns[col_idx]
                    show_pairwise_comparison(asset1, asset2)

        def on_mouse_move(event):
            if event.inaxes == ax:
                canvas.get_tk_widget().config(cursor="hand2")
            else:
                canvas.get_tk_widget().config(cursor="")

        # 2. Clear container
        for widget in container.winfo_children():
            widget.destroy()
        
        # 3. Build Heatmap Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr.values, cmap="coolwarm", norm=Normalize(-1, 1))
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        
        ax.set_title(f"Asset Return Correlation ({method.capitalize()})", 
                     fontweight='bold', pad=25)
        
        # Hint text (Standard text used to avoid Linux Glyph warnings)
        fig.text(0.5, 0.92, "INFO: Click any cell to view detailed pairwise analysis", 
                 ha='center', fontsize=9, color='gray' if not is_dark else '#aaaaaa',
                 fontstyle='italic')

        # Annotate cells with correlation values
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                       ha="center", va="center", fontsize=9,
                       color="white" if abs(corr.values[i, j]) > 0.5 else "black" if not is_dark else "white")
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        
        # 4. Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        
        # Connect interactive events
        canvas.mpl_connect('button_press_event', on_click)
        canvas.mpl_connect('motion_notify_event', on_mouse_move)
        
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, container)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    def show_pairwise_comparison(asset1, asset2):
        """Display detailed pairwise comparison plot with improved spacing."""
        for widget in container.winfo_children():
            widget.destroy()
        
        # Use constrained_layout to prevent labels/titles overlapping
        fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])
        
        ret1 = returns_df[asset1]
        ret2 = returns_df[asset2]
        correlation = corr.loc[asset1, asset2]
        
        # --- 1. Scatter plot with Regression ---
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(ret1, ret2, alpha=0.5, s=30, edgecolors='w', linewidth=0.5)
        
        z = np.polyfit(ret1, ret2, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ret1.min(), ret1.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Linear Fit")
        
        ax1.set_xlabel(f"{asset1} Returns")
        ax1.set_ylabel(f"{asset2} Returns")
        ax1.set_title(f"Scatter Analysis (Overall Correlation: {correlation:.3f})", fontweight='bold')
        ax1.grid(True, alpha=0.2)
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # --- 2. Time series comparison ---
        ax2 = fig.add_subplot(gs[1, 0])
        cumulative1 = (1 + ret1).cumprod()
        cumulative2 = (1 + ret2).cumprod()
        ax2.plot(returns_df.index, cumulative1, label=asset1)
        ax2.plot(returns_df.index, cumulative2, label=asset2)
        ax2.set_title("Relative Growth (Cumulative)", fontsize=10)
        ax2.set_xlabel(f"Year")
        ax2.set_ylabel(f"{portfolio_obj.base_currency}")
        ax2.legend(fontsize='x-small')
        ax2.grid(True, alpha=0.2)
        
        # --- 3. Rolling correlation ---
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_corr = ret1.rolling(window=12).corr(ret2)
        ax3.plot(returns_df.index, rolling_corr, color='purple', alpha=0.8)
        ax3.axhline(y=correlation, color='r', linestyle='--', linewidth=1, label='Mean')
        ax3.set_title("12-Month Rolling Correlation", fontsize=10)
        ax3.set_xlabel(f"Year")
        ax3.set_ylabel(f"Correlation")
        ax3.set_ylim(-1.1, 1.1)
        ax3.grid(True, alpha=0.2)
        
        # --- 4. Distribution comparison ---
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(ret1, bins=25, alpha=0.5, label=asset1, density=True)
        ax4.hist(ret2, bins=25, alpha=0.5, label=asset2, density=True)
        ax4.set_title("Return Frequency Distributions", fontsize=10)
        ax4.set_xlabel(f"Monthly Returns")
        ax4.set_ylabel(f"Frequency")
        ax4.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax4.grid(True, alpha=0.2)
        
        # --- 5. Statistics table ---
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        stats_data = [
            ['Metric', asset1, asset2],
            ['Ann. Mean', f'{ret1.mean()*12:.1%}', f'{ret2.mean()*12:.1%}'],
            ['Ann. Vol', f'{ret1.std()*np.sqrt(12):.1%}', f'{ret2.std()*np.sqrt(12):.1%}'],
            ['Sharpe', f'{ret1.mean()/ret1.std()*np.sqrt(12):.2f}', 
             f'{ret2.mean()/ret2.std()*np.sqrt(12):.2f}'],
            ['Worst Mo.', f'{ret1.min():.1%}', f'{ret2.min():.1%}'],
        ]
        
        table = ax5.table(cellText=stats_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.9) 
        
        # Define colors based on mode
        header_bg = '#3a3f5b' if is_dark else '#eeeeee'
        data_bg = '#1e1e1e' if is_dark else 'white'  # Force dark gray background
        text_color = 'white' if is_dark else 'black'
        edge_color = '#444444' if is_dark else '#cccccc'
        
        # Style EVERY cell in the table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(edge_color) # Clean up the borders
            
            if row == 0:
                # Header styling
                cell.set_facecolor(header_bg)
                cell.get_text().set_color(text_color)
                cell.get_text().set_weight('bold')
            else:
                # Data cell styling
                cell.set_facecolor(data_bg)
                cell.get_text().set_color(text_color)

        fig.suptitle(f"Pairwise Analysis: {asset1} vs {asset2}", 
                     fontsize=14, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
        
        # Navigation bar with Back Button
        nav_frame = ttk.Frame(container)
        nav_frame.pack(side="bottom", fill="x", pady=5)
        
        back_btn = ttk.Button(nav_frame, text="← Back to Correlations", command=show_heatmap)
        back_btn.pack(side="left", padx=10)
        
        toolbar = NavigationToolbar2Tk(canvas, nav_frame)
        toolbar.update()
        toolbar.pack(side="left", fill="x", expand=True)

    # Trigger initial view
    show_heatmap()