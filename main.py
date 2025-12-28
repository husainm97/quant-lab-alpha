# main.py

import gui
import tkinter as tk

def main():
    root = tk.Tk()
    app = gui.PortfolioGUI(root)  # Assuming gui.py defines PortfolioGUI
    root.mainloop()

if __name__ == "__main__":
    main()
