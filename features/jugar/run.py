import tkinter as tk
from tkinter import ttk

class MainUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create the UI elements here.
        self.create_widgets()
    
    def create_widgets(self):
        label = ttk.Label(self, text="Hello, World")
        label.pack(padx=10, pady=10)
        
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Modern UI")
        self.geometry("300x200")
        
        # Create the MainUI instance
        main_ui = MainUI(self)
        main_ui.pack(side="top", fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
