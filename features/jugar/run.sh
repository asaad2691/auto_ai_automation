# Update to main.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess

class MainUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create the UI elements here.
        self.create_widgets()
    
    def create_widgets(self):
        self.command_entry = ttk.Entry(self, width=50)
        self.command_entry.pack(padx=10, pady=10)
        
        submit_button = ttk.Button(self, text="Submit", command=self.submit_command)
        submit_button.pack(padx=10, pady=10)
        
        self.output_area = scrolledtext.ScrolledText(self, width=50, height=10)
        self.output_area.pack(padx=10, pady=10)
    
    def submit_command(self):
        command = self.command_entry.get()
        try:
            if not command:
                raise Exception("Please enter a command.")
            
            # Run the CLI function here
            output = subprocess.check_output(["python", "run.py"] + command.split(), stderr=subprocess.STDOUT)
        except Exception as e:
            output = str(e).encode()
        
        self.output_area.delete("1.0", tk.END)  # Clear the text area
        self.output_area.insert(tk.INSERT, output.decode())  # Insert the command output

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Modern UI")
        self.geometry("500x400")
        
        # Create the MainUI instance
        main_ui = MainUI(self)
        main_ui.pack(side="top", fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
