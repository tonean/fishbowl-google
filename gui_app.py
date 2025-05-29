import tkinter as tk
from tkinter import ttk, scrolledtext
from main import call_agent

class AIAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant")
        self.root.geometry("800x600")
        
        # Configure style
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#f0f0f0")
        
        # Main container
        self.main_frame = ttk.Frame(root, style="Custom.TFrame", padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            width=70,
            height=20,
            font=("Arial", 11)
        )
        self.chat_display.pack(pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input area
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X, pady=5)
        
        self.message_entry = ttk.Entry(
            self.input_frame,
            font=("Arial", 11)
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind("<Return>", self.send_message)
        
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Welcome message
        self.add_message("AI Assistant", "I can help you with screen analysis, game strategies, puzzles, and more!")
    
    def add_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        message = self.message_entry.get().strip()
        if message:
            self.add_message("You", message)
            self.message_entry.delete(0, tk.END)
            
            try:
                response = call_agent(message)
                self.add_message("AI Assistant", response)
            except Exception as e:
                self.add_message("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = AIAssistantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 