import tkinter as tk

root = tk.Tk()
root.title("GUI Test")
root.geometry("300x200")

label = tk.Label(root, text="If you can see this, tkinter is working!")
label.pack(pady=50)

root.mainloop()