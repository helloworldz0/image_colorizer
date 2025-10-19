import tkinter as tk

def on_run_click():
    print("Run program")

def on_stop_click():
    print("Stop program")

def on_view_click():
    print("View images")

def on_enter_1(event):
    button1.config(bg="lightblue", fg="black")

def on_enter_2(event):
    button2.config(bg="lightblue", fg="black")

def on_enter_3(event):
    button3.config(bg="lightblue", fg="black")

def on_leave_1(event):
    button1.config(bg="#f0f0f0", fg="black")

def on_leave_2(event):
    button2.config(bg="#f0f0f0", fg="black")

def on_leave_3(event):
    button3.config(bg="#f0f0f0", fg="black")

root = tk.Tk()
root.title("My Simple GUI")
root.geometry("1280x720")

label = tk.Label(root, text="GUI for Image Colorizer!", font=("Arial", 50))
label.pack(pady=25)

button1 = tk.Button(root, text="Run", command=on_run_click, width=25, height=5)
button1.pack(pady=25)

button2 = tk.Button(root, text="Stop", command=on_stop_click, width=25, height=5)
button2.pack(pady=25)

button3 = tk.Button(root, text="View", command=on_view_click, width=25, height=5)
button3.pack(pady=25)

button1.bind("<Enter>", on_enter_1)
button1.bind("<Leave>", on_leave_1)

button2.bind("<Enter>", on_enter_2)
button2.bind("<Leave>", on_leave_2)

button3.bind("<Enter>", on_enter_3)
button3.bind("<Leave>", on_leave_3)

root.mainloop()