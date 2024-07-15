print("running")

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TCL_LIBRARY'] = r'C:\Users\Dennis\AppData\Local\Programs\Python\Python312\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Dennis\AppData\Local\Programs\Python\Python312\tcl\tk8.6'

import keras
import tkinter as tk
import numpy as np
import threading
import time

W_WIDTH, W_HEIGHT = 1200, 600
C_WIDTH, C_HEIGHT = 560, 560

# Initialize the grid and model globally
grid = np.zeros((28, 28)).astype("float32")
model = keras.models.load_model("models/nn.keras")

def create_window():
    root = tk.Tk() # window creation
    root.title("Digit Recognizer") # window title
    root.geometry(f"{W_WIDTH}x{W_HEIGHT}") # window size
    root.configure(bg="light gray") # window background color
    return root


def create_canvas(root):
    # canvas creation
    canvas = tk.Canvas(root, width=C_WIDTH, height=C_HEIGHT, bg="black")
    canvas.place(x=28, rely=0.5, anchor="w")

    for i in range(1, 28):
        canvas.create_line(i*20, 0, i*20, 560, fill="#505050")
        canvas.create_line(0, i*20, 560, i*20, fill="#505050")
    return canvas


def paint(event):
    # map the mouse position to the grid
    x = int(event.x / 20)
    y = int(event.y / 20)

    # draw a rectangle on the grid
    grid[y, x] = 1.0
    canvas.create_rectangle(x*20, y*20, x*20+20, y*20+20, fill="white")


def ai_predict():
    while True:
        time.sleep(1)
        # make a prediction using the model
        pred = model.predict(np.array([grid]))

        # print the prediction number + accuracy
        print(f"Predicted Digit: {np.argmax(pred)} | Accuracy: {np.max(pred)*100:.2f}%")


if __name__ == "__main__":
    print("running")
    #TK GUI
    root = create_window()

    canvas = create_canvas(root)
    canvas.bind("<B1-Motion>", paint)  # event listener for mouse movement

    ai_pred = threading.Thread(target=ai_predict, daemon=True)
    ai_pred.start()

    root.mainloop()