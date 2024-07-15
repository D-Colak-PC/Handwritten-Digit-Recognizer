import io
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TCL_LIBRARY'] = r'C:\Users\Dennis\AppData\Local\Programs\Python\Python312\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Dennis\AppData\Local\Programs\Python\Python312\tcl\tk8.6'

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import keras

W_WIDTH, W_HEIGHT = 1200, 600
C_WIDTH, C_HEIGHT = 560, 560
CIRCLE_RADIUS = 20


def create_window():
    root = tk.Tk() # window creation
    root.title("Digit Recognizer") # window title
    root.geometry(f"{W_WIDTH}x{W_HEIGHT}") # window size
    root.configure(bg="light gray") # window background color
    return root


def create_canvas():
    # canvas creation
    canvas = tk.Canvas(root, width=C_WIDTH, height=C_HEIGHT, bg="black")
    canvas.place(x=28, rely=0.5, anchor="w")
    return canvas


def paint(event):
    # tkinter
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(
        x1, y1, x2, y2,
        fill="white",
        width=CIRCLE_RADIUS*2,
        outline="white",
        tags="paint"
    )

    # draw a circle with CIRLCE_RADIUS with PIL
    draw.circle(xy=(event.x, event.y), radius=CIRCLE_RADIUS, fill=255, outline=255)


def clear():
    canvas.delete("paint")
    draw.rectangle([0, 0, C_WIDTH, C_HEIGHT], fill=0)


def create_clear_button():
    clear_button = tk.Button(
        root,
        text="Clear",
        font=("Fira Code Bold", 18),
        width=10,
        height=1,
        bg="#0000ff",
        activebackground="#5050ff",
        command=lambda: clear()
    )
    clear_button.place(x=40 + C_WIDTH, rely=0.3, anchor="w")
    return clear_button


def update_prediction_labels(prediction, accuracy):
    prediction_label.config(
        text=f"Prediction:\n{prediction} ({np.round((accuracy * 100), 2)}%)"
    )


def compute(grid, image):
    # resize image to 28x28 pixels
    image = image.resize((28, 28))
    # convert rgb to grayscale
    image = image.convert('L')
    image = np.array(image)
    # reshaping for model normalization
    image = image.reshape(1, 28, 28, 1)
    image = (image / 255.0).astype("float32")

    # predicting the class
    res = model.predict([image])[0]
    print("Prediction: ", np.argmax(res), max(res))

    update_prediction_labels(np.argmax(res), max(res))
    return np.argmax(res), max(res)


def create_compute_button():
    compute_button = tk.Button(
        root,
        text="Compute",
        font=("Fira Code Bold", 18),
        width=10,
        height=1,
        bg="#aa0000",
        activebackground="#aa5050",
        command=lambda: compute(grid, image)
    )
    compute_button.place(x=40 + C_WIDTH, rely=0.7, anchor="w")
    return compute_button


def create_prediction_label():
    prediction_label = tk.Label(
        root,
        text="Prediction: ",
        font=("Fira Code", 30),
        bg="white",
        fg="black"
    )
    prediction_label.place(relx=0.7, rely=0.2, anchor="w")
    return prediction_label


if __name__ == "__main__":
    # model
    model = keras.models.load_model('models/nn_1.keras')

    # internal grid
    grid = np.zeros((28, 28)).astype("float32")

    # IL CANVAS
    image = Image.new("L", (C_WIDTH, C_HEIGHT), "black")
    draw = ImageDraw.Draw(image)

    #TK GUI
    root = create_window()

    canvas = create_canvas()
    canvas.bind("<B1-Motion>", paint)  # event listener for mouse movement

    clear_button = create_clear_button()

    compute_button = create_compute_button()

    prediction_label = create_prediction_label()

    root.mainloop()
