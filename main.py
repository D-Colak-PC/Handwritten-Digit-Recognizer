import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import keras
import threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
W_WIDTH, W_HEIGHT = 1200, 600
C_WIDTH, C_HEIGHT = 560, 560
BRUSH_RADIUS = 20
PREDICTION_INTERVAL = 100  # milliseconds

# Global variables
model = keras.models.load_model('models/cnn.keras')
root = tk.Tk()
canvas = None
prediction_label = None
bar_canvas = None
figure = None
ax = None
image = None
draw = None
ai_image_label = None  # Label to display the processed image

# Control flag to avoid overlapping predictions
prediction_running = False

def setup_root():
    global root
    root.title("Digit Recognizer")
    root.geometry(f"{W_WIDTH}x{W_HEIGHT}")
    root.configure(bg="light gray")

def setup_canvas():
    global canvas
    canvas = tk.Canvas(root, width=C_WIDTH, height=C_HEIGHT, bg="black")
    canvas.place(x=28, rely=0.5, anchor="w")
    canvas.bind("<B1-Motion>", paint)

def setup_image():
    global image, draw
    image = Image.new("L", (C_WIDTH, C_HEIGHT), "black")
    draw = ImageDraw.Draw(image)

def setup_widgets():
    global prediction_label
    clear_button = tk.Button(
        root, text="Clear", font=("Fira Code Bold", 18),
        width=10, height=1, bg="#0000ff", activebackground="#5050ff",
        command=clear)
    clear_button.place(x=40 + C_WIDTH, rely=0.3, anchor="w")

    compute_button = tk.Button(
        root, text="Compute", font=("Fira Code Bold", 18),
        width=10, height=1, bg="#aa0000", activebackground="#aa5050",
        command=compute)
    compute_button.place(x=40 + C_WIDTH, rely=0.7, anchor="w")

    prediction_label = tk.Label(
        root, text="Prediction: ", font=("Fira Code", 30),
        bg="white", fg="black")
    prediction_label.place(relx=0.7, rely=0.2, anchor="w")

def setup_bar_graph():
    global figure, ax, bar_canvas
    figure = plt.Figure(figsize=(4, 3), dpi=100)
    ax = figure.add_subplot(111)
    ax.set_ylim([0, 100])
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar_canvas = FigureCanvasTkAgg(figure, root)
    bar_canvas.get_tk_widget().place(relx=0.7, rely=0.5, anchor="center")

def setup_ai_image():
    global ai_image_label
    ai_image_label = tk.Label(root)
    ai_image_label.place(relx=0.7, rely=0.8, anchor="center")

def update_bar_graph(probabilities):
    ax.clear()
    digits = list(range(10))
    percentages = probabilities * 100
    ax.bar(digits, percentages, color='blue')
    ax.set_ylim([0, 100])
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar_canvas.draw()

def update_ai_image(processed):
    # Enlarge the 28x28 image for better visibility (e.g., 5x enlargement)
    enlarged = processed.resize((140, 140), Image.NEAREST)
    photo = ImageTk.PhotoImage(enlarged)
    ai_image_label.config(image=photo)
    ai_image_label.image = photo

def paint(event):
    x, y = event.x, event.y
    # Draw on the Tkinter canvas
    canvas.create_oval(
        x - 1, y - 1, x + 1, y + 1,
        fill="white", width=BRUSH_RADIUS * 2, outline="white", tags="paint")
    # Draw on the PIL image (thread-safe)
    bbox = [x - BRUSH_RADIUS, y - BRUSH_RADIUS, x + BRUSH_RADIUS, y + BRUSH_RADIUS]
    with threading.Lock():
        draw.ellipse(bbox, fill=255, outline=255)

def clear():
    canvas.delete("paint")
    with threading.Lock():
        draw.rectangle([0, 0, C_WIDTH, C_HEIGHT], fill=0)
    prediction_label.config(text="Prediction: ")
    ax.clear()
    ax.set_ylim([0, 100])
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticks([])
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar_canvas.draw()
    # Clear the processed image display
    ai_image_label.config(image='')

def compute():
    with threading.Lock():
        img_copy = image.copy()
    prediction, accuracy, res, processed = run_prediction(img_copy)
    update_ui(prediction, accuracy, res, processed)
    print("Prediction:", prediction, accuracy)

def continuous_prediction():
    global prediction_running
    if not prediction_running:
        prediction_running = True
        threading.Thread(target=run_prediction_thread, daemon=True).start()
    root.after(PREDICTION_INTERVAL, continuous_prediction)

def run_prediction_thread():
    global prediction_running
    with threading.Lock():
        img_copy = image.copy()
    prediction, accuracy, res, processed = run_prediction(img_copy)
    root.after(0, lambda: update_ui(prediction, accuracy, res, processed))
    print("Real-Time Prediction:", prediction, accuracy)
    prediction_running = False

def update_ui(prediction, accuracy, res, processed):
    prediction_label.config(
        text=f"Prediction:\n{prediction} ({np.round(accuracy * 100, 2)}%)")
    update_bar_graph(res)
    update_ai_image(processed)

def run_prediction(img):
    # Preprocess the image: resize to 28x28 and convert to grayscale
    processed = img.resize((28, 28)).convert('L')
    arr = np.array(processed).reshape(1, 28, 28, 1) / 255.0
    res = model.predict(arr)[0]
    return np.argmax(res), np.max(res), res, processed

def main():
    setup_root()
    setup_canvas()
    setup_image()
    setup_widgets()
    setup_bar_graph()
    setup_ai_image()
    # Start continuous background prediction every 50ms.
    continuous_prediction()
    root.mainloop()

if __name__ == "__main__":
    main()
