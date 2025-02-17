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
BRUSH_RADIUS = 10
PREDICTION_INTERVAL = 50  # milliseconds
PADDING = 20  # uniform vertical padding for right panel widgets

class DigitRecognizerApp:
    def __init__(self):
        self.model = keras.models.load_model('models/cnn.keras')
        self.prediction_running = False
        self.last_x, self.last_y = None, None
        self.image_lock = threading.Lock()
        self.image = Image.new("L", (C_WIDTH, C_HEIGHT), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_root()
        self.setup_canvas()
        self.setup_control_buttons()  # Clear and Compute buttons remain unchanged
        self.setup_right_panel()      # New right panel for prediction label, bar graph, and AI image
        self.continuous_prediction()

    def setup_root(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")
        self.root.geometry(f"{W_WIDTH}x{W_HEIGHT}")
        self.root.configure(bg="light gray")

    def setup_canvas(self):
        self.canvas = tk.Canvas(self.root, width=C_WIDTH, height=C_HEIGHT, bg="black")
        self.canvas.place(x=28, rely=0.5, anchor="w")
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def setup_control_buttons(self):
        clear_button = tk.Button(
            self.root, text="Clear", font=("Fira Code Bold", 18),
            width=10, height=1, bg="#0000ff", activebackground="#5050ff",
            command=self.clear
        )
        clear_button.place(x=40 + C_WIDTH, rely=0.3, anchor="w")

        compute_button = tk.Button(
            self.root, text="Compute", font=("Fira Code Bold", 18),
            width=10, height=1, bg="#aa0000", activebackground="#aa5050",
            command=self.compute
        )
        compute_button.place(x=40 + C_WIDTH, rely=0.7, anchor="w")

    def setup_right_panel(self):
        # Create a frame on the right side to hold the prediction label, bar graph, and AI image.
        self.right_frame = tk.Frame(self.root, bg="light gray")
        self.right_frame.place(relx=0.8, rely=0.5, anchor="center")
        
        # Prediction label
        self.prediction_label = tk.Label(
            self.right_frame, text="Prediction: ", font=("Fira Code", 30),
            bg="white", fg="black"
        )
        self.prediction_label.pack(pady=PADDING)
        
        # Bar graph setup using matplotlib
        self.figure = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.configure_axes(self.ax)
        self.bar_canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.bar_canvas.get_tk_widget().pack(pady=PADDING)
        
        # AI image label
        self.ai_image_label = tk.Label(self.right_frame)
        self.ai_image_label.pack(pady=PADDING)

    def configure_axes(self, ax):
        ax.set_ylim([0, 100])
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(10))
        ax.set_yticks([])
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)

    def update_bar_graph(self, probabilities):
        self.ax.clear()
        self.configure_axes(self.ax)
        digits = list(range(10))
        percentages = probabilities * 100
        self.ax.bar(digits, percentages, color='blue')
        self.bar_canvas.draw()

    def update_ai_image(self, processed):
        # Resize the processed image so its height equals the bar graph's height.
        bar_height = int(self.figure.get_figheight() * self.figure.dpi)
        enlarged = processed.resize((bar_height, bar_height), Image.NEAREST)
        photo = ImageTk.PhotoImage(enlarged)
        self.ai_image_label.config(image=photo)
        self.ai_image_label.image = photo

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.last_x is not None and self.last_y is not None:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=BRUSH_RADIUS*2, fill="white",
                                    capstyle=tk.ROUND, smooth=True)
            with self.image_lock:
                self.draw.line((self.last_x, self.last_y, x, y), fill=255, width=BRUSH_RADIUS*2)
            self.last_x, self.last_y = x, y
        else:
            self.last_x, self.last_y = event.x, event.y

    def end_draw(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        with self.image_lock:
            self.draw.rectangle([0, 0, C_WIDTH, C_HEIGHT], fill=0)
        self.prediction_label.config(text="Prediction: ")
        self.ax.clear()
        self.configure_axes(self.ax)
        self.bar_canvas.draw()
        self.ai_image_label.config(image='')

    def compute(self):
        with self.image_lock:
            img_copy = self.image.copy()
        prediction, accuracy, res, processed = self.run_prediction(img_copy)
        self.update_ui(prediction, accuracy, res, processed)
        print("Prediction:", prediction, accuracy)

    def continuous_prediction(self):
        if not self.prediction_running:
            self.prediction_running = True
            threading.Thread(target=self.run_prediction_thread, daemon=True).start()
        self.root.after(PREDICTION_INTERVAL, self.continuous_prediction)

    def run_prediction_thread(self):
        with self.image_lock:
            img_copy = self.image.copy()
        prediction, accuracy, res, processed = self.run_prediction(img_copy)
        self.root.after(0, lambda: self.update_ui(prediction, accuracy, res, processed))
        print("Real-Time Prediction:", prediction, accuracy)
        self.prediction_running = False

    def update_ui(self, prediction, accuracy, res, processed):
        # Format the accuracy to two decimal places and update prediction label text.
        self.prediction_label.config(
            text=f"Prediction:\n{prediction} ({accuracy * 100:.2f}%)"
        )
        self.update_bar_graph(res)
        self.update_ai_image(processed)

    def preprocess_image(self, img, size=28, margin=4):
        np_img = np.array(img)
        coords = np.argwhere(np_img > 0)
        if coords.size == 0:
            return Image.new("L", (size, size), "black")
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = img.crop((x0, y0, x1, y1))
        w, h = cropped.size
        scale = (size - 2 * margin) / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.ANTIALIAS
        resized = cropped.resize((new_w, new_h), resample_filter)
        new_img = Image.new("L", (size, size), "black")
        offset_x = (size - new_w) // 2
        offset_y = (size - new_h) // 2
        new_img.paste(resized, (offset_x, offset_y))
        return new_img

    def run_prediction(self, img):
        processed = self.preprocess_image(img, size=28, margin=4)
        arr = np.array(processed).reshape(1, 28, 28, 1) / 255.0
        res = self.model.predict(arr)[0]
        return np.argmax(res), np.max(res), res, processed

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.run()
