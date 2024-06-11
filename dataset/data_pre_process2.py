import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import cv2
import numpy as np

def read_image_as_array(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    byte_array = np.asarray(bytearray(binary_data), dtype=np.uint8)
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
    return image

def get_mask(img_0, img_1, threshold=0):
    difference = cv2.absdiff(img_0, img_1)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_difference, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

class ImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparison Tool")

        self.img1_path = None
        self.img2_path = None

        self.label1 = tk.Label(root, text="Image 1: Not Selected")
        self.label1.pack()

        self.label2 = tk.Label(root, text="Image 2: Not Selected")
        self.label2.pack()

        self.select_button1 = tk.Button(root, text="Select Image 1", command=self.select_image1)
        self.select_button1.pack()

        self.select_button2 = tk.Button(root, text="Select Image 2", command=self.select_image2)
        self.select_button2.pack()

        self.compare_button = tk.Button(root, text="Compare Images", command=self.compare_images)
        self.compare_button.pack()

        self.clear_button = tk.Button(root, text="Clear Images", command=self.clear_images)
        self.clear_button.pack()

        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

    def select_image(self, image_number):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            if image_number == 1:
                self.img1_path = file_path
                self.label1.config(text=f"Image 1: {file_path}")
            elif image_number == 2:
                self.img2_path = file_path
                self.label2.config(text=f"Image 2: {file_path}")

    def select_image1(self):
        self.select_image(1)

    def select_image2(self):
        self.select_image(2)

    def on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if len(files) >= 2:
            self.img1_path = files[0]
            self.img2_path = files[1]
            self.label1.config(text=f"Image 1: {files[0]}")
            self.label2.config(text=f"Image 2: {files[1]}")
            self.compare_images()
        elif len(files) == 1:
            if self.img1_path is None:
                self.img1_path = files[0]
                self.label1.config(text=f"Image 1: {files[0]}")
            else:
                self.img2_path = files[0]
                self.label2.config(text=f"Image 2: {files[0]}")
            if self.img1_path and self.img2_path:
                self.compare_images()

    def compare_images(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showerror("Error", "Please select both images before comparing.")
            return

        img_0 = read_image_as_array(self.img1_path)
        img_1 = read_image_as_array(self.img2_path)

        if img_0 is None or img_1 is None:
            messagebox.showerror("Error", "Failed to read one or both images.")
            return

        mask = get_mask(img_0, img_1)
        mask_image = Image.fromarray(mask)
        mask_image.thumbnail((600, 400))

        mask_tk = ImageTk.PhotoImage(mask_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=mask_tk)
        self.canvas.image = mask_tk

    def clear_images(self):
        self.img1_path = None
        self.img2_path = None
        self.label1.config(text="Image 1: Not Selected")
        self.label2.config(text="Image 2: Not Selected")
        self.canvas.delete("all")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()
