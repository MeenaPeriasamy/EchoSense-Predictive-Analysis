import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_img():
    global img, image_display, results_label
    # Clear previous image and results
    for widget in frame.winfo_children():
        if widget not in [load, title_label]:
            widget.destroy()

    img_path = filedialog.askopenfilename(initialdir="/", title="Select an Image",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    if img_path:
        img = Image.open(img_path)
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Display the image
        image_display = ImageTk.PhotoImage(img.resize((250, 250), Image.Resampling.LANCZOS))
        display_label = Label(frame, image=image_display)
        display_label.image = image_display
        display_label.pack()

        results_label = Label(frame, text="", font=("Helvetica", 14))
        results_label.pack(side=tk.TOP, pady=20)

        # Predict
        predict(img_array)

def predict(img):
    class_names = ['Abnormal Heartbeat', 'Myocardial Infarction', 'Normal', 'History of Myocardial Infarction']
    subcategories = {
        0: "\n1. Atrial fibrillation\n2. Tachycardia\n3. Bradycardia\n4. Ventricular fibrillation\n5. Premature contractions",
        1: "\n1. Acute Myocardial Infarction (AMI)\n2. STEMI\n3. NSTEMI\n4. Arrhythmias\n5. Heart Failure\n6. Cardiogenic Shock\n7. Ventricular Septal Defect\n8. Papillary Muscle Rupture",
        2: "\nNo abnormalities",
        3: "\n1. Pathological Q Waves\n2. T Wave Inversions\n3. ST Segment Changes"
    }
    result = model.predict(img)
    predicted_class = np.argmax(result)
    main_category = class_names[predicted_class]
    details = ''.join(subcategories[predicted_class])
    results_label.config(text=f"Main Category: {main_category}\nPossible Conditions: {details}", justify=tk.LEFT)

root = tk.Tk()
root.title("EchoSense - ECG Analysis")
root.state('zoomed')  # Full screen mode
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)
title_label = Label(frame, text="Upload ECG Image for Analysis", font=("Helvetica", 18, "bold"))
title_label.pack(side=tk.TOP, pady=20)
load = Button(frame, text="Upload Image", command=load_img, font=("Helvetica", 14, "bold"), relief="raised", activebackground="grey", bg="lightgrey")
load.pack(side=tk.TOP, pady=10)
results_label = Label(frame, text="", font=("Helvetica", 12))
results_label.pack(side=tk.TOP, pady=20)
model = load_model('your path model.h5')

root.mainloop()