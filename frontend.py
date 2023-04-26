import tkinter as tk
from tkinter import messagebox
import fake_news_classifier  # Import your function for predicting fake news

def classify_news():
    # Get the input from the title and text entry fields
    title = title_entry.get()
    text = text_entry.get("1.0", "end-1c")

    # Calling classifier function
    is_fake = fake_news_classifier.classify(title, text)

    # Show the prediction result in a message box
    if is_fake==[1]:
        messagebox.showinfo("Prediction Result", "The news is FAKE!")
    else:
        messagebox.showinfo("Prediction Result", "The news is REAL.")

# Create a new Tkinter window

window = tk.Tk()
window.title("Fake News Classifier")

# Create form labels and input fields
title_label = tk.Label(window, text="Title:")
title_entry = tk.Entry(window, width=50)

text_label = tk.Label(window, text="Text:")
text_entry = tk.Text(window, width=50, height=10)

# Add the form elements to the window using a grid layout
title_label.grid(row=0, column=0)
title_entry.grid(row=0, column=1)

text_label.grid(row=1, column=0)
text_entry.grid(row=1, column=1)

# Add a classify button to the form
classify_button = tk.Button(window, text="Classify", command=classify_news)
classify_button.grid(row=2, column=1)


# Start the Tkinter event loop
window.mainloop()