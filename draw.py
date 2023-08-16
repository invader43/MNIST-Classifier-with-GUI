# Import the required libraries
from tkinter import *
from tkinter import ttk
from tkinter.colorchooser import askcolor
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

PATH = 'model.ckpt'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# Create an instance of the tkinter frame or window
win = Tk()

# Set the size of the window
win.geometry("600x600")  # Increased height to accommodate more buttons

# Variables to track the previous point, drawing state, color, and width
prev_x, prev_y = None, None
drawing = False
draw_color = "black"
draw_width = 20

# Define a function to start drawing
def start_drawing(event):
    global prev_x, prev_y, drawing
    prev_x, prev_y = event.x, event.y
    drawing = True

# Define a function to continue drawing
def continue_drawing(event):
    global prev_x, prev_y, drawing
    if drawing and prev_x is not None and prev_y is not None:
        x, y = event.x, event.y
        canvas.create_line(prev_x, prev_y, x, y, fill=draw_color, width=draw_width, capstyle=ROUND, smooth=TRUE)
        prev_x, prev_y = x, y

# Define a function to stop drawing
def stop_drawing(event):
    global drawing
    drawing = False

# Define a function to change the drawing color
def change_color():
    global draw_color
    color = askcolor(color=draw_color)[1]
    if color:
        draw_color = color

# Define a function to change the drawing width
def change_width(new_width):
    global draw_width
    draw_width = new_width

def perform_inference(canvas):
    # Load your external model and checkpoint as before
    from torchmodel import NeuralNet
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    print("loading Checkpoint")
    checkpoint = torch.load(PATH)
    print("loaded Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("loaded state dict")
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Rescale the canvas drawing to match model input size
    drawing = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(drawing.encode('utf-8')))
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)

    model.eval()  # Set the model to evaluation mode



# Create a canvas widget
canvas = Canvas(win, width=300, height=300, background="white")
canvas.grid(row=0, column=0, columnspan=2)

# Bind events for drawing
canvas.bind('<Button-1>', start_drawing)
canvas.bind('<B1-Motion>', continue_drawing)
canvas.bind('<ButtonRelease-1>', stop_drawing)

# Create buttons
clear_button = Button(win, text="Clear Canvas", command=lambda: canvas.delete("all"))
clear_button.grid(row=1, column=0, padx=10, pady=10)

exit_button = Button(win, text="Exit", command=win.quit)
exit_button.grid(row=1, column=1, padx=10, pady=10)

color_button = Button(win, text="Change Color", command=change_color)
color_button.grid(row=2, column=0, padx=10, pady=10)

width_buttons = [
    Button(win, text="Small", command=lambda: change_width(5)),
    Button(win, text="Medium", command=lambda: change_width(10)),
    Button(win, text="Large", command=lambda: change_width(20))
]

for idx, button in enumerate(width_buttons):
    button.grid(row=2, column=idx + 1, padx=10, pady=10)

# Create a button for performing inference
inference_button = Button(win, text="Perform Inference", command=lambda: perform_inference(canvas))
inference_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)


# Create a label to display inference result
result_label = Label(win, text="", font=("Helvetica", 16))
result_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# ... (previous code)

# Create a button for performing inference
inference_button = Button(win, text="Perform Inference", command=lambda: perform_inference(canvas))
inference_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Create a Label widget for displaying prediction
prediction_label = Label(win, text="Prediction = 1", font=("Helvetica", 16))
prediction_label.grid(row=0, column=2, padx=10, pady=10)

# ... (rest of the code)

win.mainloop()

