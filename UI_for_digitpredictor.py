
# Run this file to output

from tkinter import *

from process_user_input import predict_from_image
from PIL import ImageTk, Image

def display_value():
    x=predict_from_image()

    cv = Canvas(root, width=200, height=200, bg='white')

    cv.pack()

    gif1 = PhotoImage(file='image.png')
    # image not visual
    cv.create_image(0, 0, image=gif1, anchor=NW)
    # assigned the gif1 to the canvas object
    cv.gif1 = gif1
    w1. config(text="\nPredicted value", font="Verdana 8")
    w2.config(text=str(x), font="Verdana 15 bold",fg = "red")


root=Tk()
root.geometry("550x450+100+100")
heading=Label(root, text="\n Hand Written Digit recognition Using Neural Network\n",font="Verdana 13 bold",fg = "green")
heading.pack()

w1 = Label(root, text="\n\nPress Train to train the neural network\n")

w1.pack()
w2 = Label(root, text="\nPress Predict to predict the drawn digit\n\n")
w2.pack()

button1=Button(text="Train")
button1.pack()

button2=Button(text="predict",command=display_value)
button2.pack()

root.mainloop()