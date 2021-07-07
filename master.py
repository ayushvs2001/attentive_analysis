from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk
from tkinter import *
from processing_data import data_processing_interface
from model_building import model_building_interface
from model_implementation import model_implementation_interface
from student_database import open_person_database

from tkinter import messagebox
from tkinter import filedialog

root = ThemedTk(theme="radiance")
root.title("Attentive Analysis")
root.geometry("400x430")

# label at the top of person database name
my_label = Label(root, text="Attentive Analysis", font=("Times New Roman", 18, "bold"))
my_label.pack(padx=(40, 60), pady=(10, 20))

data_processing_btn = Button(root, text="Data Processing", font=("Times New Roman", 12, "bold"),
                             command=lambda: data_processing_interface(root))
data_processing_btn.pack(padx=30, pady=20, ipadx=60, ipady=3)

model_building_btn = Button(root, text="Model Building", font=("Times New Roman", 12, "bold"),
                            command=lambda: model_building_interface(root))
model_building_btn.pack(padx=30, pady=20, ipadx=63, ipady=3)

model_implementation_btn = Button(root, text="Model Implementation", font=("Times New Roman", 12, "bold"),
                                  command=lambda: model_implementation_interface(root))
model_implementation_btn.pack(padx=30, pady=20, ipadx=35, ipady=3)

model_implementation_btn = Button(root, text="Student Database", font=("Times New Roman", 12, "bold"),
                                  command=lambda: open_person_database(root))
model_implementation_btn.pack(padx=30, pady=20, ipadx=54, ipady=3)

ttk.Button(root, text="Quit", command=root.destroy).pack()
root.mainloop()
