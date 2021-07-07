import sqlite3
import numpy
import io
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import cv2
import face_recognition


def add(name, encoding):
    """
    This function add the data of student into the database
    :param name: name of person
    :param encoding: data of there image
    :return:
    """

    # create a database or connect to one
    conn = sqlite3.connect("person_database.bd", detect_types=sqlite3.PARSE_DECLTYPES)
    # create cursor
    c = conn.cursor()

    # we are saving encoding of image using BytesIO object
    out = io.BytesIO()
    numpy.save(out, encoding)
    out.seek(0)

    c.execute("INSERT INTO person_info VALUES (:name, :encoding)",
              {
                  "name": name.title(),
                  "encoding": sqlite3.Binary(out.read())
              })

    conn.commit()
    conn.close()


def delete(no):
    """
    This function remove the data of student from the database
    :param no: Id of student
    :return:
    """

    conn = sqlite3.connect("person_database.bd")
    c = conn.cursor()

    # delete a record
    c.execute(f"DELETE from person_info WHERE oid= " + str(no))

    conn.commit()
    conn.close()


def show():
    """
    It is used to return data of student
    """

    quality_list = []

    conn = sqlite3.connect("person_database.bd")
    c = conn.cursor()

    c.execute("SELECT *, oid FROM person_info")
    records = c.fetchall()

    conn.commit()
    conn.close()

    for record in records:
        quality_list.append(str(record[2]) + " " + str(record[0]))

    return quality_list


def open_person_database(root):
    """
    second window for person database
    :param root: instance of main interface
    :return:
    """
    try:
        root.withdraw()
        top = Toplevel()
        top.title("PERSON DATABASE")
        top.geometry("430x420")

        variable = StringVar(top)
        quality_combo = ttk.Combobox(top, width=27, font=("Times New Roman", 9, "bold"), textvariable=variable)

        # label at the top of person database name
        my_label = Label(top, text="Student Database", font=("Times New Roman", 18, "bold"))
        my_label.grid(row=0, column=1, padx=(60, 0), pady=(10, 0))

        # label and input field for name && Add button
        name_label = Label(top, text="NAME/ID :", font=("Times New Roman", 12, "bold"))
        name_label.grid(row=1, column=0, padx=(10, 0), pady=(10, 0))

        name = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        name.grid(row=1, column=1, padx=20, pady=(20, 0))

        # label and input field for path and button right to input field for browsing to img
        path_name_label = Label(top, text="PATH :", font=("Times New Roman", 12, "bold"))
        path_name_label.grid(row=2, column=0, padx=(20, 10), pady=(10, 0))

        path_name = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        path_name.grid(row=2, column=1, padx=(7, 10), pady=(10, 0))

        def path_of_image():
            root.filename = filedialog.askopenfilename(initialdir="/", title="select a image",
                                                       filetypes=(("jpeg images", "*.jpg"), ("png images", "*.png")))
            path_name.insert(0, root.filename)

        path_btn = Button(top, text="->", fg="black", bg="white", command=path_of_image)
        path_btn.grid(row=2, column=2, columnspan=2, pady=(25, 0), padx=(0, 0), ipady=1, ipadx=6)

        def add_data():
            if name.get() == "" or path_name.get() == "":
                messagebox.showerror("Message", "Please Fill All Required DATA")
            else:
                # read img
                img = cv2.imread(path_name.get())
                resize_crop_img = cv2.resize(img, (100, 100))

                # find encoding
                encoding = face_recognition.face_encodings(resize_crop_img)

                if encoding:
                    # pass to add function in person database.py
                    add(name.get(), encoding[0])
                    messagebox.showinfo("Message", "Data ADDED Successfully")
                else:
                    messagebox.showerror("error", "Image is not valid")

                name.delete(0, len(name.get()))  # delete the name in name input filled after hitting the add button
                path_name.delete(0, len(path_name.get()))

        add_btn = Button(top, text="ADD Data", font=("Times New Roman", 12, "bold"), bg="gray", command=add_data)
        add_btn.grid(row=3, column=0, columnspan=2, pady=10, padx=(70,0), ipadx=47)

        # label and input field for oid and delete button
        oid_label = Label(top, text="S-ID : ", font=("Times New Roman", 12, "bold"))
        oid_label.grid(row=4, column=0, pady=(20, 0))

        oid = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        oid.grid(row=4, column=1, pady=(20, 0))

        def delete_data():  # this function is used to delete data from the person_info table
            if oid.get() == "":
                messagebox.showerror("Message", "Please select record \n you want to DELETE")
            else:
                delete(oid.get())
                messagebox.showinfo("Message", "Data Deleted Successfully")
                oid.delete(0)

        delete_btn = Button(top, text="DELETE Data", font=("Times New Roman", 12, "bold"), bg="gray", command=delete_data)
        delete_btn.grid(row=5, column=0, columnspan=2, pady=(20, 10), padx=(60, 0), ipadx=30)

        def callback_func(event):  # this function get selected item from the combo box and load into oid i/p box
            """when the item choose from the combobox it load to choice"""
            choice = quality_combo.get()
            choice = int((choice.strip())[0])

            # put the data choose into oid input field
            oid.insert(0, choice)

        def show_data():  # this function used to show data in combo box whenever show button hit
            quality_combo['values'] = ()
            quality_combo['values'] = tuple(show())

            quality_combo.grid(row=7, column=0, columnspan=2, pady=(10, 0), padx=10, ipadx=30)
            quality_combo.bind("<<ComboboxSelected>>", callback_func)  # call callback function when record selected

        # show and exit button
        show_btn = Button(top, text="SHOW DATA", font=("Times New Roman", 12, "bold"), bg="gray", command=show_data)
        show_btn.grid(row=6, column=0, columnspan=2, pady=10, padx=(0, 150), ipadx=22)

        def hide_open2():
            root.deiconify()
            top.destroy()

        exit2_btn = Button(top, text="EXIT", font=("Times New Roman", 12, "bold"), bg="gray", command=hide_open2)
        exit2_btn.grid(row=6, column=1, columnspan=2, pady=20, padx=(160, 0), ipadx=50)

    except:
        messagebox.showerror("Message", "Error in the open_person database")
