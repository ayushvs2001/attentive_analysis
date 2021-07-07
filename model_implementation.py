import os
from ttkthemes import ThemedTk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import keras as k
import cv2
import numpy as np
import datetime
import time
import face_recognition
import xlsxwriter
import sqlite3
import io

MODEL_PATH = ''
current_model = None
ATTENTIVENESS = ['Attentive', 'Not_Attentive']
DETECTOR = cv2.CascadeClassifier("classifier/lbpcascade_frontalface_improved.xml")


def load_model(model_path):
    """
    This function used to load model

    :param model_path: path of model
    """
    global current_model, MODEL_PATH

    # if model path is empty then assign model path equal to default model path
    if model_path == "":
        model_path = './model/model.h5'

    # model path is empty and current model is None means we have to load the default model
    if model_path == './model/model.h5' and current_model is None:
        MODEL_PATH = model_path
        if os.path.exists(model_path):
            current_model = k.models.load_model(model_path)

    # if model path is not equal to MODEL_Path then we have to load model as user changes the model path
    elif model_path != MODEL_PATH:
        MODEL_PATH = model_path
        if os.path.exists(model_path):
            current_model = k.models.load_model(model_path)

    # if model path is same as MODEL_PATH and current model is same, means the user using same model for testing
    elif model_path == MODEL_PATH and current_model:
        return current_model
    else:
        messagebox.showerror("Error", "Model Not Fount")

    return current_model


def predict_image_label(test_image):
    """
    Function used to run model on image at a time

    :param test_image: three dimensional array
    :return:
    """
    global current_model
    test_image = np.expand_dims(test_image, axis=0)
    result = current_model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'not_attentive'
    else:
        prediction = 'attentive'
    return prediction


def result_file(student_list, attentive_counts, total_frame, path):
    """
    It is used to add data in xlsx file
    :param student_list:  Students ID
    :param attentive_counts: Each student attentive frame count
    :param total_frame: total no of frame in video
    :param path: path where we are going to store result
    """

    file_name = f"{path}/Attentive_info.xlsx"

    # create a file object
    workbook = xlsxwriter.Workbook(file_name)

    # used to add work shit using worksheet object
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, "ID")
    worksheet.write(0, 2, "No of Attentive frames")
    worksheet.write(0, 4, "No of Non Attentive Frames")
    worksheet.write(0, 6, "Attentiveness")
    row = 1
    column = 0
    for i in range(len(student_list)):
        worksheet.write(row, column, student_list[i])
        worksheet.write(row, column + 2, attentive_counts[i])
        worksheet.write(row, column + 4, total_frame-attentive_counts[i])
        worksheet.write(row, column + 6, attentive_counts[i]/total_frame > 0.65)
        row += 1

    workbook.close()


def increase_dim(x1, y1, x2, y2, shape):
    """
    This function used to increase co-ordinate of face in images properly

    :param y1, x2, y2, x1: co-ordinate of face in images
    :param shape: width and height of image
    :return: proper increased co-ordinates
    """
    if y1 - 0.105 * y1 > 0:
        y1 = int(y1 - 0.105 * y1)

    if x1 - 0.065 * x1 > 0:
        x1 = int(x1 - 0.07 * x1)

    if y2 + 0.065 * y2 < shape[0]:
        y2 = int(y2 + 0.07 * y2)

    if x2 + 0.065 * x2 < shape[1]:
        x2 = int(x2 + 0.065 * x2)

    return x1, y1, x2, y2


def on_video(video_path, model_path, model_dir_path, save):
    """
    Function used to implement model on the video or webcam

    :param video_path: path of video or empty string(in case of webcam)
    :param model_path: path of model
    :param model_dir_path: directory where model present
    :param save: flag denoting Yes or No
    """
    global current_model
    load_model(model_path)

    # taking data from model_info file
    if model_dir_path == "":
        model_dir_path = "./model"
    f = open(f"{model_dir_path}/model_info.txt", "r")
    data = f.read()
    f.close()
    i = data.index("dimension") + 12
    n = int(data[i: i + 3])

    fourcc, out = None, None
    flag = (save == "Yes")

    # fetching data from the database
    conn = sqlite3.connect("person_database.bd")
    c = conn.cursor()

    c.execute("SELECT *, oid FROM person_info")
    records = c.fetchall()

    class_names = []
    encode_list_known = []

    # load encoding of faces into list
    for record in records:
        class_names.append(record[0])
        out = io.BytesIO(record[1])
        out.seek(0)
        data = np.load(out)
        encode_list_known.append(data)

    conn.commit()
    conn.close()

    try:
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)

        # set width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        # creating directory for storing output
        dir_format = str(datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S'))
        if not os.path.exists('./output'):
            os.mkdir(os.path.join('./', "output"))
        os.mkdir(os.path.join('./output/', dir_format))

        if flag:
            # define the codec and create videoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            out = cv2.VideoWriter(f"./output/{dir_format}/processed_video.avi", fourcc, 15.0, (width, height))

        _continue = True
        total_frame = 0
        attentive_count = [0] * len(class_names)
        while True:
            success, original_img = cap.read()
            _continue != _continue
            total_frame += 1
            if success and _continue:
                gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                cos = DETECTOR.detectMultiScale(gray_img, 1.06, 3)

                # process on each face in the image
                for i in cos:
                    x1, y1, x2, y2 = round(i[0] * 0.901038160701588), round(i[1] * 0.9086847599164927), \
                                     round((i[2] + i[0]) * 1.05864893845164), round((i[3] + i[1]) * 1.0579661016949153)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    face = cv2.resize(gray_img[y1:y2, x1:x2], (n, n))
                    face = np.expand_dims(face, axis=-1)
                    result = predict_image_label(face)

                    # identify img
                    identify_img = cv2.resize(gray_img[y1:y2, x1:x2], (0, 0), None, 0.25, 0.25)
                    identify_img = cv2.cvtColor(identify_img, cv2.COLOR_BGR2RGB)

                    cv2.putText(original_img, result, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    encode_cur_frame = face_recognition.face_encodings(identify_img)

                    if encode_cur_frame:
                        matches = face_recognition.compare_faces(encode_list_known, encode_cur_frame[0])
                        face_dis = face_recognition.face_distance(encode_list_known, encode_cur_frame[0])
                        match_index = np.argmin(face_dis)
                        if matches[match_index] and face_dis[match_index] < 0.5:
                            if result == "attentive":
                                attentive_count[match_index] += 1

                # quit on press 'q'
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                cv2.imshow('webcam', original_img)
                if flag:
                    out.write(original_img)
            else:
                time.sleep(1)

        # de-allocate any memory usage
        cv2.destroyAllWindows()

        result_file(class_names, attentive_count, total_frame, f"./output/{dir_format}")
        messagebox.showinfo("info", "Processing On Given Data Completed")

    except Exception as e:
        messagebox.showerror("error", f"Please try again\n {e}")


def on_video_anon(video_path, model_path, model_dir_path, save):
    """
    Function used to implement model on the video or webcam

    :param video_path: path of video or empty string(in case of webcam)
    :param model_path: path of model
    :param model_dir_path: directory where model present
    :param save: flag denoting Yes or No
    """
    global current_model
    load_model(model_path)

    if model_dir_path == "":
        model_dir_path = "./model"
    f = open(f"{model_dir_path}/model_info.txt", "r")
    data = f.read()
    f.close()
    i = data.index("dimension") + 12
    n = int(data[i: i + 3])

    fourcc, out = None, None
    flag = save == "Yes"

    # creating directory for storing output
    dir_format = str(datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S'))
    if not os.path.exists('./output'):
        os.mkdir(os.path.join('./', "output"))
    os.mkdir(os.path.join('./output/', dir_format))


    try:
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        if flag:
            # define the codec and create videoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(f"./output/{dir_format}/processed_video.avi", fourcc, 15.0, (width, height))

        total_frame = 0
        attentive_count = 0

        while True:
            success, original_img = cap.read()
            if success:
                gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                cos = DETECTOR.detectMultiScale(gray_img, 1.06, 3)

                for i in cos:
                    x1, y1, x2, y2 = round(i[0] * 0.901038160701588), round(i[1] * 0.9086847599164927), \
                                     round((i[2] + i[0]) * 1.05864893845164), round((i[3] + i[1]) * 1.0579661016949153)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    face = cv2.resize(gray_img[y1:y2, x1:x2], (n, n))
                    face = np.expand_dims(face, axis=-1)
                    result = predict_image_label(face)
                    cv2.putText(original_img, result, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    total_frame += 1
                    if result == "attentive":
                        attentive_count += 1

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                cv2.imshow('Image two', original_img)
                if flag:
                    out.write(original_img)
            else:
                break

        # de-allocate any memory usage
        cv2.destroyAllWindows()


        # saving result
        info = open(f"./output/{dir_format}/attentive_info.txt", "a")
        info.write(f"\nNo of Attentive Frames - {attentive_count}")
        info.write(f"\nNo of Not Attentive  Frames = {total_frame-attentive_count}")
        info.write(f"\nAttentive = {attentive_count/total_frame > 0.65}")

    except Exception as e:
        messagebox.showerror("error", f"Please try again\n {e}")


def model_implementation_interface(root):
    """
    This function provide the first interface for model building phase.
    :param root: instance of main window
    """
    try:
        root.withdraw()
        top = ThemedTk(theme="aqua")
        top.title("Model Building")
        top.geometry("600x440")

        # label at the top of person database name
        my_label = Label(top, text="Model Implementation", font=("Times New Roman", 18, "bold"))
        my_label.grid(row=0, column=1, padx=(0, 90), pady=(10, 20))

        def valid_model_dir(path):
            """"""
            if "/" in path and f"{path.rsplit('/', 1)[1]}.h5" in os.listdir(path):
                return True
            else:
                return False

        def path_of_dir():
            top.folder_name = filedialog.askdirectory(title="select dataset directory", initialdir="./")
            if valid_model_dir(top.folder_name):
                model_dir_path.delete(0, "end")
                model_dir_path.insert(0, top.folder_name)
            else:
                messagebox.showerror("error", "please select valid directory")

        # label and input field for path and button right to input field for browsing to data source folder
        model_path_label = Label(top, text="Model Directory Path:", font=("Times New Roman", 12, "bold"))
        model_path_label.grid(row=1, column=0, padx=(0, 10), pady=(15, 10))

        model_dir_path = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        model_dir_path.grid(row=1, column=1, padx=(13, 10), pady=(15, 10))

        path_btn = Button(top, text="->", command=path_of_dir)
        path_btn.grid(row=1, column=2, columnspan=2, pady=(10, 0), padx=(0, 0), ipady=1, ipadx=6)

        def show_object(click):
            if click == "WebCam":
                object_label.grid_remove()
                object_path.grid_remove()
                object_path_btn.grid_remove()
            else:
                object_label.grid(row=7, column=0, padx=(0, 10), pady=(15, 10))
                object_path.grid(row=7, column=1, padx=(13, 10), pady=(15, 10))
                object_path_btn.grid(row=7, column=2, columnspan=2, pady=(10, 0), padx=(0, 0), ipady=1, ipadx=6)

        # label and option menu for object
        options = ['Video', 'WebCam']

        object_clicked = StringVar(top)
        object_clicked.set("Video")

        object_menu_label = Label(top, text="Object :", font=("Times New Roman", 12, "bold"))
        object_menu_label.grid(row=2, column=0, rowspan=2, padx=(10, 10), pady=(15, 15))

        object_menu = OptionMenu(top, object_clicked, *options, command=show_object)
        object_menu.config(width=40)
        object_menu.grid(row=2, column=1, rowspan=2, padx=10, pady=(15, 15))

        # label and option menu for Anonymous Implementation
        options = ['Yes', 'No']

        anon_clicked = StringVar(top)
        anon_clicked.set("Yes")

        anon_menu_label = Label(top, text="Anonymous Implementation :", font=("Times New Roman", 12, "bold"))
        anon_menu_label.grid(row=4, column=0, rowspan=2, padx=(10, 10), pady=(15, 15))

        anon_menu = OptionMenu(top, anon_clicked, *options)
        anon_menu.config(width=40)
        anon_menu.grid(row=4, column=1, rowspan=2, padx=10, pady=(15, 15))


        def path_of_object():
            top.file_path = filedialog.askopenfilename(initialdir="./", title="select a image",
                                                       filetypes=(("choose video", "*.mp4"),))
            object_path.delete(0, "end")
            object_path.insert(0, top.file_path)

        # label and input field for path and button right to input field for selecting object
        object_label = Label(top, text="Object Path:", font=("Times New Roman", 12, "bold"))
        object_label.grid(row=7, column=0, padx=(0, 10), pady=(15, 10))

        object_path = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        object_path.grid(row=7, column=1, padx=(13, 10), pady=(15, 10))

        object_path_btn = Button(top, text="->", command=path_of_object)
        object_path_btn.grid(row=7, column=2, columnspan=2, pady=(10, 0), padx=(0, 0), ipady=1, ipadx=6)

        # label and input field for no of conv2d and maxpool2d layer
        options = ['Yes', 'No']

        store_clicked = StringVar(top)
        store_clicked.set("No")

        store_label = Label(top, text="Save :", font=("Times New Roman", 12, "bold"))
        store_label.grid(row=9, column=0, rowspan=2, padx=(10, 10), pady=(15, 15))

        store = OptionMenu(top, store_clicked, *options)
        store.config(width=40)
        store.grid(row=9, column=1, rowspan=2, padx=10, pady=(15, 15))

        def implement():
            """ This function used implement model on the images or videos """

            # model_dir_path whether model dir exist and directory contain model
            if (os.path.exists(model_dir_path.get()) and valid_model_dir(model_dir_path.get())) \
                    or not model_dir_path.get():
                if anon_clicked.get() == "No":
                    if object_clicked.get() == "WebCam":
                        model_path = ''
                        if model_dir_path.get():
                            model_path = f"{model_dir_path.get()}/{(model_dir_path.get()).rsplit('/', 1)[1]}.h5"
                        on_video("", model_path, model_dir_path.get(), store_clicked.get())
                    else:
                        if os.path.exists(object_path.get()):
                            if (object_path.get()).rsplit('.', 1)[1] == "mp4":
                                model_path = ''
                                if model_dir_path.get():
                                    model_path = f"{model_dir_path.get()}/{(model_dir_path.get()).rsplit('/', 1)[1]}.h5"
                                on_video(object_path.get(), model_path, model_dir_path.get(), store_clicked.get())
                        else:
                            messagebox.showerror("error", "please select valid object path")
                else:
                    if object_clicked.get() == "WebCam":
                        model_path = ''
                        if model_dir_path.get():
                            model_path = f"{model_dir_path.get()}/{(model_dir_path.get()).rsplit('/', 1)[1]}.h5"
                        on_video_anon("", model_path, model_dir_path.get(), store_clicked.get())
                    else:
                        if os.path.exists(object_path.get()):
                            if (object_path.get()).rsplit('.', 1)[1] == "mp4":
                                model_path = ''
                                if model_dir_path.get():
                                    model_path = f"{model_dir_path.get()}/{(model_dir_path.get()).rsplit('/', 1)[1]}.h5"
                                on_video_anon(object_path.get(), model_path, model_dir_path.get(), store_clicked.get())
                        else:
                            messagebox.showerror("error", "please select valid object path")

            else:
                messagebox.showerror("error", "please select valid model directory")

        implement_btn = Button(top, text="Test Model", font=("Times New Roman", 12, "bold"), bg="gray", command=implement)
        implement_btn.grid(row=21, column=0, pady=10, padx=20, ipadx=30)

        def hide_open2():
            root.deiconify()
            top.destroy()

        exit2_btn = Button(top, text="EXIT", font=("Times New Roman", 12, "bold"), bg="gray", command=hide_open2)
        exit2_btn.grid(row=21, column=1, columnspan=2, pady=25, padx=(60, 0), ipadx=50)

    except Exception as e:
        messagebox.showerror("Message", "Error in the data processing interface")
