import os
import numpy as np
import face_recognition
import cv2
from ttkthemes import ThemedTk
from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import messagebox
from tkinter import filedialog
import math

SUB_DIRS = ["attentive", "not_attentive"]
SETS = ['train_set', 'test_set']
IMG_FORMAT = ['png', 'jpg', 'jpeg']
DETECTOR = cv2.CascadeClassifier("classifier/lbpcascade_frontalface_improved.xml")


def create_structure(dataset_name):
    """
    This function creates the directories which is used during the model building

    :param dataset_name: name of dataset we have to created
    """
    flag = True
    try:
        while flag:
            if not os.path.exists("./" + dataset_name):
                os.mkdir(f'./{dataset_name}')
                path = os.path.join('./', dataset_name)
                flag = False
            else:
                messagebox.showerror("Message", "Directory already exist, please give another name.")

        for set in SETS:
            for j in SUB_DIRS:
                os.makedirs(dataset_name + "/" + set + "/" + j)

    except:
        path = False


def check_availability(img_path):
    """
    This function return true if face found in image

    :param img_path: path of image
    :return: True if face is found
    """
    # loading gray image
    gray_image = cv2.imread(img_path, 0)

    # check whether img give empty list or not
    flag = face_recognition.face_locations(gray_image)
    if flag:
        return True
    return False


def find_dimensions_not_attentive_imgs(y1, x2, y2, x1, shape):
    """
    This function used to increase co-ordinate of face in not attentive images properly

    :param y1, x2, y2, x1: co-ordinate of face in images
    :param shape: width and height of image
    :return: proper increased co-ordinates
    """
    if y1 - 0.20 * y1 > 0:
        y1 = int(y1 - 0.20 * y1)
    elif y1 - 0.1 * y1 > 0:
        y1 = int(y1 - 0.1 * y1)

    if x1 - 0.1 * x1 > 0:
        x1 = int(x1 - 0.1 * x1)

    if y2 + 0.1 * y2 < shape[0]:
        y2 = int(y2 + 0.1 * y2)

    if x2 + 0.1 * x2 < shape[1]:
        x2 = int(x2 + 0.1 * x2)

    return y1, x2, y2, x1


def find_dimensions_attentive_imgs(y1, x2, y2, x1, shape):
    """
    This function used to increase co-ordinate of face in attentive images properly

    :param y1, x2, y2, x1: co-ordinate of face in images
    :param shape: width and height of image
    :return: proper increased co-ordinates
    """
    if y1 - 0.20 * y1 > 0:
        y1 = int(y1 - 0.20 * y1)
    elif y1 - 0.15 * y1 > 0:
        y1 = int(y1 - 0.15 * y1)
    elif y1 - 0.1 * y1 > 0:
        y1 = int(y1 - 0.1 * y1)

    # manipulate x1
    if x1 - 0.1 * x1 > 0:
        x1 = int(x1 - 0.1 * x1)

    # manipulate y2
    if y2 + 0.1 * y2 < shape[0]:
        y2 = int(y2 + 0.1 * y2)

    # manipulate x2
    if x2 + 0.1 * x2 < shape[1]:
        x2 = int(x2 + 0.1 * x2)

    return y1, x2, y2, x1


def loading_data(source_path_name, dataset_path, attentive, not_attentive, image_count, train_rate, dimension,
                 next_instance, root):
    """
    This function used to show the loading of images with progress bar

    :param source_path_name: directory path containing the attentive and not attentive images
    :param dataset_path: directory where we will going to insert images
    :param attentive: list of name of valid attentive images
    :param not_attentive: list of name of valid not attentive images
    :param image_count: no of images we want to load for each class(attentive or not attentice)
    :param train_rate: no of images on which we want to train data
    :param dimension: dimension of images
    :param next_instance: instance of next window
    :param root: instance of main window
    :return:
    """

    # dictionary to store the four destination path
    dest_path = {}
    for s in SETS:
        for d in SUB_DIRS:
            dest_path[f"{s}_{d}"] = os.path.join(os.path.join(dataset_path, s), d)

    train_img_count = math.ceil(int(image_count) * float(train_rate[0]) * 0.1)
    test_img_count = image_count - train_img_count

    def loading_faces(source_image_set_path, dest_image_set_path, source_image_set):
        """
        This is function write data into destination directory.

        :param source_image_set_path: directory from where images are coming
        :param dest_image_set_path: directory we created to insert the valid images
        :param source_image_set: list of valid images
        """
        dimensions_of_img = find_dimensions_not_attentive_imgs
        if 'attentive' in dest_image_set_path:
            dimensions_of_img = find_dimensions_attentive_imgs
        for image_name in source_image_set:

            # loading gray image
            gray_image = cv2.imread(source_image_set_path + "/" + image_name, 0)

            # find co-ordinates of faces in images
            y1, x2, y2, x1 = dimensions_of_img(*face_recognition.face_locations(gray_image)[0], np.shape(gray_image))

            # crop image and resize to particular dimension
            crop_img = gray_image[y1:y2, x1:x2]
            resize_crop_img = cv2.resize(crop_img, (int(dimension[0:3]), int(dimension[0:3])))

            # load images from source to destination directory
            cv2.imwrite(dest_image_set_path + "/" + image_name, resize_crop_img)

    # building progress bar
    next_instance.destroy()
    progress = ThemedTk(theme="aqua")
    progress.title("Progress")

    info_label = Label(progress, text="Building of Training set is on progress", font=("Times New Roman", 12, "bold"))
    info_label.pack(pady=10)
    progress_bar = Progressbar(progress, orient=HORIZONTAL, length=220, mode='determinate')
    progress_bar.pack(pady=20)

    progress_bar['value'] = 0
    progress.update()

    # create the dataset structure contain the training and testing set
    create_structure(dataset_path)

    # training of attentive images
    loading_faces(source_path_name["attentive"], dest_path["train_set_attentive"], attentive[:train_img_count])

    progress_bar['value'] = 25
    progress.update()

    # training of not attentive images
    loading_faces(source_path_name["not_attentive"], dest_path["train_set_not_attentive"],
                  not_attentive[:train_img_count])

    progress_bar['value'] = 50
    info_label['text'] = 'Building of Testing set is on progress'
    progress.update()

    # testing of attentive images
    loading_faces(source_path_name["attentive"], dest_path["test_set_attentive"], attentive[-test_img_count:])

    progress_bar['value'] = 75
    progress.update()

    # testing of not attentive images
    loading_faces(source_path_name["not_attentive"], dest_path["test_set_not_attentive"],
                  not_attentive[-test_img_count:])

    progress_bar['value'] = 100
    progress.update()
    info_label['text'] = 'Data Processing is completed'
    progress.destroy()
    root.deiconify()

    info = open(f"{dataset_path}/dataset_info.txt", "a")
    info.write(f"source directory path - {source_path_name['attentive'].rsplit('//')[0]}")
    info.write('\n\n#########  dataset parameter ##########')
    info.write(f"\ndataset name - {dataset_path}")
    info.write(f"\nimage count - {image_count}")
    info.write(f"\ntrain rate - {train_rate}")
    info.write(f"\ndimension - {dimension}")

    info.close()

    messagebox.showinfo("info", "Data Processing is Completed")


def next_page_interface(source_path_name, dataset_path, attentive, not_attentive, root):
    """
    This function provide the first interface for data processing phase.

    :param source_path_name: directory path containing the attentive and not attentive images
    :param dataset_path: directory where we will going to insert images
    :param attentive: list of name of valid attentive images
    :param not_attentive: list of name of valid not attentive images
    :param root: instance of main window
    """
    try:
        next_instance = ThemedTk(theme="aqua")
        next_instance.title("Data Processing")
        next_instance.geometry("630x400")

        # label at the top of person database name
        my_label = Label(next_instance, text="Data Processing", font=("Times New Roman", 18, "bold"))
        my_label.grid(row=0, column=0, columnspan=2, padx=(35, 0), pady=(10, 30))

        # label and input field for image count
        image_count_label = Label(next_instance, text="Image Count For Processing:", font=("Times New Roman", 12, "bold"))
        image_count_label.grid(row=14, column=0, padx=(10, 10), pady=(10, 20))

        total_count = DoubleVar(next_instance)
        default_len = min(len(attentive), len(not_attentive)) // 2
        total_count.set(default_len)

        image_count = Scale(next_instance, from_=200, to=default_len, sliderlength=20, length=270, orient=HORIZONTAL,
                            variable=total_count)
        image_count.grid(row=14, column=1, padx=(0, 0), pady=(10, 20))

        # label and input field for training rate
        options = ['50%', '60%', '70%', '80%']

        training_rate_clicked = StringVar(next_instance)
        training_rate_clicked.set("50%")

        training_rate_label = Label(next_instance, text="Image count For Training :", font=("Times New Roman", 12, "bold"))
        training_rate_label.grid(row=16, column=0, padx=(10, 10), pady=(10, 0))

        training_rate = OptionMenu(next_instance, training_rate_clicked, *options)
        training_rate.config(width=40)
        training_rate.grid(row=16, column=1, padx=20, pady=(10, 0))

        # label and input field for dimension of images
        options = ['64 x 64', '100 x 100', '128 x 128']

        dimension_clicked = StringVar(next_instance)
        dimension_clicked.set("100 x 100")

        dimension_label = Label(next_instance, text="Dimension :", font=("Times New Roman", 12, "bold"))
        dimension_label.grid(row=18, column=0, padx=(30, 10), pady=(30, 30))

        dimension = OptionMenu(next_instance, dimension_clicked, *options)
        dimension.config(width=40)
        dimension.grid(row=18, column=1, padx=20, pady=(30, 30))

        def default_setting():
            """ This function changes total images count, training rate  and dimension of images """
            total_count.set(default_len)
            training_rate_clicked.set('50%')
            dimension_clicked.set('100 x 100')

        default_btn = Button(next_instance, text="Default Model", font=("Times New Roman", 12, "bold"), bg="gray",
                             command=default_setting)
        default_btn.grid(row=20, column=0, pady=10, padx=10, ipadx=30)

        # show and exit button
        process_btn = Button(next_instance, text="Process Data", font=("Times New Roman", 12, "bold"),
                             command=lambda: loading_data(source_path_name, dataset_path, attentive, not_attentive,
                                                          image_count.get(), training_rate_clicked.get(),
                                                          dimension_clicked.get(), next_instance, root))
        process_btn.grid(row=20, column=1, pady=10, padx=10, ipadx=30)

        def hide_open2():
            root.deiconify()
            next_instance.destroy()

        exit2_btn = Button(next_instance, text="EXIT", font=("Times New Roman", 12, "bold"), bg="gray",
                           command=hide_open2)
        exit2_btn.grid(row=22, column=0, columnspan=2, pady=10, padx=10, ipadx=60)

    except:
        messagebox.showerror("Message", "Error in the data processing interface")


def filter_dataset(source_path, dataset_path, progress_bar, info_label, progress, root):
    """
    This is function used to obtain the valid images from the source path.

    :param source_path: directory path containing the attentive and not attentive images
    :param dataset_path: directory where we will going to insert images
    :param progress_bar: instance of progress bar
    :param info_label: label on progress bar provide information about currently filtering set
    :param progress: window contain progress_bar and info_label widget
    :param root: instance of main window
    """
    # dictionary to store two source path
    source_path_name = {}
    for d in SUB_DIRS:
        source_path_name[f"{d}"] = os.path.join(source_path, d)

    if not os.path.exists(source_path + "/" + SUB_DIRS[0]) and not os.path.exists(source_path + "/" + SUB_DIRS[1]):
        messagebox.showerror("Message", "Please check whether source directory, \n \
                             must contain 'attentive' and 'not_attentive' dataset")
    else:
        attentive = set()
        not_attentive = set()

        total_img = len(os.listdir(source_path + "/" + SUB_DIRS[0])) + len(os.listdir(source_path + "/" + SUB_DIRS[1]))
        i = 0

        # for attentive images in format particular format and availability of face
        for image in os.listdir(source_path + "/" + SUB_DIRS[0]):
            if len(image.split(".")) == 2 and image.split(".")[1] in IMG_FORMAT \
                    and check_availability(source_path + "/" + SUB_DIRS[0] + "/" + image):
                attentive.add(image)
            i += 1
            progress_bar['value'] = int((i / total_img) * 100)
            progress.update()

        info_label['text'] = 'Not Attentive set filtering is on progress'

        # for not attentive images
        for image in os.listdir(source_path + "/" + SUB_DIRS[1]):
            if len(image.split(".")) == 2 and image.split(".")[1] in IMG_FORMAT \
                    and check_availability(source_path + "/" + SUB_DIRS[1] + "/" + image):
                not_attentive.add(image)
            i += 1
            progress_bar['value'] = int((i / total_img) * 100)
            progress.update()

        info_label['text'] = 'Filtering is completed'
        progress.destroy()

        attentive, not_attentive = list(attentive), list(not_attentive)

        if len(attentive) > 200 and len(not_attentive) > 200:
            next_page_interface(source_path_name, dataset_path, attentive, not_attentive, root)
        else:
            messagebox.showerror("Message", "Valid Image Count Is Less Than 100")


def data_processing_interface(root):
    """
    This function provide the first interface for data processing phase.

    :param root: instance of main window
    """
    try:
        root.withdraw()
        top = ThemedTk(theme="aqua")
        top.title("Data Processing")
        top.geometry("500x300")

        # label at the top of person database name
        my_label = Label(top, text="Data Processing", font=("Times New Roman", 18, "bold"))
        my_label.grid(row=0, column=1, padx=(50, 50), pady=(10, 20))

        def path_of_image():
            """ open file explorer to select directory """
            top.folder_name = filedialog.askdirectory(title="select directory",
                                                      initialdir="C:/Users/Ayush sagore/JUPITER NOTEBOOK ML/CNN Model/"
                                                                 "test_dataset/")
            path_name.insert(0, top.folder_name)

        # label and input field for path and button right to input field for browsing to data source folder
        path_name_label = Label(top, text="Source Path :", font=("Times New Roman", 12, "bold"))
        path_name_label.grid(row=1, column=0, padx=(0, 10), pady=(20, 10))

        path_name = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        path_name.grid(row=1, column=1, padx=(13, 10), pady=(20, 10))

        path_btn = Button(top, text="->", command=path_of_image)
        path_btn.grid(row=1, column=2, columnspan=2, pady=(15, 0), padx=(0, 0), ipady=1, ipadx=6)

        # label and input field for folder name
        folder_name_label = Label(top, text="Folder Name :", font=("Times New Roman", 12, "bold"))
        folder_name_label.grid(row=2, column=0, padx=(10, 10), pady=(20, 10))

        folder_name = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        folder_name.grid(row=2, column=1, padx=20, pady=(20, 10))

        def hide_open2():
            root.deiconify()
            top.destroy()

        exit2_btn = Button(top, text="EXIT", font=("Times New Roman", 12, "bold"), bg="gray",  command=hide_open2)
        exit2_btn.grid(row=20, column=1, columnspan=2, pady=25, padx=(100, 0), ipadx=50)

        def check_given_data():
            """ This function check whether data given by user valid or not. """
            if os.path.exists(path_name.get()) and folder_name.get() != "":
                # if True:
                progress = ThemedTk(theme="aqua")
                progress.title("Progress")

                top.withdraw()

                info_label = Label(progress, text="Attentive set filtering is on progress", font=("Times New Roman", 12, "bold"))
                info_label.pack(pady=10)
                progress_bar = Progressbar(progress, orient=HORIZONTAL, length=220, mode='determinate')
                progress_bar.pack(pady=20)
                filter_dataset(path_name.get(), folder_name.get(), progress_bar, info_label, progress, root)

            else:
                messagebox.showerror("Message", "Please enter valid directory path\n \
                                                and folder name")

        # show and exit button
        next_page_btn = Button(top, text="Next Page", font=("Times New Roman", 12, "bold"), bg="gray",
                               command=check_given_data)  # next_page_interface(top, root)
        next_page_btn.grid(row=20, column=0, columnspan=2, pady=25, padx=(0, 200), ipadx=22)

    except Exception as e:
        messagebox.showerror("Message", "Error in the data processing interface")
