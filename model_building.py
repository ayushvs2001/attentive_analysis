import os
from ttkthemes import ThemedTk
from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import messagebox
from tkinter import filedialog
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import shutil
import re


def train_model(path_name, model_name, no_of_layer, batch_sized, units, epoch, root):
    """
    Used to obtain trained model

    :param path_name: Processed data set
    :param model_name: name of model
    :param no_of_layer: no. of Conv2D and MaxPool Layer
    :param batch_sized: no. of images on which model trained per iteration
    :param units: no. of neurons to update weight per iteration
    :param epoch: no. of iteration
    :param root: instance of main window
    """
    f = open(f"{path_name}/dataset_info.txt", "r")
    data = f.read()
    f.close()
    i = data.index("dimension") + 12
    n = int(data[i: i + 3])
    batch_sized = int(batch_sized)
    no_of_layer = int(no_of_layer)

    # Preprocessing the Training set
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.1,
                                       rotation_range=30,
                                       width_shift_range=0.15,
                                       height_shift_range=0.15,
                                       zoom_range=0.1,
                                       horizontal_flip=False)

    training_set = train_datagen.flow_from_directory(f'{path_name}/train_set',
                                                     target_size=(n, n),
                                                     batch_size=batch_sized,  # no of images
                                                     class_mode='binary',
                                                     color_mode='grayscale')

    #  Preprocessing the Training set
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(f'{path_name}/test_set',
                                                target_size=(n, n),
                                                batch_size=batch_sized,  # no of images
                                                class_mode='binary',
                                                color_mode='grayscale')

    # Building the CNN
    # Initialising the CNN

    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[n, n, 1]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    while no_of_layer-1:
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        no_of_layer -= 1

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=int(units), activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    image_count = 0
    pattern = re.compile('(?<=image count - )\d*')
    for i, line in enumerate(open(f"{path_name}/dataset_info.txt")):
        for match in re.finditer(pattern, line):
            image_count = int(line[match.start():match.end()])
    steps_per_epoch = image_count // batch_sized
    epoch = int(epoch)

    # show progress bar
    root.withdraw()
    progress = ThemedTk(theme="aqua")
    progress.title("Progress")
    progress.geometry("500x200")
    info_label = Label(progress, text="Training is on progress", font=("Times New Roman", 12, "bold"))
    info_label.pack(pady=10)
    progress_bar = Progressbar(progress, orient=HORIZONTAL, length=220, mode='determinate')
    progress_bar.pack(pady=20)
    progress_bar['value'] = 0
    progress.update()

    for i in range(epoch-1):
        cnn.fit(x=training_set, validation_data=test_set, epochs=1, steps_per_epoch=steps_per_epoch)
        progress_bar['value'] = int(((i + 1) / epoch) * 100)
        progress.update()

    history = cnn.fit(x=training_set, validation_data=test_set, epochs=1, steps_per_epoch=steps_per_epoch)
    progress_bar['value'] = 100
    progress.update()

    # make directoory to story the model and model_info.txt
    os.mkdir(os.path.join('./', model_name))

    # save model
    cnn.save(f"{model_name}/{model_name}.h5")

    # copy dataset info file
    shutil.copyfile(f"{path_name}/dataset_info.txt", f"./{model_name}/model_info.txt")

    info = open(f"./{model_name}/model_info.txt", "a")
    info.write('\n\n#########  model summary ##########\n')
    cnn.summary(print_fn=lambda x: info.write(x + '\n'))

    a, b = cnn.evaluate(test_set)
    info.write(f"\n\nLoss - {a*100: .2f}%")
    info.write(f"\nAccuracy - {b*100: .2f}%")

    messagebox.showinfo("Success", f'Model is successfully trained.\nLoss - {round(a, 2)}, Acuuracy - {round(b, 2)}')
    root.deiconify()
    progress.destroy()


def model_building_interface(root):
    """
    This function provide the first interface for model building phase phase.

    :param root: instance of main window
    """
    try:
        root.withdraw()
        top = ThemedTk(theme="aqua")
        top.title("Model Building")
        top.geometry("600x670")

        # label at the top of person database name
        my_label = Label(top, text="Model Building", font=("Times New Roman", 18, "bold"))
        my_label.grid(row=0, column=1, padx=(0, 90), pady=(10, 20))

        def valid_dataset_dir(files):
            if "dataset_info.txt" in files:
                return True
            else:
                return False

        def path_of_dir():
            top.folder_name = filedialog.askdirectory(title="select dataset directory", initialdir="./")
            if top.folder_name != "" and valid_dataset_dir(os.listdir(top.folder_name)):
                path_name.insert(0, top.folder_name)
            else:
                messagebox.showerror("error", "please select valid directory")

        # label and input field for path and button right to input field for browsing to data source folder
        path_name_label = Label(top, text="Dataset Path :", font=("Times New Roman", 12, "bold"))
        path_name_label.grid(row=1, column=0, padx=(0, 10), pady=(15, 10))

        path_name = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        path_name.grid(row=1, column=1, padx=(13, 10), pady=(15, 10))

        path_btn = Button(top, text="->", command=path_of_dir)
        path_btn.grid(row=1, column=2, columnspan=2, pady=(10, 0), padx=(0, 0), ipady=1, ipadx=6)

        # label and input field for folder name
        model_name_label = Label(top, text="Model Name :", font=("Times New Roman", 12, "bold"))
        model_name_label.grid(row=2, column=0, padx=(10, 10), pady=(15, 10))

        model_name = Entry(top, width=25, font=("Times New Roman", 12, "bold"))
        model_name.grid(row=2, column=1, padx=20, pady=(15, 10))

        # label and input field for batch size
        options = ['20', '25', '32', '100']

        batch_sized_clicked = StringVar(top)
        batch_sized_clicked.set("")

        batch_sized_label = Label(top, text="Batch sized :", font=("Times New Roman", 12, "bold"))
        batch_sized_label.grid(row=3, column=0, padx=(10, 10), pady=(15, 15))

        batch_sized = OptionMenu(top, batch_sized_clicked, *options)
        batch_sized.config(width=40)
        batch_sized.grid(row=3, column=1, padx=10, pady=(15, 15))

        # label and input field for no of conv2d and maxpool2d layer
        options = ['2', '3', '4']

        layer_clicked = StringVar(top)
        layer_clicked.set("")

        layer_label = Label(top, text="No. of Conv2d\nMaxPool2d Layer :", font=("Times New Roman", 12, "bold"))
        layer_label.grid(row=5, column=0, rowspan=2, padx=(10, 10), pady=(15, 15))

        layer = OptionMenu(top, layer_clicked, *options)
        layer.config(width=40)
        layer.grid(row=5, column=1, rowspan=2, padx=10, pady=(15, 15))

        # label and input field for no of unit in dense layer
        options = ['500', '750', '1000', '1500']

        units_clicked = StringVar(top)
        units_clicked.set("")

        units_label = Label(top, text="No. of Unit\nin Dense Layer :", font=("Times New Roman", 12, "bold"))
        units_label.grid(row=8, column=0, rowspan=2, padx=(10, 10), pady=(15, 15))

        units = OptionMenu(top, units_clicked, *options)
        units.config(width=40)
        units.grid(row=8, column=1, rowspan=2, padx=10, pady=(15, 15))

        # label and input field for no of epoch
        options = ['20', '25', '50', '75']

        epoch_clicked = StringVar(top)
        epoch_clicked.set("")

        epoch_label = Label(top, text="No. of Epoch :", font=("Times New Roman", 12, "bold"))
        epoch_label.grid(row=11, column=0, padx=(10, 10), pady=(15, 15))

        epoch = OptionMenu(top, epoch_clicked, *options)
        epoch.config(width=40)
        epoch.grid(row=11, column=1, padx=10, pady=(15, 15))

        def default_setting():
            """ Used to set default parameter for model"""
            if path_name != "":
                # check whether dataset dir exist and the directory contain the dataset_info.txt file
                if os.path.exists(path_name.get()) and valid_dataset_dir(os.listdir(path_name.get())):
                    layer_clicked.set("3")
                    batch_sized_clicked.set('32')

                    units_clicked.set('750')
                    epoch_clicked.set('50')
                else:
                    messagebox.showerror("error", "please select valid dataset directory")
            else:
                messagebox.showerror("error", "please select dataset directory")

        default_btn = Button(top, text="Default Settings", font=("Times New Roman", 12, "bold"), bg="gray",
                             command=default_setting)
        default_btn.grid(row=20, column=0, pady=10, padx=(20, 0), ipadx=20)

        def compile_model():
            """ check the data given by user and call train function """
            if os.path.exists(path_name.get()) and valid_dataset_dir(os.listdir(path_name.get())) and model_name.get():
                default_setting()
                top.withdraw()
                train_model(path_name.get(), model_name.get(), layer_clicked.get(), batch_sized_clicked.get(),
                            units_clicked.get(), epoch_clicked.get(), root)
                top.destroy()

        build_btn = Button(top, text="Build Model", font=("Times New Roman", 12, "bold"),  bg="gray",
                           command=compile_model)
        build_btn.grid(row=20, column=1, pady=10, padx=(80, 0), ipadx=30)

        def hide_open2():
            root.deiconify()
            top.destroy()

        exit2_btn = Button(top, text="EXIT", font=("Times New Roman", 12, "bold"), bg="gray", command=hide_open2)
        exit2_btn.grid(row=21, column=0, columnspan=2, pady=25, ipadx=50)

    except Exception as e:
        messagebox.showerror("Message", "Error in the data processing interface")
