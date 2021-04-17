import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
import os
import matplotlib.image as mpimg
import cv2
from sklearn import preprocessing
import Optical_Flow as flow
from clr_callback import CyclicLR
import padding

# TODO: CYLCIC LEARNING EXAMPLE
# MIN_LR = 1e-7
# MAX_LR = 1e-2
# BATCH_SIZE = 64
# STEP_SIZE = 8
# CLR_METHOD = "triangular2"
# clr = CyclicLR(
#             mode=CLR_METHOD,
#             base_lr=MIN_LR,
#             max_lr=MAX_LR,
#             step_size=STEP_SIZE * (train_images.shape[0] // BATCH_SIZE))
#
#     history = model.fit(train_images, train_labels,
#                             epochs=number_of_epochs,
#                             validation_data=(test_train_images, test_train_labels),
#                             callbacks=[clr])

# TODO: REGULIZER EXAMPLE
# layer = layers.Dense(
#         units=64,
#         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#         bias_regularizer=regularizers.l2(1e-4),
#         activity_regularizer=regularizers.l2(1e-5)
#     )

XSIZE = 128
YSIZE = 128
STANFORD_LEARNING_RATE = 0.01
MAX_HEIGHT = 965
MAX_WIDTH = 997

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def Standford40():
    with open('data/Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        # print(f'Train files ({len(train_files)}):\n\t{train_files}')
        # print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

    # TODO: DO A TRAIN VALIDATION SPLIT OF 10%(4000)
    # TODO: 10 images per category
    train_val_files = []
    train_val_labels = []
    strat_count = 0
    total_count = 0
    step_count = 0
    for file in train_files:
        if strat_count <= 9:
            train_val_files.append(file)
            train_val_labels.append(train_labels[strat_count + (step_count * 100)]) # get the label with the corresponding file
            strat_count = strat_count + 1
            total_count = total_count + 1
        elif total_count < 99:
            total_count = total_count + 1
        else:
            total_count = 0
            strat_count = 0
            step_count = step_count + 1

    with open('data/Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        # print(f'Test files ({len(test_files)}):\n\t{test_files}')
        # print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')

    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    # print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    # image_no = 234  # change this to a number between [0, 3999] and you can see a different training image
    # img = mpimg.imread(f'data/Stanford40/JPEGimages/{train_files[image_no]}')
    # imgplot = plt.imshow(img)
    # plt.show()
    # print(f'An image with the label - {train_labels[image_no]}')

    #Encodes the labels from strings to a number
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(action_categories)

    # print(label_encoder.transform(["cooking"])) gives the number from string
    # print(label_encoder.inverse_transform([5])) gives the string from the number

    #Load in the training files and encode the labels
    train_files_nd = []
    for x in train_files:
        img = cv2.imread("data/Stanford40/JPEGimages/"+x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resized_image = cv2.resize(img, (XSIZE, YSIZE), interpolation=cv2.INTER_NEAREST)
        # resized_image = padding.add_padding(img, MAX_HEIGHT, MAX_WIDTH)
        resized_image = padding.pad_and_resize(img, YSIZE, XSIZE)
        train_files_nd.append(resized_image)
    train_files = np.asarray(train_files_nd)
    train_labels = label_encoder.transform(train_labels)

    train_val_files_nd = []
    for x in train_val_files:
        img = cv2.imread("data/Stanford40/JPEGimages/"+x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = padding.pad_and_resize(img, YSIZE, XSIZE)
        train_val_files_nd.append(resized_image)

    valid_images = np.asarray(train_val_files_nd)
    valid_labels = label_encoder.transform(train_val_labels)

    #load in the testing files and encode the labels
    test_files_nd = []
    for x in test_files:
        img = cv2.imread("data/Stanford40/JPEGimages/" + x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resized_image = cv2.resize(img, (XSIZE, YSIZE), interpolation=cv2.INTER_NEAREST)
        # resized_image = padding.add_padding(img, MAX_HEIGHT, MAX_WIDTH)
        resized_image = padding.pad_and_resize(img, YSIZE, XSIZE)
        test_files_nd.append(resized_image)
    test_files = np.asarray(test_files_nd)
    test_labels = label_encoder.transform(test_labels)

    # # TODO: SPLIT THESE CORRECTLY
    # (train_files, train_labels) = train_files[400:], train_labels[400:]
    # (valid_images, valid_labels) = train_files[:400], train_labels[:400]

    return train_files, train_labels, valid_images, valid_labels, test_files, test_labels

def TV_HI():
    set_1_indices = [
        [2, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        [1, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 47, 48],
        [2, 3, 4, 11, 12, 15, 16, 17, 18, 20, 21, 27, 29, 30, 31, 32, 33, 34, 35, 36, 42, 44, 46, 49, 50],
        [1, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 29, 31, 35, 36, 38, 39, 40, 41, 42]]
    set_2_indices = [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 22, 23, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39],
                     [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 50],
                     [1, 5, 6, 7, 8, 9, 10, 13, 14, 19, 22, 23, 24, 25, 26, 28, 37, 38, 39, 40, 41, 43, 45, 47, 48],
                     [2, 3, 4, 5, 6, 15, 19, 20, 21, 25, 27, 28, 30, 32, 33, 34, 37, 43, 44, 45, 46, 47, 48, 49, 50]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [int(c) for c in range(len(classes)) for i in set_1_indices[c]]
    # print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
    # print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')   

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [int(c) for c in range(len(classes)) for i in set_2_indices[c]]
    # print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
    # print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')

    # video_no = 55
    # print(f'data/TV-HI/{set_2[video_no]}')
    # cap = cv2.VideoCapture(f'data/TV-HI/{set_2[video_no]}')
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #
    #     gray = cv2.cvtColor(frame, cv2.COLORMAP_HOT)
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # print(f'\n\nA video with the label - {set_2_label[video_no]}\n')
    # cap.release()
    # cv2.destroyAllWindows()
    return set_1, set_1_label, set_2, set_2_label

def get_history(model, valid_test_images, valid_test_labels, train_images, train_labels):
    history = model.fit(train_images,
                        train_labels,
                        batch_size=8,
                        epochs=10,
                        verbose=1,
                        validation_data=(valid_test_images, valid_test_labels))
    return history


def plot_training_loss(history):
    plt.plot(np.log(history.history["loss"]), label='training')
    plt.plot(np.log(history.history["val_loss"]), label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.ylim([0.4, 0.6])
    plt.title("Loss Model Stanford")
    plt.legend(loc='upper right')
    plt.show()


def stanford_model(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(YSIZE, XSIZE, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(40, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=STANFORD_LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if verbose == 1:
        model.summary()
    return model


def transfer_stanford_to_tvhi_model(st_model, verbose=0):
    if verbose == 1:
        st_model.summary()

    # Remove trainability from Stanford layers
    for layer in st_model.layers:
        layer.trainable = False

    # Add new output layer to the Stanford layer instead of the original output layer
    new_output_layer = layers.Dense(5, activation='softmax', name="newlayer")(st_model.layers[-2].output)
    tvhi_transfer_model = tf.keras.models.Model(st_model.input, new_output_layer)

    # Compile new model with 1/10 of the original learning rate
    opt = tf.keras.optimizers.Adam(lr=STANFORD_LEARNING_RATE/10)
    tvhi_transfer_model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    if verbose == 1:
        tvhi_transfer_model.summary()

    return tvhi_transfer_model


def optical_flow_model(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(YSIZE, XSIZE, 40)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(40, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=STANFORD_LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if verbose == 1:
        model.summary()
    return model


def main():
    standford_train_images, standford_train_labels, standford_valid_images, standford_valid_labels, standford_test_images, standford_test_labels = Standford40()

    tvhi_train_files, tvhi_train_labels, tvhi_test_files, tvhi_test_labels = TV_HI()
    ct = 1
    for val in tvhi_train_files:
        if val.find("highFive") != -1:
            ct = ct + 1

    print(ct)

    # np.save("train_flow_labels.npy", tvhi_train_labels)
    tvhi_train_labels = np.load("train_flow_labels.npy", allow_pickle=True)

    # np.save("test_flow_labels.npy", tvhi_test_labels)
    tvhi_test_labels = np.load("test_flow_labels.npy", allow_pickle=True)

    # train_stacked_videos = flow.get_video_flow_stacks(tvhi_train_files)
    # np.save("train_flow_stacks.npy", train_stacked_videos)
    tvhi_train_flow_tmp = np.load("train_flow_stacks.npy", allow_pickle=True)

    # test_stacked_videos = flow.get_video_flow_stacks(tvhi_test_files)
    # np.save("test_flow_stacks.npy", test_stacked_videos)
    tvhi_test_flow = np.load("test_flow_stacks.npy", allow_pickle=True)

    # train_middle_frames = flow.get_middle_frames(tvhi_train_files)
    # np.save("train_middle_frames.npy", train_middle_frames)
    train_middle_frames_tmp = np.load("train_middle_frames.npy", allow_pickle=True)

    # test_middle_frames = flow.get_middle_frames(tvhi_test_files)
    # np.save("test_middle_frames.npy", test_middle_frames)
    test_middle_frames = np.load("test_middle_frames.npy", allow_pickle=True)


    # TODO: SPLIT THESE CORRECTLY
    middle_frames_train, flow_stacks_train, flow_labels_train = train_middle_frames_tmp[15:], tvhi_train_flow_tmp[15:], np.array(tvhi_train_labels)[15:]
    middle_frames_valid, flow_stacks_valid, flow_labels_valid = train_middle_frames_tmp[:15], tvhi_train_flow_tmp[:15], np.array(tvhi_train_labels)[:15]

    # h, w = padding.get_max_size(train_middle_frames)
    # h, w = padding.get_max_size(test_middle_frames)
    # h, w = padding.get_max_size(standford_train_images, h, w)
    # h, w = padding.get_max_size(standford_valid_images, h, w)
    # h, w = padding.get_max_size(standford_test_images, h, w)

    # st_model = stanford_model()
    # st_model.summary()
    # history = get_history(st_model, standford_valid_images, standford_valid_labels, standford_train_images, standford_train_labels)
    # plot_training_loss(history)
    # st_model.save('Models/stanford_model')
    # st_model = tf.keras.models.load_model('Models/stanford_model')
    # tvhi_transfer_model = transfer_stanford_to_tvhi_model(st_model, verbose=1)

    opt_flow_model = optical_flow_model()
    opt_flow_model.summary()
    history = get_history(opt_flow_model, flow_stacks_valid, flow_labels_valid, flow_stacks_train, flow_labels_train)
    # plot_training_loss(history)
    # opt_flow_model.save('Models/opt_flow_model')
    # opt_flow_model = tf.keras.models.load_model('Models/opt_flow_model')

if __name__ == "__main__":
    main()