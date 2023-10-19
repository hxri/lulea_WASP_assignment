from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda, Add, Dropout, BatchNormalization
import tensorflow as tf
from keras.metrics import MeanIoU
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import keras.backend as K
import glob
import cv2
from keras.utils import to_categorical
from collections import namedtuple
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__=="__main__": 

    #############################################################################
    # Custom model architecture
    #############################################################################

    # Input layers for RGB image and depth map
    input_rgb = Input(shape=(256, 256, 3))
    input_depth = Input(shape=(256, 256, 1))

    # Encoder for RGB image
    conv1_rgb = Conv2D(64, 3, activation='relu', padding='same')(input_rgb)
    d11 = BatchNormalization()(conv1_rgb)
    pool1_rgb = MaxPooling2D(pool_size=(2, 2))(d11)
    conv2_rgb = Conv2D(128, 3, activation='relu', padding='same')(pool1_rgb)
    d12 = BatchNormalization()(conv2_rgb)
    pool2_rgb = MaxPooling2D(pool_size=(2, 2))(d12)
    conv3_rgb = Conv2D(256, 3, activation='relu', padding='same')(pool2_rgb)
    d13 = BatchNormalization()(conv3_rgb)
    pool3_rgb = MaxPooling2D(pool_size=(2, 2))(d13)

    # Encoder for depth map
    conv1_depth = Conv2D(64, 3, activation='relu', padding='same')(input_depth)
    d21 = BatchNormalization()(conv1_depth)
    pool1_depth = MaxPooling2D(pool_size=(2, 2))(d21)
    conv2_depth = Conv2D(128, 3, activation='relu', padding='same')(pool1_depth)
    d22 = BatchNormalization()(conv2_depth)
    pool2_depth = MaxPooling2D(pool_size=(2, 2))(d22)
    conv3_depth = Conv2D(256, 3, activation='relu', padding='same')(pool2_depth)
    d23 = BatchNormalization()(conv3_depth)
    pool3_depth = MaxPooling2D(pool_size=(2, 2))(d23)

    # Combine encoder outputs
    lambda_layer = Lambda(lambda x: x[0] * x[1])([pool3_rgb, pool3_depth])

    # Bottleneck layer
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(lambda_layer)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv4)
    concat1 = Concatenate()([up1, conv3_rgb, conv3_depth])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(concat1)
    d31 = BatchNormalization()(conv5)
    up2 = UpSampling2D(size=(2, 2))(d31)
    concat2 = Concatenate()([up2, conv2_rgb, conv2_depth])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(concat2)
    d32 = BatchNormalization()(conv6)
    up3 = UpSampling2D(size=(2, 2))(d32)
    concat3 = Concatenate()([up3, conv1_rgb, conv1_depth])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(concat3)
    d33 = BatchNormalization()(conv7)


    # Output layer
    output = Conv2D(35, 1, activation='softmax')(d33)  # 19 classes for Cityscapes

    # Create the model
    model = Model(inputs=[input_rgb, input_depth], outputs=output)

    def mean_iou(y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Calculate IoU for each class
        class_iou = []
        for i in range(1, num_classes):  # Assuming class 0 is the background
            true_class = tf.cast(tf.equal(y_true, i), dtype=tf.float32)
            pred_class = tf.cast(tf.equal(y_pred, i), dtype=tf.float32)
            intersection = tf.reduce_sum(true_class * pred_class, axis=[1, 2])
            union = tf.reduce_sum(true_class + pred_class, axis=[1, 2]) - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            class_iou.append(iou)
        
        # Compute the mean IoU for non-background classes
        mean_iou = tf.reduce_mean(class_iou)
        return mean_iou

    def f1_score(y_true, y_pred):
        smooth = 1e-5  # Small constant to avoid division by zero
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        intersection = K.sum(y_true * y_pred)
        f1 = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
        return f1

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Compile the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[mean_iou,
                                                                            f1_score], run_eagerly=True)

    model.summary()

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    #############################################################################
    # Data encoding and preprocessing
    #############################################################################

    # a label and all meta information
    Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).

        'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                        # images for training.
                        # For training, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )


    lbs = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , 34 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]

    Id2Color = { lb.id : np.asarray(lb.color) for lb in lbs }

    def encode(mask, mapping):
        distance = np.full([mask.shape[0], mask.shape[1]], 99999) 
        category = np.full([mask.shape[0], mask.shape[1]], None)   

        for id, color in mapping.items():
            dist = np.sqrt(np.linalg.norm(mask - color.reshape([1,1,-1]), axis=-1))
            condition = distance > dist
            distance = np.where(condition, dist, distance)
            category = np.where(condition, id, category)
        
        return category

    def decode(mask_enc, mapping):
        SIZE = (128,128,3)
        color_enc = np.zeros(SIZE)
        for i in range(SIZE[0]):
            for j in range(SIZE[1]):
                color_enc[i,j,:] = mapping[np.argmax(mask_enc[i,j])]
                color_enc = color_enc.astype('uint8')
        return color_enc


    # Define batch size and the number of classes
    batch_size = 8
    num_classes = 35  # Adjust this to match the number of classes in your dataset
    dlim = 0


    # Define the paths to your image, depth, and labels folders for training and validation
    train_image_folder = './original_data/leftImg8bit/train/*/*.png'
    train_depth_folder = './original_data/disparity/train/**/*.png'
    train_labels_folder = './original_data/gtFine/train/**/*_gtFine_color.png'

    val_image_folder = './original_data/leftImg8bit/val/*/*.png'
    val_depth_folder = './original_data/disparity/val/**/*.png'
    val_labels_folder = './original_data/gtFine/val/**/*_gtFine_color.png'

    test_image_folder = './original_data/leftImg8bit/test/*/*.png'
    test_depth_folder = './original_data/disparity/test/**/*.png'
    test_labels_folder = './original_data/gtFine/test/**/*_gtFine_color.png'

    # List all the file names in the training and validation folders
    train_image_files = sorted(glob.glob(train_image_folder, recursive=True))
    train_depth_files = sorted(glob.glob(train_depth_folder, recursive=True))
    train_labels_files = sorted(glob.glob(train_labels_folder, recursive=True))

    val_image_files = sorted(glob.glob(val_image_folder, recursive=True))
    val_depth_files = sorted(glob.glob(val_depth_folder, recursive=True))
    val_labels_files = sorted(glob.glob(val_labels_folder, recursive=True))

    test_image_files = sorted(glob.glob(test_image_folder, recursive=True))
    test_depth_files = sorted(glob.glob(test_depth_folder, recursive=True))
    test_labels_files = sorted(glob.glob(test_labels_folder, recursive=True))

    # Define a data generator function
    def data_generator(image_folder, depth_folder, labels_folder, batch_size, num_classes, dlim, target_size=(128, 128)):
        image_files = sorted(glob.glob(image_folder, recursive=True))
        depth_files = sorted(glob.glob(depth_folder, recursive=True))
        labels_files = sorted(glob.glob(labels_folder, recursive=True))
        
        while True:
            for offset in range(0, len(image_files), batch_size):
                batch_image = []
                batch_depth = []
                batch_labels = []
                batch_indices = list(range(offset, min(offset + batch_size, len(image_files))))
                for i in batch_indices:
                    # Load the depth and labels using NumPy
                    depth = cv2.imread(depth_files[i], 0)
                    labels = cv2.imread(labels_files[i])
                    
                    # Load the image using PIL or NumPy depending on the file format
                    
                    image = cv2.imread(image_files[i])
                    image = cv2.resize(image, target_size)

                    depth = cv2.resize(depth, target_size)
                    depth = np.expand_dims(depth, axis=-1)
                    labels = cv2.resize(labels, target_size)
                    labels = encode(labels, Id2Color)
                    # labels = np.expand_dims(labels, axis=-1)
                    
                    batch_image.append(image)
                    batch_depth.append(depth)
                    batch_labels.append(labels)
                # print(np.shape(batch_labels))
                x = [np.array(batch_image), np.array(batch_depth)]
                y = to_categorical(np.array(batch_labels), num_classes=num_classes)  # One-hot encode labels if needed
                yield x, y


    # Create data generators for training and validation
    train_generator = data_generator(train_image_folder, train_depth_folder, train_labels_folder, batch_size, num_classes, 800)
    val_generator = data_generator(val_image_folder, val_depth_folder, val_labels_folder, batch_size, num_classes, 200)
    test_generator = data_generator(test_image_folder, val_depth_folder, val_labels_folder, batch_size, num_classes, 100)


    # Calculate steps per epoch for training and validation
    train_steps_per_epoch = len(train_image_files) // batch_size
    val_steps_per_epoch = len(val_image_files) // batch_size
    test_steps_per_epoch = len(test_image_files) // batch_size


    #############################################################################
    # Training
    #############################################################################

    num_epochs = 25

    model_checkpoint = ModelCheckpoint('unet_model.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=num_epochs,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=[model_checkpoint, early_stopping]
    )

    # Save the trained model
    model.save('final_unet_model.h5')

    #############################################################################
    # Plot graphs
    #############################################################################

    # Plot training & validation loss values
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss') 
    plt.grid()

    # Plot training & validation accuracy values
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mean_iou'], label='Training Mean IoU')
    plt.plot(history.history['val_mean_iou'], label='Validation Mean IoU')
    plt.legend()
    plt.title('Mean IoU')
    plt.grid()

    # Plot training & validation mean_iou values
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='Training F1 Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
    plt.legend()
    plt.title('F1 Score')
    plt.grid()

    plt.show()

    #############################################################################
    # Load trained model
    #############################################################################

    model = tf.keras.models.load_model('final_unet_model.h5', compile=None)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Compile the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[mean_iou,
                                                                            f1_score], run_eagerly=True)


    #############################################################################
    # Testing
    #############################################################################

    example_index = 0

    # Load a batch of data from the validation generator
    val_batch = next(val_generator)

    # Separate the input data (images and depth) from the labels
    input_data, actual_labels = val_batch

    ss = decode(actual_labels[example_index], Id2Color)
    # print(np.unique(ss))

    # Predict with the model
    predicted_labels = model.predict(input_data)
    pp = decode(predicted_labels[example_index], Id2Color)

    # Display the actual label
    actual_label = ss
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.title("Actual Label")
    plt.imshow(actual_label)
    plt.colorbar()

    # Display the predicted label
    predicted_label = pp
    plt.subplot(1, 3, 2)
    plt.title("Predicted Label")
    plt.imshow(predicted_label)
    plt.colorbar()

    # Display the image
    actual_image = input_data[example_index]
    plt.subplot(1, 3, 3)
    plt.title("Actual Image")
    plt.imshow(actual_image[0])

    plt.show()