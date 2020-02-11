import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import random
import os
print(os.listdir("./input"))


FAST_RUN = True
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

# Prepare Traning Data
filenames = os.listdir("./input/train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({'filename': filenames,'category': categories})


def train_test(fp32_or_16): 
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    earlystop = EarlyStopping(patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=2, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    batch_size=64

    # Traning Generator
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        "./input/train/", 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    #Validation Generator
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        "./input/train/", 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    #See how our generator
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df, 
        "./input/train/", 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )

    epochs=25 if FAST_RUN else 50
    history = model.fit_generator(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
    )

    # Save Model
    model.save_weights("model."+fp32_or_16+"h5")

    # Prepare Testing Data
    test_filenames = os.listdir("./input/test/")
    test_df = pd.DataFrame({'filename': test_filenames})
    nb_samples = test_df.shape[0]

    # Create Testing Generator
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        "./input/test/", 
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    # Predict
    predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

    #For categoral classication the prediction will come with probability of each category. So we will pick the category that have the highest probability with numpy average max
    test_df['category'] = np.argmax(predict, axis=-1)

    #We will convert the predict category back into our generator classes by using `train_generator.class_indices`. It is the classes that image generator map while converting data into computer vision
    label_map = dict((v,k) for k,v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)

    #From our prepare data part. We map data with `{1: 'dog', 0: 'cat'}`. Now we will map the result back to dog is 1 and cat is 0
    test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

    # Submission
    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['category']
    submission_df.drop(['filename', 'category'], axis=1, inplace=True)
    submission_df.to_csv('submission'+fp32_or_16+'.csv', index=False)


import os, sys

if not len(sys.argv) == 2:  # select the test
        print("error in argument: python lattice.py [TEST_NAME]")
        print("ex:python lattice.py fp32")
        exit()

if sys.argv[1]=="fp32":
    # #non-quanitized
    from mine_f32 import my_net_32
    model = my_net_32()
    train_test("fp32")
    print ("normal (fp32 is done)")

if sys.argv[1]=="fp16":
    #quantized
    from mine_f16 import my_net_16
    model = my_net_16()
    train_test("fp16")
    print("quanitzed is done!!!")
else:
    print("wrong test!!!")