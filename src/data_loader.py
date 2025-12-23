from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE


def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    return train_data, val_data
