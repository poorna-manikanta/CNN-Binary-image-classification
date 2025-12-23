import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

MODEL_PATH = os.path.join(MODEL_DIR, "binary_mobilenet_model.h5")
