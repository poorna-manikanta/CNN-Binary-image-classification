from data_loader import load_data
from model_mobilenet import build_mobilenet_model
from config import EPOCHS, MODEL_PATH
from tensorflow.keras.callbacks import EarlyStopping


def train():
    train_data, val_data = load_data()
    model = build_mobilenet_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()
