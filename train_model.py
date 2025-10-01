import tensorflow as tf
import pandas as pd
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

IMG_SIZE = (160,160)
BATCH = 16
EPOCHS = 20
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.pkl"

# CSV loader function
def load_dataset(data_dir, csv_file):
    df = pd.read_csv(csv_file)
    df['label_code'] = df['label'].astype('category').cat.codes
    label_mapping = dict(enumerate(df['label'].astype('category').cat.categories))

    file_paths = [os.path.join(data_dir, f) for f in df['filename']]
    labels = df['label_code'].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def process_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    dataset = dataset.map(process_image).shuffle(len(df)).batch(BATCH)
    return dataset, label_mapping

# Simple CNN model
def build_simple_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main training function
def main():
    train_dataset, train_labels = load_dataset('train', 'train_labels.csv')
    valid_dataset, valid_labels = load_dataset('valid', 'valid_labels.csv')

    num_classes = len(train_labels)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    model = build_simple_cnn(input_shape, num_classes)
    print(model.summary())

    # Save label mapping
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(train_labels, f)

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, callbacks=[checkpoint])
    print("Training complete. Best model saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
