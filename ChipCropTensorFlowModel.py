import cv2
import numpy as np
import tensorflow as tf
import os

# Normalize RGB values to align with an ideal reference
def normalize_rgb(rgb, reference_rgb, ideal_rgb):
    normalized_rgb = (rgb / reference_rgb) * ideal_rgb
    return normalized_rgb

# Detect and crop chips from a strip
def detect_and_crop_chips(image_path, num_chips=5, chip_width=50, chip_height=50, threshold=100):
    """
    Detects and crops chips from a single image of a 5×1 strip.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and focus on potential chip regions
    chip_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if chip_width * 0.8 < w < chip_width * 1.2 and chip_height * 0.8 < h < chip_height * 1.2:
            chip_contours.append((x, y, w, h))

    # Sort contours by x-coordinate (left to right)
    chip_contours = sorted(chip_contours, key=lambda b: b[0])

    # Ensure exactly num_chips are detected
    if len(chip_contours) != num_chips:
        raise ValueError(f"Expected {num_chips} chips, but found {len(chip_contours)}.")

    # Crop and process each chip
    cropped_chips = []
    for (x, y, w, h) in chip_contours:
        # Expand the region slightly
        x1, y1 = max(0, x - 5), max(0, y - 5)
        x2, y2 = min(image.shape[1], x + w + 5), min(image.shape[0], y + h + 5)
        cropped_chip = image[y1:y2, x1:x2]

        # Resize to uniform dimensions
        resized_chip = cv2.resize(cropped_chip, (chip_width, chip_height))
        
        # Optional: Apply Gaussian blur to reduce noise
        blurred_chip = cv2.GaussianBlur(resized_chip, (5, 5), 0)
        
        cropped_chips.append(blurred_chip)

    return cropped_chips

# Process image with a strip of chips
def process_image_with_strip(image_path, chip_width, chip_height, rgb_ideal):
    # Detect and crop chips
    cropped_chips = detect_and_crop_chips(image_path, chip_width=chip_width, chip_height=chip_height)

    # Extract RGB values from each cropped chip
    rgb_values = []
    for chip in cropped_chips:
        avg_color = np.mean(chip, axis=(0, 1))  # Average RGB of the chip
        rgb_values.append(avg_color)

    # Normalize RGB values
    rgb_reference = rgb_values[0]  # Assume the first chip is the reference
    normalized_rgb_values = [normalize_rgb(rgb, rgb_reference, rgb_ideal) for rgb in rgb_values]

    return normalized_rgb_values

# Train a simple TensorFlow model
def train_model(data_path, input_shape, model_path):
    # Load and preprocess data
    images, labels = [], []
    for filename in os.listdir(data_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(data_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, input_shape[:2])
            images.append(img)
            labels.append(int(filename.split('_')[0]))  # Example label parsing from filename

    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)

    # Define and compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # For regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(images, labels, epochs=10, validation_split=0.2)

    # Save the trained model
    model.save(model_path)
    return model

# Predict with a trained model
def predict_with_model(image_path, model_path, input_shape):
    model = tf.keras.models.load_model(model_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape[:2])
    image = np.expand_dims(image / 255.0, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction[0][0]

# Example Usage
if __name__ == "__main__":
    # RGB normalization example
    rgb_ideal = [255, 255, 255]  # Assume ideal is pure white

    # Process a strip of chips
    try:
        normalized_rgb_values = process_image_with_strip(
            image_path="strip_image.png",
            chip_width=50,
            chip_height=50,
            rgb_ideal=rgb_ideal
        )
        print("Normalized RGB values:", normalized_rgb_values)
    except ValueError as e:
        print("Error:", e)

    # Train a model
    model_path = "chip_model.h5"
    input_shape = (50, 50, 3)
    train_model(data_path="training_data", input_shape=input_shape, model_path=model_path)

    # Predict using the trained model
    prediction = predict_with_model("new_chip_image.png", model_path, input_shape)
    print("Prediction:", prediction)
