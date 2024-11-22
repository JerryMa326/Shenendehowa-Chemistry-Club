import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Step 1: Locate Registration Marks
def find_registration_marks(image_path, threshold=100):
    """
    Locate registration marks in the image using contour detection.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    registration_marks = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                registration_marks.append((cX, cY))

    return sorted(registration_marks, key=lambda x: x[0])


# Step 2: Calculate Chip ROIs
def calculate_chip_rois(registration_marks, chip_width, chip_height):
    """
    Calculate the ROIs for each chip based on registration marks.
    """
    rois = []
    for mark in registration_marks:
        x, y = mark
        roi = {'x': x, 'y': y - chip_height // 2, 'width': chip_width, 'height': chip_height}
        rois.append(roi)
    return rois


# Step 3: Extract RGB Data
def get_rgb_from_rois(image_path, rois):
    """
    Extract average RGB values from specified ROIs.
    """
    image = cv2.imread(image_path)
    rgb_values = []
    for roi in rois:
        x, y, width, height = roi['x'], roi['y'], roi['width'], roi['height']
        cropped = image[max(0, y):y+height, max(0, x):x+width]  # Handle edge cases
        if cropped.size == 0:
            rgb_values.append([0, 0, 0])  # Fallback for invalid ROIs
        else:
            avg_color = np.mean(cropped, axis=(0, 1))  # Average over width and height
            rgb_values.append(avg_color)
    return rgb_values


# Step 4: Normalize RGB Values
def normalize_rgb(rgb_sample, rgb_reference, rgb_ideal):
    """
    Normalize the RGB values of a sample based on a reference chip's RGB values.
    """
    correction_factor = np.array(rgb_ideal) / np.array(rgb_reference)
    normalized_rgb = np.array(rgb_sample) * correction_factor
    return np.clip(normalized_rgb, 0, 255)


# Step 5: Process the Image
def process_image(image_path, chip_width, chip_height, rgb_ideal):
    """
    Process the image to extract, normalize, and analyze RGB data.
    """
    registration_marks = find_registration_marks(image_path)
    rois = calculate_chip_rois(registration_marks, chip_width, chip_height)
    rgb_values = get_rgb_from_rois(image_path, rois)

    if len(rgb_values) == 0:
        raise ValueError("No RGB values could be extracted from the image. Check the registration marks.")

    rgb_reference = rgb_values[0]  # Assuming the first chip is the reference
    normalized_rgb_values = [normalize_rgb(rgb, rgb_reference, rgb_ideal) for rgb in rgb_values]
    return normalized_rgb_values


# Step 6: Train kNN Model
def train_knn_model(data, known_concentrations):
    """
    Train a kNN regression model to predict concentrations.
    """
    if len(data) != len(known_concentrations):
        print("Length mismatch detected. Adjusting lengths...")
        min_length = min(len(data), len(known_concentrations))
        data = data[:min_length]
        known_concentrations = known_concentrations[:min_length]

    df = pd.DataFrame(data, columns=['R', 'G', 'B'])
    df['Concentration'] = known_concentrations

    X = df[['R', 'G', 'B']].values
    y = df['Concentration'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("Model Performance:")
    print("  MAE:", mean_absolute_error(y_test, y_pred))
    print("  MSE:", mean_squared_error(y_test, y_pred))

    return knn, X_test, y_test


# Step 7: Visualize Predictions
def visualize_predictions(model, X_test, y_test):
    """
    Visualize the predictions made by the kNN model.
    """
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Concentration')
    plt.ylabel('Predicted Concentration')
    plt.title('kNN Model: Actual vs Predicted Concentrations')
    plt.legend()
    plt.show()


# Step 8: Predict New Samples
def predict_new_samples(model, new_rgb_values):
    """
    Predict concentrations for new samples based on their RGB values.
    """
    predictions = model.predict(new_rgb_values)
    for i, rgb in enumerate(new_rgb_values):
        print(f"RGB: {rgb}, Predicted Concentration: {predictions[i]:.2f}")


# Main Script
if __name__ == "__main__":
    # Define image path and chip parameters
    image_path = "/Users/jerryma/Documents/vsCode/Shenendehowa-Chemistry-Club/strip_image.png"
    chip_width = 50
    chip_height = 50
    rgb_ideal = [128, 128, 128]  # Ideal RGB for the reference chip
    known_concentrations = [0.0, 0.1, 0.2, 0.5, 1.0]  # Example values (repeat if needed)

    try:
        # Step 1-5: Process the image
        normalized_rgb_values = process_image(image_path, chip_width, chip_height, rgb_ideal)

        # Step 6: Train the kNN model
        knn_model, X_test, y_test = train_knn_model(normalized_rgb_values, known_concentrations)

        # Step 7: Visualize predictions
        visualize_predictions(knn_model, X_test, y_test)

        # Step 8: Predict new samples
        new_rgb_values = [[130, 120, 110], [150, 140, 130]]  # Example new RGB values
        predict_new_samples(knn_model, new_rgb_values)

    except Exception as e:
        print(f"Error: {e}")