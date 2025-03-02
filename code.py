import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Function to load and preprocess image
def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies thresholding & morphological transformations.
    """
    img = Image.open(image_path)
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations for noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    kernelc = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    kernelo = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    dilated = cv2.dilate(threshold, kernel, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernelc, iterations=3)
    opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernelo, iterations=8)

    return img, opening

# Function to detect and classify shapes
def detect_shapes(img, processed_img):
    """
    Identifies and classifies geometric shapes in an image using contour detection.
    """
    contours, _ = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

            shape_name = ""
            if len(approx) == 3:
                shape_name = 'Triangle'
            elif len(approx) == 4:
                shape_name = 'Rectangle'
            elif len(approx) == 5:
                shape_name = 'Pentagon'
            elif len(approx) == 6:
                shape_name = 'Hexagon'
            else:
                shape_name = 'Circle'

            cv2.putText(img, shape_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img

# Main execution
if __name__ == "__main__":
    image_path = 'shapes3.jpeg'  # Update with your actual image file
    img, processed_img = preprocess_image(image_path)
    result_img = detect_shapes(img, processed_img)

    # Display the result
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Save the output
    cv2.imwrite("output_shapes_detected.jpg", result_img)
