import numpy as np
import matplotlib.pyplot as plt
import cv2


def draw_red_square(image, center_position, square_width, key, count):
    if key == "merge":
        color = [0, 0, 255]
    if key == "split":
        color = [0, 255, 255]
    if key == "fn":
        color = [255, 0, 0]
    if key == "fp":
        color = [0, 255, 0]
    # Calculate the coordinates for the square
    x, y = center_position
    half_width = square_width // 2

    # Draw the red square border
    clip_y_min = np.clip(y - half_width + 1, 0, None)
    clip_x_min = np.clip(x - half_width, 0, None)
    clip_y_max = np.clip(y + half_width, None, 255)
    clip_x_max = np.clip(x + half_width + 1, None, 255)

    image[clip_y_min:clip_y_max, clip_x_min] = color  # Left border
    image[clip_y_min:clip_y_max, clip_x_max] = color  # Right border
    image[clip_y_min, clip_x_min:clip_x_max] = color  # Top border
    image[clip_y_max, clip_x_min:clip_x_max] = color  # Bottom border
    
    image = cv2.putText(
        image,
        str(count),
        (clip_x_max - 20, clip_y_min + 15),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=color,
        thickness=2,
    )

    return image
