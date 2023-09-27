import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from ultralytics import YOLO
import mlflow


# set to True to draw anotations
draw_anotations = False

#mlflow server --backend-store-uri sqlite:///mlflow.db
# mlflow.set_tracking_uri("sqlite:////home/jioy/PROJECTS/Tensorboard/mlflow.db")
mlflow.set_experiment("New annotation")


# load the model
model = YOLO('datasets/runs/pose/train18/weights/best.pt')

# path to the folder containing the images
image_folder = 'datasets/dataset/train/images'

# read all images and annotations in image_folder
images = os.listdir(image_folder)
annotations = [image.replace('tif', 'txt') for image in images]

# arrays of results
x_diff_array  = np.zeros([13, len(images)])
y_diff_array = np.zeros([13, len(images)])

# loop over all images
for ind in range(len(images)):

    # read image and annotation
    image = cv2.imread(os.path.join(image_folder, images[ind]))

    # run model
    results = model(image)

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        yolo_keypoints = result.keypoints  # Keypoints object for pose outputs

        # draw rectangle using opencv
        cv2.rectangle(image, (int(boxes.xyxy[0][0]), int(boxes.xyxy[0][1])), (int(boxes.xyxy[0][2]), int(boxes.xyxy[0][3])), (0, 255, 0), 3)
        
        # draw keypoints using opencv
        for i in range(len(yolo_keypoints.xyn[0])):
            cv2.circle(image, (int(yolo_keypoints.xy[0][i][0]), int(yolo_keypoints.xy[0][i][1])), 9, (0, 255, 0), -1)


    annotation_path = os.path.join(image_folder.replace(
        'images', 'labels'), annotations[ind])

    # Load YOLO annotations
    with open(annotation_path, 'r') as f:
        labels = f.readline().strip().split(' ')

    # Get the rectangle data and draw the rectangle
    class_id, x_center, y_center, width, height = map(float, labels[:5])
    image_height, image_width = image.shape[:2]

    x_min = int((x_center - width / 2) * image_width)
    x_max = int((x_center + width / 2) * image_width)
    y_min = int((y_center - height / 2) * image_height)
    y_max = int((y_center + height / 2) * image_height)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # Green rectangle

    # Get the keypoints data and draw the keypoints
    keypoints = labels[5:]
    keypoint_size = 9  # Diameter of the drawn keypoints
    keypoint_color = (255, 0, 0)  # Red keypoints

    for i in range(0, len(keypoints), 3):
        x, y, visibility = map(float, keypoints[i:i + 3])
        if visibility == 2:  # Visible keypoint
            x = int(x * image_width)
            y = int(y * image_height)
            cv2.circle(image, (x, y), keypoint_size, keypoint_color, -1)

            # calculate distance between predicted and ground truth keypoints
            x_diff = abs(x - yolo_keypoints.xy[0][int(i/3)][0])
            y_diff = abs(y - yolo_keypoints.xy[0][int(i/3)][1])

            x_diff_array[int(i/3), ind] = x_diff
            y_diff_array[int(i/3), ind] = y_diff


    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if draw_anotations:
        # Plot the image using Matplotlib
        plt.imshow(image_rgb)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()

keys = [
    't1_top', 't1_bottom', 't2_top', 't2_bottom', 
    'weld_position', 'weld_center', 'weld_left', 'weld_right', 'weld_bottom', 
    'concavity', 't1_top_new', 't2_top_left', 't2_top_right'
]

results = {}

for key in keys:
    for prefix, array_index in zip(['mean', 'std', 'max'], range(3)):
        x_key = f'{prefix}_error_{key}_x'
        y_key = f'{prefix}_error_{key}_y'
        e_key = f'{prefix}_error_{key}_e'
        
        results[x_key] = np.mean(x_diff_array[array_index, :])
        results[y_key] = np.mean(y_diff_array[array_index, :])
        results[e_key] = np.sqrt(np.mean(x_diff_array[array_index, :]**2 + y_diff_array[array_index, :]**2))


mlflow.log_metrics(results)