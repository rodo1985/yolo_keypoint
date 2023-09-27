import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# load the model
model = YOLO('datasets/runs/pose/train18/weights/best.pt')

# path to the folder containing the images
image_folder = 'datasets/dataset/val/images'

# read all images and annotations in image_folder
images = os.listdir(image_folder)
annotations = [image.replace('tif', 'txt') for image in images]

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

    print(len(keypoints)/3)

    for i in range(0, len(keypoints), 3):
        x, y, visibility = map(float, keypoints[i:i + 3])
        if visibility == 2:  # Visible keypoint
            x = int(x * image_width)
            y = int(y * image_height)
            cv2.circle(image, (x, y), keypoint_size, keypoint_color, -1)

            # calculate distance between predicted and ground truth keypoints
            x_diff = abs(x - int(yolo_keypoints.xy[0][int(i/3)][0]))
            y_diff = abs(y - int(yolo_keypoints.xy[0][int(i/3)][1]))
            print(x_diff, y_diff)


    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image using Matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()
