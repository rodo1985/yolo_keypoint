import json
import cv2
from matplotlib import pyplot as plt
import os
import base64
import numpy as np
import shutil
from tqdm import tqdm

# set to True to draw anotations
draw_anotations = False

# paths
image_folder = 'datasets/13_points_dataset_2/val/images'
output_folder = image_folder.replace('13_points_dataset_2', 'dataset')

# read all images and annotations in image_folder
images = os.listdir(image_folder)
annotations = [image.replace('tif', 'json') for image in images]

# if folder exists delete them
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    shutil.rmtree(output_folder.replace('images', 'labels'))

# create folders
os.makedirs(output_folder)
os.makedirs(output_folder.replace('images', 'labels'))

# loop over all images
for ind in tqdm(range(len(images))):

    # read image and annotation
    # image = cv2.imread(os.path.join(image_folder, images[ind]))
    annotation = os.path.join(image_folder.replace(
        'images', 'annotations'), annotations[ind])
        
    # Open the JSON file
    with open(annotation) as f:
        # Load the JSON data
        json_data = json.load(f)

    # Decoding the base64 string
    img_bytes = base64.b64decode(json_data['imageData'])

    # Converting the bytes to a NumPy array
    nparr = np.frombuffer(img_bytes, np.uint8)

    # Reading the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # save image
    cv2.imwrite(os.path.join(output_folder, images[ind]), image)

    # list to store labels
    labels = []

    for data in json_data['shapes']:

        # save labels
        if data['label'] == 'bbox':

            # get coordinates
            x_min = int(data['points'][0][0])
            y_min = int(data['points'][0][1])
            x_max = int(data['points'][1][0])
            y_max = int(data['points'][1][1])

            # get normalized values of center, width and height
            x_min_norm = (x_min + (x_max - x_min)/2) / image.shape[1]
            y_min_norm = (y_min + (y_max - y_min)/2) / image.shape[0]
            x_max_norm = (x_max - x_min) / image.shape[1]
            y_max_norm = (y_max - y_min) / image.shape[0]

            # draw rectangle
            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

            # store normalized label
            label = '0 ' + str(x_min_norm) + ' ' + str(y_min_norm) + \
                ' ' + str(x_max_norm) + ' ' + str(y_max_norm)

        else:
            # get coordinates
            x = int(data['points'][0][0])
            y = int(data['points'][0][1])

            if x == 0 and y == 0:
                break

            # draw circle
            cv2.circle(image, (x, y), 9, (255, 0, 0), -1)

            # get normalized values between 0 to 1
            x_norm = x / image.shape[1]
            y_norm = y / image.shape[0]

            # add to label the normalized values and visibility
            label += ' ' + str(x_norm) + ' ' + str(y_norm) + ' 2'

    # append label to labels list
    labels.append(label)

    # save labels
    with open(os.path.join(output_folder.replace('images', 'labels'), images[ind].replace('tif', 'txt')), 'w') as f:
        for label in labels:
            f.write(label + '\n')

    if draw_anotations:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()
