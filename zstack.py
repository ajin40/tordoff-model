import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from pythonabm.backend import check_direct


def zstack(csv, z=25, single_slice_loc=0, single_slice=False, size=[325, 325, 325], background=(0, 0, 0), origin_bottom=True, image_quality=3250, cell_rad=0.5):
    df = pd.read_csv(csv, sep=',')

    locations_x = df['locations[0]']
    locations_y = df['locations[1]']
    locations_z = df['locations[2]']
    colors_0 = df['colors[0]']
    colors_1 = df['colors[1]']
    colors_2 = df['colors[2]']
    colors = np.vstack((colors_0, colors_1, colors_2))
    if single_slice:
        x_size = image_quality
        scale = x_size / size[0]
        y_size = math.ceil(scale * size[1])
        # create the agent space background image and apply background color
        image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
        image[:, :] = background
        indices = []
        # flatten 3d image in slice
        for index in range(len(locations_z)):
            if single_slice_loc < locations_z[index] < single_slice_loc+z:
                indices.append(index)
        zslice_x = np.array(locations_x[indices])
        zslice_y = np.array(locations_y[indices])
        colors_slice = colors[:, indices]

        for index in range(len(zslice_x)):
            # get xy coordinates, the axis lengths, and color of agent
            x, y = int(scale * zslice_x[index]), int(scale * zslice_y[index])
            major, minor = int(scale * cell_rad), int(scale * cell_rad)
            color = (int(colors_slice[2][index]), int(colors_slice[1][index]), int(colors_slice[0][index]))

            # draw the agent and a black outline to distinguish overlapping agents
            image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
            image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

        # if the origin should be bottom-left flip it, otherwise it will be top-left
        if origin_bottom:
            image = cv2.flip(image, 0)
        image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
        file_name = f"{'10122021_3d_800'}_zslice_{single_slice_loc}_thickness_{z}.png"
        cv2.imshow(file_name, image)

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        if not cv2.imwrite('Outputs/10122021_3d/' + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression]):
            raise Exception("Could not write image")
    else:
        num_slices = int(size[2] / z)

        # fig, ax = plt.subplots(5, 3, sharex=True, sharey=True, squeeze=True)
        for i in range(num_slices):
            # get the size of the array used for imaging in addition to the scaling factor
            x_size = image_quality
            scale = x_size / size[0]
            y_size = math.ceil(scale * size[1])
            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            image[:, :] = background
            indices = []
            # flatten 3d image in slice
            for index in range(len(locations_z)):
                if z*i< locations_z[index] < z*(i+1):
                    indices.append(index)
            #indices = z * i <= locations_z
            #indices += locations_z < z * (i + 1)
            # print(indices)
            zslice_x = np.array(locations_x[indices])
            zslice_y = np.array(locations_y[indices])
            colors_slice = colors[:, indices]

            for index in range(len(zslice_x)):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * zslice_x[index]), int(scale * zslice_y[index])
                major, minor = int(scale * cell_rad), int(scale * cell_rad)
                color = (int(colors_slice[2][index]), int(colors_slice[1][index]), int(colors_slice[0][index]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{'10122021_3d_800'}_image_{i}.png"
            cv2.imshow(file_name, image)

            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)
            if not cv2.imwrite('Outputs/10122021_3d/' + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression]):
                raise Exception("Could not write image")


if __name__ == "__main__":
    zstack('/Users/andrew/PycharmProjects/tordoff_model/Outputs/10122021_3d_80Hek_7000/10122021_3d_80Hek_7000_values/10122021_3d_80Hek_7000_values_600.csv',single_slice_loc=162, z=0.5, single_slice=True)
