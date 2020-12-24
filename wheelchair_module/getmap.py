"""
This module get map information from image.
This module consists of 
    1. getting coordinate of polygon obstacle's vertext
        - input: image
        - output: coordinate
    2. making obstacles into shapely format
        - input: coordinate
        - output: shapely format polygon
    3. getting coordinate of waypoints
        - input: image
        - output: coordinate
"""
import numpy as np
import cv2
from shapely import geometry

class GetMap:
    def __init__(self, map_size, gap_set, gap_waypoint):
        self.map_size = map_size
        self.gap_set = gap_set
        self.gap_waypoint = gap_waypoint

    def get_waypoint(self, image_set, order_set=[]):
        new_image_set, order_set = self.rgb_constraint(image_set, self.gap_waypoint)
        waypoint_coordinate_set = self.get_coordinate(new_image_set, order_set=order_set)
        return waypoint_coordinate_set[0]

    def get_obstacle_set(self, image_set, order_set=[]):
        new_image_set, order_set = self.rgb_constraint(image_set, self.gap_set)
        obstacle_coordinate_set = self.get_coordinate(new_image_set, order_set=order_set)
        poly_set = self.get_poly(obstacle_coordinate_set)
        return poly_set

    def get_poly(self, coordinate_set):
        poly_set = []
        for coordinate in coordinate_set:
            poly = geometry.Polygon(coordinate)
            poly_set.append(poly)
        return poly_set

    def rgb_constraint(self, image_set, gap_set, criteria='r', 
                                    let=(255, 255, 255, 0)):
        """
        rgb_constraint is used for getting image editted RGB constraints.

        Parameters
        ----------
        lb : list
            ``lb`` is lower bound of RGB. It has three components which represent
            lower bound of red, blue, green respectively. Default is [0, 0, 0].
        ub : list
            ``ub`` is upper bound of RGB. It has three components which represent
            upper bound of red, blue, green respectively.
            Default is [255, 255, 255].
        let : tuple
            ``let`` is what will be RGBA value of image's pixels if the RGB of
            pixels don't satisfy RGB constraints. Default is (255, 255, 255, 0)
            which means white.
        kwargs : image
            ``kwargs`` is image loaded like Imgge.open('image.png').
            Keyward will be name of image saved with '_rgb_ediited'.

        Returns
        -------
        new_image : image
            ``new_image`` is the image editted according to RGB constraints.
        """
        new_image_set = []
        order_set = []
        k = 0
        for image in image_set:
            image_rgba = image.convert("RGBA")
            image_rgb = image_rgba.convert("RGB")
            data = image_rgb.getdata()
            newData = []
            order = []
            gap = gap_set[k]
            for item in data:
                if criteria == 'r':
                    if item[0] == 255:
                        if item[1] % gap[0] == 0 and item[2] % gap[1] == 0:
                            part1 = item[1] // gap[0]
                            part2 = item[2] // gap[1]
                            num = (255 // gap[1] + 1) * part1 + part2
                            if num in order:
                                newData.append(let)
                            else:
                                order.append(num)
                                newData.append(item)
                        else:
                            newData.append(let)                   
                    else:
                        newData.append(let)
                elif criteria == 'g':
                    if item[1] == 255:
                        if item[0] % gap[0] == 0 and item[2] % gap[1] == 0:
                            part1 = item[0] // gap[0]
                            part2 = item[2] // gap[1]
                            num = (255 // gap[1] + 1) * part1 + part2
                            if num in order:
                                newData.append(let)
                            else:    
                                order.append(num)
                                newData.append(item)     
                        else:
                            newData.append(let)                         
                    else:
                        newData.append(let)
                elif criteria == 'b':
                    if item[2] == 255:
                        if item[0] % gap[0] == 0 and item[1] % gap[1] == 0:
                            part1 = item[0] // gap[0]
                            part2 = item[1] // gap[1]
                            num = (255 // gap[1] + 1) * part1 + part2
                            if num in order:
                                newData.append(let)
                            else:    
                                order.append(num)
                                newData.append(item)     
                        else:
                            newData.append(let)                        
                    else:
                        newData.append(let)
            k += 1

            image_rgb.putdata(newData)
            new_image_set.append(image_rgb)
            order_set.append(order)
        return new_image_set, order_set

    def get_coordinate(self, image_set, order_set=[]):
        """
        get_binary_map is used for getting binary map and obstacle's coordinate.

        Parameters
        ----------
        img_name : str
            ``img_name`` is name of file what we want to get binary map
            and obstacle's cooridnate.

        Returns
        -------
        obstacle_coordinate : ndarray
            ``np.array(obstacle_coordinate)`` is ndarray of obstacle's coordinate.

        bw_img : image
            ``bw_img`` is the image converted from ``img`` to binary image.
        """
        obstacle_coordinate_set = []
        k = 0
        for image_temp in image_set:
            image = np.asarray(image_temp)
            column_length = image.shape[0]
            row_length = image.shape[1]

            row_size = self.map_size[0]
            column_size = self.map_size[1]

            d_row = row_size / row_length
            d_column = column_size / column_length
            obstacle = []
            indx = np.where(image[:][:] != 255)
            indx_x = indx[0][0:-1:2]
            indx_y = indx[1][0:-1:2]
            coordinate = [indx_y * d_row, (column_length - indx_x) * d_column]
            if order_set:
                obstacle_reorder = [None]*np.size(order_set[k])
                j = 0
                for i in order_set[k]:
                    obstacle_reorder[i] = [coordinate[0][j], coordinate[1][j]]
                    j += 1
                k += 1
                obstacle = obstacle_reorder

            obstacle_coordinate_set.append(obstacle)
        return obstacle_coordinate_set
