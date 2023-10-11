from PIL import Image
import cv2
import numpy as np


def search_similar_parts(image, region, sensitivity, rotation, filter_color):
    tolerance_fraction = 0.15

    start_x = round(region.get('start_x'));
    start_y = round(region.get('start_y'))
    end_x = round(region.get('end_x'))
    end_y = round(region.get('end_y'))

    x1 = min(start_x, end_x)
    y1 = min(start_y, end_y)
    x2 = max(start_x, end_x)
    y2 = max(start_y, end_y)

    # selected_region = image.crop((x1, y1, x2, y2))
    selected_region = image[y1:y2, x1:x2]

    # prevent from searching for a single color
    selected_region_2d = selected_region.reshape(-1, 3)
    unique_colors = np.unique(selected_region_2d, axis=0)
    if unique_colors.shape[0] == 1:
        return []

    selected_array = np.array(selected_region)

    if not selected_array.any(): return []

    if selected_array.shape == (0, 0):  return []

    # Convert the selected region to OpenCV format
    selected_region = cv2.cvtColor(selected_array, cv2.COLOR_RGB2GRAY)
    selected_region_90 = cv2.rotate(selected_region, cv2.ROTATE_90_CLOCKWISE)

    # Create a list to store rotated versions of the selected region
    rotated_regions = [
        selected_region,
    ]

    if rotation:
        rotated_regions.append(cv2.flip(selected_region, 0))
        rotated_regions.append(cv2.flip(selected_region, 1))
        rotated_regions.append(selected_region_90)
        rotated_regions.append(cv2.flip(selected_region_90, 0))
        rotated_regions.append(cv2.flip(selected_region_90, 1))
        rotated_regions.append(cv2.rotate(selected_region, cv2.ROTATE_180))
        rotated_regions.append(cv2.rotate(selected_region, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
    threshold = sensitivity
    if threshold > 0.9 : threshold = 0.9

    if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR image to grayscale if it's not already
        search_image_color = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # The image is already in grayscale format
        search_image_color = image

    # Create a set to store unique match positions
    unique_match_positions = []


    for rotated_region in rotated_regions:
        result = cv2.matchTemplate(search_image_color, rotated_region, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        # Iterate through match positions and add them to the set
        for pt_rotated in zip(*loc[::-1]):
            if (unique_match_positions.__len__() > 200): break
            
            x, y = pt_rotated
            h, w = rotated_region.shape
            new_rect = (x, y, x + w, y + h)

            cv2.rectangle(search_image_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

            is_unique = True
            new_area = w * h

            for existing_x1, existing_y1, existing_x2, existing_y2 in unique_match_positions:
                existing_rect = (existing_x1, existing_y1, existing_x2, existing_y2)
                intersection_w = min(new_rect[2], existing_rect[2]) - max(new_rect[0], existing_rect[0])
                intersection_h = min(new_rect[3], existing_rect[3]) - max(new_rect[1], existing_rect[1])

                if intersection_w > 0 and intersection_h > 0:
                    intersection_area = intersection_w * intersection_h
                    existing_area = (existing_rect[2] - existing_rect[0]) * (existing_rect[3] - existing_rect[1])

                    if intersection_area / min(new_area, existing_area) > tolerance_fraction:
                        is_unique = False
                        break

            # for existing_x1, existing_y1, existing_x2, existing_y2 in unique_match_positions:
            #     existing_w = existing_x2 - existing_x1
            #     existing_h = existing_y2 - existing_y1
            #     if (abs(x - existing_x1) <= tolerance_fraction * existing_w and abs(y - existing_y1) <= tolerance_fraction * existing_h ):
            #         is_unique = False
            #         break

            # If the position is unique, add it to the set
            if is_unique:  unique_match_positions.append([x, y, (x + w), (y + h)])

    # cv2.rectangle(search_image_color, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)
    # search_image_color = Image.fromarray(search_image_color)
    # search_image_color.show()
            
    return unique_match_positions