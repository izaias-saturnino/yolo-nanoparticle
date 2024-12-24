import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import sys
import time

image_types = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

data_file = "data.yaml"
model_paths = ["yolov8n_black_particles-obb_phase_1.pt", "yolov8n_black_particles-obb_phase_2.pt"]
confidence = 0.5
metric_sets = ["train", "val", "test"]
save_results = False
dir_path = "data-obb"
oriented_bb = True

def read_results(dir_path):
    results = {}
    for file in os.listdir(dir_path):
        result = []
        if file.endswith(".txt"):
            with open(os.path.join(dir_path, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    line = [float(x) for x in line]
                    if len(line) > 2:
                        line = line[1:]
                        result.append(line)
        file = os.path.splitext(file)[0]
        results[file] = result
    return results

def load_image_sizes(dir_path):
    image_sizes = {}
    for file in os.listdir(dir_path):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            image = cv2.imread(os.path.join(dir_path, file))
            # get the image name
            file = os.path.split(file)[1]
            # remove the extension
            file = os.path.splitext(file)[0]
            image_sizes[file] = (image.shape[1], image.shape[0])
    return image_sizes

def get_IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    low_x1 = x1 - w1 / 2
    high_x1 = x1 + w1 / 2
    low_y1 = y1 - h1 / 2
    high_y1 = y1 + h1 / 2
    low_x2 = x2 - w2 / 2
    high_x2 = x2 + w2 / 2
    low_y2 = y2 - h2 / 2
    high_y2 = y2 + h2 / 2
    low_x = max(low_x1, low_x2)
    high_x = min(high_x1, high_x2)
    low_y = max(low_y1, low_y2)
    high_y = min(high_y1, high_y2)
    intersection = max(0, high_x - low_x) * max(0, high_y - low_y)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union

def rotated_rect_area(obb):
    return obb.size[0] * obb.size[1]

def obb_intersection(obb1, obb2):
    # Get the four vertices of the rotated rectangles
    vertices1 = cv2.boxPoints(obb1)
    vertices2 = cv2.boxPoints(obb2)
    
    # Find the intersection polygon
    intersection_polygon = cv2.intersectConvexConvex(vertices1, vertices2)
    
    if intersection_polygon[0] > 0:
        # Calculate the area of the intersection polygon
        intersection_area = cv2.contourArea(intersection_polygon[1])
    else:
        intersection_area = 0.0
    
    return intersection_area

def to_el_list(obb, image_width, image_height):
    if len(obb) == 8:
        return obb
    elif len(obb) == 4:
        new_obb = []
        for el in obb:
            new_obb.append(el[0]/image_width)
            new_obb.append(el[1]/image_height)
        return new_obb
    else:
        raise ValueError("Invalid number of elements in the oriented bounding box.")

def get_IoU_oriented(box1, box2, image_width, image_height):

    box1 = to_el_list(box1, image_width, image_height)
    box2 = to_el_list(box2, image_width, image_height)

    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1 = box1
    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2 = box2

    center1 = ((x1_1 + x3_1) / 2, (y1_1 + y3_1) / 2)
    center2 = ((x1_2 + x3_2) / 2, (y1_2 + y3_2) / 2)
    size1 = ((x1_1 - x2_1) ** 2 + (y1_1 - y2_1) ** 2) ** 0.5, ((x1_1 - x4_1) ** 2 + (y1_1 - y4_1) ** 2) ** 0.5
    size2 = ((x1_2 - x2_2) ** 2 + (y1_2 - y2_2) ** 2) ** 0.5, ((x1_2 - x4_2) ** 2 + (y1_2 - y4_2) ** 2) ** 0.5
    angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1) * 180 / np.pi
    angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2) * 180 / np.pi

    obb1 = cv2.RotatedRect(center1, size1, angle1)
    obb2 = cv2.RotatedRect(center2, size2, angle2)

    cv2.rotatedRectangleIntersection(obb1, obb2)

    obb1_area = rotated_rect_area(obb1)
    obb2_area = rotated_rect_area(obb2)
    intersection_area = obb_intersection(obb1, obb2)

    union_area = obb1_area + obb2_area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area

def get_metric(model, metric_set, expected_results):
    images_path = os.path.join(dir_path, "images", metric_set)

    image_sizes = load_image_sizes(images_path)

    results = model.predict(images_path, conf=confidence, save=save_results, augment=True, show_labels=False, mode="val", split=metric_set)
    
    new_results = {}

    for i in range(len(results)):
        result = results[i]
        result_image = result.path
        # get the image name
        result_image = os.path.split(result_image)[1]
        # remove the extension
        result_image = os.path.splitext(result_image)[0]
        new_result = []
        for j in range(len(result)):
            if oriented_bb:
                result_box = result.obb.xyxyxyxy[j]
            else:
                result_box = result.boxes.xywhn[j]
            result_box = result_box.tolist()
            new_result.append(result_box)
        new_results[result_image] = new_result

    results = new_results

    values = []

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    instance_count = 0

    # pair the results with the expected results
    for key in results:
        if key in expected_results:
            result = results[key]
            expected_result = expected_results[key]
            for image_type in image_types:
                if os.path.exists(os.path.join(images_path, key + image_type)):
                    image_shape = cv2.imread(os.path.join(images_path, key + image_type)).shape
                    break
            image_width = image_shape[1]
            image_height = image_shape[0]
            # match the boxes with the highest IoU

            instance_count += len(expected_result)
            correspondence_dict = {}

            for expected_box in expected_result:
                max_iou = 0

                correspondence_expected = None
                for box2 in result:
                    box1 = expected_box
                    if oriented_bb:
                        iou = get_IoU_oriented(box1, box2, image_width, image_height)
                    else:
                        iou = get_IoU(box1, box2)
                    if iou > max_iou and box2 not in correspondence_dict.values():
                        max_iou = iou
                        correspondence_expected = box2

                if correspondence_expected != None:
                    true_positive_count += 1

                    correspondence_dict[tuple(expected_box)] = correspondence_expected
                    box = correspondence_expected

                    if oriented_bb:
                        box = to_el_list(box, image_width, image_height)
                        expected_box = to_el_list(expected_box, image_width, image_height)
                        
                        x1, y1, x2, y2, x3, y3, x4, y4 = box

                        d1_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
                        d2_squared = (x2 - x3) ** 2 + (y2 - y3) ** 2
                        d3_squared = (x3 - x4) ** 2 + (y3 - y4) ** 2
                        d4_squared = (x4 - x1) ** 2 + (y4 - y1) ** 2

                        max_d = max(d1_squared, d2_squared, d3_squared, d4_squared)

                        if d1_squared == max_d:
                            result_diameter = d1_squared
                        elif d2_squared == max_d:
                            result_diameter = d2_squared
                        elif d3_squared == max_d:
                            result_diameter = d3_squared
                        elif d4_squared == max_d:
                            result_diameter = d4_squared
                        
                        expected_x1, expected_y1, expected_x2, expected_y2, expected_x3, expected_y3, expected_x4, expected_y4 = expected_box

                        expected_d1_squared = (expected_x1 - expected_x2) ** 2 + (expected_y1 - expected_y2) ** 2
                        expected_d2_squared = (expected_x2 - expected_x3) ** 2 + (expected_y2 - expected_y3) ** 2
                        expected_d3_squared = (expected_x3 - expected_x4) ** 2 + (expected_y3 - expected_y4) ** 2
                        expected_d4_squared = (expected_x4 - expected_x1) ** 2 + (expected_y4 - expected_y1) ** 2

                        expected_max_d = max(expected_d1_squared, expected_d2_squared, expected_d3_squared, expected_d4_squared)

                        if expected_d1_squared == expected_max_d:
                            expected_original_diameter = expected_d1_squared
                        elif expected_d2_squared == expected_max_d:
                            expected_original_diameter = expected_d2_squared
                        elif expected_d3_squared == expected_max_d:
                            expected_original_diameter = expected_d3_squared
                        elif expected_d4_squared == expected_max_d:
                            expected_original_diameter = expected_d4_squared

                        expected_original_diameter = np.sqrt(expected_original_diameter)
                        result_diameter = np.sqrt(result_diameter)

                        # normalize the values
                        result_diameter = result_diameter / expected_original_diameter
                        expected_diameter = 1

                        values.append((result_diameter, expected_diameter))
                    else:
                        result_width = box[2]
                        result_height = box[3]
                        
                        result_original_width = result_width * image_width
                        result_original_height = result_height * image_height
                        expected_width = expected_box[2]
                        expected_height = expected_box[3]
                        expected_original_width = expected_width * image_width
                        expected_original_height = expected_height * image_height
                        if expected_original_width > expected_original_height:
                            expected_original_diameter = expected_original_width
                        else:
                            expected_original_diameter = expected_original_height
                        if result_original_width > result_original_height:
                            result_diameter = result_original_width
                        else:
                            result_diameter = result_original_height

                        # normalize the values
                        result_diameter = result_diameter / expected_original_diameter
                        expected_diameter = 1

                        values.append((result_diameter, expected_diameter))
                else:
                    false_positive_count += 1

            for box in expected_result:
                if tuple(box) not in correspondence_dict:
                    false_negative_count += 1

            for box in result:
                if box not in correspondence_dict.values():
                    false_positive_count += 1

    if true_positive_count == 0:
        # print("No instances found.")
        return float("inf")

    stddev_diameter = np.std(values)
    ci_99 = 2.576 * stddev_diameter / np.sqrt(true_positive_count)
    mean = np.mean(values)

    result_dict = {}
    result_dict["mean"] = mean
    result_dict["ci_99"] = ci_99
    result_dict["instance_count"] = instance_count
    result_dict["true_positive_count"] = true_positive_count
    result_dict["false_positive_count"] = false_positive_count
    result_dict["false_negative_count"] = false_negative_count
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)
    result_dict["precision"] = precision
    result_dict["recall"] = recall
    result_dict["f1_score"] = 2 * precision * recall / (precision + recall)

    return result_dict

for metric_set in metric_sets:
    print()
    print("Metric set:", metric_set)

    labels_path = os.path.join(dir_path, "labels", metric_set)
    expected_results = read_results(labels_path)

    for model_name in model_paths:
        print("-" * 50)
        print("Model:", model_name)
        model = YOLO(model_name)
        metric = get_metric(model, metric_set, expected_results)
        print("Metric:", metric)

        current_time = time.strftime("%Y%m%d-%H%M%S")

        try:
            shutil.move("runs", "runs_" + model_name + "_" + metric_set + "_" + current_time)
        except:
            pass
    print("-" * 50)