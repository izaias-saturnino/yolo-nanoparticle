import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import sys

image_types = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

data_file = "data.yaml"
local_model_name = "yolov8n_black_particles.pt"
base_model_name = "yolov8n.pt"

verbose = False

oriented_bb = True

# reuse the model or create a new one
new_model = True
skip_phase_1 = False
skip_phase_2 = False
save_phase_1_training_data = True
save_phase_2_training_data = True
save_phase_3_training_data = True

# model saving parameters
save_phase_1_model = True
save_phase_2_model = True
save_phase_3_model = True
save_best_model = True
save_last_model = True
# save the model after a certain number of epochs to avoid losing progress (0 to disable)
save_model_interval = 50
# save_model_interval = 1

# model training parameters
max_epochs_phase_1 = 1000
patience_phase_1 = 100
max_epochs_phase_2 = 2000
patience_phase_2 = 100
max_epochs_phase_3 = 2000
patience_phase_3 = 200

# model training test parameters
# max_epochs_phase_1 = 3
# patience_phase_1 = 1
# max_epochs_phase_2 = 3
# patience_phase_2 = 1

# confidence threshold
confidence = 0.5

# augmentation parameters (default parameters not included)
degrees = 1
if oriented_bb:
    degrees = 180
copy_paste = 0.5
mixup = 0.5
flipud = 0.5

if oriented_bb:
    data_file = data_file.replace(".yaml", "-obb.yaml")
    local_model_name = local_model_name.replace(".pt", "-obb.pt")
    base_model_name = base_model_name.replace(".pt", "-obb.pt")

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

dir_path = "data-obb"
labels_path = os.path.join(dir_path, "labels", "val")
expected_results = read_results(labels_path)

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

def get_IoU_oriented(box1, box2):
    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1 = box1

    angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1) * 180 / np.pi

    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2 = box2

    angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2) * 180 / np.pi

    obb1 = cv2.RotatedRect(((x1_1 + x3_1) / 2, (y1_1 + y3_1) / 2), ((x1_1 - x2_1) ** 2 + (y1_1 - y2_1) ** 2) ** 0.5, ((x1_1 - x4_1) ** 2 + (y1_1 - y4_1) ** 2) ** 0.5, angle1)
    obb2 = cv2.RotatedRect(((x1_2 + x3_2) / 2, (y1_2 + y3_2) / 2), ((x1_2 - x2_2) ** 2 + (y1_2 - y2_2) ** 2) ** 0.5, ((x1_2 - x4_2) ** 2 + (y1_2 - y4_2) ** 2) ** 0.5, angle2)

    intersection_area = obb1.intersection(obb2).area

    union_area = obb1.area + obb2.area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area

def get_metric(model):
    images_path = os.path.join(dir_path, "images", "val")

    image_sizes = load_image_sizes(images_path)

    results = model(images_path, conf=confidence, save=False, augment=True, show_labels=False)
    
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
            if result is None:
                continue
            if result.boxes is None:
                continue
            if oriented_bb:
                result_box = result.boxes.xyxyxyxy[j]
            else:
                result_box = result.boxes.xywhn[j]
            result_box = result_box.tolist()
            new_result.append(result_box)
        new_results[result_image] = new_result

    results = new_results

    instance_count = 0

    false_positive_count = 0
    values = []

    # pair the results with the expected results
    for key in results:
        if key in expected_results:
            result = results[key]
            expected_result = expected_results[key]
            # match the boxes with the highest IoU
            for box in result:
                max_iou = 0
                max_iou_index = -1
                for i in range(len(expected_result)):
                    if oriented_bb:
                        iou = get_IoU_oriented(box, expected_result[i])
                    else:
                        iou = get_IoU(box, expected_result[i])
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_index = i
                if max_iou_index != -1:
                    instance_count += 1
                    if oriented_bb:
                        x1 = box[0]
                        y1 = box[1]
                        x2 = box[2]
                        y2 = box[3]
                        x3 = box[4]
                        y3 = box[5]
                        x4 = box[6]
                        y4 = box[7]

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
                        
                        expected_x1 = expected_result[max_iou_index][0]
                        expected_y1 = expected_result[max_iou_index][1]
                        expected_x2 = expected_result[max_iou_index][2]
                        expected_y2 = expected_result[max_iou_index][3]
                        expected_x3 = expected_result[max_iou_index][4]
                        expected_y3 = expected_result[max_iou_index][5]
                        expected_x4 = expected_result[max_iou_index][6]
                        expected_y4 = expected_result[max_iou_index][7]

                        expected_d1_squared = (expected_x1 - expected_x2) ** 2 + (expected_y1 - expected_y2) ** 2
                        expected_d2_squared = (expected_x2 - expected_x3) ** 2 + (expected_y2 - expected_y3) ** 2
                        expected_d3_squared = (expected_x3 - expected_x4) ** 2 + (expected_y3 - expected_y4) ** 2
                        expected_d4_squared = (expected_x4 - expected_x1) ** 2 + (expected_y4 - expected_y1) ** 2

                        expected_max_d = max(d1_squared, d2_squared, d3_squared, d4_squared)

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
                        image_width = image_sizes[key][0]
                        image_height = image_sizes[key][1]
                        result_width = box[2]
                        result_height = box[3]
                        
                        result_original_width = result_width * image_width
                        result_original_height = result_height * image_height
                        expected_width = expected_result[max_iou_index][2]
                        expected_height = expected_result[max_iou_index][3]
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

    if instance_count == 0:
        # print("No instances found.")
        return float("inf")

    stddev_diameter = np.std(values)
    ci_99 = 2.576 * stddev_diameter / np.sqrt(instance_count)

    return ci_99

file_name, extension = os.path.splitext(local_model_name)

if skip_phase_2:
    local_model_name = local_model_name.replace(extension, "_phase_2" + extension)
    model = YOLO(local_model_name)
elif skip_phase_1:
    local_model_name = local_model_name.replace(extension, "_phase_1" + extension)
    model = YOLO(local_model_name)
else:
    if new_model:
        print("Creating a new model.")
        model = YOLO(base_model_name)
    else:
        try:
            model = YOLO(local_model_name)
        except FileNotFoundError:
            print("Model", local_model_name, "not found. Creating a new model.")
            model = YOLO(base_model_name)

    # phase 1 (no augmentation)
    model.train(data=data_file, epochs=max_epochs_phase_1, patience=patience_phase_1, save_period=save_model_interval, verbose=False, show_labels=False)

    if save_phase_1_model:
        model.save(file_name + "_phase_1" + extension)

    # rename runs folder to avoid overwriting
    shutil.move("runs", "runs_phase_1")

if save_phase_1_training_data:
    metric = get_metric(model)

    # save metadata to file
    with open("training_phase_1.txt", "w") as f:
        try:
            f.write(("model.epoch:" + str(model.epoch)))
            f.write("\n")
        except Exception:
            pass
        try:
            f.write(("model.metrics:" + str(model.metrics)))
            f.write("\n")
        except Exception:
            pass

# phase 2 (with augmentation)
if not skip_phase_2:
    model.train(data=data_file, epochs=max_epochs_phase_1, patience=patience_phase_1, save_period=save_model_interval, verbose=False, show_labels=False, degrees=degrees, copy_paste=copy_paste, mixup=mixup, flipud=flipud)

    if save_phase_2_model:
        model.save(file_name + "_phase_2" + extension)

        shutil.move("runs", "runs_phase_2")

    if save_phase_2_training_data:
        metric = get_metric(model)

        # save metadata to file
        with open("training_phase_2.txt", "w") as f:
            try:
                f.write(("model.epoch:" + str(model.epoch)))
                f.write("\n")
            except Exception:
                pass
            try:
                f.write(("model.metrics:" + str(model.metrics)))
                f.write("\n")
            except Exception:
                pass

# phase 3 (with shear)
# model.train(data=data_file, epochs=max_epochs_phase_3, patience=patience_phase_3, save_period=save_model_interval, verbose=False, degrees=degrees, copy_paste=copy_paste, mixup=mixup, flipud=flipud, shear=shear)

# if save_phase_3_model:
#     model.save(file_name + "_phase_3" + extension)

#     shutil.move("runs", "runs_phase_3")

# if save_phase_3_training_data:
#     metric = get_metric(model)

#     # save metadata to file
#     with open("training_phase_3.txt", "w") as f:
#         try:
#             f.write(("model.epoch:" + str(model.epoch)))
#             f.write("\n")
#         except Exception:
#             pass
#         try:
#             f.write(("model.metrics:" + str(model.metrics)))
#             f.write("\n")
#         except Exception:
#             pass

# old code

# best_metric = float("inf")
# print("Training phase 2.")
# print("Training for", max_epochs_phase_2, "epochs.")
# current_epoch = 0
# best_epoch = 0
# patience_counter = 0
# for i in range(0, max_epochs_phase_2, save_model_interval):
#     for j in range(0, save_model_interval):
#         current_epoch += 1
#         print("Epoch", current_epoch)

#         if verbose == False:
#             sys.stdout = open(os.devnull, 'w')
#         model.train(data=data_file, epochs=1, patience=0, save=False, verbose=False, degrees=degrees, copy_paste=copy_paste, mixup=mixup, flipud=flipud, shear=shear)

#         metric = get_metric(model)
#         if verbose == False:
#             sys.stdout = sys.__stdout__

#         # delete the runs folder
#         shutil.rmtree("runs")

#         patience_counter += 1

#         if metric < best_metric:
#             print("New best metric:", metric)
#             patience_counter = 0
#             best_metric = metric
#             if save_best_model:
#                 model.save(file_name + "_phase_2_best"  + extension)
#                 best_epoch = current_epoch
#                 with open("best_epoch.txt", "w") as f:
#                     f.write(str(best_metric))
#                     f.write("\n")
#                     f.write(str(best_epoch))
#                     f.write("\n")

#         if patience_counter >= patience_phase_2 and patience_phase_2 > 0:
#             break

#     if patience_counter >= patience_phase_2 and patience_phase_2 > 0:
#         break

#     if save_model_interval > 0:
#         model.save(file_name + "_phase_2_epoch_" + str(i+save_model_interval) + extension)

# if save_last_model:
#     model.save(file_name + "_phase_2_epoch_" + str(i) + extension)

# if save_phase_2_training_data:
#     metric = get_metric(model)

#     # save metadata to file
#     with open("training_phase_2.txt", "w") as f:
#         try:
#             f.write(("model.epoch:" + str(model.epoch)))
#             f.write("\n")
#         except Exception:
#             pass
#         try:
#             f.write(("model.metrics:" + str(model.metrics)))
#             f.write("\n")
#         except Exception:
#             pass
#         f.write(("best epoch:" + str(best_epoch)))
#         f.write("\n")