from rcnn import *

import torch
import os
import random
import numpy as np
import configparser
import csv
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from numpy.lib.stride_tricks import sliding_window_view


class NonNeuralMethod:
    def __init__(self):
        self.image_width = 1920
        self.image_height = 1080
        self.n_components = 100
        self.img_center = np.array([self.image_width/2, self.image_height/2])

        self.train_data_path = "D:/football-tracker/train/train"
        self.test_data_path = "D:/football-tracker/test/test"

        self.train_data_obj = []
        self.train_data_obj_train = []
        self.train_data_obj_val = []

        self.pca_model = PCA(n_components=self.n_components)
        self.svm_model = SVC(kernel="rbf", class_weight="balanced", probability=True) # Rbf er en kerne som håndterer ikke lineære systemer :)

        self.pca_ball_model = PCA(n_components=1)
        self.ball_eigen_template = []

    def train(self, ratio, ball_non_ball_ratio=3, ball_size=32):
        self._split_train_val(ratio)

        ball_radius = int(ball_size/2)

        image_paths = []
        centers_ball = []
        centers_non_ball = []

        for train_data_obj in self.train_data_obj_train:
            image_path = train_data_obj['path']
            print(image_path)


            center_ball = []
            center_non_ball = []
            if len(train_data_obj['boxes']) == 0:
                continue

            corners = np.array(train_data_obj['boxes'][0])
            center = np.array([(corners[0]+corners[2])/2, (corners[1]+corners[3])/2], dtype=np.int32)

            center_ball.append(center)

            for i in range(ball_non_ball_ratio):
                non_ball = np.array([np.random.randint(0, self.image_width - ball_size),
                                     np.random.randint(0, self.image_height - ball_size)])

                diff = np.linalg.norm(center - non_ball)

                if diff > ball_size:
                    center_non_ball.append(non_ball)

            centers_ball.append(center_ball)
            centers_non_ball.append(center_non_ball)
            image_paths.append(image_path)

        snippets = []
        snippets_only_ball = []
        classes = []

        print("Loading data...")

        for i in range(len(image_paths)):
            if len(centers_ball[i]) == 0 and len(centers_non_ball[i]) == 0:
                continue

            img = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
            for center_ball in centers_ball[i]:
                snip = img[center_ball[1]-ball_radius:center_ball[1]+ball_radius,
                           center_ball[0]-ball_radius:center_ball[0]+ball_radius]

                if snip.shape == (ball_size, ball_size):
                    # cv2.imshow("snip", snip)
                    # test = img.copy()
                    # cv2.circle(test, center_ball, 16, (255, 255, 255), thickness=1)
                    # cv2.imshow("test", test)
                    # cv2.waitKey(0)
                    #
                    # print(center_ball)
                    snippets.append(snip.flatten())
                    snippets_only_ball.append(snip.flatten())
                    classes.append(1)

            for center_non_ball in centers_non_ball[i]:
                snip = img[center_non_ball[1]-ball_radius:center_non_ball[1]+ball_radius,
                           center_non_ball[0]-ball_radius:center_non_ball[0]+ball_radius]

                if snip.shape == (ball_size, ball_size):
                    snippets.append(snip.flatten())
                    classes.append(0)

            print(f"\rProgress: {i}/{len(image_paths)}, total datapoints: {len(classes)}", end="", flush=True)

        print("\rDone loading data")

        print("Getting PCA")

        snippets_pca = self.pca_model.fit_transform(snippets)

        self.pca_ball_model.fit(snippets_only_ball)
        first_eigenvector = self.pca_ball_model.components_[0]
        eigen_template = first_eigenvector.reshape(32, 32)
        self.ball_eigen_template = cv2.normalize(eigen_template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        print("Done getting PCA, now doing the svm thing :)")

        self.svm_model.fit(snippets_pca, classes)

    def detect(self, path, threshold=0.95, patch_size=(32, 32), top_k=50):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image from path: {path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_patch, w_patch = patch_size

        heatmap = cv2.matchTemplate(gray, self.ball_eigen_template, cv2.TM_CCOEFF_NORMED)
        heatmap = cv2.normalize(heatmap,None,0,1.0, cv2.NORM_MINMAX)

        heatmap = 1 - heatmap

        working_map = heatmap.copy()
        map_h, map_w = working_map.shape[:2]

        snippets = []
        saved_coordinates = []

        # 2. Extract patches from most probable locations
        for _ in range(top_k):

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(working_map)

            if max_val <= 0:
                break

            peak_x, peak_y = max_loc

            center_x = peak_x + (w_patch // 2)
            center_y = peak_y + (h_patch // 2)

            snip = gray[
                   peak_y:peak_y + h_patch,
                   peak_x:peak_x + w_patch
                   ]

            if snip.shape != (h_patch, w_patch):
                snip = cv2.resize(snip, (w_patch, h_patch))

            snippets.append(
                snip.flatten().astype(np.float32)
            )

            saved_coordinates.append(
                (center_x, center_y)
            )

            # Non-maximum suppression on heatmap
            y_start = max(0, peak_y - h_patch // 2)
            y_end = min(map_h, peak_y + h_patch // 2)

            x_start = max(0, peak_x - w_patch // 2)
            x_end = min(map_w, peak_x + w_patch // 2)

            working_map[y_start:y_end, x_start:x_end] = 0

        # No candidate patches found
        if not snippets:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,), dtype=torch.float32)
            }

        # 3. Classify with PCA + SVM
        snippets_array = np.asarray(snippets)

        snippets_pca = self.pca_model.transform(
            snippets_array
        )

        probs = self.svm_model.predict_proba(
            snippets_pca
        )[:, 1]

        boxes = []
        labels = []
        scores = []

        # 4. Convert detections to TorchVision format
        for (cx, cy), prob in zip(saved_coordinates, probs):

            if prob < threshold:
                continue

            x1 = cx - w_patch / 2
            y1 = cy - h_patch / 2
            x2 = cx + w_patch / 2
            y2 = cy + h_patch / 2

            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # Football class
            scores.append(float(prob))

        if len(boxes) == 0:
            return [{
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,), dtype=torch.float32)
            }]

        return [{
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32)
        }]

    def save_model(self, name):
        joblib.dump(self.pca_model, f"{name}_pca.pkl")
        joblib.dump(self.pca_ball_model, f"{name}_pca_ball.pkl")
        joblib.dump(self.svm_model, f"{name}_svm.pkl")

    def load_model(self, name):
        self.pca_model = joblib.load(f"{name}_pca.pkl")
        self.pca_ball_model = joblib.load(f"{name}_pca_ball.pkl")
        self.svm_model = joblib.load(f"{name}_svm.pkl")

        first_eigenvector = self.pca_ball_model.components_[0]
        eigen_template = first_eigenvector.reshape(32, 32)
        self.ball_eigen_template = cv2.normalize(eigen_template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _read_train_folder(self, folder):
        full_path = self.train_data_path + "/" + folder + "/gameinfo.ini"

        game_info = configparser.ConfigParser()
        game_info.read(full_path)

        num_tracklets = game_info.getint('Sequence', 'num_tracklets')

        ball_ids = []

        for i in range(1, num_tracklets + 1):
            tracklet_value = game_info.get('Sequence', f'trackletID_{i}')

            if "ball" in tracklet_value:
                ball_ids.append(i)

        img_folder_dir = self.train_data_path + "/" + folder + "/img1"
        tracking_data_dir = self.train_data_path + "/" + folder + "/gt/gt.txt"

        image_ids = os.listdir(img_folder_dir)
        raw_tracking_data = {}

        for ball_id in ball_ids:
            raw_tracking_data = raw_tracking_data | extract_tracking_data(tracking_data_dir, ball_id)

        training_data_from_folder = []

        for image_id in image_ids:
            image_no = int(image_id.split('.')[0])
            if image_no in raw_tracking_data:
                data = {
                    'path': img_folder_dir + "/" + image_id,
                    'boxes': torch.tensor(
                        [raw_tracking_data[image_no]],
                        dtype=torch.float32
                    ),
                    "labels": torch.tensor(
                        [1],
                        dtype=torch.int64,
                    )
                }
            else:
                data = {
                    'path': img_folder_dir + "/" + image_id,
                    "boxes": torch.empty(
                        (0, 4),
                        dtype=torch.float32
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.int64
                    )
                }
            training_data_from_folder.append(data)

        return training_data_from_folder

    def _split_train_val(self, ratio):
        for folder in os.listdir(self.train_data_path):
            self.train_data_obj.extend(self._read_train_folder(folder))

        random.shuffle(self.train_data_obj)

        split = int(len(self.train_data_obj) * ratio)

        self.train_data_obj_train = self.train_data_obj[:split]
        self.train_data_obj_val = self.train_data_obj[split:]


def extract_tracking_data(file_path, target_tracking_id):
    tracking_dict = {}

    with open(file_path, mode="r") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            frame_id = int(row[0])
            tracking_id = int(row[1])
            borders = [int(x) for x in row[2:6]]

            if tracking_id == target_tracking_id:
                tracking_dict[frame_id] = borders
                tracking_dict[frame_id][2] += tracking_dict[frame_id][0]
                tracking_dict[frame_id][3] += tracking_dict[frame_id][1]

    return tracking_dict