from rcnn import *

import torch
import os
import random
import numpy as np
import configparser
import csv
import cv2


class RCNNDataHandler:
    def __init__(self):
        self.train_data_path = "D:/football-tracker/train/train"
        self.test_data_path = "D:/football-tracker/test/test"

        self.train_data_obj = []
        self.train_data_obj_train = []
        self.train_data_obj_val = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = create_model(1)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)


    def epoch(self, batch_size, ratio, max_runs = 200):
        self._split_train_val(ratio)

        image_paths = []
        targets = []

        for train_data_obj in self.train_data_obj_train:
            image_paths.append(train_data_obj['path'])
            targets.append({"boxes": train_data_obj['boxes'], "labels": train_data_obj['labels']})

        total_items = len(image_paths)
        batches = int(np.floor(total_items / batch_size))

        batches = min([batches, max_runs])

        ids = list(range(total_items))
        random.shuffle(ids)

        for i in range(batches):
            batch_ids = ids[i * batch_size : (i + 1) * batch_size]
            batch_imgs = []
            batch_targets = []
            for batch_id in batch_ids:
                img = cv2.imread(image_paths[batch_id])
                img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

                batch_imgs.append(img)
                batch_targets.append(targets[batch_id])

            loss = train_step(model=self.model,
                optimizer=self.optimizer,
                images=batch_imgs,
                targets=batch_targets,
                device=self.device,
            )
            print(f"Training batch run with loss {loss}. ({i}/{batches}, {i/batches*100}%)")

    def detect(self, path):
        img = cv2.imread(path)
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        resp = predict(self.model, [img], self.device, 0.15)
        return resp

    def save_model(self, name):
        torch.save(self.model.state_dict(), f"{name}.pth")

    def load_model(self, name):
        self.model.load_state_dict(torch.load(f"{name}.pth"))
        self.model.eval()

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

