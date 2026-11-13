from bytetracker import FootballByteTracker
from detect_camera_change import DetectCameraChange
from rcnn_data_handler import RCNNDataHandler
from non_neural import NonNeuralMethod

import cv2
import os
import numpy as np

def nn_basted_method(train=False):
    tracker = FootballByteTracker(football_label=1)

    network = RCNNDataHandler()

    if train:
        network.epoch(4, 0.8, max_runs=10000)
        network.save_model("1epoch_updated")
        return

    network.load_model("1epoch_updated")

    folder_path = "D:/football-tracker/test/test/SNMOT-150/img1/"
    imgs = os.listdir(folder_path)

    for img in imgs:
        img_path = f"{folder_path}{img}"
        resp = network.detect(img_path)
        result = tracker.update(resp[0])

        test = cv2.imread(img_path)

        if result is not None:
            for box, tracker_id in zip(result.xyxy, result.tracker_id):
                pos1 = (int(box[0]), int(box[1]))
                pos2 = (int(box[2]), int(box[3]))

                cv2.rectangle(test, pos1, pos2, (255, 0, 0), 3)

        for box in resp[0]["boxes"].cpu().numpy():
            pos1 = (int(box[0]), int(box[1]))
            pos2 = (int(box[2]), int(box[3]))

            #cv2.rectangle(test, pos1, pos2, (255, 0, 255), 2)

        cv2.imshow("test", test)
        cv2.waitKey(30)

def non_nn_basted_method(train=False, file_name="5percent_data_used"):
    tracker = FootballByteTracker(football_label=1)
    non_network = NonNeuralMethod()

    if train:
        non_network.train(0.03)
        non_network.save_model(file_name)
        print("Done saving")
        return

    non_network.load_model(file_name)

    folder_path = "D:/football-tracker/test/test/SNMOT-150/img1/"
    imgs = os.listdir(folder_path)

    for img in imgs:
        img_path = f"{folder_path}{img}"
        resp = non_network.detect(img_path)
        result = tracker.update(resp[0])

        test = cv2.imread(img_path)

        if result is not None:
            for box, tracker_id in zip(result.xyxy, result.tracker_id):
                pos1 = (int(box[0]), int(box[1]))
                pos2 = (int(box[2]), int(box[3]))

                cv2.rectangle(test, pos1, pos2, (255, 0, 0), 3)

        for box in resp[0]["boxes"].cpu().numpy():
            pos1 = (int(box[0]), int(box[1]))
            pos2 = (int(box[2]), int(box[3]))

            # cv2.rectangle(test, pos1, pos2, (255, 0, 255), 2)

        cv2.imshow("test", test)
        cv2.waitKey(30)

if __name__ == '__main__':
    # task = ManualMethod()
    # task.load_video("")
    # task.run()
    # non_nn_basted_method(train=False, file_name="3percent_32_32_size_40_pc2")
    nn_basted_method(train=False)

