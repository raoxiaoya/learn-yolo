from ultralytics import YOLO
from util import util
import cv2
from collections import defaultdict
import numpy as np
import threading


def detect():
    model = YOLO("yolo11n.pt")
    results = model(["imgs/20250210101802.png",
                    "imgs/20240731142129234.jpg", "imgs/cat.jpg"])

    for k, result in enumerate(results):
        # print(type(result)) # <class 'ultralytics.engine.results.Results'>
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        # result.show()  # display to screen

        filename = util.generate_random_string(15)
        result.save(filename="imgs_result/"+filename+".jpg")  # save to disk

        # im = result.plot()  # numpy.ndarray
        # cv2.imshow("test"+str(k), im)

    # cv2.waitKey(0)


def segment():
    model = YOLO("yolo11n-seg.pt")
    results = model(["imgs/20250210101802.png"])

    for k, result in enumerate(results):
        filename = util.generate_random_string(15)
        result.save(filename="imgs_result/"+filename+".jpg")


def classify():
    model = YOLO("yolo11m-cls.pt")
    results = model(["imgs/20250210101802.png"])

    for k, result in enumerate(results):
        print(result.probs)
        result.show()


def pose():
    model = YOLO("yolo11n-pose.pt")  # load an official model
    results = model("imgs/20250210101802.png")

    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)
        print(result.keypoints)
        result.show()


def track():
    model = YOLO("yolo11n.pt")

    results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
    # results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack
    for result in results:
        print(result.boxes.id)  # print track IDs
        result.show()


def track2():
    model = YOLO("yolo11n.pt")
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO11 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def track3():
    model = YOLO("yolo11n.pt")
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def track4():
    # Define model names and video sources
    MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # local video, 0 for webcam


    def run_tracker_in_thread(model_name, filename):
        """
        Run YOLO tracker in its own thread for concurrent processing.

        Args:
            model_name (str): The YOLO11 model object.
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
        """
        model = YOLO(model_name)
        results = model.track(filename, save=True, stream=True)
        for r in results:
            pass

    # Create and start tracker threads using a for loop
    tracker_threads = []
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # Wait for all tracker threads to finish
    for thread in tracker_threads:
        thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()


def obb():
    model = YOLO("yolo11n-obb.pt")
    results = model(["imgs/boats.jpg"])

    for result in results:
        # print(result.obb)

        # xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
        # xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
        # names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
        # confs = result.obb.conf  # confidence score of each box

        filename = util.generate_random_string(15)
        result.save(filename="imgs_result/"+filename+".jpg")

if __name__ == "__main__":
    obb()
