from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection

def detectOnImage(input_path):
    model_path = "./models/yolo-tiny.h5"
    output_path = "./output/result.jpg"
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(model_path)
    detector.loadModel()
    detection = detector.detectObjectsFromImage(
        input_image=input_path,
        output_image_path=output_path)

    for item in detection:
        print(f"{item['name']}:{item['percentage_probability']}")

def detectOnVideo(input_path):
    model_path = "./models/yolo-tiny.h5"
    output_path = "./output/videoResult.mp4"
    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(model_path)
    detector.loadModel()
    detection = detector.detectObjectsFromVideo(
        input_file_path=input_path,
        output_file_path=output_path,
        frames_per_second=20,
        minimum_percentage_probability=30,
        log_progress=True)
    print(detection)