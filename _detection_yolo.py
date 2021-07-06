import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.78, allow_growth=True)
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from imageai.Detection import VideoObjectDetection
import os
# from matplotlib import pyplot as plt
import time
from cv2 import cv2
import json
# import socket
# import pafy

execution_path = os.getcwd()

fps = 25  # Please do not change

videoInputPath = os.path.join(execution_path, "video\\input")
videoOutputPath = os.path.join(execution_path, 'video\\output')
videos = os.listdir(videoInputPath)

modelName = 'sample-models/yolo.h5'
modelPath = os.path.join(modelName)


detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, modelPath))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=True, bus=True, truck=False, motorcycle=True)

forFrame_list = list()
forSecond_list = list()
forSeconds_list = list()
forMinute_list = list()


for video in videos:
    if os.path.isdir(os.path.join(videoInputPath, video)):
        continue
    videoName, ext = os.path.splitext(video)
    outputVideoName = videoName + '_detected_yolo'
    output_stats = os.path.join(videoOutputPath, videoName)

    with open(output_stats + '_for_second_yolo.json', 'w') as fn:
        # writess: str = time.strftime("%Y-%m-%d %H:%M:%S %z", time.gmtime()) + '\n\n'
        fn.write('')

    def forFrame(frame_number, output_array, output_count, returned_frame):
        # print(frame_number)
        # buffer_frame = "buffer/" + str(int(time.time() % (60 * fps))) + ".jpg"
        # cv2.imwrite(buffer_frame, returned_frame)
        # print("Output for each object : ", output_array)
        # print("Output count for unique objects : ", output_count)
        # print("------------END OF A FRAME --------------")
        # frame = cv2.cvtColor(returned_frame, cv2.COLOR_RGB2BGR)
        # frame = returned_frame
        # os.environ["frame_numer"] = frame
        cv2.imshow("frame", returned_frame)
        # cv2.imwrite((str(int(frame_number % 30)) + ".jpg"), frame)

        # os.path.join(time.time() % 3)
        if cv2.waitKey(1) == ord('q'):
            # capture.release()
            exit()


    def forSecond(second_number, output_arrays, count_arrays, average_count, returned_frame):
        print("SECOND: ", second_number)
        # print(average_count)
        forSecond_list = {
            'second': second_number,
            'objects': average_count
        }
        # print(forSecond_list)

        forSeconds_list.append(forSecond_list)

        # json_formatted_str = json.dumps(forSecond_list, indent=2)
        # with open(os.path.join(output_stats + '_for_second.json'), 'a') as fn:
        #     fn.write(json_formatted_str)

        json_formatted_str = json.dumps(forSeconds_list, indent=2)
        with open(os.path.join(output_stats + '_for_second_yolo.json'), 'w') as fn:
            fn.write(json_formatted_str)

        if cv2.waitKey(1) == ord('q'):
            exit()



    def forMinute(minute_number, output_arrays, count_arrays, average_output_count, detected_copy):
        f = open(output_stats, 'a')
        writes = time.strftime("%Y-%m-%d %H:%M:%S %z", time.gmtime()) + ": " + str(count_arrays) + '\n' + '    average: ' + str(average_output_count) + '\n\n'
        f.write(writes)
        f.close()
        # print("MINUTE : ", minute_number)
        # print("Array for the outputs of each frame ", output_arrays)
        # print("Array for output count for unique objects in each frame : ", count_arrays)
        # print("Output average count for unique objects in the last minute: ", average_output_count)
        # print("------------END OF A MINUTE --------------")


    detector.detectCustomObjectsFromVideo(
        input_file_path=os.path.join(videoInputPath, video),
        # camera_input=camera5,
        save_detected_video=True,
        output_file_path=os.path.join(videoOutputPath, outputVideoName),
        codec="X264",
        custom_objects=custom_objects,
        frames_per_second=15, frame_detection_interval=1,
        log_progress=False, minimum_percentage_probability=50, return_detected_frame=True, thread_safe=True,
        # per_minute_function=forMinute,
        per_second_function=forSecond
        # per_frame_function=forFrame
    )
    print(json.dumps(forSeconds_list, indent=2))
