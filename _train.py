# import warnings
# warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from imageai.Detection import VideoObjectDetection
import os
import time
from imageai.Detection.Custom import DetectionModelTrainer
from modules.utils import split_dataset

execution_path = os.getcwd()

datasetName = 'dataset2'
datasetDir = os.path.join(execution_path, datasetName)

objects = [
    'car'
]

# split_dataset(datasetName, 15)

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=datasetDir)
trainer.setGpuUsage("1")
trainer.setTrainConfig(object_names_array=objects, batch_size=4, num_experiments=150,
                        train_from_pretrained_model=os.path.join(execution_path, 'sample-models/pretrained-yolov3.h5'),
                        accum_iters=1, lr=1e-5)
trainer.trainModel()


# if __name__ == '__main__':
#     trainer.trainModel()