import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import os
from imageai.Detection.Custom import DetectionModelTrainer
import json


execution_path = os.getcwd()

datasetDir = os.path.join(execution_path, 'dataset2')
# modelsDir = os.path.join(datasetDir, 'models/v1')
modelsDir = os.path.join(datasetDir, 'models\\20210207-1714')
jsonFile = os.path.join(datasetDir, 'json\\detection_config.json')
evaluationJson = os.path.join(modelsDir, 'evaluation.json')

# with open(evaluationJson, 'w') as fn:
#     fn.write("")

json_formatted_str = str()

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(datasetDir, False)

metrics = trainer.evaluateModel(model_path=modelsDir, json_path=jsonFile, iou_threshold=0.5, object_threshold=0.50, nms_threshold=0.4)

results = list()
for result in metrics:
    print(result)
    results.append(result)
    # json_formatted_str = json.dumps(result, indent=2)
    # with open(evaluationJson, 'a') as fn:
    #     fn.write(json_formatted_str)
    #     fn.write("\n")

    json_formatted_str = json.dumps(results, indent=2)
    with open(evaluationJson, 'w') as fn:
        fn.write(json_formatted_str)