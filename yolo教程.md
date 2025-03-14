yolo教程









[YOLOv12入门教程](https://mp.weixin.qq.com/s/2YUXLl2X_4t396frEDgOCg)

[YOLO系列发展历程：从YOLOv1到YOLO11，目标检测技术的革新与突破](https://mp.weixin.qq.com/s/Fol-Y_C46Yervorfx07z5Q)

[YOLOv12 初学者使用教程](https://mp.weixin.qq.com/s/VSoG9YOr1dGh5WaHJlraow)

[手把手教你玩转YOLOv12目标检测：从环境配置、模型训练、验证、推理的全流程指南](https://mp.weixin.qq.com/s/V_AqZOIfVL8MyPjI5Iw8cw)

[YOLOv12保姆级使用教程重磅来袭，教你如何找熊猫](https://mp.weixin.qq.com/s/8rKXUYzMWpVObQKItgpHRg)



YOLO（You Only Look Once）

state-of-the-art (SOTA)

可以实现的功能：分类classify，检测detect，姿态估计Pose estimation 

YOLOv12：https://github.com/sunsmarterjie/yolov12



目前最新的是YOLOV12，2025年2月发布

YOLOV12分为两个版本：Turbo，v1.0

**Turbo (default version)**:

| Model                                                        | size (pixels) | mAPval 50-95 | Speed T4 TensorRT10 | params (M) | FLOPs (G) |
| ------------------------------------------------------------ | ------------- | ------------ | ------------------- | ---------- | --------- |
| [YOLO12n](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt) | 640           | 40.4         | 1.60                | 2.5        | 6.0       |
| [YOLO12s](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt) | 640           | 47.6         | 2.42                | 9.1        | 19.4      |
| [YOLO12m](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt) | 640           | 52.5         | 4.27                | 19.6       | 59.8      |
| [YOLO12l](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt) | 640           | 53.8         | 5.83                | 26.5       | 82.4      |
| [YOLO12x](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12x.pt) | 640           | 55.4         | 10.38               | 59.3       | 184.6     |

[**v1.0**](https://github.com/sunsmarterjie/yolov12/tree/V1.0):

| Model                                                        | size (pixels) | mAPval 50-95 | Speed T4 TensorRT10 | params (M) | FLOPs (G) |
| ------------------------------------------------------------ | ------------- | ------------ | ------------------- | ---------- | --------- |
| [YOLO12n](https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt) | 640           | 40.6         | 1.64                | 2.6        | 6.5       |
| [YOLO12s](https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt) | 640           | 48.0         | 2.61                | 9.3        | 21.4      |
| [YOLO12m](https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12m.pt) | 640           | 52.5         | 4.86                | 20.2       | 67.5      |
| [YOLO12l](https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12l.pt) | 640           | 53.7         | 6.77                | 26.4       | 88.9      |
| [YOLO12x](https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt) | 640           | 55.2         | 11.79               | 59.1       | 199.0     |



-----









[YOLO11详解](https://mp.weixin.qq.com/s/iTwprX2crSc13Sahtsfa_g)

[YOLO模型综述：YOLO11及其前身的全面基准研究](https://mp.weixin.qq.com/s/QFomq1oM2Smgm8OY9DaaRQ)

[YOLOv11入门到入土使用教程(含结构图)](https://blog.csdn.net/StopAndGoyyy/article/details/143169639)

[Ultralytics：YOLO11使用教程](https://blog.csdn.net/FriendshipTang/article/details/142772535)

[YOLOv11超详细环境搭建以及模型训练（GPU版本）](https://blog.csdn.net/2401_85556416/article/details/143378148)



#### 一、YOLO11简介

[ultralytics](https://github.com/ultralytics/ultralytics)：是一个框架，通过它可以使用YOLO11模型，会自动下载模型到当前目录（自己手动在浏览器下载要快很多），支持YOLOv2到YOLO11。

YOLO11的模型都不大，yolo11n.pt 为5.4M，yolo11x.pt 为109M。

ultralytics文档：https://docs.ultralytics.com/

模型下载地址：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

`ultralytics`为`YOLO11`在不同功能上训练了一批模型，由于不同数据集的特点，所以针对不同的功能会选择在不同数据集上训练。

YOLO11 [Detect](https://docs.ultralytics.com/tasks/detect/), [Segment](https://docs.ultralytics.com/tasks/segment/) and [Pose](https://docs.ultralytics.com/tasks/pose/) models pretrained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco/) dataset are available here, as well as YOLO11 [Classify](https://docs.ultralytics.com/tasks/classify/) models pretrained on the [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) dataset. [Track](https://docs.ultralytics.com/modes/track/) mode is available for all Detect, Segment and Pose models. All [Models](https://docs.ultralytics.com/models/) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.



**模型指标**

size (pixels)：图片像素 640 * 640

mAP：平均精度

Speed：推理时间，毫秒

params (M)：参数量，百万

FLOPs (G)：每秒浮点运算次数。G表示10亿，可以用来衡量算法/模型复杂度，理论上该数值越高越好



**模型规格**

模型后缀`n, s, m, l, x`代表模型越来越大。

YOLO11m 在准确性、效率、模型大小之间取得了最佳平衡



**模型功能**

Detect：物体检测

Segment：图像分割

Classify：分类

Pose：人体姿态推断

OBB：定向物体检测，检测框有一定的倾斜角度

Track：物体追踪

![image-20250313105036683](D:\dev\php\magook\trunk\server\md\img\image-20250313105036683.png)



**模型列表**

<details open><summary>Detection (COCO)</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [COCO](https://docs.ultralytics.com/datasets/detect/coco/), which include 80 pre-trained classes.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640                   | 39.5                 | 56.1 ± 0.8                     | 1.5 ± 0.0                           | 2.6                | 6.5               |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640                   | 47.0                 | 90.0 ± 1.2                     | 2.5 ± 0.0                           | 9.4                | 21.5              |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640                   | 51.5                 | 183.2 ± 2.0                    | 4.7 ± 0.1                           | 20.1               | 68.0              |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640                   | 53.4                 | 238.6 ± 1.4                    | 6.2 ± 0.1                           | 25.3               | 86.9              |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640                   | 54.7                 | 462.8 ± 6.7                    | 11.3 ± 0.2                          | 56.9               | 194.9             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>Segmentation (COCO)</summary>

See [Segmentation Docs](https://docs.ultralytics.com/tasks/segment/) for usage examples with these models trained on [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/), which include 80 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                   | 38.9                 | 32.0                  | 65.9 ± 1.1                     | 1.8 ± 0.0                           | 2.9                | 10.4              |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                   | 46.6                 | 37.8                  | 117.6 ± 4.9                    | 2.9 ± 0.0                           | 10.1               | 35.5              |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                   | 51.5                 | 41.5                  | 281.6 ± 1.2                    | 6.3 ± 0.1                           | 22.4               | 123.3             |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                   | 53.4                 | 42.9                  | 344.2 ± 3.2                    | 7.8 ± 0.2                           | 27.6               | 142.2             |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                   | 54.7                 | 43.8                  | 664.5 ± 3.2                    | 15.8 ± 0.7                          | 62.1               | 319.0             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val segment data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>Classification (ImageNet)</summary>

See [Classification Docs](https://docs.ultralytics.com/tasks/classify/) for usage examples with these models trained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/), which include 1000 pretrained classes.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 224 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLO11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) | 224                   | 70.0             | 89.4             | 5.0 ± 0.3                      | 1.1 ± 0.0                           | 1.6                | 0.5                      |
| [YOLO11s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt) | 224                   | 75.4             | 92.7             | 7.9 ± 0.2                      | 1.3 ± 0.0                           | 5.5                | 1.6                      |
| [YOLO11m-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-cls.pt) | 224                   | 77.3             | 93.9             | 17.2 ± 0.4                     | 2.0 ± 0.0                           | 10.4               | 5.0                      |
| [YOLO11l-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-cls.pt) | 224                   | 78.3             | 94.3             | 23.2 ± 0.3                     | 2.8 ± 0.0                           | 12.9               | 6.2                      |
| [YOLO11x-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt) | 224                   | 79.5             | 94.9             | 41.4 ± 0.9                     | 3.8 ± 0.0                           | 28.4               | 13.7                     |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

</details>

<details><summary>Pose (COCO)</summary>

See [Pose Docs](https://docs.ultralytics.com/tasks/pose/) for usage examples with these models trained on [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/), which include 1 pre-trained class, person.

| Model                                                                                          | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) | 640                   | 50.0                  | 81.0               | 52.4 ± 0.5                     | 1.7 ± 0.0                           | 2.9                | 7.6               |
| [YOLO11s-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt) | 640                   | 58.9                  | 86.3               | 90.5 ± 0.6                     | 2.6 ± 0.0                           | 9.9                | 23.2              |
| [YOLO11m-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt) | 640                   | 64.9                  | 89.4               | 187.3 ± 0.8                    | 4.9 ± 0.1                           | 20.9               | 71.7              |
| [YOLO11l-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt) | 640                   | 66.1                  | 89.9               | 247.7 ± 1.1                    | 6.4 ± 0.1                           | 26.2               | 90.7              |
| [YOLO11x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) | 640                   | 69.5                  | 91.1               | 488.0 ± 13.9                   | 12.1 ± 0.2                          | 58.8               | 203.3             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

</details>

<details><summary>OBB (DOTAv1)</summary>

See [OBB Docs](https://docs.ultralytics.com/tasks/obb/) for usage examples with these models trained on [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2/#dota-v10/), which include 15 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>test<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) | 1024                  | 78.4               | 117.6 ± 0.8                    | 4.4 ± 0.0                           | 2.7                | 17.2              |
| [YOLO11s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt) | 1024                  | 79.5               | 219.4 ± 4.0                    | 5.1 ± 0.0                           | 9.7                | 57.5              |
| [YOLO11m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt) | 1024                  | 80.9               | 562.8 ± 2.9                    | 10.1 ± 0.4                          | 20.9               | 183.5             |
| [YOLO11l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt) | 1024                  | 81.0               | 712.5 ± 5.0                    | 13.5 ± 0.6                          | 26.2               | 232.0             |
| [YOLO11x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt) | 1024                  | 81.3               | 1408.6 ± 7.7                   | 28.6 ± 1.0                          | 58.8               | 520.2             |

- **mAP<sup>test</sup>** values are for single-model multiscale on [DOTAv1](https://captain-whu.github.io/DOTA/index.html) dataset. <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test` and submit merged results to [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** averaged over DOTAv1 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`

</details>



### 一、使用YOLO11模型

环境准备

```bash
python >= 3.8
pytorch >= 1.8

conda create -n yolo11 python=3.11.4
conda activate yolo11
pip install ultralytics
```

创建项目 `learn-yolo`

下载 `ultralytics`代码，因为有一些东西需要参考，主要是`ultralytics`目录。



#### 1、Detect

参考：https://docs.ultralytics.com/tasks/detect/

从图像或视频流中，识别对象的位置和类别。

物体检测器的输出是一组包围图像中物体的边框，以及每个边框的类标签和置信度分数。如果您需要识别场景中感兴趣的物体，但又不需要知道物体的具体位置或确切形状，那么物体检测就是一个不错的选择。

Predict：https://docs.ultralytics.com/modes/predict/

我们知道，Detect 功能是在 COCO 数据集上训练的，那这个 COCO 数据集可以识别哪些东西呢。我们打开`ultralytics/cfg/datasets/coco.yaml`文件，里面列出了 80 个物体名称，

```python
from ultralytics import YOLO
import cv2

def detect():
    model = YOLO("yolo11n.pt")
    results = model(["imgs/20250210101802.png", "imgs/20240731142129234.jpg", "imgs/cat.jpg"])

    for result in results:
        # print(type(result)) # <class 'ultralytics.engine.results.Results'>
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk
```

```bash
results 是一个列表，result 类型为 <class 'ultralytics.engine.results.Results'>

Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        speed (Dict): Dictionary containing inference speed information.
        names (Dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Updates the Results object with new detection data.
        cpu: Returns a copy of the Results object with all tensors moved to CPU memory.
        numpy: Converts all tensors in the Results object to numpy arrays.
        cuda: Moves all tensors in the Results object to GPU memory.
        to: Moves all tensors to the specified device and dtype.
        new: Creates a new Results object with the same image, path, names, and speed attributes.
        plot: Plots detection results on an input RGB image.
        show: Displays the image with annotated inference results.
        save: Saves annotated inference results image to file.
        verbose: Returns a log string for each task in the results.
        save_txt: Saves detection results to a text file.
        save_crop: Saves cropped detection images to specified directory.
        summary: Converts inference results to a summarized dictionary.
        to_df: Converts detection results to a Pandas Dataframe.
        to_json: Converts detection results to JSON format.
        to_csv: Converts detection results to a CSV format.
        to_xml: Converts detection results to XML format.
        to_html: Converts detection results to HTML format.
        to_sql: Converts detection results to an SQL-compatible format.
```

`show()`是打开图片查看器展示图片。

`plot()`是返回结果图片的 `numpy.ndarray`数组，这样就可以使用`cv2`来处理图片了，其已经安装了`opencv-python`。对于视频流，那最好是用`cv2.imshow`，使用同一个窗口名称，就只有一个窗口。

```python
def detect():
    model = YOLO("yolo11n.pt")
    results = model(["imgs/20250210101802.png", "imgs/20240731142129234.jpg", "imgs/cat.jpg"])
    for k, result in enumerate(results):
            im = result.plot()  # numpy.ndarray
            cv2.imshow("test"+str(k), im)

    cv2.waitKey(0)
```

输出信息

```bash
0: 640x640 1 person, 1 sports ball, 1 tennis racket, 138.6ms
1: 640x640 3 persons, 3 cars, 138.6ms
2: 640x640 1 cat, 138.6ms
Speed: 6.8ms preprocess, 138.6ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 640)
```

![image-20250313152319292](D:\dev\php\magook\trunk\server\md\img\image-20250313152319292.png)

![image-20250313161612139](D:\dev\php\magook\trunk\server\md\img\image-20250313161612139.png)

![image-20250313161632257](D:\dev\php\magook\trunk\server\md\img\image-20250313161632257.png)



#### 2、Segment

实例分割比对象检测更进一步，它涉及识别图像中的单个对象，并将它们从图像的其余部分中分割出来。

实例分割模型的输出是一组掩码或轮廓，这些掩码或轮廓勾勒出图像中的每个对象，并且每个对象都附有类别标签和置信度分数。当你不仅需要知道图像中对象的位置，还需要了解它们的确切形状时，实例分割就非常有用。

![image-20250313152852101](D:\dev\php\magook\trunk\server\md\img\image-20250313152852101.png)

```python
def segment():
    model = YOLO("yolo11n-seg.pt")
    results = model(["imgs/20250210101802.png"])

    for k, result in enumerate(results):
        filename = util.generate_random_string(15)
        result.save(filename="imgs_result/"+filename+".jpg")
```

![image-20250313165311348](D:\dev\php\magook\trunk\server\md\img\image-20250313165311348.png)



#### 3、Pose

姿态估计是一项涉及识别图像中特定点的位置的任务，这些点通常被称为关键点。关键点可以代表对象的各种部分，如关节、标志点或其他显著特征。关键点的位置通常以一组2D [x, y]或3D [x, y, 可见性]坐标表示。

姿态估计模型的输出是一组代表图像中对象上的关键点的点，通常还附有每个点的置信度分数。当你需要识别场景中对象的具体部分及其相互之间的位置关系时，姿态估计是一个很好的选择。

使用的是 `coco-pose`数据集。

人的姿态被定义为17个点，并标记每个点在图像中的坐标

```bash
Nose
Left Eye
Right Eye
Left Ear
Right Ear
Left Shoulder
Right Shoulder
Left Elbow
Right Elbow
Left Wrist
Right Wrist
Left Hip
Right Hip
Left Knee
Right Knee
Left Ankle
Right Ankle
```



```python
def pose():
    model = YOLO("yolo11n-pose.pt")  # load an official model
    results = model("imgs/20250210101802.png")

    for result in results:
        print(result.keypoints)
        result.show()
```

![image-20250313180124063](D:\dev\php\magook\trunk\server\md\img\image-20250313180124063.png)

```bash
ultralytics.engine.results.Keypoints object with attributes:

conf: tensor([[0.9966, 0.9942, 0.9561, 0.9619, 0.3857, 0.9993, 0.9988, 0.9974, 0.9947, 0.9763, 0.9706, 0.9999, 0.9998, 0.9985, 0.9982, 0.9850, 0.9847]])
data: tensor([[[2.1565e+02, 3.4999e+01, 9.9656e-01],
         [2.1984e+02, 2.7018e+01, 9.9415e-01],
         [2.0907e+02, 3.1001e+01, 9.5614e-01],
         [2.3481e+02, 2.5937e+01, 9.6186e-01],
         [0.0000e+00, 0.0000e+00, 3.8572e-01],
         [2.5573e+02, 4.5721e+01, 9.9928e-01],
         [2.0819e+02, 6.0532e+01, 9.9878e-01],
         [2.8809e+02, 6.5891e+01, 9.9740e-01],
         [1.5987e+02, 8.1759e+01, 9.9467e-01],
         [3.2300e+02, 1.0970e+02, 9.7626e-01],
         [1.1831e+02, 8.0122e+01, 9.7056e-01],
         [2.5918e+02, 1.3346e+02, 9.9990e-01],
         [2.0929e+02, 1.4076e+02, 9.9985e-01],
         [3.3731e+02, 1.8365e+02, 9.9854e-01],
         [1.6218e+02, 1.9817e+02, 9.9822e-01],
         [3.9673e+02, 2.3521e+02, 9.8505e-01],
         [9.3972e+01, 2.5031e+02, 9.8471e-01]]])
has_visible: True
orig_shape: (299, 478)
shape: torch.Size([1, 17, 3])
xy: tensor([[[215.6517,  34.9987],
         [219.8361,  27.0179],
         [209.0718,  31.0013],
         [234.8076,  25.9374],
         [  0.0000,   0.0000],
         [255.7305,  45.7213],
         [208.1883,  60.5316],
         [288.0875,  65.8907],
         [159.8745,  81.7594],
         [323.0020, 109.6960],
         [118.3133,  80.1224],
         [259.1849, 133.4618],
         [209.2867, 140.7613],
         [337.3135, 183.6483],
         [162.1798, 198.1695],
         [396.7317, 235.2071],
         [ 93.9718, 250.3072]]])
xyn: tensor([[[0.4512, 0.1171],
         [0.4599, 0.0904],
         [0.4374, 0.1037],
         [0.4912, 0.0867],
         [0.0000, 0.0000],
         [0.5350, 0.1529],
         [0.4355, 0.2024],
         [0.6027, 0.2204],
         [0.3345, 0.2734],
         [0.6757, 0.3669],
         [0.2475, 0.2680],
         [0.5422, 0.4464],
         [0.4378, 0.4708],
         [0.7057, 0.6142],
         [0.3393, 0.6628],
         [0.8300, 0.7866],
         [0.1966, 0.8371]]])
```

`conf`：每个点的得分

`xy`：每一个点的x,y坐标

`xyn`：坐标标准化

`data`：x, y, visibility



#### 4、Classify

图像分类是这三个任务中最简单的，它涉及将整张图像归类到一组预定义的类别中的一种。

图像分类器的输出是一个单一的类别标签和一个置信度分数。当你只需要知道图像属于哪个类别，而不需要知道该类别的对象在图像中的具体位置或它们的确切形状时，图像分类就非常有用。

使用的是`imageNet`数据集，里面有1000个类别。

```python
def classify():
    model = YOLO("yolo11m-cls.pt")
    results = model(["imgs/20250210101802.png"])

    for k, result in enumerate(results):
        print(result.probs)
        result.show()
```

`imgs/20250210101802.png`是上面的打羽毛球的图片。

输出结果

```bash
......
orig_shape: None
shape: torch.Size([1000])
top1: 752
top1conf: tensor(0.9599)
top5: [752, 852, 890, 722, 422]
top5conf: tensor([0.9599, 0.0154, 0.0079, 0.0037, 0.0026])
```

`top1: 752`代表分类ID，打开`ultralytics\cfg\datasets\ImageNet.yaml`文件，搜索到`752: racket`，就是球拍的意思。

`top1conf: tensor(0.9599)`代表 top1 的得分。

`top5`和`top5conf`代表 top5 的分类。

从得分来看，top1 远远超过了其他四个。

![image-20250313181129539](D:\dev\php\magook\trunk\server\md\img\image-20250313181129539.png)



#### 5、Track

https://docs.ultralytics.com/modes/track/

![image-20250314141356363](D:\dev\php\magook\trunk\server\md\img\image-20250314141356363.png)

在视频分析领域中，对象跟踪是一项至关重要的任务，它不仅能够识别帧内对象的位置和类别，而且还能在视频进行过程中为每个检测到的对象维持一个唯一的ID。该技术的应用范围非常广泛，从 surveillance（监控）和安全到实时体育分析等领域都有涉及。

来自Ultralytics跟踪器的输出与标准对象检测一致，但增加了对象ID的价值。这使得在视频流中跟踪对象并执行后续分析变得容易。

Ultralytics YOLO 提供了两个跟踪算法，你需要在 YAML 配置文件中设置`tracker=tracker_type.yaml`，可选的值为`botsort.yaml 和 bytetrack.yaml`，默认值是`botsort`。

你可以使用训练好的`Detect / Segment / Pose`模型，在视频流上运行追踪器，比如`YOLO11n, YOLO11n-seg, YOLO11n-pose`。

```python
def track():
    model = YOLO("yolo11n.pt")
	
    # Tracking with default tracker
    results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  
    
    # with ByteTrack
    # results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  
    
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

```



#### 6、OBB

Oriented Bounding Boxes Object Detection，定向物体检测，画出来的边界框带有一个倾斜角度。

定向对象检测比传统对象检测更进一步，它引入了一个额外的角度来更精确地定位图像中的对象。

定向对象检测器的输出是一组旋转的边界框，这些框精确地包围了图像中的对象，并且每个框都附有类别标签和置信度分数。当你需要识别场景中的感兴趣对象，但不需要知道对象的确切位置或其确切形状时，对象检测是一个很好的选择。

使用的数据集是 `DOTAv1`

```python
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
```

![image-20250314171720310](D:\dev\php\magook\trunk\server\md\img\image-20250314171720310.png)

![image-20250314172505181](D:\dev\php\magook\trunk\server\md\img\image-20250314172505181.png)





`requirements.txt`

```bash
certifi==2022.12.7
charset-normalizer==2.1.1
colorama==0.4.6
contourpy==1.2.0
cycler==0.12.1
filelock==3.9.0
fonttools==4.50.0
fsspec==2024.6.1
huggingface-hub==0.23.4
idna==3.4
Jinja2==3.1.2
kiwisolver==1.4.5
MarkupSafe==2.1.3
matplotlib==3.8.3
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
opencv-python==4.9.0.80
packaging==24.0
pandas==2.2.1
pillow==10.2.0
psutil==5.9.8
py-cpuinfo==9.0.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
requests==2.28.1
scipy==1.12.0
seaborn==0.13.2
six==1.16.0
sympy==1.12
thop==0.1.1.post2209072238
torch==2.0.0+cu118
torchaudio==2.0.1+cu118
torchvision==0.15.1+cu118
tqdm==4.66.2
typing_extensions==4.8.0
tzdata==2024.1
ultralytics==8.1.34
urllib3==1.26.13
```

```bash
pip install -r requirements.txt
```



