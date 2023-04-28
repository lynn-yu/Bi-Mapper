# Bi-Mapper: Holistic BEV Semantic Mapping for Autonomous Driving  

Siyu Li, Kailun Yang, Hao Shi, Jiaming Zhang, Jiacheng Lin, Zhifeng Teng, and Zhiyong Li

## Motivation

![img1](https://github.com/lynn-yu/Bi-Mapper/blob/main/pic/img1.png)

### Abstract

A semantic map of the road scene, covering fundamental road elements, is an essential ingredient in autonomous driving systems. It provides important perception foundations for positioning and planning when rendered in the Bird’sEye-View (BEV). Currently, the learning of prior knowledge, the hypothetical depth, has two sides in the research of BEV scene understanding. It can guide learning of translating front perspective views into BEV directly on the help of calibration parameters. However, it suffers from geometric distortions in the representation of distant objects. In this paper, we propose a Bi-Mapper framework for top-down road-scene semantic understanding. The dual streams incorporate global view and local prior knowledge, which are learned asynchronously according to the learning timing. At the same time, an Across-Space Loss (ASL) is designed to mitigate the negative impact of geometric distortions. Extensive results verify the effectiveness of each module in the proposed Bi-Mapper framework. Compared with exiting road mapping networks, the proposed Bi-Mapper achieves 5.0 higher IoU on the nuScenes dataset. Moreover, we verify the generalization performance of Bi-Mapper in a realworld driving scenario.   

### Method
![img2](https://github.com/lynn-yu/Bi-Mapper/blob/main/pic/img2.png)

### Result

![img3](https://github.com/lynn-yu/Bi-Mapper/blob/main/pic/img3.png)

### Update

2023.04.28 Init repository.



### Contact

Feel free to contact me if you have additional questions or have interests in collaboration. Please drop me an email at  lsynn@hnu.edu.cn
