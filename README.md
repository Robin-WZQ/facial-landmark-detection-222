# facial-landmark-detection-222
> a pytorch inference code of facial landmark detection based on [PIPNET](https://github.com/jhb86253817/PIPNet), supporting 222 points and multiple face conditions.

### Key points
<div align=center>
<img src=test_meanface.jpg width="380" height="460"/>
</div>

### Results
> Our model perform well in many face conditions.

- Images
    - <details><summary> 👨‍👩‍👧‍👦 Age </summary><p><div align="center">
    <div align=center>
    <img src=https://github.com/Robin-WZQ/facial-landmark-detection-222/assets/60317828/e583df4a-c24f-4a12-bb0d-438b455e8cbd width="380" height="380"/>
    </div>
    
    - <details><summary> 🥸 Occlusion </summary><p><div align="center">
    <div align=center>
    <img src=https://github.com/Robin-WZQ/facial-landmark-detection-222/assets/60317828/9f3adadc-bea0-40d4-acc8-f01bdc547d2a width="380" height="380"/>
    </div>  
    
    - <details><summary> 🔆 Illumination </summary><p><div align="center">
    <div align=center>
    <img src=https://github.com/Robin-WZQ/facial-landmark-detection-222/assets/60317828/61435623-83a9-40a1-9e03-f1b7eef39502 width="380" height="380"/>
    </div>  
  
    - <details><summary> 🔭 Large pose </summary><p><div align="center">
    <div align=center>
    <img src=https://github.com/Robin-WZQ/facial-landmark-detection-222/assets/60317828/cccee92f-a671-45fd-ad88-d6ed1136df10 width="380" height="380"/>
    </div>  

- Videos

### Installation
The code was tested with Anaconda, Python 3.7. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/Robin-WZQ/facial-landmark-detetion-222.git
    cd facial-landmark-detetion-222
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install -r requirements.txt
    ```
### Usage
- for an image
    ```
    python lib/image_test.py --img_path your_image_path
    ```

- for a video
    ```
    python lib/video_test.py --video_path your_video_path
    ```

### About the model
> Here lists some details of our model.

### Acknowledgement
```
@article{JLS21,
  title={Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild},
  author={Haibo Jin and Shengcai Liao and Ling Shao},
  journal={International Journal of Computer Vision},
  publisher={Springer Science and Business Media LLC},
  ISSN={1573-1405},
  url={http://dx.doi.org/10.1007/s11263-021-01521-4},
  DOI={10.1007/s11263-021-01521-4},
  year={2021},
  month={Sep}
}
```
