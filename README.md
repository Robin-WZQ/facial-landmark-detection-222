# Facial Landmark Detection 👀🔍
> a pytorch inference code of facial landmark detection based on [PIPNET](https://github.com/jhb86253817/PIPNet), supporting 222 points and multiple face conditions.

### Key points
<div align=center>
<img src=pictures/test_meanface.jpg width="380" height="460"/>
</div>

### Results
> Our model perform well in many face conditions.

- Images
    - <details><summary> 👨‍👩‍👧‍👦 Age </summary><p><div align="center">
        <div align=center>
        <img src=pictures/2.png width="380" height="380"/>
        </div>
    
    - <details><summary> 🥸 Occlusion </summary><p><div align="center">
        <div align=center>
        <img src=pictures/1.png width="380" height="380"/>
        </div>  
    
    - <details><summary> 🔆 Illumination </summary><p><div align="center">
        <div align=center>
        <img src=pictures/3.png width="380" height="380"/>
        </div>  
  
    - <details><summary> 🔭 Large pose </summary><p><div align="center">
        <div align=center>
        <img src=pictures/4.png width="380" height="380"/>
        </div>  
    
    - <details><summary> 😃 Extreme expression </summary><p><div align="center">
        <div align=center>
        <img src=pictures/6.png width="570" height="380"/>
        </div>  

- Videos
     - <details><summary> 📽️ Video </summary><p><div align="center">
        <div align=center>
        <img src=pictures/video1.gif width="426" height="240"/>
        </div>  

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
    > We highly recommend using an aligned photo.
    ```
    python lib/image_test.py --img_path your_image_path
    ```

- for a video
    > We support a real-time face alignment algorithm that uses the key points of the previous frame to align the face of the current frame
    ```
    python lib/video_test.py --video_path your_video_path
    ```

### About the model
> Here lists some details of our model.
  
<div align=center>
<img src=pictures/detail.png width="480" height="240"/>
</div>  
    
   P.S. * is the original parameter of PIPNET.

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
