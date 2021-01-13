## 3d-pose-baseline

This is the code for an attempt to plug the code from  
Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little.
_A simple yet effective baseline for 3d human pose estimation._
In ICCV, 2017. https://arxiv.org/pdf/1705.03098.pdf.

into Android. 

The code in this repository was mostly written by
[Julieta Martinez](https://github.com/una-dinosauria),
[Rayat Hossain](https://github.com/rayat137)
[Javier Romero](https://github.com/libicocco) and
[Arash Hosseini](https://github.com/ArashHosseini).  

I have modified it to use TfLite Posenet as the 2d keypoint model, and to be runnable in an Android app. The 2d->3d model used is one pretrained on SH detections, downloadable [here](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing).

From the paper by Martinez et al.:
"We provide a strong baseline for 3d human pose estimation that also sheds light
on the challenges of current approaches. Our model is lightweight and we strive
to make our code transparent, compact, and easy-to-understand."

### Dependencies

* [h5py](http://www.h5py.org/)
* [tensorflow](https://www.tensorflow.org/) 1.0 or later

### First of all
1. You need access to the [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset. If you don't have access, you can't run this code.
2. Clone this repository and get the data.
3. If you got this far, wait for further instructions. The code is under construction.

You can test the pre-trained model by downloading it at the link above, decompressing the file at the top level of this project, and calling

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --sample --load 4874200`

You should see something like this:  
![Visualization example](/imgs/viz_example.png?raw=1)