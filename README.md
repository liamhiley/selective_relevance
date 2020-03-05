# selective_relevance
A code base for extracting dimension based relevance from multi-modal explanations for deep learning models.

Activity Recognition models (such as *[[Learning Spatiotemporal
Features with 3d convolutional
network](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)
Tran, D., Bourdev, L., Fergus, R., Torresani, L. and Paluri, M.]*) process human activity videos as 3D cubes of (X,Y,T) shape. In
this way they effectively learn spatio-temporal features as 3D edges/textures.
When applying post-hoc explainability techniques like LRP
(https://github.com/albermax/innvestigate) and GradCAM
(https://github.com/ramprs/grad-cam) to these videos, the resulting explanations
display heatmaps over each frame. While the input agnosticism of these methods
makes for very simple adaptation to video, their inherent use-case is image
data. 
The Selective Relevance method is a post-processing step that seeks to unpair
the spatial and temporal aspects of such an explanation, a functionality that is
not present in the base explainability techniques.

The basic workflow for generating a selective explanation is as follows:
- Read a video from a file using OpenCV
``` python
tensor,frame_list = video.get_input(video_path,input_frame_shape,channelwise_mean)
```
- Load in a model of a predefined architecture with pretrained weights

``` python
# model_architecture is a string that should match a key in the model_dict dictionary which will link to a torch model class in the models folder
mdl = model.get_model(model_architecture, num_classes, weights_path, (lowest_val,highest_val))
```
- Pass this input through and generate an explanation from the models inference

``` python
exp = model.get_exp(tensor,mdl)
```
- Run the explanation through the Selective Relevance process with a degree of
selectivity sigma 

