# Lane Segmentation and Object Detection/Classification

In this project, we aim to *Lane Segmentation* and *Traffic Signs Classification* in the images.

Some of the technologies used in the project; **Python, OpenCV, Pytorch, TensorFlow, YOLOv4**

The results of the project can be viewed in the video below;

<p  align="center">
<a href = "https://youtu.be/0WAls6WQne8">
<img  src="images/video.png"  width="">
</a>
</p>

### The project consists of 2 main parts;
- [Lane Segmentation](#lane-segmentation)
- [Traffic Sign Detection and Classification](#traffic-sign-detection-and-classification)

## Lane Segmentation

Many steps in the Lane Segmentation section have the same content as the Drivable Area Segmentation project.

**[Click for the GitHub repository of the Drivable Area Detection project. ](https://github.com/recepayddogdu/Freespace_Segmentation-Ford_Otosan_Intern)**

### Json to Mask
JSON files are obtained as a result of highway images labeled by Ford Otosan Annotation Team. The JSON files contain the locations of the *Solid Line* and *Dashed Line* classes.

A mask was created with the data in the JSON file to identify the pixels of the lines in the image.

The `fillPoly` function from the cv2 library was used to draw the masks.

    for obj in json_dict["objects"]: #To access each list inside the json_objs list
        if obj['classTitle']=='Solid Line':
           cv2.polylines(mask,np.array([obj['points']['exterior']],dtype=np.int32),False,color=1,thickness=14)
        elif obj['classTitle']=='Dashed Line':       
           cv2.polylines(mask,np.array([obj['points']['exterior']],dtype=np.int32),False,color=2,thickness=9)

Mask example;
<p  align="center">
<img  src="images/lane_segmentation/maskonimg.png"  width="">
</p> 
