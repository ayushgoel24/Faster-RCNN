# Faster RCNN: A two-stage object detector

Faster RCNN is two-stage RCNN based object classifier. The first stage involves the use of Region Proposal Network(RPN) which is run once per image. It is responsible for:
1. classifying whether an anchor box is object or not
2. predicting the transform from anchor box to proposal box.

The second stage runs once per region proposed by RPN. It involves:
1. Cropping features: ROI pool/align
2. Classifying proposals as background or object
3. Predicting transform from proposal box to object box

For this project, a subset of COCO dataset was used to detect 3 types of objects: Vehicles, People, and Animals.

## Part A: Training RPN
No. of Anchors used: 1
<table>
  <tr>
      <td align = "center"> <img src="./static/images/1. RPN_backbone_a.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Backbone used for RPN</td>
  </tr>
</table>

**Intermediate** **Layer:**  Kernel Size: (3,3,256), Batch Norm, ReLU, Padding = Same

**RPN Classifier Head:** Kernel Size: (1,1,1), Sigmoid, Padding = Same

**RPN Regressor Head:**  Kernel Size: (1,1,4), Padding = Same

### RPN Losses:
**Classifier** **Loss:** Binary Cross Entropy

**Regressor** Loss:**  Smooth L1 Loss. **Note:** To minimize the bias towards negative classes, the positive(object) and negative(not object) anchors are subsampled in a ratio as close as possible to 1:1. Also, only a constant size minibatch is used for training regressor.

### Post Processing:
Boxes going out of the image boundary are removed. After that top K boxes(top objectness score boxes) are kept which is followed by Non-Maximum Suppression. The top N boxes after NMS are then kept as final proposals.

### Results:
<table>
  <tr>
      <td align = "center"> <img src="./static/images/2. Pre_NMS_1_a.png"> </td>
      <td align = "center"> <img src="./static/images/3. Post_NMS_1_a.png"> </td>
  </tr>
  <tr>
      <td align = "center"> <img src="./static/images/4. Pre_NMS_2_a .png"> </td>
      <td align = "center"> <img src="./static/images/5. Post_NMS_2_a.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Pre NMS, K = 20</td>
      <td align = "center"> POST NMS, N = 5s </td>
  </tr>
</table>

### Loss curves:
<table>
  <tr>
      <td align = "center"> <img src="./static/images/6. Classifier_Loss_a.png"> </td>
      <td align = "center"> <img src="./static/images/7. Regressor_Loss_a.png"> </td>
  </tr>
  <tr>
      <td align = "center"> RPN Classifier Loss Curve</td>
      <td align = "center"> RPN Regressor Loss Curve</td>
  </tr>
</table>

## Part B: Training second stage networks
The RPN trained in the Part A is a simpler version that keeps all the necessary components. Due to it's simplicity, for the second stage, a pretrained Feature Pyramid Network and RPN are used. Top 200 proposals from RPN are kept for the second stage\
**ROI** **align:** torchvision.ops.RoiAlign is used for this\
**Intermediate** **Layer:** 2 linear layers with ReLU activation. Output size of both layers: 1024\
**Box** **Head** **Classifier:** Linear layer with output size C+1 followed by softmax\
**Box** **Head** **Regressor:** Linear layer with output 4*C

### Losses
**Classifier's** **Loss:** Cross Entropy Loss/
**Regressor** **Loss:** Smooth L1 loss **Sampling:** Similar to RPN with a small change. For this case, the ratio of object to no-object boxes as close to 3:1

### Post Processing
Boxes going out of the image boundary and having object confidence lower than 0.5 are removed. After that top K boxes(top objectness score boxes) are kept which is followed by Non-Maximum Suppression for each class independently. The top N boxes after NMS are then kept as final proposals.

### Results
<table>
  <tr>
      <td align = "center"> <img src="./static/images/1. Pre_NMS_1_b.png"> </td>
      <td align = "center"> <img src="./static/images/2. Post_NMS_1_b.png"> </td>
  </tr>
  <tr>
      <td align = "center"> <img src="./static/images/3. Pre_NMS_2_b.png"> </td>
      <td align = "center"> <img src="./static/images/4. Post_NMS_2_b.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Pre NMS</td>
      <td align = "center"> POST NMS</td>
  </tr>
</table>

### Loss curves:
<table>
  <tr>
      <td align = "center"> <img src="./static/images/5. Classifier_Loss_b.png"> </td>
      <td align = "center"> <img src="./static/images/6. Regressor_Loss_b.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Classifier Loss Curve</td>
      <td align = "center"> Regressor Loss Curve</td>
  </tr>
</table>

### Precision Recall Curve:
**Average** **Precision** **Values**:\
1. Vehicle: 0.7057
2. Person:  0.7585
3. Animal:  0.8233
<table>
  <tr>
      <td align = "center"> <img src="./static/images/7. Precision_Recall_plots_b.png"> </td>
  </tr>
  <tr>
      <td align = "center"> Precision Recall curves</td>
  </tr>
</table>



