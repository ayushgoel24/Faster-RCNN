import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):

    return iou



def iou(anchors, gt):               #Dimensions Anchors: (n_proposals,4), gt (ground_truth_boxes, 4)
    #Extracting centers and h w
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    px1 =  anchors[:,0].reshape(-1,1)                                            #[n_proposals,1]                                        
    py1 =  anchors[: ,1].reshape(-1,1)                                           #[n_proposals,1]
    px2  = anchors[: ,2].reshape(-1,1)                                           #[n_proposals,1]
    py2  = anchors[: ,3].reshape(-1,1)                                           #[n_proposals,1]
    
    
    gx1  = gt[:,0].reshape(-1,1)                                                 #[ground_truth_boxes,1]
    gy1  = gt[:,1].reshape(-1,1)                                                 #[ground_truth_boxes,1]
    gx2  = gt[:,2].reshape(-1,1)                                                 #[ground_truth_boxes,1] 
    gy2  = gt[:,3].reshape(-1,1)                                                 #[ground_truth_boxes,1]
    
    #Box format [x1 y1 x2 y2]
    box1 =  [px1, py1, px2, py2]
    box2 =  [gx1, gy1, gx2, gy2]

    xA = torch.max(box1[0], box2[0].T)                                            #[n_proposals,ground_truth_boxes] 
    yA = torch.max(box1[1], box2[1].T)                                            #[n_proposals,ground_truth_boxes] 
  
    xB = torch.min(box1[2], box2[2].T)                                            #[n_proposals,ground_truth_boxes] 
    yB = torch.min(box1[3], box2[3].T)                                            #[n_proposals,ground_truth_boxes] 
  
    area_intersection = torch.max(xB-xA, torch.zeros(xB.shape, dtype=xB.dtype,device = device)) * torch.max(yB-yA, torch.zeros(yB.shape, dtype=yB.dtype, device = device))
  
    area_union = (box1[2]-box1[0]) * (box1[3]-box1[1]) + ((box2[2]-box2[0]) * (box2[3]-box2[1])).T - area_intersection
  
    iou = torch.div(area_intersection+1,area_union+1)
    return iou   #[n_proposals, ground_truth_boxes]


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)

def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):
	wp = flatten_proposals[:,2] - flatten_proposals[:,0]
	hp = flatten_proposals[:,3] - flatten_proposals[:,1]

	x_p = (flatten_proposals[:,2] + flatten_proposals[:,0])/2
	y_p = (flatten_proposals[:,3] + flatten_proposals[:,1])/2

	box= torch.zeros(regressed_boxes_t.shape, device=device)
	box[:,0] = regressed_boxes_t[:,0]*wp + x_p - torch.exp(regressed_boxes_t[:,2])*wp/2
	box[:,1] = regressed_boxes_t[:,1]*hp + y_p - torch.exp(regressed_boxes_t[:,3])*hp/2
	box[:,2] = regressed_boxes_t[:,0]*wp + x_p + torch.exp(regressed_boxes_t[:,2])*wp/2
	box[:,3] = regressed_boxes_t[:,1]*hp + y_p + torch.exp(regressed_boxes_t[:,3])*hp/2

	return box
