import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
# from rpn import *
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset
        #############################################
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.img    = h5py.File(path[0],'r')['data']               #Array:           Shape: (len(dataset),3,300,400)
        self.mask   = h5py.File(path[1],'r')['data']               #Array:           Shape: (total_masks_in dataset, 300, 400)
        self.labels = np.load(path[2], allow_pickle=True)          #Array of arrays: Shape: (len(dataset))  Each array: Shape: (num_labels for that image)
        self.bbox   = np.load(path[3], allow_pickle=True)          #Array of arrays: Shape: (len(dataset))  Each array: Shape: (num_labels for that image, 4)
        #Transforms
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

        self.aligned_masks = [] #List of mask tensors    i.e.  len(dataset) (num_masks_for_an_image, 300,400)

        i = 0
        for l in range(self.labels.shape[0]):
            num_labels = self.labels[l].size
            # if l ==0:
            #     self.aligned_masks.append([])
            #     continue
            mask_clubbed = np.zeros((num_labels,300,400))#,dtype = torch.int)
            for mask_idx in range(num_labels):
                mask_clubbed[mask_idx,:,:] = self.mask[i,:,:]
                i+=1
            self.aligned_masks.append(mask_clubbed)

    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        img   = self.img[index,:,:,:]                             #Array: Shape: (3,300,400)
        mask  = self.aligned_masks[index]                         #Array: Shape: (num_masks_for_an_image, 300,400)  
        label = self.labels[index]                                #Array: Shape: (num_labels)                       
        bbox  = self.bbox[index]                                  #Array: Shape: (num_labels,4)
        
        img   = torch.tensor(img.astype(np.float),  dtype = torch.float).to(self.device)
        mask  = torch.tensor(mask.astype(np.float), dtype = torch.float).to(self.device)
        label = torch.tensor(label,                 dtype = torch.float).to(self.device)
        bbox  = torch.tensor(bbox.astype(np.float), dtype = torch.float).to(self.device)


        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)

        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return transed_img, label, transed_mask, transed_bbox, index



    # This function preprocess the given image, mask, box by rescaling them appropriately
    #Input: 
    #        img:  (3,300,400)
    #        mask: (n_box,300,400)
    #        bbox: (n_box,4)
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################

        scale_factor_x = 800/300
        scale_factor_y = 1066/400

        img =  img/255.0 #Normalization
        img =  torch.unsqueeze(img, 0)                                                 #(3,300,400)   -> (1,3,300,400)
        img = torch.nn.functional.interpolate(img, size=(800, 1066), mode='bilinear')  #(1,3,300,400) -> (1,3,800,1066)
        img =  self.normalize(img[0])                                                  #(1,3,800,1066)-> (3,800,1066)
        img =  torch.nn.functional.pad(img, pad=(11,11), mode='constant',value=0)      #(3,800,1066)  -> (3,800,1088)
        img = img.squeeze(0)                                                           #(3,800,1066)  -> (1,3,800,1088)

        mask = mask.unsqueeze(0)                                                                     #(n_box,300,400)    -> (1,n_box,300,400)
        mask = torch.nn.functional.interpolate(mask, size = (800,1066), mode = 'bilinear')           #(1,n_box,300,400)  -> (1,n_box,800,1066)
        mask = torch.nn.functional.pad(mask, pad=(11,11), mode='constant',value=0)                   #(1,n_box,800,1066) -> (1,n_box,800,1088)
        

        bbox[:,0] = bbox[:,0] * scale_factor_x
        bbox[:,2] = bbox[:,2] * scale_factor_x
        bbox[:,1] = bbox[:,1] * scale_factor_y
        bbox[:,3] = bbox[:,3] * scale_factor_y
        bbox[:,0] += 11 
        bbox[:,2] += 11 #Accounting for changes in x due to padding            

        assert img.squeeze(0).shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.squeeze(0).shape[0]
        return img.squeeze(0), mask.squeeze(0), bbox

    
    def __len__(self):
        # return len(self.imgs_data)
        return self.img.shape[0]




class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        
        out_batch = {}
        # bz = len(batch)
        
        img_list = []
        label_list = []
        mask_list = []
        bbox_list = []
        index_list = []
        
        for img, label, mask, bbox, index in batch:
          img_list.append(img)
          label_list.append(label)
          mask_list.append(mask)
          bbox_list.append(bbox)
          index_list.append(index)  

        out_batch['images'] = torch.stack(img_list,dim=0)
        out_batch['labels'] = label_list
        out_batch['masks']  = mask_list
        out_batch['bbox']   = bbox_list
        out_batch['index']  = index_list
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == "__main__":
    # file path and make a list
    imgs_path =   '../../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path =  '../../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../../data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    
    for i,batch in enumerate(train_loader,0):
        images=batch['images'][0,:,:,:]
        indexes=batch['index']
        boxes=batch['bbox']
        gt,ground_coord=rpn_net.create_batch_truth(boxes,indexes,images.shape[-2:])


        # Flatten the ground truth and the anchors
        flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,rpn_net.get_anchors())
        
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)
        fig,ax=plt.subplots(1,1)
        ax.imshow(images.permute(1,2,0))
        
        find_cor=(flatten_gt==1).nonzero()
        find_neg=(flatten_gt==-1).nonzero()
             
        for elem in find_cor:
            coord=decoded_coord[elem,:].view(-1)
            anchor=flatten_anchors[elem,:].view(-1)

            col='r'
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)

        plt.show()
 
        if(i>0):
            break

 