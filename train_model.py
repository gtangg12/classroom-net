from torch.utils.data.dataloader import random_split, DataLoader
from classroomnet.classroomnet import create_classroom_net
from datalake.datalake import Datalake
from teachers.spvnas import get_projected_features_from_point_clouds
import cv2 
import time
import numpy as np
import torch    
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle

# torch.cuda.set_enabled_lms(True)

def draw_bounding_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        image = cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), (255, 0, 0), 2)
    return image


def imshow(name, image, enc='RGB'):
    image = image.astype('float32')
    image /= np.max(image)
    if enc == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, image)
    cv2.waitKey(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = create_classroom_net(2, 96, [(0, 1, 0, 1), (0, 1, 0, 1)], [76, 5], [48, 48, 48], 64, 10)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=8)

data = Datalake(50000, ['image', 'bounding_boxes', 'object_classes', 'object_depths', 'object_class_mask', 'image_point_cloud_map_unscaled'], 'datalake/data')
# print(data[0])

trainset, valset = random_split(data, (48000, 2000))

trainloader = DataLoader(trainset, batch_size=8, collate_fn=Datalake.collate_fn, shuffle=True)
testloader = DataLoader(valset, batch_size=8, collate_fn=Datalake.collate_fn, shuffle=False)

num_epochs = 2

num_batches = 0
train_losses = []
val_losses = []

for epochs in range(num_epochs):
    try:
        num_batches = 0
        for data_batch in trainloader:
            
            num_batches += 1

            data_dict = [x for x, _, _ in data_batch]
            data_instance = {
                'image': torch.stack([torch.tensor(x['image']) for x in data_dict], dim=0),
                'bounding_boxes': [x['bounding_boxes'] for x in data_dict],
                'object_classes': [x['object_classes'] for x in data_dict],
                'object_depths': [x['object_depths'] for x in data_dict],
                'object_class_mask': torch.stack([torch.tensor(x['object_class_mask']) for x in data_dict], dim=0),
                'image_point_cloud_map_unscaled': [x['image_point_cloud_map_unscaled'] for x in data_dict],
            }

            

            st = time.time()
            optimizer.zero_grad()

            image = data_instance['image']
            bounding_boxes = data_instance['bounding_boxes']
            object_classes = data_instance['object_classes']
            object_depths = data_instance['object_depths']
            mask = data_instance['object_class_mask']
            pc = data_instance['image_point_cloud_map_unscaled']

            # draw_bounding_boxes(image, bounding_boxes)
            # imshow('Image', image)
            # print(object_depths)
            # print(object_classes)

            #print(bounding_boxes)

            image_reshape = torch.permute(image, (0, 3, 1, 2)) / 256
            image_reshape.to(device)

            #print(image_reshape)

            targets = []

            for idx, bbox_sample in enumerate(bounding_boxes):
                bbox = torch.tensor(bbox_sample)
                x1 = bbox[:, 0:1]
                x2 = bbox[:, 1:2]
                y1 = bbox[:, 2:3]
                y2 = bbox[:, 3:4]
                bbox = torch.cat([x1, y1, x2, y2], dim=1).to(device)

                labels = torch.tensor(object_classes[idx]).to(device)
                depths = torch.tensor(object_depths[idx]).to(device)

                #print(bbox, labels, depths)

                keep_idxs = (x2[:, 0]-x1[:, 0]>4) & (y2[:, 0]-y1[:, 0]>4)
                bbox, labels, depths = bbox[keep_idxs], labels[keep_idxs], depths[keep_idxs]
                
                # if (x2 - x1 > 4) and (y2 - y1 > 4):
                targets.append({'boxes': bbox, 'labels': labels, 'depths': depths})

            # model.eval()

            print(time.time() - st, 'time preprocessing')
            st = time.time()

            #print(image_reshape.shape, targets[0]['boxes'].shape, targets[0]['labels'].shape, targets[0]['depths'].shape)
            l, z = model(image_reshape, targets)
            #print('pc shape', pc.shape)
            print(time.time() - st, 'time student')
            st = time.time()

            pc = [torch.flip(torch.tensor(pc_), (1,)) for pc_ in pc]
            feats_3d = get_projected_features_from_point_clouds(pc).detach()
            
            print(time.time() - st, 'time 3d')
            st = time.time() 

            distill_loss_3d = F.mse_loss(z[0], feats_3d)
            distill_loss_mask = F.mse_loss(z[1], mask.cuda().detach())
            print('distill_loss_3d', distill_loss_3d)
            print('distill_loss_mask', distill_loss_mask)
            total_loss = distill_loss_3d + distill_loss_mask
            for k, v in l.items():
                print(k, v)
                total_loss += v
            
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

            print(time.time() - st, 'time optimize')

            train_losses.append((distill_loss_3d.item(), distill_loss_mask.item(), total_loss.item()))


            if num_batches%499 == 1:
                torch.save(model.state_dict(), f'last_model{num_batches}.pt')

                for data_batch in testloader:

                    data_dict = [x for x, _, _ in data_batch]
                    data_instance = {
                        'image': torch.stack([torch.tensor(x['image']) for x in data_dict], dim=0),
                        'bounding_boxes': [x['bounding_boxes'] for x in data_dict],
                        'object_classes': [x['object_classes'] for x in data_dict],
                        'object_depths': [x['object_depths'] for x in data_dict],
                        'object_class_mask': torch.stack([torch.tensor(x['object_class_mask']) for x in data_dict], dim=0),
                        'image_point_cloud_map_unscaled': [x['image_point_cloud_map_unscaled'] for x in data_dict],
                    }

                    
                    optimizer.zero_grad()

                    image = data_instance['image']
                    bounding_boxes = data_instance['bounding_boxes']
                    object_classes = data_instance['object_classes']
                    object_depths = data_instance['object_depths']
                    mask = data_instance['object_class_mask']
                    pc = data_instance['image_point_cloud_map_unscaled']

                    # draw_bounding_boxes(image, bounding_boxes)
                    # imshow('Image', image)
                    # print(object_depths)
                    # print(object_classes)

                    #print(bounding_boxes)

                    image_reshape = torch.permute(image, (0, 3, 1, 2)) / 256
                    image_reshape.to(device)

                    #print(image_reshape)

                    targets = []

                    for idx, bbox_sample in enumerate(bounding_boxes):
                        bbox = torch.tensor(bbox_sample)
                        x1 = bbox[:, 0:1]
                        x2 = bbox[:, 1:2]
                        y1 = bbox[:, 2:3]
                        y2 = bbox[:, 3:4]
                        bbox = torch.cat([x1, y1, x2, y2], dim=1).to(device)

                        labels = torch.tensor(object_classes[idx]).to(device)
                        depths = torch.tensor(object_depths[idx]).to(device)

                        #print(bbox, labels, depths)

                        keep_idxs = (x2[:, 0]-x1[:, 0]>4) & (y2[:, 0]-y1[:, 0]>4)
                        bbox, labels, depths = bbox[keep_idxs], labels[keep_idxs], depths[keep_idxs]
                        
                        # if (x2 - x1 > 4) and (y2 - y1 > 4):
                        targets.append({'boxes': bbox, 'labels': labels, 'depths': depths})

                    # model.eval()

                    #print(image_reshape.shape, targets[0]['boxes'].shape, targets[0]['labels'].shape, targets[0]['depths'].shape)
                    l, z = model(image_reshape, targets)
                    #print('pc shape', pc.shape)

                    pc = [torch.flip(torch.tensor(pc_), (1,)) for pc_ in pc]
                    feats_3d = get_projected_features_from_point_clouds(pc).detach()

                    distill_loss_3d = F.mse_loss(z[0], feats_3d)
                    distill_loss_mask = F.mse_loss(z[1], mask.cuda().detach())
                    print('distill_loss_3d', distill_loss_3d)
                    print('distill_loss_mask', distill_loss_mask)
                    total_loss = distill_loss_3d + distill_loss_mask
                    for k, v in l.items():
                        print(k, v)
                        total_loss += v
                    
                    val_losses.append((distill_loss_3d.item(), distill_loss_mask.item(), total_loss.item()))
    except KeyboardInterrupt:
        break


torch.save(model.state_dict(), 'last_model_overall.pt')

with open('train_losses.pkl', 'wb+') as f:
    pickle.dump(train_losses, f)

with open('val_losses.pkl', 'wb+') as f:
    pickle.dump(val_losses, f)
