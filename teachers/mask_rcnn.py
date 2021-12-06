import torch
import torchvision

MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
MODEL.eval()
MODEL.cuda()


def images_to_masks(images):
    """
    Given a list of images, returns a list of masks.
    """
    results = MODEL(images)

    image_segs = []
    for image, result in zip(images, results):
        image_seg = torch.zeros((5, *image.shape[1:]))

        n_boxes = len(result['boxes'])
        labels_numpy = result['labels'].cpu().detach().numpy()

        masks_numpy_raw = result['masks'].cpu().detach().numpy()
        masks_numpy = result['masks'].cpu().detach().numpy() > 0.1
        for i in range(n_boxes-1, -1, -1):
            if labels_numpy[i] < 5:
                image_seg[labels_numpy[i]][masks_numpy[i][0]] += masks_numpy_raw[i][0][masks_numpy[i][0]]
        image_segs.append(image_seg)
    return torch.stack(image_segs, dim=0)


def generate_and_store_masks(root_dir):
    """
    Generates and stores all masks
    """
    pass

    



