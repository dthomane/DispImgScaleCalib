import cv2

import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes


class model_internimage:
    def __init__(self, config, checkpoint, device='cuda'):

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(config, checkpoint=None, device=device)
        self.checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
        
        if 'CLASSES' in self.checkpoint.get('meta', {}):
            self.model.CLASSES = self.checkpoint['meta']['CLASSES']
        else:
            self.model.CLASSES = get_classes('cityscapes')

    def predict(self, img):

        orig_size = img.shape[::-1]

        img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_LINEAR)
        result = inference_segmentor(self.model, img)
        seg = np.array(result[0])
        seg = cv2.resize(seg, orig_size, interpolation = cv2.INTER_LINEAR)

        return seg
        