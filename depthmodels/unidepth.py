import os
import numpy as np

import torch
from unidepth.models import UniDepthV2


# Load model and preprocessing transform
class model_unidepth:
    def __init__(self, prediction='depth', device="cuda"):
        self.prediction=prediction
        
        type_ = "l"  # available types: s, b, l
        name = f"unidepth-v2-vit{type_}14"
        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

        self.model = self.model.to(device)

    def predict(self, img):

        img = np.array(img)
        rgb = torch.from_numpy(img)

        if rgb.size()[2] == 4:
            rgb = rgb[:,:,:3] # remove alpha channel
        
        rgb = rgb.permute(2, 0, 1) # C, H, W

        predictions = self.model.infer(rgb)

        if self.prediction == 'depth': # Depth
            depth = predictions["depth"][0,0].detach().cpu().numpy()
            return depth
            
        elif self.prediction == 'points':
            points = predictions["points"][0].detach().cpu().numpy().transpose(1,2,0)
            return points
        
        else:
            print(f'ERROR: Unidepth predictiontype must be depth or points but is {self.prediction}')
            assert False