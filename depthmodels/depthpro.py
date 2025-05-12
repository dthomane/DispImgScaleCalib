import os
import numpy as np

import torch
import depth_pro

# Load model and preprocessing transform
class model_depthpro:
    def __init__(self, checkpoint_path, f_px = 7267.95450880415, device="cuda"):
        self.f_px = torch.tensor(f_px)
        config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = os.path.join(checkpoint_path)
        self.model, self.transform = depth_pro.create_model_and_transforms(config=config, device=torch.device(device))
        self.model.eval()

    def predict(self, img):

        img = np.array(img)
        img = self.transform(img)

        prediction = self.model.infer(img, f_px=self.f_px)
        depth = prediction["depth"].detach().cpu().numpy() # Depth in [m].

        return depth

