from PIL import Image

import torch
from infer import InferenceHelper


class model_adabins:
    def __init__(self, checkpoint_path, device="cuda"):
        self.inference = InferenceHelper(pretrained_path=checkpoint_path, dataset=self.dataset, device=device)

    def predict(self, img):
        img = Image.fromarray(img)
        _, predicted_depth = self.inference.predict_pil(img)
        depth = torch.nn.functional.interpolate(torch.from_numpy(predicted_depth), size=img.size[::-1], mode="bicubic", align_corners=False).squeeze().cpu().numpy()

        return depth