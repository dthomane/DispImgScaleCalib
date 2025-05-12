from PIL import Image

import torch
import torchvision.transforms as transforms

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


# Load model and preprocessing transform
class model_depthanything:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.kwargs = {}

        overwrite = {**self.kwargs, "pretrained_resource": checkpoint_path}
        config = get_config('zoedepth', "eval", 'kitti', **overwrite)

        self.model = build_model(config)
        self.model = self.model.to(self.device)

        self.model.eval()

    def predict(self, img):
        
        img = Image.fromarray(img)
        orig_size = img.size

        img = img.resize((518, 392))
        
        img = transforms.PILToTensor()(img).type(torch.FloatTensor)
        img = img.unsqueeze(0)

        img = img.cuda()
        pred = self.model(img, **self.kwargs)
        depth = pred['metric_depth'].detach().cpu()

        depth = torch.nn.functional.interpolate(depth, size=orig_size[::-1], mode="bicubic", align_corners=False).squeeze().numpy()

        return depth

