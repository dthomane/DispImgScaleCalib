import torch

class model_midas:
    def __init__(self, model_type = "DPT_Large", device = "cuda"):
        self.model_type = model_type # or: "DPT_Hybrid", or: "MiDaS_small"
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def predict(self, img):

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False).squeeze()
        disp_map = prediction.squeeze().cpu().numpy()

        return disp_map