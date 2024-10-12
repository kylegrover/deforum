import os
import torch
import cv2
from .depth_pro import depth_pro
from PIL import Image
import torchvision.transforms as T

from ... import logger
from ...utils.file_dl_util import download_file_with_checksum

class AppleDepthPro:
    def __init__(self, models_path, device, half_precision=True):
        self.device = device
        self.half_precision = half_precision
        
        self.model_filename = 'apple_depthpro_model.pt'
        self.model_checksum = '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
        self.model_url = 'https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt?download=true'
        
        self.resize_px = 512  # Adjust based on Apple DepthPro requirements
        
        self.transform = T.Compose([
            T.Resize((self.resize_px, self.resize_px)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        download_file_with_checksum(url=self.model_url, expected_checksum=self.model_checksum, 
                                    dest_folder=models_path, dest_filename=self.model_filename)
        
        self.load_model(models_path, self.model_filename)
        if half_precision:
            self.model = self.model.half()

    def load_model(self, models_path, model_filename):
        model_file = os.path.join(models_path, model_filename)
        logger.info(f"Loading Apple DepthPro model from {model_filename}...")
        self.model, _ = depth_pro.create_model_and_transforms(model_path=model_file)
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval().to(self.device)

    def predict(self, prev_img_cv2, half_precision):
        print(f"Input image shape: {prev_img_cv2.shape}")
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2, cv2.COLOR_BGR2RGB))
        img_input = self.transform(img_pil).unsqueeze(0).to(self.device)
        print(f"Transformed input shape: {img_input.shape}")
        
        if self.device != "cpu":
            img_input = img_input.to(memory_format=torch.channels_last)
        if half_precision:
            img_input = img_input.half()
        
        with torch.no_grad():
            prediction = self.model.infer(img_input) # optional: f_px = focal length
            depth = prediction["depth"]  # Depth in [m]
            print(f"Raw depth shape: {depth.shape}")
            
            # Ensure depth has the correct number of dimensions
            if depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif depth.dim() == 3:
                depth = depth.unsqueeze(0)  # Add batch dimension
            print(f"Adjusted depth shape: {depth.shape}")
            
            depth_tensor = torch.nn.functional.interpolate(
                depth,
                size=prev_img_cv2.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).to(self.device)
            print(f"Final depth tensor shape: {depth_tensor.shape}")
        
        return depth_tensor.detach().clone()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device, memory_format=torch.channels_last if device == torch.device("cuda") else None)

    def delete(self):
        del self.model
        torch.cuda.empty_cache()