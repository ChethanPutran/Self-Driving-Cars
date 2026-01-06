import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np

class PerceptionDLModels:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_yolo(self, version='yolov5s'):
        """
        Load YOLO model for object detection
        """
        import torch.hub
        # Load YOLOv5 from torch hub
        model = torch.hub.load('ultralytics/yolov5', version, pretrained=True)
        model = model.to(self.device)
        model.eval()
        return model
    
    def detect_objects_yolo(self, image, model, conf_threshold=0.5):
        """
        Detect objects using YOLO
        """
        # Run inference
        results = model(image)
        
        # Parse results
        detections = []
        if hasattr(results, 'xyxy'):
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf > conf_threshold:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': int(cls),
                        'class_name': results.names[int(cls)]
                    })
        
        return detections, results
    
    class UNet(nn.Module):
        """U-Net for semantic segmentation"""
        def __init__(self, n_channels=3, n_classes=3):  # Road, lane, background
            super().__init__()
            
            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            # Encoder
            self.enc1 = double_conv(n_channels, 64)
            self.enc2 = double_conv(64, 128)
            self.enc3 = double_conv(128, 256)
            self.enc4 = double_conv(256, 512)
            
            self.pool = nn.MaxPool2d(2)
            
            # Bottleneck
            self.bottleneck = double_conv(512, 1024)
            
            # Decoder
            self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = double_conv(1024, 512)
            
            self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = double_conv(512, 256)
            
            self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = double_conv(256, 128)
            
            self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = double_conv(128, 64)
            
            # Output
            self.conv_out = nn.Conv2d(64, n_classes, 1)
            
        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            
            # Bottleneck
            bottleneck = self.bottleneck(self.pool(enc4))
            
            # Decoder with skip connections
            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.dec4(dec4)
            
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec3(dec3)
            
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            
            return self.conv_out(dec1)
    
    def segment_road(self, image, model):
        """
        Segment road and lanes using U-Net
        """
        # Prepare image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        
        # Create colored segmentation map
        color_map = {
            0: [0, 0, 0],      # Background - black
            1: [255, 0, 0],    # Road - red
            2: [0, 255, 0]     # Lane - green
        }
        
        h, w = pred.shape
        seg_map = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            seg_map[pred == class_id] = color
        
        return seg_map, pred
    
    class Monodepth2(nn.Module):
        """Simplified Monodepth2 for self-supervised depth estimation"""
        def __init__(self):
            super().__init__()
            
            # Encoder (ResNet18 based)
            backbone = models.resnet18(pretrained=True)
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            
            # Decoder
            self.upconv4 = self._upconv(512, 256)
            self.upconv3 = self._upconv(256, 128)
            self.upconv2 = self._upconv(128, 64)
            self.upconv1 = self._upconv(64, 32)
            
            # Depth prediction heads
            self.disp4 = nn.Conv2d(256, 1, 3, padding=1)
            self.disp3 = nn.Conv2d(128, 1, 3, padding=1)
            self.disp2 = nn.Conv2d(64, 1, 3, padding=1)
            self.disp1 = nn.Conv2d(32, 1, 3, padding=1)
            
        def _upconv(self, in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            # Encoder
            features = []
            for layer in self.encoder:
                x = layer(x)
                features.append(x)
            
            # Decoder with skip connections
            x = self.upconv4(x)
            disp4 = torch.sigmoid(self.disp4(x))
            
            x = self.upconv3(x)
            disp3 = torch.sigmoid(self.disp3(x))
            
            x = self.upconv2(x)
            disp2 = torch.sigmoid(self.disp2(x))
            
            x = self.upconv1(x)
            disp1 = torch.sigmoid(self.disp1(x))
            
            return [disp1, disp2, disp3, disp4]
    
    def estimate_depth(self, image, model):
        """
        Estimate depth from single image
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            disparities = model(img_tensor)
            # Use finest disparity map
            disp = disparities[0].squeeze().cpu().numpy()
        
        # Convert disparity to depth (assuming camera parameters)
        # depth = 1.0 / (disp + 1e-6)
        depth = disp  # Simplified for demo
        
        # Normalize for visualization
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        return depth, depth_colormap
    
    def multi_task_learning(self, image, task='all'):
        """
        Multi-task learning for perception
        """
        # This would be a single model predicting:
        # 1. Object detection
        # 2. Semantic segmentation
        # 3. Depth estimation
        # 4. Motion prediction
        
        class MultiTaskPerception(nn.Module):
            def __init__(self):
                super().__init__()
                # Shared backbone
                backbone = models.resnet50(pretrained=True)
                self.shared = nn.Sequential(*list(backbone.children())[:-2])
                
                # Task-specific heads
                self.detection_head = nn.Sequential(
                    nn.Conv2d(2048, 1024, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 5 * 4)  # 5 objects * 4 coordinates
                )
                
                self.segmentation_head = nn.Sequential(
                    nn.Conv2d(2048, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 3, 1)  # 3 classes
                )
                
                self.depth_head = nn.Sequential(
                    nn.Conv2d(2048, 1024, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 1, 1)  # Depth map
                )
            
            def forward(self, x):
                shared_features = self.shared(x)
                
                detection = self.detection_head(shared_features)
                segmentation = self.segmentation_head(shared_features)
                depth = self.depth_head(shared_features)
                
                return detection, segmentation, depth
        
        # For demo, return placeholder
        return None