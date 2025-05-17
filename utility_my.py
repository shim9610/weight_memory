"""
ViTClassifier: Utility Module for Vision Transformer-based Classification.
Author: [SSY]
Created: [2025-05-06]
Description:
    Defines a custom classifier model built upon a pre-trained Vision Transformer (ViT) encoder.
    Extracts the [CLS] token from the transformer encoder output and applies additional
    fully-connected layers to perform classification tasks.
Usage:
    from utility_my import ViTClassifier
"""
import torch
from torch.optim import AdamW
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import AutoImageProcessor, ViTMAEModel
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from collections import OrderedDict
from tqdm import tqdm
class TrainViTClassifier():
    def __init__(self, device,mode="classifier",weight_ref="best_result.pt" ,num_classes=10,size=224,dataset='cifar10',batch_size=512,num_workers=8,keep_labels=None):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        if dataset=='cifar10':
            # Load CIFAR-10 dataset
            full_train = CIFAR10(root='./data', train=True, transform=self.transform, download=True)
            full_test = CIFAR10(root='./data', train=False, transform=self.transform, download=True)

                        # ---------- 라벨 필터링 ----------
            if keep_labels is not None:     # e.g. [0, 2, 8]  ← airplane, bird, ship
                tr_idx = [i for i, t in enumerate(full_train.targets)
                          if t in keep_labels]
                te_idx = [i for i, t in enumerate(full_test.targets)
                          if t in keep_labels]
                self.train_dataset = Subset(full_train, tr_idx)
                self.test_dataset  = Subset(full_test,  te_idx)
            else:
                self.train_dataset = full_train
                self.test_dataset = full_test
                
        else: 
            self.train_dataset = None
            self.test_dataset = None
        
        # Create DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        if mode=="classifier":
            # Load the ViT-MAE model and processor
            self.processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
            self.encoder = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
            #freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            # define the ViTClassifier
            self.model = ViTClassifier(self.encoder, num_classes=num_classes).to(device)
        elif mode == "encoder":
                        # Load the ViT-MAE model and processor
            self.processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
            self.encoder = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
            # define the ViTClassifier
            self.model = ViTClassifier(self.encoder,mode="freeze" ,num_classes=num_classes).to(device)
            load_head_only(self.model, weight_ref, prefix="classifier.")
            for p in self.model.classifier.parameters():       # head 완전 고정
                p.requires_grad = False
    def train(self,name='result',limit=10,max_epochs=100):
        device = self.device
        model = self.model
        processor = self.processor
        train_loader = self.train_loader
        test_loader = self.test_loader
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        model.train()
        # Early stopping parameters
        best_loss = float('inf')  # minimum loss for early stopping
        early_stop_counter=0
        # Training loop
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader):
                labels = labels.to(device)
                # convert images to PIL format
                images_list = [transforms.ToPILImage()(img) for img in images]
                # image processor
                inputs = processor(images=images_list, return_tensors="pt").to(device)
                pixel_values = inputs["pixel_values"]
                optimizer.zero_grad()
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation loop
            model.eval()  # Evaluation mode
            val_loss = 0
            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    labels = labels.to(device)
                    # convert images to PIL format
                    images_list = [transforms.ToPILImage()(img) for img in images]
                    # image processor
                    inputs = processor(images=images_list, return_tensors="pt").to(device)
                    pixel_values = inputs["pixel_values"]
                    logits = model(pixel_values)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
            val_loss_avg = val_loss / len(test_loader)
            print(f'Epoch {epoch+1},Train Loss: {avg_loss:.8f},Validation Loss: {val_loss_avg:.8f}')
            early_stop_counter+=1
            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                torch.save(model.state_dict(), 'best_'+name+'.pt')
                print(f"🚩 Best model saved (Epoch {epoch+1},Train Loss: {avg_loss:.8f},Validation Loss: {val_loss_avg:.8f})")
                early_stop_counter=0
            if early_stop_counter>limit:
                print(f"Early stopping at epoch {epoch+1} with validation loss: {val_loss_avg:.8f}")
                break
        # final model saving
        torch.save(model.state_dict(), name+'.pt')
    # -------------------------------------------------------------
    # XPU-optimized training loop (no PIL, bf16 autocast, non-blocking transfers)
    # -------------------------------------------------------------
    def train_xpu(self, name='result', patience=5, max_epochs=100):
        device, model, processor = self.device, self.model, self.processor
        train_loader, test_loader = self.train_loader, self.test_loader

        criterion  = nn.CrossEntropyLoss()
        optimizer  = AdamW(model.parameters(), lr=1e-4)

        best_val, patience_ctr = float('inf'), 0

        for epoch in range(max_epochs):
            # -------- train --------
            model.train()
            running = 0.0
            for imgs, lbls in tqdm(train_loader, desc=f'E{epoch+1}-train', leave=False):
                imgs  = imgs.to(device, non_blocking=True)
                lbls  = lbls.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "xpu"):
                    px = processor(images=imgs,
                            return_tensors="pt",
                            do_rescale=False,  # <-- 중복 스케일링 끔
                            do_resize=False,
                            do_center_crop=False)["pixel_values"].to(device, non_blocking=True)
                    logit = model(px)
                    loss  = criterion(logit, lbls)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()

            tr_loss = running / len(train_loader)

            # -------- validate --------
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for imgs, lbls in tqdm(test_loader, desc=f'E{epoch+1}-val', leave=False):
                    imgs = imgs.to(device, non_blocking=True)
                    lbls = lbls.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "xpu"):
                        px = processor(images=imgs,
                            return_tensors="pt",
                            do_rescale=False,  # <-- 중복 스케일링 끔
                            do_resize=False,
                            do_center_crop=False)["pixel_values"].to(device, non_blocking=True)
                        logit = model(px)
                        val_running += criterion(logit, lbls).item()

            val_loss = val_running / len(test_loader)
            print(f'Epoch {epoch+1}: train {tr_loss:.6f}  |  val {val_loss:.6f}')

            # -------- early-stopping / checkpoint --------
            if val_loss < best_val:
                best_val, patience_ctr = val_loss, 0
                torch.save(model.state_dict(), f'best_{name}.pt')
                print(f'🚩 best updated → {best_val:.6f}')
            else:
                patience_ctr += 1
                if patience_ctr > patience:
                    print(f'⏹ early stop at epoch {epoch+1}')
                    break

        torch.save(model.state_dict(), f'{name}.pt')


        
class ViTClassifier(nn.Module):
    def __init__(self, encoder,mode="classifier" ,num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, 256), # Intermediate linear layer
            nn.GELU(),                                  # Non-linear activation
            nn.Dropout(0.2),                            # Dropout for regularization    
            nn.Linear(256, num_classes)                 # Final classification layer    
        )
        if mode == "freeze":
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        # Forward pass through encoder to obtain feature embeddings
        encoder_outputs = self.encoder(pixel_values)  
        # Use [CLS] token embedding (first token) for classification          
        pooled_output = encoder_outputs.last_hidden_state[:, 0]
        # Compute logits through classification head
        logits = self.classifier(pooled_output)
        
        return logits
    
    
    
def load_head_only(model, ckpt_path, prefix="classifier."):
    """
    model        : ViTClassifier 인스턴스
    ckpt_path    : torch.save(...) 했던 전체 state-dict 경로
    prefix       : head 모듈 이름(기본 'classifier.')
    """
    full_sd = torch.load(ckpt_path, map_location="cpu")       # ① 전체 dict 로드
    if "state_dict" in full_sd:                               #  (배터리포함 세이브일 경우)
        full_sd = full_sd["state_dict"]

    # ② prefix 에 맞는 키만 추려서 접두사 잘라 줌
    head_sd = OrderedDict()
    for k, v in full_sd.items():
        if k.startswith("module."):           # DataParallel 저장본 처리
            k = k[len("module."):]
        if k.startswith(prefix):
            head_sd[k[len(prefix):]] = v      # 'classifier.fc.weight' → 'fc.weight'

    # ③ 분류기 서브모듈에 주입
    model.classifier.load_state_dict(head_sd, strict=True)
