import datetime
import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR  # 🌟 核心修改 1: 改用 CosineAnnealingLR (餘弦退火) 讓學習率平滑下降
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DetrImageProcessor,
    RTDetrV2Config,
    RTDetrV2ForObjectDetection,
)


class DigitDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        h, w = image_np.shape[:2]

        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            x_min, y_min, bbox_w, bbox_h = ann['bbox']

            x_min = max(0.0, float(x_min))
            y_min = max(0.0, float(y_min))
            x_max = min(float(w), x_min + float(bbox_w))
            y_max = min(float(h), y_min + float(bbox_h))

            bbox_w = max(0.0, x_max - x_min)
            bbox_h = max(0.0, y_max - y_min)

            if bbox_w > 0 and bbox_h > 0:
                boxes.append([x_min, y_min, bbox_w, bbox_h])
                labels.append(ann['category_id'])

        if self.transform:
            transformed = self.transform(
                image=image_np,
                bboxes=boxes,
                class_labels=labels
            )
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        formatted_annotations = []
        for box, label in zip(boxes, labels):
            formatted_annotations.append({
                "bbox": box,
                "category_id": label,
                "area": box[2] * box[3],
                "iscrowd": 0
            })

        target = {
            "image_id": img_id,
            "annotations": formatted_annotations
        }

        return image_np, target


processor = DetrImageProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd",
    do_resize=False,
    do_pad=True
)


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    batch_dict = processor(
        images=images, annotations=targets, return_tensors="pt"
    )
    return batch_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的裝置: {device}")

    # 🌟 核心修改 2: 提升解析度到 800，幫助小物件特徵提取
    TARGET_MAX_SIZE = 800

    # 🌟 核心修改 3: 優化 Data Augmentation (資料擴增) 與防呆機制
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=TARGET_MAX_SIZE),
            A.PadIfNeeded(
                min_height=TARGET_MAX_SIZE,
                min_width=TARGET_MAX_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
            # 加入輕微的平移、縮放與旋轉，增加泛化能力 (限制在10度內避免數字變形失真)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
            ),
            A.GaussianBlur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_area=0.1  # 移除 min_visibility，保留極小 min_area=0.1 防呆
        )
    )

    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=TARGET_MAX_SIZE),
            A.PadIfNeeded(
                min_height=TARGET_MAX_SIZE,
                min_width=TARGET_MAX_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
        ],
        bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_area=0.1
        )
    )

    train_dir = ''
    train_json = ''
    val_dir = ''
    val_json = ''

    train_dataset = DigitDataset(
        img_dir=train_dir,
        annotation_file=train_json,
        transform=train_transform
    )
    val_dataset = DigitDataset(
        img_dir=val_dir,
        annotation_file=val_json,
        transform=val_transform
    )

    BATCH_SIZE = 7
    ACCUMULATION_STEPS = 4

    # 🌟 核心修改 4: 加入 persistent_workers=True 讓 worker 常駐，消除 Epoch 間的等待卡頓
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    config = RTDetrV2Config(
        use_pretrained_backbone=True,
        backbone="resnet50",
        backbone_kwargs={"out_indices": [2, 3, 4]},
        num_labels=11,
    )
    model = RTDetrV2ForObjectDetection(config)
    model.to(device)

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": 1e-5
        },
    ]
    optimizer = AdamW(param_dicts, lr=2e-4, weight_decay=1e-4)

    num_epochs = 80

    # 🌟 核心修改 5: 實作 CosineAnnealingLR 排程器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # 讓學習率在 80 個 Epoch 內平滑降至最低點
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda')

    print("🚀 開始 RT-DETRv2 終極加速訓練 (800 解析度 + 餘弦退火優化版)...")

    for epoch in range(num_epochs):
        # ========================
        #    Training Phase
        # ========================
        model.train()
        train_total_loss = 0.0
        train_total_batches = len(train_dataloader)

        print(f"\n========== Epoch [{epoch+1}/{num_epochs}] 開始 ==========")

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(
                device, non_blocking=True
            )
            pixel_mask = (
                batch["pixel_mask"].to(device, non_blocking=True)
                if "pixel_mask" in batch else None
            )
            labels = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()}
                for t in batch["labels"]
            ]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels
                )
                loss = outputs.loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            train_total_loss += outputs.loss.item()

            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0 or
                    (batch_idx + 1) == train_total_batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=0.1
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            if ((batch_idx + 1) % 100 == 0 or
                    (batch_idx + 1) == train_total_batches):
                current_lr = optimizer.param_groups[0]['lr']
                current_time = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(
                    f"[{current_time}] Train | "
                    f"Epoch: [{epoch+1}/{num_epochs}] | "
                    f"Batch: [{batch_idx+1}/{train_total_batches}] | "
                    f"Loss: {outputs.loss.item():.4f} | LR: {current_lr:.6f}"
                )

        train_avg_loss = train_total_loss / train_total_batches

        # ========================
        #    Validation Phase
        # ========================
        model.eval()
        val_total_loss = 0.0
        val_total_batches = len(val_dataloader)

        print(f"--- 開始驗證 Epoch {epoch+1} ---")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                pixel_values = batch["pixel_values"].to(
                    device, non_blocking=True
                )
                pixel_mask = (
                    batch["pixel_mask"].to(device, non_blocking=True)
                    if "pixel_mask" in batch else None
                )
                labels = [
                    {k: v.to(device, non_blocking=True) for k, v in t.items()}
                    for t in batch["labels"]
                ]

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels
                    )
                    val_total_loss += outputs.loss.item()

        val_avg_loss = val_total_loss / val_total_batches

        print(
            f"--> Epoch {epoch+1} 總結 | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Val Loss: {val_avg_loss:.4f}"
        )

        # 🌟 核心修改 6: 更新 CosineAnnealingLR (無須傳入 val_loss，每個 Epoch 固定踩一步)
        scheduler.step()

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), "hw2_10_best.pth")
            print("=> 破紀錄！在驗證集找到更好的模型，權重已儲存 (hw2_10_best.pth)！")
        else:
            print("=> 驗證集 Loss 沒有創新低，繼續訓練！")

        if (epoch + 1) % 5 == 0:
            fixed_save_path = f"hw2_10_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), fixed_save_path)
            print(
                f"=> [固定儲存] 達到 5 的倍數 Epoch，"
                f"模型已額外儲存為 ({fixed_save_path})"
            )


if __name__ == "__main__":
    main()