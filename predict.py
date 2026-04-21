import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DetrImageProcessor,
    RTDetrV2Config,
    RTDetrV2ForObjectDetection,
)


class TestDataset(Dataset):
    def __init__(self, img_dir, target_size=800):
        self.img_dir = img_dir
        self.target_size = target_size
        self.img_names = [
            f for f in os.listdir(img_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.resize_transform = A.LongestMaxSize(max_size=target_size)
        self.pad_transform = A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        orig_size = torch.tensor([h, w])

        # 1. 先做等比例縮放，並紀錄「補邊前」的真實大小
        image = self.resize_transform(image=image)['image']
        res_h, res_w = image.shape[:2]
        resized_size = torch.tensor([res_h, res_w])

        # 2. 再做補邊到 800x800
        image = self.pad_transform(image=image)['image']

        return image, img_name, orig_size, resized_size


def create_collate_fn(processor):
    def collate_fn(batch):
        images = [item[0] for item in batch]
        img_names = [item[1] for item in batch]
        orig_sizes = torch.stack([item[2] for item in batch])
        resized_sizes = torch.stack([item[3] for item in batch])

        # 圖片已經是完美的 800x800，處理器直接轉 Tensor 即可
        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_pad=True
        )

        return {
            "pixel_values": inputs['pixel_values'],
            "pixel_mask": (
                inputs['pixel_mask'] if 'pixel_mask' in inputs else None
            ),
            "orig_size": orig_sizes,
            "resized_size": resized_sizes,
            "img_name": img_names
        }
    return collate_fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 路徑請確認是否正確
    test_img_dir = ''
    model_path = ''
    output_json = ''
    batch_size = 24
    TARGET_MAX_SIZE = 800

    processor = DetrImageProcessor.from_pretrained(
        "PekingU/rtdetr_r50vd",
        do_resize=False,
        do_pad=True
    )

    config = RTDetrV2Config(
        use_pretrained_backbone=False,
        backbone="resnet50",
        backbone_kwargs={"out_indices": [2, 3, 4]},
        num_labels=11,
    )
    model = RTDetrV2ForObjectDetection(config)

    print(f"載入權重: {model_path}...")
    state_dict = torch.load(
        model_path, map_location='cpu', weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = TestDataset(test_img_dir, target_size=TARGET_MAX_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=create_collate_fn(processor)
    )

    predictions = []
    print(f"開始推論 {len(dataset)} 張圖片...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = (
                batch['pixel_mask'].to(device)
                if batch['pixel_mask'] is not None else None
            )
            orig_sizes = batch['orig_size'].to(device)
            resized_sizes = batch['resized_size'].to(device)
            img_names = batch['img_name']

            # 欺騙 processor，讓它認為原始圖片就是 800x800
            padded_sizes = torch.tensor(
                [[TARGET_MAX_SIZE, TARGET_MAX_SIZE]] * len(pixel_values)
            ).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask
                )

            results = processor.post_process_object_detection(
                outputs, target_sizes=padded_sizes, threshold=0.01
            )

            for i, result in enumerate(results):
                scores = result["scores"]
                boxes = result["boxes"]
                labels = result["labels"] + 1

                orig_h, orig_w = orig_sizes[i].tolist()
                res_h, res_w = resized_sizes[i].tolist()

                pad_y = (TARGET_MAX_SIZE - res_h) // 2
                pad_x = (TARGET_MAX_SIZE - res_w) // 2

                boxes[:, 0] -= pad_x
                boxes[:, 1] -= pad_y
                boxes[:, 2] -= pad_x
                boxes[:, 3] -= pad_y

                # 扣除邊界後，再按比例放大回原始解析度
                scale_x = orig_w / res_w
                scale_y = orig_h / res_h

                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y

                img_name = img_names[i]
                try:
                    img_id = int(os.path.splitext(img_name)[0])
                except ValueError:
                    img_id = batch_idx * batch_size + i

                if len(scores) > 0:
                    for score, label, box in zip(scores, labels, boxes):
                        x_min, y_min, x_max, y_max = box.tolist()
                        w = x_max - x_min
                        h = y_max - y_min

                        # 加入安全鎖：強制過濾掉偶發的「負數框」或是超出邊界太多的幽靈框
                        x_min = max(0.0, min(orig_w, x_min))
                        y_min = max(0.0, min(orig_h, y_min))
                        w = max(1.0, min(orig_w - x_min, w))
                        h = max(1.0, min(orig_h - y_min, h))

                        predictions.append({
                            "image_id": img_id,
                            "category_id": int(label.item()),
                            "bbox": [
                                round(float(x_min), 2),
                                round(float(y_min), 2),
                                round(float(w), 2),
                                round(float(h), 2)
                            ],
                            "score": round(float(score.item()), 6)
                        })

        if (batch_idx + 1) % 5 == 0:
            print(f"已處理 {(batch_idx + 1) * batch_size} 張圖片...")

    print(f"儲存結果至 {output_json}...")
    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=4)

    print("🎉 推論完成！")


if __name__ == "__main__":
    main()