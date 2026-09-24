"""
Faster R-CNN 快速训练演示 — 小数据集 + 少量 Epoch
用于生成训练曲线图和对比分析素材
"""
import os, sys, time, json, random
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────
BASE = r'd:\AAA夹心的代码\AAAAAAAlidongyuan\Projects1'
DATASET_ROOT = os.path.join(BASE, 'archive')
OUTPUT_DIR = os.path.join(BASE, 'rcnn_training_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
BATCH_SIZE = 4
MAX_SAMPLES = 200      # 只用200张图快速跑
NUM_CLASSES = 6        # 6个类别（不含背景）
CLASS_NAMES = ['boots','gloves','goggles','helmet','person','vest']
CN_NAMES = ['安全靴','手套','护目镜','安全帽','人员','反光背心']
LR = 0.005

print(f'设备: {DEVICE}')
print(f'类别数: {NUM_CLASSES}')
print(f'Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | 样本: {MAX_SAMPLES}')

# ── 数据加载 ──────────────────────────────────────────
def load_dataset(split='train', max_samples=None):
    """加载YOLO格式数据集，限制样本数"""
    img_dir = os.path.join(DATASET_ROOT, split, 'images')
    lbl_dir = os.path.join(DATASET_ROOT, split, 'labels')
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if max_samples and len(files) > max_samples:
        files = random.sample(files, max_samples)

    images, targets = [], []
    for fname in files:
        # 加载图像
        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # 加载YOLO标注
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    # YOLO归一化坐标 → 像素坐标 [x1,y1,x2,y2]
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        if not boxes:  # 跳过无标注图像
            boxes.append([0, 0, 1, 1])
            labels.append(0)

        images.append(torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0)
        targets.append({
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([hash(fname) % 100000]),
        })

    return images, targets

print('加载训练集...')
train_imgs, train_tgts = load_dataset('train', MAX_SAMPLES)
print(f'训练集: {len(train_imgs)} 张图像')

print('加载验证集...')
val_imgs, val_tgts = load_dataset('valid', MAX_SAMPLES // 2)
print(f'验证集: {len(val_imgs)} 张图像')

# ── 构建模型 ──────────────────────────────────────────
backbone = resnet_fpn_backbone('resnet50', weights=None)
model = FasterRCNN(
    backbone=backbone,
    num_classes=NUM_CLASSES + 1,  # +1 for background
    rpn_anchor_generator=AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    ),
)
model.to(DEVICE)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

print(f'模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# ── 训练循环 ──────────────────────────────────────────
history = {'epoch': [], 'train_loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    indices = list(range(len(train_imgs)))
    random.shuffle(indices)

    t0 = time.time()
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        images = [train_imgs[j].to(DEVICE) for j in batch_idx]
        targets = [{k: v.to(DEVICE) for k, v in train_tgts[j].items()} for j in batch_idx]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    lr_scheduler.step()
    avg_train_loss = epoch_loss / (len(indices) / BATCH_SIZE)

    # ── 验证 ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_imgs), BATCH_SIZE):
            batch_idx = list(range(i, min(i+BATCH_SIZE, len(val_imgs))))
            images = [val_imgs[j].to(DEVICE) for j in batch_idx]
            targets = [{k: v.to(DEVICE) for k, v in val_tgts[j].items()} for j in batch_idx]
            loss_dict = model(images, targets)
            val_loss += sum(loss.item() for loss in loss_dict.values())
    avg_val_loss = val_loss / (len(val_imgs) / BATCH_SIZE)

    elapsed = time.time() - t0
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)

    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.0f}s')

    # 保存checkpoint
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'rcnn_epoch_{epoch+1}.pth'))

# ── 绘制训练曲线 ──────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(history['epoch'], history['train_loss'], 'o-', color='#1565C0', lw=2, markersize=8, label='训练损失 (Train Loss)')
ax.plot(history['epoch'], history['val_loss'], 's-', color='#FF5722', lw=2, markersize=8, label='验证损失 (Val Loss)')
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Faster R-CNN 训练收敛曲线 (200样本 × 5 Epochs)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(history['epoch'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rcnn_training_curve.png'), dpi=200, facecolor='white')
plt.close()
print(f'训练曲线已保存: rcnn_training_curve.png')

# ── 保存训练历史 ──────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

# ── 最终模型保存 ──────────────────────────────────────
final_path = os.path.join(OUTPUT_DIR, 'rcnn_demo_final.pth')
torch.save(model.state_dict(), final_path)
print(f'最终模型: {final_path}')
print(f'输出目录: {OUTPUT_DIR}')
print('训练完成！')
