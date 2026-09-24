# 基于深度学习的施工安全防护装备（PPE）检测系统

实时视频监控场景下的安全防护装备违规检测系统：基于 RT-DETR / Faster R-CNN 双模型对安全帽、反光衣等防护装备进行检测，结合目标跟踪与时序投票抑制误报，内置多级报警规则、告警图片留档、CSV 日志与钉钉通知，并提供 Web 仪表盘实时查看。

## 功能特性

- **实时检测**：Flask + Socket.IO 推流，浏览器端实时查看检测画面与标注结果
- **双模型支持**：RT-DETR（Ultralytics）与 Faster R-CNN（TorchVision）可在界面中切换对比
- **稳定性优化**：目标跟踪（Tracker）+ 时序投票（Temporal Voter）+ 防抖（Debouncer）降低单帧误检
- **装备关联**：将防护装备关联到具体人员，判断"人—装备"违规组合（如未戴安全帽）
- **多级报警**：按违规类型与持续时间分级触发，告警截图自动保存至 `alarm_img/`，记录写入 `alarm_log.csv`
- **消息通知**：违规事件推送钉钉群机器人
- **Web 界面**：监控页、数据仪表盘、训练信息页

## 技术栈

| 类别 | 组件 |
|------|------|
| 后端 | Python、Flask、Flask-SocketIO、Flask-CORS |
| 检测模型 | Ultralytics RT-DETR、TorchVision Faster R-CNN |
| 图像处理 | OpenCV、Pillow、NumPy |
| 前端 | Jinja2 模板、原生 JS、CSS |

## 目录结构

```
├── app.py                  # Flask 主应用（路由 + Socket.IO 实时推流）
├── config.py               # 全局配置（模型注册、阈值、检测类别等）
├── detection/              # 检测引擎
│   ├── engine.py           #   检测引擎入口（模型加载与推理调度）
│   ├── faster_rcnn.py      #   Faster R-CNN 推理封装
│   ├── preprocessor.py     #   图像预处理（缩放、CLAHE 等）
│   ├── tracker.py          #   目标跟踪
│   ├── temporal_voter.py   #   时序投票抑制误报
│   ├── debouncer.py        #   报警防抖
│   ├── ppe_association.py  #   人—装备关联与违规判定
│   └── annotator.py        #   结果绘制
├── alarm/                  # 报警模块（规则引擎 + 日志）
├── notification/           # 钉钉通知
├── templates/              # 前端页面（监控 / 仪表盘 / 训练信息）
├── static/                 # 前端静态资源（JS / CSS）
├── notebooks/              # 训练与数据处理 notebook
│   ├── train_ppe.ipynb             # RT-DETR 主训练流程
│   ├── train_yolov8l_kaggle.ipynb  # Kaggle 上的 YOLOv8L 训练
│   ├── train_multi_dataset_kaggle.ipynb
│   ├── rcnn_training_demo.ipynb    # Faster R-CNN 训练示例
│   ├── augment_dataset.ipynb       # 数据增强
│   └── flowchart_notebook.ipynb    # 论文/汇报图表绘制
├── train_rcnn_demo.py      # Faster R-CNN 训练脚本
├── test_vest.py            # 反光衣检测快速测试脚本
├── 方法总结.md              # 预处理 / 增强 / 训练策略方法沉淀
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> GPU 环境请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本安装 `torch` / `torchvision`。

### 2. 准备模型权重

仓库不包含模型权重与数据集（体积过大）。两种方式获取：

- **自行训练**：参考 `notebooks/` 下的训练 notebook（含数据增强、多数据集训练流程），训练完成后将 RT-DETR 权重放到项目根目录并命名为 `best_rfdter.pt`；
- **已有权重**：直接将 `best_rfdter.pt`（RT-DETR）和 `faster_rcnn_epoch_5.pth`（Faster R-CNN）放到项目根目录。

权重路径、类别名等均在 `config.py` 的 `MODELS` 中注册，可按需修改。

### 3. 启动

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5000`（以启动日志实际端口为准），上传视频或图片即可开始检测；监控页可切换模型、调整置信度阈值与检测类别。

### 4. 钉钉通知（可选）

在 `notification/dingtalk.py` 中配置机器人 Webhook 与加签密钥即可启用违规推送。

## 说明

- 数据集（Roboflow 导出的 PPE 数据集、FIRC 数据集等）、模型权重、训练日志均不入库，详见 `.gitignore`；
- `alarm_img/`、`alarm_log.csv`、`uploads/`、`outputs/` 为运行时自动生成的目录，首次运行会自动创建。
