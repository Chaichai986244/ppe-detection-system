from ultralytics import YOLO
import cv2

model = YOLO('best_yolo11.pt')
img_path = '反光衣.jpg'

results = model(img_path, conf=0.25, iou=0.45)

vest_found = False
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = result.names[cls_id]
        print(f"  类别: {cls_name} (id={cls_id}), 置信度: {conf:.4f}")
        if cls_name == 'Reflective-Jacket':
            vest_found = True

if vest_found:
    print("\n✅ 模型成功检出了反光衣 (vest)!")
else:
    print("\n❌ 模型没有检出反光衣 (vest)")

result_img = results[0].plot()
cv2.imwrite('反光衣_result.jpg', result_img)
print(f"\n检测结果图已保存: 反光衣_result.jpg")