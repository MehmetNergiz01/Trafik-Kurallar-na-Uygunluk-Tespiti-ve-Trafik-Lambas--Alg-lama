import cv2
import numpy as np

# Trafik lambasının rengini belirlemek için fonksiyon
def detect_traffic_light_color(traffic_light_roi):
    # Trafik lambasının bölgesindeki renklerin ortalamasını al
    average_color = np.mean(traffic_light_roi, axis=(0, 1))

    # Ortalama renk tonlarına göre trafik lambasının rengini belirle
    # Bu sadece bir örnek, gerçek uygulamada daha sofistike bir yöntem kullanılabilir
    if average_color[0] > 100 and average_color[1] < 100 and average_color[2] < 100:
        return "Red"
    elif average_color[0] < 100 and average_color[1] > 100 and average_color[2] < 100:
        return "Green"
    else:
        return "Unknown"

# YOLO modelini yükle
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# getUnconnectedOutLayers() yönteminin döndüğü indisleri bir liste içine alın
output_layer_indices = net.getUnconnectedOutLayers()
if not isinstance(output_layer_indices, list):
    output_layer_indices = [output_layer_indices]

output_layers = [layer_names[i[0] - 1] for i in output_layer_indices]

# Giriş görüntüsünü yükle
image = cv2.imread("2115.jpg")

# Giriş görüntüsünün yüklendiğini kontrol et
if image is None:
    print("Giriş görüntüsü yüklenemedi!")
    exit()

height, width, _ = image.shape

# Giriş görüntüsünü YOLO için hazırla
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Forward pass yaparak nesneleri algıla
outs = net.forward(net.getUnconnectedOutLayersNames())

# Algılanan nesneleri işle
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Güvenilirlik eşiği
            # Nesnenin orta noktası ve boyutları
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Dikdörtgen koordinatları
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression uygula
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Trafik lambalarını ve nesneleri işle
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (255, 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        # Trafik lambası bölgesini belirle
        if label == "traffic light":
            traffic_light_roi = image[y:y + h, x:x + w]

            # Trafik lambasının rengini belirle
            traffic_light_color = detect_traffic_light_color(traffic_light_roi)

            # Trafik lambasının rengini ekranda göster
            cv2.rectangle(image, (x, y - 20), (x + 100, y), (0, 255, 255), -1)
            cv2.putText(image, "Traffic Light Color: " + traffic_light_color, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Kırmızı ışıkta bekleyen araçları kontrol et
            if traffic_light_color == "Red":
                for j in range(len(boxes)):
                    if j in indexes and (classes[class_ids[j]] == "car" or classes[class_ids[j]] == "person"):
                        x_obj, y_obj, w_obj, h_obj = boxes[j]
                        # Kırmızı ışıkta bekleyen araçları yeşil göster
                        cv2.rectangle(image, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), (0, 255, 0), 2)
                        cv2.putText(image, "Follows Traffic Rule!", (x_obj, y_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Görüntünün genişliğini artır
new_width = 1250  # Yeni genişlik değeri
scale_ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * scale_ratio)
image = cv2.resize(image, (new_width, new_height))

# Sonucu göster
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
