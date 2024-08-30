import pickle
from flask import Flask, render_template, request
import os
from random import random
import cv2
import torch, json

# Detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load(
    "/home/minhthanh/directory_env/object_tracking/yolov5",
    "custom",
    path="/home/minhthanh/directory_env/object_tracking/yolov5/weights/best.pt",
    source="local"  # Local repo
)
model.conf = 0.25
model.max_det = 1000
model.eval()

def infer(frame):

    results = model(frame)
    data = results.pandas().xyxy[0].to_json(orient="records")
    json_data = json.loads(data)

    if not json_data:
        return frame, 0

    for record in json_data:
        confidence = round(record['confidence'], 2)
        class_id = record['class']
        left = int(record['xmin'])
        top = int(record['ymin'])
        right = int(record['xmax'])
        bottom = int(record['ymax'])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    return frame, len(json_data)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
         try:
            image = request.files['file']
            if image:
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                frame = cv2.imread(path_to_save)

                detection_frame, ndet = infer(frame)

                if ndet!=0:
                    cv2.imwrite(path_to_save, detection_frame)

                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = ndet)
                else:
                    return render_template('index.html', msg='Khong nhan dien duoc vat the')
            else:
                return render_template('index.html', msg='Hay chon file de tai len')

         except Exception as ex:
            print(ex)
            return render_template('index.html', msg='Khong nhan dien duoc vat the')

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)