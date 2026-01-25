import torch
import cv2
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms

NAME = "Shiv"   # 👈 your name here
DB_FILE = "face_db.pkl"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

cap = cv2.VideoCapture(0)
print("📸 Look at the camera and press S to save your face")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    face = mtcnn(pil_img)

    if face is not None:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face)

        cv2.putText(frame, "Press S to register face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('s') and face is not None:
        data = {NAME: embedding.cpu()}
        with open(DB_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Face registered as {NAME}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
