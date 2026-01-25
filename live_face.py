import torch
import cv2
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import pyttsx3
import time
import threading

# =========================
# VOICE ENGINE (THREAD-SAFE)
# =========================
class VoiceAnnouncer:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 1.0)
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while self.running:
            with self.lock:
                if self.queue:
                    name = self.queue.pop(0)
                    print(f"🔊 Announcing: {name}")
                    self.engine.say(name)
                    self.engine.runAndWait()
            time.sleep(0.1)
    
    def announce(self, name):
        with self.lock:
            if name not in self.queue:  # Avoid duplicate announcements in queue
                self.queue.append(name)
    
    def stop(self):
        self.running = False

# =========================
# CONFIG
# =========================
DB_FILE = "face_db.pkl" 
THRESHOLD = 0.6
CAMERA_INDEX = 0
COOLDOWN_SECONDS = 5  # Wait time before re-announcing same person

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# MODELS
# =========================
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# LOAD DATABASE
# =========================
with open(DB_FILE, "rb") as f:
    face_db = pickle.load(f)
print(f"📦 Loaded {len(face_db)} registered people")

# =========================
# INITIALIZE VOICE
# =========================
announcer = VoiceAnnouncer()

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
print("🧠 Face recognition started | Press Q to quit")

# Track last announcement time for each person
last_announcement_time = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    boxes, probs = mtcnn.detect(pil_img)
    
    current_time = time.time()
    detected_names = []  # Track all detected names in this frame
    
    if boxes is not None and probs is not None:
        for box, prob in zip(boxes, probs):
            if prob < 0.9:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            face_img = pil_img.crop((x1, y1, x2, y2))
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = resnet(face_tensor)
            
            name = "Unknown"
            best_score = 0.0
            
            for known_name, emb_list in face_db.items():
                for known_emb in emb_list:
                    score = F.cosine_similarity(
                        emb,
                        known_emb.to(device)
                    ).item()
                    if score > best_score:
                        best_score = score
                        name = known_name
            
            if best_score < THRESHOLD:
                name = "Unknown"
            
            # Add to detected names (only known people)
            if name != "Unknown":
                detected_names.append(name)
            
            # Draw
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{name} ({best_score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
    
    # =========================
    # 🔊 VOICE ANNOUNCEMENT
    # =========================
    # Announce each detected person if cooldown has passed
    for name in set(detected_names):  # Use set to avoid duplicates in same frame
        last_time = last_announcement_time.get(name, 0)
        
        if current_time - last_time >= COOLDOWN_SECONDS:
            announcer.announce(name)
            last_announcement_time[name] = current_time
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
announcer.stop()