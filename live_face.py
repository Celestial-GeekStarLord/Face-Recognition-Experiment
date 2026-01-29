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
            if name not in self.queue:
                self.queue.append(name)
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

# =========================
# CONFIG
# =========================
DB_FILE = "face_db.pkl" 
THRESHOLD = 0.6
CAMERA_INDEX = 0
COOLDOWN_SECONDS = 3  # Cooldown after announcement
ABSENCE_THRESHOLD = 2.0  # How long person must be gone before re-announcing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# MODELS
# =========================
mtcnn = MTCNN(
    keep_all=True, 
    device=device,
    post_process=False
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# LOAD DATABASE
# =========================
try:
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
    print(f"📦 Loaded {len(face_db)} registered people")
except FileNotFoundError:
    print(f"❌ Database file '{DB_FILE}' not found. Please register faces first.")
    exit(1)

# =========================
# INITIALIZE VOICE
# =========================
announcer = VoiceAnnouncer()

# =========================
# TRACKING STATE
# =========================
# Track when each person was last SEEN (not announced)
last_seen_time = {}
# Track when each person was last ANNOUNCED
last_announcement_time = {}

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("🧠 Face recognition started | Press Q to quit")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Process every frame for detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        try:
            boxes, probs = mtcnn.detect(pil_img)
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            boxes, probs = None, None
        
        detected_names_this_frame = set()
        
        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
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
                
                # Update last seen time for known people
                if name != "Unknown":
                    detected_names_this_frame.add(name)
                    last_seen_time[name] = current_time
                
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
        # 🔊 VOICE ANNOUNCEMENT LOGIC
        # =========================
        # For each detected person, check if we should announce
        for name in detected_names_this_frame:
            last_announced = last_announcement_time.get(name, 0)
            time_since_announcement = current_time - last_announced
            
            # Check if person was absent (not seen for ABSENCE_THRESHOLD seconds)
            # This allows re-announcement when they return
            was_absent = (name not in last_seen_time or 
                         (current_time - last_seen_time[name]) > ABSENCE_THRESHOLD)
            
            # Announce if:
            # 1. Never announced before, OR
            # 2. Cooldown has passed AND person was absent
            should_announce = (
                last_announced == 0 or 
                (time_since_announcement >= COOLDOWN_SECONDS and was_absent)
            )
            
            if should_announce:
                announcer.announce(name)
                last_announcement_time[name] = current_time
                print(f"✅ Queued announcement for {name}")
        
        # Display FPS
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Quitting...")
            break

except KeyboardInterrupt:
    print("\n👋 Interrupted by user")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
finally:
    print("🧹 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    announcer.stop()
    print("✅ Cleanup complete")