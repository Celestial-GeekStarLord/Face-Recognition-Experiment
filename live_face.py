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
COOLDOWN_SECONDS = 3  # Minimum time between announcements of same person
ABSENCE_FRAMES = 15  # Number of consecutive frames person must be absent

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
# Track when each person was last announced
last_announcement_time = {}
# Track if person is currently present
person_present = {}
# Track consecutive frames person has been absent
absence_counter = {}

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("🧠 Face recognition started | Press Q to quit")
print(f"⚙️ Settings: Cooldown={COOLDOWN_SECONDS}s, Absence={ABSENCE_FRAMES} frames")

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
                
                # Track detected names
                if name != "Unknown":
                    detected_names_this_frame.add(name)
                
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
        # 🔊 IMPROVED ANNOUNCEMENT LOGIC
        # =========================
        
        # Get all registered people
        all_known_people = set(face_db.keys())
        
        # Update presence tracking for all known people
        for person in all_known_people:
            if person in detected_names_this_frame:
                # Person is detected in this frame
                was_absent_before = not person_present.get(person, False)
                person_present[person] = True
                absence_counter[person] = 0
                
                # Check if we should announce
                last_announced = last_announcement_time.get(person, 0)
                time_since_announcement = current_time - last_announced
                
                # Announce if:
                # 1. Never announced before (first detection ever), OR
                # 2. Person was absent and just returned (re-entry)
                should_announce = (
                    last_announced == 0 or 
                    (was_absent_before and time_since_announcement >= COOLDOWN_SECONDS)
                )
                
                if should_announce:
                    announcer.announce(person)
                    last_announcement_time[person] = current_time
                    print(f"✅ Queued announcement for {person} (return: {was_absent_before})")
                    
            else:
                # Person NOT detected in this frame
                if person in person_present and person_present[person]:
                    # Increment absence counter
                    absence_counter[person] = absence_counter.get(person, 0) + 1
                    
                    # Mark as absent if threshold reached
                    if absence_counter[person] >= ABSENCE_FRAMES:
                        person_present[person] = False
                        print(f"👋 {person} has left the frame")
        
        # Display status
        status_y = 30
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Show who's present
        if detected_names_this_frame:
            status_y += 25
            cv2.putText(
                frame,
                f"Present: {', '.join(detected_names_this_frame)}",
                (10, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
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
    import traceback
    traceback.print_exc()
finally:
    print("🧹 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    announcer.stop()
    print("✅ Cleanup complete")