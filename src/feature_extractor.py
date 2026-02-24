import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ["TORCH_HOME"] = "./data/models/torch_cache"
os.environ["YOLO_CONFIG_DIR"] = "./data/models/yolo_cache"

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, extensions=None):
        self.root_dir = root_dir
        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.paths = []
        root_path = Path(root_dir)
        if root_path.exists():
            for path in root_path.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return path.name, image


def collate_fn(batch):
    names, images = zip(*batch)
    return list(names), list(images)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_model_local(model_name: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        cache_dir=local_dir,
    )
    return local_dir


def safe_normalize(tensor: torch.Tensor):
    norms = tensor.norm(p=2, dim=-1, keepdim=True)
    return tensor / torch.where(norms == 0, torch.ones_like(norms), norms)


def build_models(device: torch.device):
    model_name = "openai/clip-vit-large-patch14"
    local_model_dir = ensure_model_local(model_name, "./data/models/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained(local_model_dir, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(local_model_dir, local_files_only=True)
    clip_model.to(device)
    clip_model.eval()

    yolo_model = YOLO("./data/models/yolov8n.pt").to(device)
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    return clip_model, processor, yolo_model, mtcnn, resnet


def clip_feature(image: Image.Image, clip_model: CLIPModel, processor: CLIPProcessor, device: torch.device):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_outputs.pooler_output
        if hasattr(clip_model, "visual_projection") and pooled.shape[-1] == clip_model.visual_projection.in_features:
            features = clip_model.visual_projection(pooled)
        else:
            features = pooled
    features = safe_normalize(features)
    return features.squeeze(0)


def get_largest_person_crop(image: Image.Image, yolo_model: YOLO):
    results = yolo_model(image, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    cls = boxes.cls
    if cls is None:
        return None
    person_mask = cls == 0
    if person_mask.sum().item() == 0:
        return None
    xyxy = boxes.xyxy[person_mask]
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(torch.argmax(areas).item())
    x1, y1, x2, y2 = xyxy[idx].tolist()
    width, height = image.size
    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(width, int(x2))
    bottom = min(height, int(y2))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def face_feature(image: Image.Image, mtcnn: MTCNN, resnet: InceptionResnetV1, device: torch.device):
    face = mtcnn(image)
    if face is None:
        return torch.zeros(512, device=device)
    if face.ndim == 3:
        face = face.unsqueeze(0)
    face = face.to(device)
    with torch.no_grad():
        features = resnet(face)
    features = safe_normalize(features)
    return features.squeeze(0)


def extract_hybrid_feature(
    image: Image.Image,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    yolo_model: YOLO,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: torch.device,
):
    v1 = clip_feature(image, clip_model, processor, device)
    crop = get_largest_person_crop(image, yolo_model)
    if crop is None:
        v2 = torch.zeros_like(v1)
    else:
        v2 = clip_feature(crop, clip_model, processor, device)
    v3 = face_feature(image, mtcnn, resnet, device)
    return torch.cat([v1, v2, v3], dim=-1)


def extract_hybrid_features(images, models, device: torch.device):
    clip_model, processor, yolo_model, mtcnn, resnet = models
    features = [
        extract_hybrid_feature(image, clip_model, processor, yolo_model, mtcnn, resnet, device)
        for image in images
    ]
    return torch.stack(features, dim=0)


def extract_features(
    input_dir: str,
    output_path: str,
    batch_size: int,
):
    start_time = time.time()
    device = get_device()
    print(f"Device: {device.type}")
    models = build_models(device)

    dataset = ImageFolderDataset(input_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    features = {}
    for names, images in tqdm(loader, desc="Extracting", unit="batch"):
        hybrid = extract_hybrid_features(images, models, device)
        for name, vector in zip(names, hybrid):
            features[name] = vector.detach().cpu()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"✅ 特征提取总耗时: {time.time() - start_time:.2f} 秒")
    return len(features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./data/raw_images")
    parser.add_argument("--output_path", default="./data/features.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    count = extract_features(
        input_dir=args.input_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    print(f"Extracted {count} embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
