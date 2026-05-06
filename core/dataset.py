"""
Dataset for CineVLA v3.1 — RGB-only, no depth required.

Supports:
  - Single-frame samples (from DataDoP) with synthetic multi-view augmentations
  - Multi-frame samples from extracted frame sequences
  - MP4 video files (auto-extract frames via cv2)

Each sample returns:
  'frames': [T, 3, H, W] RGB frame sequence
  'poses':  [N, 7] ground-truth trajectory
  'c2ws':   [N, 4, 4] camera-to-world matrices
  'text':   str caption
"""

import os, json, glob, random, subprocess, tempfile
import cv2, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from core.utils import matrix_to_quaternion


class CineVLADataset(Dataset):
    def __init__(self, path: str, split_txt: str, pose_length=30,
                 test=False, test_size=1, num_frames=8, image_size=224,
                 use_video_frames=False):
        self.path = path
        self.pose_length = pose_length
        self.test = test
        self.num_frames = num_frames
        self.image_size = image_size
        self.use_video_frames = use_video_frames

        with open(split_txt) as f:
            valid = {x.strip() for x in f}

        basedirs = []
        for idx in os.listdir(path):
            d = os.path.join(path, idx)
            if not os.path.isdir(d): continue
            for f in glob.glob(os.path.join(d, '*_transforms_cleaning.json')):
                base = f.replace('_transforms_cleaning.json', '')
                name = f"{idx}/{os.path.basename(base)}"
                if name in valid and self._check(base):
                    basedirs.append(base)

        basedirs = sorted(basedirs)
        random.seed(42); random.shuffle(basedirs)
        n = len(basedirs)
        self.items = basedirs[:-test_size] if not test else basedirs[-test_size:]

        self.captions = {}
        for b in self.items:
            cf = b + '_caption.json'
            if os.path.exists(cf):
                info = json.load(open(cf))
                self.captions[b] = info.get('Concise Interaction', '') or info.get('Movement', '')

        print(f'CineVLA Dataset: {len(self.items)} samples (RGB-only, {num_frames} frames/sample)')

    @staticmethod
    def _check(base):
        return os.path.exists(base + '_rgb.png') and os.path.exists(base + '_transforms_cleaning.json')

    def __len__(self): return len(self.items)

    def _load_rgb(self, path, Ht=224, Wt=224):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img = img[..., [2, 1, 0]]  # BGR → RGB
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        return self._resize(t, Ht, Wt)

    @staticmethod
    def _resize(t, Ht, Wt):
        h, w = t.shape[1], t.shape[2]
        if h > Ht: t = t[:, (h - Ht) // 2:(h - Ht) // 2 + Ht, :]
        if w > Wt: t = t[:, :, (w - Wt) // 2:(w - Wt) // 2 + Wt]
        if t.shape[1] < Ht or t.shape[2] < Wt:
            p = torch.zeros(3, Ht, Wt)
            p[:, (Ht - t.shape[1]) // 2:(Ht - t.shape[1]) // 2 + t.shape[1],
              (Wt - t.shape[2]) // 2:(Wt - t.shape[2]) // 2 + t.shape[2]] = t
            t = p
        return t

    def _extract_video_frames(self, video_path):
        """Extract num_frames evenly-spaced frames from MP4."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: return None

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            frame = frame.astype(np.float32) / 255.
            frame = frame[..., [2, 1, 0]]  # BGR → RGB
            t = torch.from_numpy(frame).permute(2, 0, 1).float()
            frames.append(self._resize(t, self.image_size, self.image_size))
        cap.release()

        if len(frames) < 2: return None
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # pad with last frame
        return torch.stack(frames[:self.num_frames])

    def _synthetic_sequence(self, rgb):
        """Build a pseudo frame sequence from a single image via augmentations."""
        frames = [rgb]
        for _ in range(self.num_frames - 1):
            aug = rgb.clone()
            # Random crop + resize (simulates viewpoint shift)
            s = random.uniform(0.85, 0.98)
            h, w = int(224 * s), int(224 * s)
            y = random.randint(0, 224 - h)
            x = random.randint(0, 224 - w)
            patch = aug[:, y:y + h, x:x + w]
            aug = F.interpolate(patch.unsqueeze(0), (224, 224), mode='bilinear',
                                align_corners=False).squeeze(0)
            # Random brightness/contrast
            aug = torch.clamp(aug * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0, 1)
            frames.append(aug)
        return torch.stack(frames)

    def __getitem__(self, idx):
        base = self.items[idx]
        try:
            j = json.load(open(base + '_transforms_cleaning.json'))
            frames_json = j['frames']
            H, W = j['h'], j['w']
            fx, fy = j['fl_x'], j['fy']
            cx, cy = j['cx'], j['cy']
            N = self.pose_length
            indices = np.arange(len(frames_json))[:120:120 // N][:N]

            c2ws, poses_7d = [], []
            for i in indices:
                m = np.array(frames_json[i]['transform_matrix'])
                c2ws.append(m)
                R = torch.from_numpy(m[:3, :3])
                T = torch.from_numpy(m[:3, 3])
                q = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)
                poses_7d.append(torch.cat([q, T]))

            poses_7d = torch.stack(poses_7d)
            c2ws_np = np.stack(c2ws)

            # Normalize
            ref = np.linalg.inv(c2ws_np[0])
            for i in range(N):
                c2ws_np[i] = ref @ c2ws_np[i]
            Tn = np.linalg.norm(c2ws_np[:, :3, 3], axis=-1).max() + 1e-5
            c2ws_np[:, :3, 3] /= Tn
            poses_7d[:, 4:7] /= Tn

            # ── Frame sequence (NO depth) ──
            primary_rgb = self._load_rgb(base + '_rgb.png')
            # Check for multi-frame directory
            frames_dir = base + '_frames'
            if self.use_video_frames and os.path.isdir(frames_dir):
                frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))[:self.num_frames]
                if len(frame_files) >= 2:
                    frames = torch.stack([self._load_rgb(f) for f in frame_files])
                else:
                    frames = self._synthetic_sequence(primary_rgb)
            else:
                frames = self._synthetic_sequence(primary_rgb)

            text = self.captions.get(base, '')

            return {
                'frames': frames,                               # [T, 3, 224, 224]
                'poses': poses_7d.float(),                      # [N, 7]
                'c2ws': torch.from_numpy(c2ws_np).float(),     # [N, 4, 4]
                'intrinsics': torch.tensor([fx, fy, cx, cy]).float(),
                'text': text,
                'path': base,
            }
        except Exception as e:
            print(f"Error {base}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.items)))




def collate_fn(batch):
    T = batch[0]['frames'].shape[0]
    frames = torch.stack([b['frames'][:T] for b in batch])
    poses = torch.stack([b['poses'] for b in batch])
    c2ws = torch.stack([b['c2ws'] for b in batch])
    intr = torch.stack([b['intrinsics'] for b in batch])
    return {
        'frames': frames, 'poses': poses, 'c2ws': c2ws,
        'intrinsics': intr,
        'text': [b['text'] for b in batch],
        'paths': [b['path'] for b in batch],
    }
