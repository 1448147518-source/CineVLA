"""
Dataset for CineVLA v3.1 — RGB-only, no depth required.

Input paradigm:
  img_0 (_rgb.png) + text (_caption.json) + music (optional) → initial planning
  img_i sequence (_frames/ or _video.mp4) → subsequent closed-loop refinement

Frame source priority (mandatory — one of the two MUST exist):
  1. _video.mp4 — auto-extract evenly-spaced frames (takes priority if both present)
  2. _frames/   — directory of PNG frame files

No padding or synthetic augmentation fallback.  If neither video nor _frames/
exists, training/inference exits with an error.
"""

import os, json, glob, random
import cv2, numpy as np
import torch
from torch.utils.data import Dataset
from core.utils import matrix_to_quaternion


class CineVLADataset(Dataset):
    def __init__(self, path: str, split_txt: str, pose_length=30,
                 test=False, test_size=1, num_frames=8, image_size=224):
        self.path = path
        self.pose_length = pose_length
        self.test = test
        self.num_frames = num_frames
        self.image_size = image_size

        with open(split_txt) as f:
            valid = {x.strip() for x in f}

        basedirs = []
        for idx in os.listdir(path):
            d = os.path.join(path, idx)
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, '*_transforms_cleaning.json')):
                base = f.replace('_transforms_cleaning.json', '')
                name = f"{idx}/{os.path.basename(base)}"
                if name in valid and self._check(base):
                    basedirs.append(base)

        basedirs = sorted(basedirs)
        random.seed(42)
        random.shuffle(basedirs)
        n = len(basedirs)
        self.items = basedirs[:-test_size] if not test else basedirs[-test_size:]

        self.captions = {}
        for b in self.items:
            cf = b + '_caption.json'
            if os.path.exists(cf):
                info = json.load(open(cf))
                self.captions[b] = info.get('Concise Interaction', '') or info.get('Movement', '')

        print(f'CineVLA Dataset: {len(self.items)} samples '
              f'(RGB-only, {num_frames} frames/sample)')

    # ── Validation ──

    @staticmethod
    def _check(base):
        has_rgb = os.path.exists(base + '_rgb.png')
        has_traj = os.path.exists(base + '_transforms_cleaning.json')
        has_frames = os.path.isdir(base + '_frames')
        has_video = os.path.exists(base + '_video.mp4')
        return has_rgb and has_traj and (has_frames or has_video)

    def __len__(self):
        return len(self.items)

    # ── Image I/O ──

    def _load_rgb(self, path, Ht=224, Wt=224):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img = img[..., [2, 1, 0]]  # BGR → RGB
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        return self._resize(t, Ht, Wt)

    @staticmethod
    def _resize(t, Ht, Wt):
        h, w = t.shape[1], t.shape[2]
        if h > Ht:
            t = t[:, (h - Ht) // 2:(h - Ht) // 2 + Ht, :]
        if w > Wt:
            t = t[:, :, (w - Wt) // 2:(w - Wt) // 2 + Wt]
        if t.shape[1] < Ht or t.shape[2] < Wt:
            p = torch.zeros(3, Ht, Wt)
            p[:, (Ht - t.shape[1]) // 2:(Ht - t.shape[1]) // 2 + t.shape[1],
              (Wt - t.shape[2]) // 2:(Wt - t.shape[2]) // 2 + t.shape[2]] = t
            t = p
        return t

    # ── Frame extraction ──

    def _extract_video_frames(self, video_path):
        """Extract num_frames evenly-spaced frames from MP4.  No padding."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < self.num_frames:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame.astype(np.float32) / 255.
            frame = frame[..., [2, 1, 0]]  # BGR → RGB
            t = torch.from_numpy(frame).permute(2, 0, 1).float()
            frames.append(self._resize(t, self.image_size, self.image_size))
        cap.release()

        if len(frames) < self.num_frames:
            return None
        return torch.stack(frames)  # [num_frames, 3, H, W]

    # ── Main ──

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

            # Normalize to first frame's coordinate frame
            ref = np.linalg.inv(c2ws_np[0])
            for i in range(N):
                c2ws_np[i] = ref @ c2ws_np[i]
            Tn = np.linalg.norm(c2ws_np[:, :3, 3], axis=-1).max() + 1e-5
            c2ws_np[:, :3, 3] /= Tn
            poses_7d[:, 4:7] /= Tn

            # ── Frame sequence (video > _frames/ — mandatory) ──
            video_path = base + '_video.mp4'
            frames_dir = base + '_frames'

            if os.path.exists(video_path):
                frames = self._extract_video_frames(video_path)
                if frames is None:
                    raise RuntimeError(
                        f"Video {video_path} has fewer than {self.num_frames} "
                        f"usable frames"
                    )
            elif os.path.isdir(frames_dir):
                frame_files = sorted(
                    glob.glob(os.path.join(frames_dir, '*.png')))
                if len(frame_files) < self.num_frames:
                    raise RuntimeError(
                        f"{frames_dir}: {len(frame_files)} PNG frames found, "
                        f"need >= {self.num_frames}"
                    )
                frames = torch.stack(
                    [self._load_rgb(f) for f in frame_files[:self.num_frames]])
            else:
                raise RuntimeError(
                    f"Sample {base} must have _video.mp4 or _frames/ directory"
                )

            text = self.captions.get(base, '')
            music = base + '_music.mp3' if os.path.exists(base + '_music.mp3') else None

            return {
                'frames': frames,
                'poses': poses_7d.float(),
                'c2ws': torch.from_numpy(c2ws_np).float(),
                'intrinsics': torch.tensor([fx, fy, cx, cy]).float(),
                'text': text,
                'music_path': music,
                'path': base,
            }
        except RuntimeError:
            raise
        except Exception as e:
            print(f"Error {base}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.items)))


def collate_fn(batch):
    T = batch[0]['frames'].shape[0]
    frames = torch.stack([b['frames'][:T] for b in batch])
    poses = torch.stack([b['poses'] for b in batch])
    c2ws = torch.stack([b['c2ws'] for b in batch])
    intr = torch.stack([b['intrinsics'] for b in batch])
    music = [b['music_path'] for b in batch]
    return {
        'frames': frames, 'poses': poses, 'c2ws': c2ws,
        'intrinsics': intr,
        'text': [b['text'] for b in batch],
        'music_path': music[0] if all(m == music[0] for m in music) else music,
        'paths': [b['path'] for b in batch],
    }
