"""Data provider for CineVLA v3 — supports synthetic intermediate views."""

import os, json, glob, random
import cv2, numpy as np
import torch
from torch.utils.data import Dataset
from core.utils import matrix_to_quaternion


class CineVLADataset(Dataset):
    """Loads DataDoP samples and synthesizes intermediate views for perception training."""

    def __init__(self, path: str, split_txt: str, pose_length=30, test=False, test_size=1):
        self.path = path
        self.pose_length = pose_length
        self.test = test

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
        random.seed(42); random.shuffle(basedirs)
        n = len(basedirs)
        self.items = basedirs[:-test_size] if not test else basedirs[-test_size:]

        # Preload captions
        self.captions = {}
        for b in self.items:
            cf = b + '_caption.json'
            if os.path.exists(cf):
                info = json.load(open(cf))
                self.captions[b] = info.get('Concise Interaction', '') or \
                                   info.get('Movement', '')

        print(f'CineVLA Dataset: {len(self.items)} samples')

    @staticmethod
    def _check(base):
        return all(os.path.exists(base + e) for e in
                   ['_depth.npy', '_caption.json', '_rgb.png', '_transforms_cleaning.json'])

    def __len__(self):
        return len(self.items)

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img = img[..., [2, 1, 0]]
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        return self._resize(t, 224, 224)

    @staticmethod
    def _resize(t, Ht, Wt):
        h, w = t.shape[1], t.shape[2]
        if h > Ht: t = t[:, (h - Ht) // 2: (h - Ht) // 2 + Ht, :]
        if w > Wt: t = t[:, :, (w - Wt) // 2: (w - Wt) // 2 + Wt]
        if t.shape[1] < Ht or t.shape[2] < Wt:
            p = torch.zeros(3, Ht, Wt)
            p[:, (Ht - t.shape[1]) // 2: (Ht - t.shape[1]) // 2 + t.shape[1],
              (Wt - t.shape[2]) // 2: (Wt - t.shape[2]) // 2 + t.shape[2]] = t
            t = p
        return t

    def __getitem__(self, idx):
        base = self.items[idx]
        try:
            j = json.load(open(base + '_transforms_cleaning.json'))
            frames = j['frames']
            H, W = j['h'], j['w']
            fx, fy = j['fl_x'], j['fy']
            cx, cy = j['cx'], j['cy']
            N = self.pose_length
            indices = np.arange(len(frames))[:120: 120 // N][:N]

            # Extract camera poses
            c2ws, poses_7d = [], []
            for i in indices:
                m = np.array(frames[i]['transform_matrix'])
                c2ws.append(m)
                R = torch.from_numpy(m[:3, :3])
                T = torch.from_numpy(m[:3, 3])
                q = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)
                poses_7d.append(torch.cat([q, T]))

            poses_7d = torch.stack(poses_7d)  # [N, 7]
            c2ws = np.stack(c2ws)  # [N, 4, 4]

            # Normalize
            scale = 1.0
            ref = np.linalg.inv(c2ws[0])
            for i in range(N):
                m = ref @ c2ws[i]
                c2ws[i] = m
            Tn = np.linalg.norm(c2ws[:, :3, 3], axis=-1).max()
            scale = Tn + 1e-5
            c2ws[:, :3, 3] /= scale
            poses_7d[:, 4:7] /= scale

            # Image and depth (first frame)
            rgb = self._load_rgb(base + '_rgb.png')
            depth = np.load(base + '_depth.npy').astype(np.float32)
            depth = cv2.resize(depth, (W, H)) if depth.shape[:2] != (H, W) else depth
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()

            # Resize first-frame RGB+Depth to 224
            rgb = self._resize(rgb, 224, 224)
            depth_t = torch.from_numpy(
                cv2.resize(depth_t.squeeze(0).numpy(), (224, 224))
            ).unsqueeze(0).float()

            text = self.captions.get(base, '')

            return {
                'rgb': rgb,                                    # [3, 224, 224] initial frame
                'depth': depth_t,                              # [1, 224, 224]
                'poses': poses_7d.float(),                     # [N, 7]
                'c2ws': torch.from_numpy(c2ws).float(),        # [N, 4, 4]
                'intrinsics': torch.tensor([fx, fy, cx, cy]).float(),
                'text': text,
                'path': base,
            }
        except Exception as e:
            print(f"Error {base}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.items)))


def collate_fn(batch):
    rgb = torch.stack([b['rgb'] for b in batch])
    depth = torch.stack([b['depth'] for b in batch])
    poses = torch.stack([b['poses'] for b in batch])
    c2ws = torch.stack([b['c2ws'] for b in batch])
    intr = torch.stack([b['intrinsics'] for b in batch])
    return {
        'rgb': rgb, 'depth': depth, 'poses': poses,
        'c2ws': c2ws,
        'intrinsics': intr,
        'text': [b['text'] for b in batch],
        'paths': [b['path'] for b in batch],
    }
