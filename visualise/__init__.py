"""
CineVLA Visualization Module
============================
Two independent visualization components:

1. Trajectory Visualization (inference only)
   - 3D camera path with pose frames
   - Refinement error over closed-loop steps
   - Saves to ./results/

2. Latent State Visualization (training & inference)
   - PCA projection of z_pred vs z_real
   - Prediction error over time
   - Saves to ./pred_latent/
"""

from .trajectory import plot_trajectory
from .latent import LatentLogger
