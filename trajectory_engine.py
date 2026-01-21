import numpy as np

class TrajectoryEngine:
    def __init__(self, horizon=60):
        self.horizon = horizon  # Predict 60 frames (approx 2s)

    def predict(self, tracked_objects):
        trajectories = {}

        for obj in tracked_objects:
            tid = obj['id']
            center = obj['center']
            vx, vy = obj['velocity']
            if abs(vx) < 0.5 and abs(vy) < 0.5:
                continue
            points = []
            curr_x, curr_y = center
            for t in range(1, self.horizon + 1):
                nx = curr_x + vx * t
                ny = curr_y + vy * t
                points.append((int(nx), int(ny)))
            trajectories[tid] = points
            
        return trajectories
