class RiskEngine:
    def __init__(self):
        pass

    def compute(self, tracked_objects, trajectories, obstructions, heatmap):
        scores = {}
        for obj in tracked_objects:
            scores[obj['id']] = obj.get('speed', 0) * 0.5
        return scores
