class EmergencyEngine:
    def __init__(self):
        pass

    def detect(self, tracked_objects, frame):
        emergency_routes = {}
        for obj in tracked_objects:
            if obj.get('v_type') == 'AMBULANCE':
                bx = obj['box']
                cx, cy = (bx[0]+bx[2])//2, (bx[1]+bx[3])//2
                route = [[cx, cy], [cx, 0]]
                emergency_routes[obj['id']] = route
        return emergency_routes
