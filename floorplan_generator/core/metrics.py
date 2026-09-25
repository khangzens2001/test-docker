import numpy as np

class RoomMetrics:
    def __init__(self, vertices, height_mm):
        self.vertices = vertices
        self.height_mm = height_mm
        
        self.area_m2 = 0
        self.perimeter_m = 0
        self.bbox_dims = (0, 0)
        self.wall_metrics = []
        self.shape_name = "UNKNOWN"
        self.wall_count = len(vertices)
        
        self.compute()
        
    def compute(self):
        print("[3/5] Computing room metrics...")
        
        # Bounding box
        v_arr = np.array(self.vertices)
        x_min, x_max = v_arr[:, 0].min(), v_arr[:, 0].max()
        y_min, y_max = v_arr[:, 1].min(), v_arr[:, 1].max()
        
        self.bbox_dims = (x_max - x_min, y_max - y_min)
        
        # Area (Shoelace formula)
        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        self.area_m2 = abs(area) / 2.0 / 1e6
        
        # Perimeter and walls
        perimeter = 0.0
        self.wall_metrics = []
        for i in range(n):
            j = (i + 1) % n
            dx = self.vertices[j][0] - self.vertices[i][0]
            dy = self.vertices[j][1] - self.vertices[i][1]
            length = np.sqrt(dx**2 + dy**2)
            direction = 'Hor' if abs(dx) > abs(dy) else 'Ver'
            
            perimeter += length
            wall_area = (length / 1000.0) * (self.height_mm / 1000.0)
            
            self.wall_metrics.append({
                "id": f"W{i}",
                "dir": direction,
                "length_mm": int(length),
                "area_m2": wall_area
            })
            
        self.perimeter_m = perimeter / 1000.0
        
        if n == 4:
            self.shape_name = "RECTANGULAR"
        elif n == 6:
            self.shape_name = "L-SHAPED"
        elif n == 8:
            self.shape_name = "T-SHAPED/U-SHAPED"
        else:
            self.shape_name = f"COMPLEX ({n}-gon)"
            
        print(f"       Shape: {self.shape_name}")
        print(f"       Walls: {n}")
        print(f"       BBox: {int(self.bbox_dims[0])} x {int(self.bbox_dims[1])} mm")
        print(f"       Floor Area: {self.area_m2:.2f} m²")
        print(f"       Perimeter: {self.perimeter_m:.2f} m")
        print(f"       Height: {int(self.height_mm)} mm")
