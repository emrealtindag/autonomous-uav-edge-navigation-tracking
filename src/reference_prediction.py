class ReferencePrediction:
    """
    Represents a predicted bounding box tracking entry mapped against 
    a specified target reference image asset profile.
    """
    def __init__(self,
                 reference_url: str,
                 frame_url: str,
                 top_left_x: float,
                 top_left_y: float,
                 bottom_right_x: float,
                 bottom_right_y: float,
                 ):
        self.reference_url = reference_url
        self.frame_url = frame_url
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

    # 🚀 DEAD ARGUMENT (evaulation_server) COMPLETELY CLEANED
    def create_payload(self):
        return {
            'reference': self.reference_url,
            'frame': self.frame_url,
            'top_left_x': str(self.top_left_x),
            'top_left_y': str(self.top_left_y),
            'bottom_right_x': str(self.bottom_right_x),
            'bottom_right_y': str(self.bottom_right_y)
        }