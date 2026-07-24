class DetectedTranslation:
    """
    Represents the calculated camera translation vectors in 3D space
    extracted from the computer vision tracking pipelines.
    """
    def __init__(self,
                 translation_x: float,
                 translation_y: float,
                 translation_z: float,
                 ):
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z

    def create_payload(self):
        return {
            'translation_x': self.translation_x, 
            'translation_y': self.translation_y,  
            'translation_z': self.translation_z 
        }