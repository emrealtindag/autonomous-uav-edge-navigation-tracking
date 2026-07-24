class Translation:
    """
    Handles ground truth baseline spatial mapping vectors 
    and systemic API route generations for internet-based telemetry logs.
    """
    def __init__(self, translation_x: float, translation_y: float, translation_z: float):
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z

    def create_payload(self):
        return {
            'translation_x': self.translation_x,
            'translation_y': self.translation_y,
            'translation_z': self.translation_z
        }

    @staticmethod
    def generate_api_url(cls_endpoint, cls_id, evaulation_server):
        # Guarded method to prevent trailing slash anomalies during dynamic URL construction via internet/API pathways
        checked_url = evaulation_server if evaulation_server.endswith("/") else evaulation_server + "/"
        return checked_url + cls_endpoint + str(cls_id) + "/"