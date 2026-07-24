
## 🛡️ Tactical Operations & Multi-Modal Environment Support

Designed specifically for critical edge deployments where internet dependency is a failure point, the system operates in a **100% Air-Gapped / Fully Offline Operational Mode**.

### 1. Multi-Modal Vision Framework (RGB + Thermal Integration)
The pipeline features an automatic, resolution-adaptive camera intrinsic binding matrix. The framework dynamically detects the sensor footprint and switches between spatial configurations:
*   **Daytime / Electro-Optical (RGB) Mode:** Bounded via high-resolution 4K ($3000 \times 4000$) or 1080p ($1080 \times 1920$) intrinsic matrices to track fast-moving vehicles and dynamic pedestrian anomalies[cite: 6].
*   **Night-Vision / Long-Wave Infrared (Thermal) Mode:** Bounded via a specialized $512 \times 640$ thermal intrinsic matrix, utilizing custom Gaussian-blurred Lucas-Kanade optical flow structures to maintain spatial positioning under zero-light conditions[cite: 6].

### 2. Zero-Network Dependency Engine
*   **Local Inference Backbones:** Both YOLOv8 and DINOv2 architectures run purely on local hardware memory weights (`local_files_only=True`), preventing any mid-flight internet handshake drops[cite: 6].
*   **Resilient Cloud Fallback:** The Microsoft Azure IoT Hub telemetry integration is heavily guarded; if the aircraft operates in a jammed or offline environment, the system gracefully bypasses network blocks and handles all spatial data locally without interrupting the primary flight orchestration pipeline[cite: 6].


### 🛰️ Multi-Modal Edge Inference Telemetry Logs

To verify the dual-target locking consistency, visual spatial validation, and real-time state enums, you can inspect the raw execution frame pipeline log below:

<details>
<summary>🔍 Click to expand raw inference log (frame_000272.jpg)</summary>

#### Visual Asset Proof
Repodaki görsel yolunu buraya bağlayabilirsin:
![Inference Log Visual](./images/codesnap_frame_272.png)

#### Raw JSON Payload
```json
"frame_000272.jpg": {
        "frame": "[http://127.0.0.1:5000/media/test_video/frame_000272.jpg](http://127.0.0.1:5000/media/test_video/frame_000272.jpg)",
        "detected_objects": [
            {
                "cls": "[http://127.0.0.1:5000/classes/1/](http://127.0.0.1:5000/classes/1/)",
                "landing_status": "-1",
                "moving_status": "0",
                "top_left_x": "1390.2657470703125",
                "top_left_y": "516.8533935546875",
                "bottom_right_x": "1741.726318359375",
                "bottom_right_y": "778.7890014648438"
            },
            {
                "cls": "[http://127.0.0.1:5000/classes/1/](http://127.0.0.1:5000/classes/1/)",
                "landing_status": "-1",
                "moving_status": "1",
                "top_left_x": "926.8118286132812",
                "top_left_y": "641.091796875",
                "bottom_right_x": "1052.2401123046875",
                "bottom_right_y": "773.009521484375"
            },
            {
                "cls": "[http://127.0.0.1:5000/classes/1/](http://127.0.0.1:5000/classes/1/)",
                "landing_status": "-1",
                "moving_status": "0",
                "top_left_x": "283.61846923828125",
                "top_left_y": "606.632568359375",
                "bottom_right_x": "369.40594482421875",
                "bottom_right_y": "741.3778076171875"
            },
            {
                "cls": "[http://127.0.0.1:5000/classes/2/](http://127.0.0.1:5000/classes/2/)",
                "landing_status": "-1",
                "moving_status": "1",
                "top_left_x": "996.187744140625",
                "top_left_y": "660.1854248046875",
                "bottom_right_x": "1013.9011840820312",
                "bottom_right_y": "680.6777954101562"
            }
        ],
        "detected_translations": [
            {
                "translation_x": -25.641985107686786,
                "translation_y": 78.81668618848228,
                "translation_z": 0.4252734114385854
            }
        ],
        "reference_predictions": [
            {
                "reference": "[http://127.0.0.1:5000/media/reference/reference3.JPG](http://127.0.0.1:5000/media/reference/reference3.JPG)",
                "frame": "[http://127.0.0.1:5000/media/test_video/frame_000272.jpg](http://127.0.0.1:5000/media/test_video/frame_000272.jpg)",
                "top_left_x": "1235.0",
                "top_left_y": "230.0",
                "bottom_right_x": "1440.0",
                "bottom_right_y": "360.0"
            },
            {
                "reference": "[http://127.0.0.1:5000/media/reference/reference1.JPG](http://127.0.0.1:5000/media/reference/reference1.JPG)",
                "frame": "[http://127.0.0.1:5000/media/test_video/frame_000272.jpg](http://127.0.0.1:5000/media/test_video/frame_000272.jpg)",
                "top_left_x": "1373.0",
                "top_left_y": "516.0",
                "bottom_right_x": "1739.0",
                "bottom_right_y": "748.0"
            }
        ]
    }
\```

</details>