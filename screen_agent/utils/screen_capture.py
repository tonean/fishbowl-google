import mss
from PIL import Image
import io
import base64

def capture_screen():
    """
    Captures the primary monitor and returns a PNG image as a base64-encoded string.
    """
    sct = mss.mss()
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
