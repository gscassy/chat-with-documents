import os
import socket

def is_streamlit_cloud():
    """Reliable detection for Streamlit Cloud"""
    # 1. Streamlit sets this in all cloud environments
    if "STREAMLIT_SERVER_HEADLESS" in os.environ:
        return True

    # 2. Streamlit Cloud runs on specific internal hostnames
    hostname = socket.gethostname()
    if "streamlit" in hostname or "cloud" in hostname:
        return True

    # 3. Fallback — check Streamlit's config path
    streamlit_home = os.environ.get("STREAMLIT_HOME", "")
    if "share" in streamlit_home or "streamlit" in streamlit_home:
        return True

    return False