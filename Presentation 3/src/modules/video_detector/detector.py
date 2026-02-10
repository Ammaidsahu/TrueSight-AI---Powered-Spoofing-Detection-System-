"""
Enhanced Video Deepfake Detection Module
Production-ready implementation with LNCLIP integration
"""

# Import the new LNCLIP implementation
from .lnclip_detector import LNCLIPVideoDetector

# Maintain backward compatibility
class EnhancedVideoDetector(LNCLIPVideoDetector):
    """Backward compatible enhanced video detector"""
    pass

class VideoDetector(LNCLIPVideoDetector):
    """Legacy alias for video detector"""
    pass

__all__ = ["EnhancedVideoDetector", "VideoDetector", "LNCLIPVideoDetector"]