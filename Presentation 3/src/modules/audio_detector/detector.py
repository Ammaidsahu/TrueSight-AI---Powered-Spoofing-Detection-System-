"""
Enhanced Audio Deepfake Detection Module
Wav2Vec2-based implementation with production-ready features
"""

# Import the new Wav2Vec2 implementation
from .wav2vec2_detector import Wav2Vec2AudioDetector

# Maintain backward compatibility
class EnhancedAudioDetector(Wav2Vec2AudioDetector):
    """Backward compatible enhanced audio detector"""
    pass

class AudioDetector(Wav2Vec2AudioDetector):
    """Legacy alias for audio detector"""
    pass

__all__ = ["EnhancedAudioDetector", "AudioDetector", "Wav2Vec2AudioDetector"]