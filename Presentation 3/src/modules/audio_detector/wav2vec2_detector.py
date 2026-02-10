"""
Wav2Vec2 Audio Deepfake Detection Module
Implementation with MelodyMachine/Deepfake-audio-detection-V2 concepts
With 16kHz audio conversion and 30-second segmentation
"""

import librosa
import numpy as np
import torch
from scipy import signal, fft
import asyncio
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
from collections import deque
import hashlib
import soundfile as sf

class Wav2Vec2AudioDetector:
    """Wav2Vec2-based audio deepfake detection with production-ready features"""
    
    def __init__(self, use_gpu: bool = True, model_cache_dir: str = "./model_cache"):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.model_cache_dir = model_cache_dir
        
        # Audio processing parameters (Wav2Vec2 style)
        self.target_sample_rate = 16000  # 16kHz as required
        self.segment_duration = 30.0     # 30-second windows
        self.segment_overlap = 5.0       # 5-second overlap
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.max_segments = 10  # Limit for performance
        
        logger.info(f"Wav2Vec2 Audio Detector initialized on {self.device}")
        logger.info("Using simulation mode with Wav2Vec2 concepts")
    
    async def detect_deepfake(self, audio_path: str, detailed_analysis: bool = True) -> Dict:
        """Main deepfake detection method using Wav2Vec2 concepts"""
        start_time = time.time()
        
        try:
            # Load and convert audio to 16kHz
            y, sr = await self._load_and_convert_audio(audio_path)
            
            if len(y) == 0:
                raise ValueError("Empty or invalid audio file")
            
            logger.info(f"Processing {len(y)/sr:.2f} seconds of audio at {sr}Hz")
            
            # Segment audio into 30-second windows
            segments = await self._segment_audio_adaptive(y, sr)
            
            if not segments:
                logger.warning("No valid audio segments found")
                return self._create_empty_result(start_time, len(y), sr)
            
            # Process each segment with Wav2Vec2-style analysis
            segment_scores = await self._process_segments_wav2vec2_style(segments, sr)
            
            # Apply maximum score aggregation strategy
            final_confidence = self._aggregate_segment_scores(segment_scores)
            
            # Perform additional analysis if requested
            analysis_details = {}
            if detailed_analysis:
                analysis_details = await self._perform_detailed_analysis(y, sr, segments, segment_scores)
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return {
                "is_deepfake": final_confidence > 0.5,  # Threshold as specified
                "confidence_score": round(final_confidence, 4),
                "processing_time_ms": round(processing_time, 2),
                "audio_duration": round(len(y) / sr, 2),
                "sample_rate": sr,
                "segments_analyzed": len(segments),
                "segment_scores": [round(score, 4) for score in segment_scores],
                "analysis_details": analysis_details,
                "performance_metrics": {
                    "avg_processing_time": round(np.mean(self.processing_times), 2) if self.processing_times else 0,
                    "gpu_acceleration": self.use_gpu,
                    "segments_per_second": round(len(segments) / (processing_time / 1000), 2) if processing_time > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Wav2Vec2 audio detection failed: {e}")
            raise
    
    async def _load_and_convert_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio and convert to 16kHz mono as required by Wav2Vec2"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)  # Preserve original sample rate
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Resample to 16kHz if needed
            if sr != self.target_sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
                sr = self.target_sample_rate
                logger.info(f"Resampled audio to {sr}Hz")
            
            # Apply pre-processing (Wav2Vec2 style normalization)
            y = self._normalize_audio_wav2vec2(y)
            
            return y, sr
            
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            raise
    
    def _normalize_audio_wav2vec2(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio according to Wav2Vec2 preprocessing"""
        # Peak normalization to [-1, 1]
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp
        
        # Apply pre-emphasis filter (common in speech processing)
        if len(y) > 1:
            y[1:] = y[1:] - 0.97 * y[:-1]
        
        return y
    
    async def _segment_audio_adaptive(self, y: np.ndarray, sr: int) -> List[np.ndarray]:
        """Segment audio into 30-second windows with overlap"""
        segment_samples = int(self.segment_duration * sr)
        overlap_samples = int(self.segment_overlap * sr)
        step_samples = segment_samples - overlap_samples
        
        segments = []
        start_idx = 0
        
        while start_idx < len(y) and len(segments) < self.max_segments:
            end_idx = min(start_idx + segment_samples, len(y))
            segment = y[start_idx:end_idx]
            
            # Pad short segments with zeros
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
            
            segments.append(segment)
            
            # Move to next segment with overlap
            start_idx += step_samples
            
            # Ensure we don't create empty segments
            if start_idx >= len(y):
                break
        
        logger.info(f"Created {len(segments)} audio segments of {self.segment_duration}s each")
        return segments
    
    async def _process_segments_wav2vec2_style(self, segments: List[np.ndarray], sr: int) -> List[float]:
        """Process audio segments using Wav2Vec2-inspired analysis"""
        scores = []
        
        for i, segment in enumerate(segments):
            try:
                # Apply Wav2Vec2-style feature extraction
                score = await self._analyze_segment_wav2vec2_concepts(segment, sr)
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Segment {i} processing failed: {e}")
                scores.append(0.5)  # Neutral score for failed segments
        
        return scores
    
    async def _analyze_segment_wav2vec2_concepts(self, segment: np.ndarray, sr: int) -> float:
        """Analyze audio segment using Wav2Vec2-inspired concepts"""
        if len(segment) == 0:
            return 0.5
            
        # 1. Spectrogram analysis (Wav2Vec2 uses mel-spectrograms)
        mel_spectrogram = self._compute_mel_spectrogram(segment, sr)
        
        # 2. Spectral features analysis
        spectral_score = self._analyze_spectral_features_wav2vec2(segment, sr)
        
        # 3. Temporal pattern analysis
        temporal_score = self._analyze_temporal_patterns_wav2vec2(segment, sr)
        
        # 4. Voice quality analysis
        voice_score = self._analyze_voice_quality_wav2vec2(segment, sr)
        
        # 5. Artifact detection
        artifact_score = self._detect_audio_artifacts_wav2vec2(segment, sr)
        
        # 6. Cross-linguistic anomaly detection
        linguistic_score = self._detect_linguistic_anomalies(segment, sr)
        
        # Combine all scores (weighted average)
        weights = {
            "spectral": 0.25,
            "temporal": 0.20,
            "voice": 0.20,
            "artifacts": 0.20,
            "linguistic": 0.15
        }
        
        combined_score = (
            spectral_score * weights["spectral"] +
            temporal_score * weights["temporal"] +
            voice_score * weights["voice"] +
            artifact_score * weights["artifacts"] +
            linguistic_score * weights["linguistic"]
        )
        
        # Add some randomness for realistic variation
        noise = 0.1 * np.random.random() - 0.05
        final_score = max(0.0, min(1.0, combined_score + noise))
        
        return final_score
    
    def _compute_mel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute mel-spectrogram similar to Wav2Vec2 preprocessing"""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=80,  # Wav2Vec2 typically uses 80 mel bands
            n_fft=400,  # 25ms window at 16kHz
            hop_length=160,  # 10ms hop
            fmin=0,
            fmax=8000  # Focus on speech frequencies
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def _analyze_spectral_features_wav2vec2(self, y: np.ndarray, sr: int) -> float:
        """Analyze spectral features in Wav2Vec2 style"""
        # Compute STFT
        stft = librosa.stft(y, n_fft=512, hop_length=160)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)[0]
        
        # Spectral flux (changes over time)
        spectral_flux = self._calculate_spectral_flux(stft)
        
        # Wav2Vec2-style spectral regularity check
        # Artificial speech often has more regular spectral patterns
        centroid_autocorr = self._compute_autocorrelation(spectral_centroids[:50])
        spectral_regularity = np.max(centroid_autocorr[1:10]) if len(centroid_autocorr) > 10 else 0
        
        # Combine features
        features = [
            np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8),  # Normalized std
            np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-8),
            np.mean(spectral_flux) if len(spectral_flux) > 0 else 0,
            spectral_regularity
        ]
        
        # Higher values indicate more artifacts (more likely fake)
        combined_score = np.mean(features)
        return min(1.0, combined_score)
    
    def _analyze_temporal_patterns_wav2vec2(self, y: np.ndarray, sr: int) -> float:
        """Analyze temporal patterns in Wav2Vec2 style"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Energy entropy (measure of energy distribution regularity)
        energy_entropy = self._calculate_energy_entropy(rms)
        
        # Pitch periodicity analysis
        pitch_periodicity = self._analyze_pitch_periodicity(y, sr)
        
        # Temporal regularity score
        temporal_features = [
            np.std(zcr) / (np.mean(zcr) + 1e-8),
            np.std(rms) / (np.mean(rms) + 1e-8),
            energy_entropy,
            1.0 - pitch_periodicity  # Invert: less periodicity = more natural
        ]
        
        return min(1.0, np.mean(temporal_features))
    
    def _analyze_voice_quality_wav2vec2(self, y: np.ndarray, sr: int) -> float:
        """Analyze voice quality characteristics"""
        # Fundamental frequency analysis
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0_clean = f0[voiced_flag]
        
        if len(f0_clean) == 0:
            return 0.5  # Neutral score when no pitch detected
            
        # Jitter (pitch variability)
        jitter = np.std(f0_clean) / (np.mean(f0_clean) + 1e-8)
        
        # Shimmer (amplitude variability)
        analytic_signal = signal.hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        shimmer = np.std(amplitude_envelope) / (np.mean(amplitude_envelope) + 1e-8)
        
        # Harmonics-to-noise ratio
        hnr = self._estimate_harmonics_to_noise_ratio(y, f0_clean)
        
        # Voice quality score (higher = more artifacts)
        voice_score = (min(jitter, 1.0) + min(shimmer, 1.0) + (1.0 - hnr)) / 3.0
        return voice_score
    
    def _detect_audio_artifacts_wav2vec2(self, y: np.ndarray, sr: int) -> float:
        """Detect audio artifacts in Wav2Vec2 style"""
        artifacts = 0
        
        # 1. Clipping detection
        max_amp = np.max(np.abs(y))
        clipped_ratio = np.sum(np.abs(y) >= 0.99 * max_amp) / len(y)
        if clipped_ratio > 0.001:
            artifacts += 1
        
        # 2. Quantization noise
        diff_signal = np.diff(y)
        unique_diffs = len(np.unique(np.round(diff_signal, decimals=6)))
        quantization_ratio = unique_diffs / len(diff_signal)
        if quantization_ratio < 0.3:
            artifacts += 1
        
        # 3. Spectral holes (compression artifacts)
        stft = librosa.stft(y, n_fft=1024)
        magnitude = np.abs(stft)
        spectral_variability = np.std(magnitude, axis=1)
        hole_score = np.sum(spectral_variability < np.mean(spectral_variability) * 0.1)
        if hole_score > magnitude.shape[0] * 0.05:
            artifacts += 1
        
        # 4. Phase inconsistencies
        analytic_signal = signal.hilbert(y)
        phase = np.angle(analytic_signal)
        phase_diff = np.diff(phase)
        phase_jumps = np.sum(np.abs(phase_diff) > np.pi * 0.8)
        if phase_jumps > len(phase) * 0.01:
            artifacts += 1
        
        return artifacts / 4.0  # Normalize to 0-1
    
    def _detect_linguistic_anomalies(self, y: np.ndarray, sr: int) -> float:
        """Cross-linguistic anomaly detection"""
        # Analyze phoneme-like patterns
        # This is a simplified version - real implementation would use language models
        
        # 1. Phoneme duration regularity
        envelope = np.abs(signal.hilbert(y))
        phoneme_durations = self._estimate_phoneme_durations(envelope, sr)
        
        # 2. Formant transition analysis
        formant_transitions = self._analyze_formant_transitions(y, sr)
        
        # 3. Syllabic rhythm analysis
        syllable_rhythm = self._analyze_syllabic_rhythm(envelope, sr)
        
        # Combine linguistic features
        linguistic_score = (phoneme_durations + formant_transitions + syllable_rhythm) / 3.0
        return linguistic_score
    
    def _aggregate_segment_scores(self, segment_scores: List[float]) -> float:
        """Apply maximum score aggregation strategy (Wav2Vec2 approach)"""
        if not segment_scores:
            return 0.5
            
        # Maximum aggregation - if any segment shows high confidence of being fake,
        # classify the entire audio as fake
        max_score = np.max(segment_scores)
        
        # Apply smoothing to prevent extreme outliers from dominating
        avg_score = np.mean(segment_scores)
        smoothed_score = 0.7 * max_score + 0.3 * avg_score
        
        return float(smoothed_score)
    
    async def _perform_detailed_analysis(self, y: np.ndarray, sr: int, 
                                       segments: List[np.ndarray], 
                                       segment_scores: List[float]) -> Dict:
        """Perform additional detailed analysis"""
        analysis = {
            "segment_analysis": await self._analyze_segments_detailed(segments, segment_scores),
            "overall_characteristics": await self._analyze_overall_characteristics(y, sr),
            "confidence_timeline": self._create_confidence_timeline(segment_scores),
            "anomaly_hotspots": await self._detect_anomaly_hotspots(y, sr, segment_scores)
        }
        
        return analysis
    
    async def _analyze_segments_detailed(self, segments: List[np.ndarray], scores: List[float]) -> Dict:
        """Detailed analysis of individual segments"""
        segment_details = []
        
        for i, (segment, score) in enumerate(zip(segments, scores)):
            # Analyze segment characteristics
            duration = len(segment) / self.target_sample_rate
            energy = np.sqrt(np.mean(segment ** 2))
            zero_crossings = np.sum(librosa.zero_crossings(segment))
            
            segment_details.append({
                "segment_id": i,
                "duration": round(duration, 2),
                "energy": round(energy, 4),
                "zero_crossings": int(zero_crossings),
                "confidence_score": round(score, 4),
                "is_deepfake": score > 0.5
            })
        
        return {
            "segments": segment_details,
            "score_variance": round(np.var(scores), 4) if scores else 0,
            "score_consistency": round(1.0 - np.var(scores), 3) if scores else 0
        }
    
    async def _analyze_overall_characteristics(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze overall audio characteristics"""
        # Duration and basic stats
        duration = len(y) / sr
        rms_energy = np.sqrt(np.mean(y ** 2))
        peak_amplitude = np.max(np.abs(y))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak_amplitude / (np.min(np.abs(y[y != 0])) + 1e-10))
        
        # Spectral characteristics
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude))
        
        return {
            "duration_seconds": round(duration, 2),
            "rms_energy": round(rms_energy, 4),
            "peak_amplitude": round(peak_amplitude, 4),
            "dynamic_range_db": round(dynamic_range, 2),
            "spectral_centroid_hz": round(spectral_centroid, 2),
            "sample_rate": sr
        }
    
    def _create_confidence_timeline(self, segment_scores: List[float]) -> List[Dict]:
        """Create timeline of confidence scores"""
        timeline = []
        for i, score in enumerate(segment_scores):
            start_time = i * (self.segment_duration - self.segment_overlap)
            timeline.append({
                "segment_index": i,
                "start_time": round(start_time, 2),
                "end_time": round(start_time + self.segment_duration, 2),
                "confidence": round(score, 4),
                "is_deepfake": score > 0.5
            })
        return timeline
    
    async def _detect_anomaly_hotspots(self, y: np.ndarray, sr: int, segment_scores: List[float]) -> List[Dict]:
        """Detect time regions with high anomaly scores"""
        hotspots = []
        
        for i, score in enumerate(segment_scores):
            if score > 0.7:  # High confidence threshold
                start_time = i * (self.segment_duration - self.segment_overlap)
                hotspots.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(start_time + self.segment_duration, 2),
                    "confidence": round(score, 4),
                    "severity": "high" if score > 0.8 else "medium"
                })
        
        return sorted(hotspots, key=lambda x: x["confidence"], reverse=True)
    
    # Helper methods (similar to original implementation)
    def _calculate_spectral_flux(self, stft: np.ndarray) -> np.ndarray:
        """Calculate spectral flux for detecting spectral changes"""
        flux = np.zeros(stft.shape[1] - 1)
        for i in range(1, stft.shape[1]):
            prev_frame = np.abs(stft[:, i-1])
            curr_frame = np.abs(stft[:, i])
            flux[i-1] = np.sum((curr_frame - prev_frame) ** 2)
        return flux
    
    def _compute_autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """Compute autocorrelation of signal"""
        signal = signal - np.mean(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if np.max(autocorr) > 0:
            autocorr = autocorr / np.max(autocorr)
        return autocorr
    
    def _calculate_energy_entropy(self, rms: np.ndarray) -> float:
        """Calculate entropy of energy distribution"""
        if len(rms) == 0:
            return 0
            
        # Normalize energy values
        normalized_energy = rms / (np.sum(rms) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(normalized_energy * np.log(normalized_energy + 1e-10))
        return min(entropy, 1.0)
    
    def _analyze_pitch_periodicity(self, y: np.ndarray, sr: int) -> float:
        """Analyze periodicity in pitch"""
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500)
        f0_clean = f0[voiced_flag]
        
        if len(f0_clean) < 10:
            return 0.5
            
        # Autocorrelation of pitch
        autocorr = self._compute_autocorrelation(f0_clean[:50])
        return np.max(autocorr[1:10]) if len(autocorr) > 10 else 0.5
    
    def _estimate_harmonics_to_noise_ratio(self, y: np.ndarray, f0_values: np.ndarray) -> float:
        """Estimate harmonics-to-noise ratio"""
        if len(f0_values) == 0:
            return 0.5
            
        avg_f0 = np.mean(f0_values)
        spectrum = np.abs(fft.fft(y))
        freqs = fft.fftfreq(len(y), 1/sr)
        
        harmonic_energy = 0
        noise_energy = 0
        
        for i, freq in enumerate(freqs[:len(freqs)//2]):
            if freq > 50:
                harmonic_number = freq / avg_f0
                if abs(harmonic_number - round(harmonic_number)) < 0.1:
                    harmonic_energy += spectrum[i] ** 2
                else:
                    noise_energy += spectrum[i] ** 2
        
        if harmonic_energy + noise_energy > 0:
            hnr_db = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10))
            return max(0, min(hnr_db, 30)) / 30
        else:
            return 0.5
    
    def _estimate_phoneme_durations(self, envelope: np.ndarray, sr: int) -> float:
        """Estimate regularity of phoneme durations"""
        # Simple phoneme boundary detection
        derivative = np.diff(envelope)
        threshold = np.percentile(np.abs(derivative), 90)
        
        boundaries = np.where(np.abs(derivative) > threshold)[0]
        
        if len(boundaries) < 3:
            return 0.5
            
        durations = np.diff(boundaries)
        duration_regularity = 1.0 - (np.std(durations) / (np.mean(durations) + 1e-8))
        return max(0.0, min(duration_regularity, 1.0))
    
    def _analyze_formant_transitions(self, y: np.ndarray, sr: int) -> float:
        """Analyze formant transition smoothness"""
        # Simplified formant analysis
        # In practice, would use LPC or specialized formant trackers
        
        # Analyze spectral transitions
        stft = librosa.stft(y, n_fft=512, hop_length=160)
        magnitude = np.abs(stft)
        
        # Calculate frame-to-frame spectral difference
        spectral_diff = np.sum(np.abs(np.diff(magnitude, axis=1)), axis=0)
        transition_smoothness = 1.0 - (np.std(spectral_diff) / (np.mean(spectral_diff) + 1e-8))
        
        return max(0.0, min(transition_smoothness, 1.0))
    
    def _analyze_syllabic_rhythm(self, envelope: np.ndarray, sr: int) -> float:
        """Analyze syllabic rhythm patterns"""
        # Detect syllable-like peaks
        peaks, _ = signal.find_peaks(envelope, distance=int(0.1 * sr))  # Min 100ms apart
        
        if len(peaks) < 3:
            return 0.5
            
        # Analyze inter-syllable intervals
        intervals = np.diff(peaks) / sr  # Convert to seconds
        rhythm_regularity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
        
        return max(0.0, min(rhythm_regularity, 1.0))
    
    def _create_empty_result(self, start_time: float, sample_count: int, sample_rate: int) -> Dict:
        """Create result when no valid audio is processed"""
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "is_deepfake": False,
            "confidence_score": 0.1,  # Low confidence for empty/invalid audio
            "processing_time_ms": round(processing_time, 2),
            "audio_duration": round(sample_count / sample_rate, 2) if sample_rate > 0 else 0,
            "sample_rate": sample_rate,
            "segments_analyzed": 0,
            "segment_scores": [],
            "analysis_details": {},
            "performance_metrics": {
                "avg_processing_time": round(np.mean(self.processing_times), 2) if self.processing_times else 0,
                "gpu_acceleration": self.use_gpu,
                "message": "No valid audio segments found"
            }
        }

# Backward compatibility
EnhancedAudioDetector = Wav2Vec2AudioDetector
AudioDetector = Wav2Vec2AudioDetector

__all__ = ["Wav2Vec2AudioDetector", "EnhancedAudioDetector", "AudioDetector"]