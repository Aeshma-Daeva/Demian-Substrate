"""Stable public API for the synthesized Demian v1 recurrent substrate."""

from demian_v1.audio_dynamics import FEATURE_NAMES, AudioFeatureExtractor
from demian_v1.audio_probe import DeterministicAudioCoupler, read_pcm_wav, run_audio_probe
from demian_v1.audio_stream import AudioStreamConfig, IncrementalAudioProcessor
from demian_v1.audio_lifecycle import SegmentedLiveAudioSession, SegmentedState
from demian_v1.live_audio import CaptureBackend, LiveAudioSession, LiveAudioState
from demian_v1.virtual_audio import WavCaptureBackend
from demian_v1.runtime import (
    DEMIAN_V1_ID,
    DemianV1Config,
    DemianV1Runtime,
    DemianV1Snapshot,
    deserialize_state,
    serialize_state,
)
from development.demian_v1_gate_state import (
    V1_CHANNELS,
    DemianV1GateState,
    V1Ablation,
    V1ResumeResult,
    V1State,
    clamp_v1_channel,
    clone_state,
    compare_v1_resume,
    run_v1_trace,
    surface_only_resume_state,
)

__all__ = [
    "DEMIAN_V1_ID",
    "FEATURE_NAMES",
    "V1_CHANNELS",
    "AudioFeatureExtractor",
    "DeterministicAudioCoupler",
    "DemianV1Config",
    "DemianV1GateState",
    "DemianV1Runtime",
    "DemianV1Snapshot",
    "V1Ablation",
    "V1ResumeResult",
    "V1State",
    "clamp_v1_channel",
    "clone_state",
    "compare_v1_resume",
    "deserialize_state",
    "read_pcm_wav",
    "run_v1_trace",
    "run_audio_probe",
    "AudioStreamConfig",
    "CaptureBackend",
    "IncrementalAudioProcessor",
    "LiveAudioSession",
    "LiveAudioState",
    "SegmentedLiveAudioSession",
    "SegmentedState",
    "serialize_state",
    "surface_only_resume_state",
    "WavCaptureBackend",
]
