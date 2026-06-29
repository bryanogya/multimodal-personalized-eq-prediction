import numpy as np

from configs.frequencies import TARGET_FREQ
from configs.eq import *

EQ_INDICES = [
    np.argmin(np.abs(TARGET_FREQ - freq))
    for freq in EQ_FREQS
]

def extract_audio_characteristics(mel_spectrogram):
    mel_energy = np.exp(mel_spectrogram)
    bass_energy = np.mean(mel_energy[:20])
    mid_energy = np.mean(mel_energy[20:80])
    treble_energy = np.mean(mel_energy[80:])
    
    total_energy = (
        bass_energy +
        mid_energy +
        treble_energy +
        1e-8
    )

    bass_ratio = bass_energy / total_energy
    mid_ratio = mid_energy / total_energy
    treble_ratio = treble_energy / total_energy

    return (
        bass_ratio,
        mid_ratio,
        treble_ratio
    )


def generate_target_eq(
    mel_spectrogram,
    device_response,
    preference,
    harman_target
):
    bass_pref, mid_pref, treble_pref = preference
    personalized_target = harman_target.copy()
    
    bass_mask = TARGET_FREQ < BASS_CUTOFF
    mid_mask = ((TARGET_FREQ >= BASS_CUTOFF) & (TARGET_FREQ < MID_CUTOFF))
    treble_mask = (TARGET_FREQ >= MID_CUTOFF)

    # USER PREFERENCE
    personalized_target[bass_mask] += ((bass_pref - 0.5) * PREFERENCE_BASS_GAIN)
    personalized_target[mid_mask] += ((mid_pref - 0.5) * PREFERENCE_MID_GAIN)
    personalized_target[treble_mask] += ((treble_pref - 0.5) * PREFERENCE_TREBLE_GAIN)

    # AUDIO CHARACTER
    (bass_ratio, mid_ratio, treble_ratio) = extract_audio_characteristics(mel_spectrogram)

    # Lagu terlalu bass-heavy
    if bass_ratio > BASS_HEAVY_THRESHOLD:
        personalized_target[bass_mask] -= AUDIO_BASS_ADJUST

    # Lagu bass kurang
    elif bass_ratio < BASS_LIGHT_THRESHOLD:
        personalized_target[bass_mask] += AUDIO_BASS_ADJUST

    # Lagu terlalu bright
    if treble_ratio > TREBLE_HEAVY_THRESHOLD :
        personalized_target[treble_mask] -= AUDIO_TREBLE_ADJUST

    # Lagu terlalu gelap
    elif treble_ratio < TREBLE_LIGHT_THRESHOLD:
        personalized_target[treble_mask] += AUDIO_TREBLE_ADJUST

    # DEVICE COMPENSATION
    eq_curve_raw = (personalized_target - device_response)
    
    eq_curve = np.clip(
        eq_curve_raw,
        EQ_MIN_GAIN,
        EQ_MAX_GAIN
    )

    eq_gains = np.array([
            eq_curve[idx]
            for idx in EQ_INDICES
    ])
    
    return {
        "target_eq": eq_gains.astype(np.float32),
        "target_curve": eq_curve.astype(np.float32),
        "eq_curve_raw": eq_curve_raw.astype(np.float32),
        "personalized_target": personalized_target.astype(np.float32),
        "audio_character": {
            "bass_ratio": float(bass_ratio),
            "mid_ratio": float(mid_ratio),
            "treble_ratio": float(treble_ratio)
        }
    }
    
    
    