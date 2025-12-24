# Hand Gesture Volume Control

Production-ready hand gesture volume control using MediaPipe and OpenCV.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python hand_volume_control.py
```

**Controls:**
- Move thumb and index finger closer together → Lower volume
- Spread thumb and index finger apart → Increase volume
- Press `q` to quit

## Features

✅ **OS-Independent**: Works on Windows (pycaw) and Linux (amixer)  
✅ **Robust Camera Handling**: Auto-fallback from index 0 to 1  
✅ **MediaPipe Hand Tracking**: High confidence (0.7) to avoid jitter  
✅ **Smoothing**: Only updates volume if change exceeds threshold  
✅ **Visual Debugging**: Volume bar, hand skeleton, finger line  
✅ **Console Logging**: Prints "Volume set to: X%" for debugging

## Requirements

- Python 3.8+
- Webcam
- Windows: pycaw for volume control
- Linux: amixer (ALSA utils)

## Technical Details

- **Landmarks Used**: 4 (Thumb tip), 8 (Index tip)
- **Distance Range**: 30-300 pixels
- **Volume Range**: 0-100%
- **Smoothing Threshold**: 3% minimum change
- **Detection Confidence**: 0.7
