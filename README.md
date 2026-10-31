# Hand-Volume-Control

Control your computer's volume and media playback with hand gestures, using your webcam,
[MediaPipe](https://developers.google.com/mediapipe) hand tracking and OpenCV.

## Gestures

| Gesture | Action |
|---|---|
| ✋ Open palm (all 5 fingers) | **Unlock**: gestures start working |
| ✊ Fist | **Lock**: gestures are ignored, so normal hand movements don't change anything |
| 🤏 Pinch (thumb + index) | Change the volume: fingers together = 0%, spread apart = 100% |
| ✌️ Peace sign (index + middle) | Play / pause media |

The app starts **locked**. Show an open palm first. Press `q` in the video window to quit.

## Installation

```bash
git clone https://github.com/Moryel/Hand-Volume-Control
cd Hand-Volume-Control/HandAndVolume
pip install -r requirements.txt
```

## Usage

```bash
python hand_volume_control.py
```

- **macOS:** allow camera access for your terminal app when asked
  (System Settings → Privacy & Security → Camera). Play/pause also needs
  Accessibility access for the terminal app.
- **Windows:** volume is set with `pycaw` (installed automatically on Windows).
- **Linux:** volume is set with `amixer`, or `pactl` if `amixer` isn't available.

## How it works

- **Works at any distance:** the pinch length is divided by the size of your hand
  (wrist to middle-finger base), so moving closer to or further from the camera
  doesn't change the volume.
- **Smooth:** the volume is the average of the last 5 frames, and the system volume
  is only changed when it moves by 2% or more.
- **Debounced:** lock/unlock waits 0.5 s between changes, play/pause 1 s.
- **On screen:** hand skeleton, lock state, volume bar, current gesture and FPS.
- The hand model (`hand_landmarker.task`) is included; if it's missing it is
  downloaded next to the script on first run.

## Requirements

- Python 3.9+
- A webcam
- Windows, macOS or Linux

## License

MIT, see [LICENSE](LICENSE).
