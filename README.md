The project is divided into three core parts:

1️⃣ Pose Detection – extracting human landmarks using MediaPipe

2️⃣ Pose Analysis – computing joint angles, torso orientation, and a custom shoulder-lift normalization metric

3️⃣ Visualization & Interaction – a real-time GUI with live metrics, pose quality feedback, and hold-time tracking

The entire system is designed using Python classes and object-oriented programming (OOP), with clear separation of responsibilities (pose detection, analysis logic, and UI), making the codebase modular, readable, and easy to extend.

🔍 Key features:

Real-time analysis from live webcam feed and recorded videos

Joint-angle computation for elbows, shoulders, and hips using geometric analysis

Pose classification into Poor / Average / Good based on biomechanics and stability

Robust handling of real-world noise and camera tilt
