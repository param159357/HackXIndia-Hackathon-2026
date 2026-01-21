Smart Traffic Violation Priority Engine:
Video Link 🔗 - https://drive.google.com/file/d/1s2WI-ucgKAdeT-Y8WMF3plNh4hfDEz9-/view?usp=drivesdk
The Smart Traffic Violation Priority Engine is a traffic monitoring and prioritisation system built to help authorities identify which traffic violations need immediate attention instead of treating all violations the same.

The system analyses traffic video, detects violations, assigns a risk score, and highlights high-priority cases such as emergency vehicles or dangerous driving situations on a single dashboard.

Problem Statement:
In real traffic conditions, all violations are usually logged in the same way, even though some situations are far more critical than others.

For example:
1- An ambulance stuck at a signal
2- Overspeeding in a crowded area
3- Multiple violations happening in one zone

Traffic authorities often need to manually scan data and video feeds, which slows down response time.

Proposed Solution:
This project focuses on priority, not just detection.

The system:
1- Analyses traffic video (real or simulated)
2- Detects common violations
3- Assigns a risk score to each violation
4- Gives highest priority to emergency vehicles
5- Shows area-wise alerts on an Stack interactive dashboard
This helps authorities quickly understand where the problem is and what should be handled first.

Key Features:
1- Live traffic video feed support
2- Simulation mode (works without real CCTV input)
3- Real-time violation alerts (speeding, red 4- light jump, etc.)
5- Risk-based prioritisation of violations
6- Interactive map for area-wise monitoring
7- Central dashboard for traffic control use.
8- Simple dark UI for better visibility

System Overview:
Traffic videos are loaded into the system
Violations are detected and scored
High-risk violations move up in priority
Emergency vehicles override all other alerts
Data is visualised using maps, charts, and alerts

Technology Stack:
Frontend:
HTML, CSS, JavaScript
Tailwind CSS
Leaflet.js
Chart.js
Backend / Processing:
Python
Streamlit
OpenCV
YOLO (for object detection)

Use Cases:
1- Traffic control centres
2- Emergency response monitoring
3- Smart city demonstrations
4- Academic and hackathon projects
Current Status:
This project is a working prototype intended to demonstrate the idea of priority-based traffic violation handling.

Future Scope:
Live CCTV (RTSP) integration
Database for violation history
Multi-city support
Signal control automation
Mobile dashboard

Summary:
The Smart Traffic Violation Priority Engine helps traffic authorities focus on what matters most by prioritising violations based on risk and urgency.
The goal is faster response, better visibility, and more informed decision-making.
