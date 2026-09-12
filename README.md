# Fire-Specific Search and Rescue Robot
## Low-Cost Integrated Multi-Sensor AI Fusion

> AI component of a fire-specific disaster-response and search-and-rescue robot.

---

## 1. Project Overview

This project develops the **AI software architecture** for a low-cost fire-specific search and rescue robot.

The robot is intended to operate in hazardous environments such as fire-affected indoor areas where visibility may be poor and direct human intervention may be dangerous.

The system combines:

- Environmental sensor data
- Camera-based visual information
- Sensor fusion
- Temporal analysis
- Hazard assessment
- Victim detection
- Rescue prioritization

The overall goal is to move beyond individual sensor-based decisions and create a system that can combine multiple sources of evidence before making a rescue-related assessment.

### Important development constraint

The current AI development is being performed **without the physical robot hardware**.

Therefore, sensor readings are currently generated using a **software sensor simulator**. The simulator is designed so that it can later be replaced by real ESP32 sensor readings without requiring major changes to the rest of the AI pipeline.

---

# 2. Problem Statement

Fire-rescue environments present several challenges:

- Smoke and fire can reduce camera visibility.
- Individual sensors provide only partial information about the environment.
- A victim may be difficult to detect or may be partially obscured.
- Sensor readings can fluctuate over time.
- Rescue decisions should consider multiple environmental and victim-related factors instead of relying on a single sensor.

The proposed system addresses these problems using a **multimodal AI pipeline** that combines visual and sensor information.

---

# 3. Research Gap

Existing fire-rescue and disaster-response systems often focus on individual capabilities such as:

- Fire detection
- Smoke detection
- Victim detection
- Environmental monitoring
- Gas detection
- Robotic triage
- Hazard mapping

These capabilities are frequently implemented as separate modules.

The proposed work investigates a lightweight integrated pipeline that combines:

```text
Environmental Sensors
        +
Camera-Based Vision
        +
Temporal Information
        ↓
   AI Fusion Layer
        ↓
Hazard Assessment
        +
Victim Assessment
        ↓
 Rescue Priority