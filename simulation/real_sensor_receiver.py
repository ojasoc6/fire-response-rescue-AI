import serial
import json
import threading
import time

class RealSensorReceiver:
    def __init__(self, port="COM9", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.latest_data = {
            "front_cm": -1.0,
            "left_cm": -1.0,
            "right_cm": -1.0,
            "gas_raw": 0,
            "gas_volt": 0.0,
            "gas_alert": False,
            "flame_detected": False,
            "acc_x": 0.0,
            "acc_y": 0.0,
            "acc_z": 1.0,
            "tilt": False,
            "connected": False,
            "timestamp": 0.0
        }
        self.running = False
        self.thread = None
        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        ser = None
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            with self._lock:
                self.latest_data["connected"] = True
            print(f"[RealSensorReceiver] Connected to ESP32 on {self.port}")
        except Exception as e:
            print(f"[RealSensorReceiver] Failed to connect on {self.port}: {e}")
            with self._lock:
                self.latest_data["connected"] = False
            return

        while self.running:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("source") == "esp32":
                    with self._lock:
                        self.latest_data.update({
                            "front_cm": float(data.get("front_cm", -1.0)),
                            "left_cm": float(data.get("left_cm", -1.0)),
                            "right_cm": float(data.get("right_cm", -1.0)),
                            "gas_raw": int(data.get("gas_raw", 0)),
                            "gas_volt": float(data.get("gas_volt", 0.0)),
                            "gas_alert": bool(data.get("gas_alert", False)),
                            "flame_detected": bool(data.get("flame_detected", False)),
                            "acc_x": float(data.get("acc_x", 0.0)),
                            "acc_y": float(data.get("acc_y", 0.0)),
                            "acc_z": float(data.get("acc_z", 1.0)),
                            "tilt": bool(data.get("tilt", False)),
                            "timestamp": time.time(),
                            "connected": True
                        })
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as e:
                print(f"[RealSensorReceiver] Read loop error: {e}")
                break

        if ser and ser.is_open:
            ser.close()
        with self._lock:
            self.latest_data["connected"] = False

    def get_readings(self):
        with self._lock:
            return self.latest_data.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

if __name__ == "__main__":
    receiver = RealSensorReceiver(port="COM9", baudrate=115200)
    receiver.start()
    try:
        while True:
            data = receiver.get_readings()
            print("[Live Telemetry]", data)
            time.sleep(0.2)
    except KeyboardInterrupt:
        receiver.stop()
