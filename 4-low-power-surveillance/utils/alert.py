import time

class AlertManager:
    def __init__(self, alert_classes, alert_once_per):
        self.alert_classes = alert_classes
        self.alert_once_per = alert_once_per
        self.last_alert_frame = {}

    def should_alert(self, cls_name, current_frame):
        if cls_name not in self.alert_classes:
            return False

        last_frame = self.last_alert_frame.get(cls_name, -self.alert_once_per - 1)
        if current_frame - last_frame >= self.alert_once_per:
            self.last_alert_frame[cls_name] = current_frame
            return True
        return False

    def trigger_alert(self, cls_name):
        print(f"[ALERT] {cls_name} detected at {time.strftime('%H:%M:%S')}")
