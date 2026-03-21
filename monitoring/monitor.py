import time
import os

while True:

    os.system("python monitoring/drift_detector.py")

    print("Monitoring cycle completed")

    time.sleep(10)  # run every hour