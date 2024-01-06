import time


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.is_running = False

    def start(self):
        if self.start_time is not None:
            print("Stopwatch is already running.")
            return
        self.start_time = time.time()
        self.is_running = True

    def stop(self):
        if self.start_time is None:
            print("Stopwatch is not running.")
            return
        elapsed_time = time.time() - self.start_time
        self.start_time = None
        self.is_running = False
        return elapsed_time

    def display(self):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            print(f"\rElapsed Time: {elapsed_time:,.2f} seconds", end="")


stopwatch = Stopwatch()

# Start the stopwatch
stopwatch.start()

try:
    while stopwatch.is_running:
        stopwatch.display()
        time.sleep(0.1)  # Update the display every 0.1 seconds
except KeyboardInterrupt:
    # Stop the stopwatch when a keyboard interrupt is received (Ctrl+C)
    elapsed_time = stopwatch.stop()
    print(f"\nStopped. Total Elapsed Time: {elapsed_time:,.2f} seconds")
