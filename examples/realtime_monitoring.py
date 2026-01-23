"""DASMatrix 实时监测示例
======================

演示如何使用 DASMatrix 的 Stream API 和 RealtimeVisualizer 进行高性能实时监控。
"""

import time

import numpy as np

from DASMatrix.api import Stream
from DASMatrix.visualization.realtime import RealtimeVisualizer


def process_and_visualize():
    """Run the real-time monitoring demo."""

    # Configuration
    DURATION = 10.0  # Total run duration
    FS = 1000.0  # Sampling rate
    N_CHANNELS = 100  # Number of channels
    CHUNK_SIZE = 100  # Process in 100ms chunks

    print("=== DASMatrix Real-time Monitoring Demo ===")
    print(f"Duration: {DURATION}s, Fs: {FS}Hz, Channels: {N_CHANNELS}")

    # 1. Initialize Visualizer (runs in main thread usually, or separate if using blitting carefully)
    # We use a 2-second rolling window for visualization
    viz = RealtimeVisualizer(
        n_channels=N_CHANNELS, duration=2.0, fs=FS, title="Real-time DAS Monitor"
    )

    # 2. Initialize Stream (Simulated Source)
    stream = Stream(
        source="simulation",
        buffer_duration=5.0,
        fs=FS,
        n_channels=N_CHANNELS,
        chunk_size=CHUNK_SIZE,
    )

    # 3. Define Callback
    def on_chunk(frame):
        """Callback to process each incoming chunk."""
        # A. Basic Processing (e.g., Bandpass + Envelope)
        # Note: In a real scenario, you'd maintain state (Atoms).
        # Here we just process the chunk statelessly for demo.
        processed = frame.detrend().envelope()

        # B. Event Detection
        # Simple threshold on the envelope
        events = processed.threshold_detect(sigma=4.0)
        n_events = np.sum(events)

        # C. Update Visualization
        # We need to get the data out.
        env_data = processed.data.values  # (time, channels)

        # Taking the max channel for the line plot
        line_trace = np.max(env_data, axis=1)

        # For the heatmap, we might want to push to a rolling buffer for visualization.
        # But RealtimeVisualizer.update expects a full frame for the window?
        # Actually RealtimeVisualizer implementation expects data matching its full X-axis?
        # Let's check RealtimeVisualizer.update implementation:
        # self.im.set_data(heatmap_data.T) -> expects (samples, channels)
        # It updates the WHOLE image. So we need a rolling buffer for VISUALIZATION too.
        # Ideally, Stream.view() gives us the rolling window!

        pass

    # We need a loop to update visualization from the main thread
    # The stream callback runs in a background thread.
    # So we'll use a shared buffer or just pull from stream.view() in the main loop.

    stream.start()  # Starts ingestion thread

    try:
        start_time = time.time()
        while time.time() - start_time < DURATION:
            # 1. Get recent view for visualization (e.g., last 2 seconds)
            view_frame = stream.view(duration=2.0)

            # Check if we have data (access via xarray shape)
            if view_frame.data.shape[0] > 0:
                # Process the VIEW for visualization
                # (In production, you might process raw stream for detection and just visualize raw/processed view)

                # Simple processing for viz
                # fast detrend + envelope
                # Note: creating new objects every frame, might be heavy.
                # For high performance, we'd use inplace ops or pure numpy.
                # But let's demonstrate the API.
                try:
                    proc_view = view_frame.detrend().abs()  # specific for demo speed

                    data = proc_view.data.values  # (time, channels)

                    # Pad if not enough data yet
                    target_samples = int(2.0 * FS)
                    if data.shape[0] < target_samples:
                        padding = np.zeros((target_samples - data.shape[0], N_CHANNELS))
                        data = np.vstack([padding, data])
                    elif data.shape[0] > target_samples:
                        data = data[-target_samples:]

                    # Prepare line trace (e.g. mean energy)
                    line_data = np.mean(data, axis=1)

                    # Update Visualizer
                    viz.update(heatmap_data=data, line_data=line_data)

                except Exception as e:
                    print(f"Viz Error: {e}")

            # Limit refresh rate to ~30 FPS
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop()
        viz.close()
        print("Done.")


if __name__ == "__main__":
    process_and_visualize()
