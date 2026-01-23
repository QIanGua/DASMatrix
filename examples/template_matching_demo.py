import matplotlib.pyplot as plt
import numpy as np

from DASMatrix import from_array


def main():
    fs = 1000.0
    nt, nx = 2000, 10
    data = np.random.randn(nt, nx).astype(np.float32) * 0.5

    t_temp = np.linspace(0, 0.1, 100)
    template_1d = np.sin(2 * np.pi * 30 * t_temp) * np.exp(-50 * t_temp)

    data[500:600, 2] += 2.0 * template_1d
    data[1200:1300, 5] += 2.5 * template_1d

    frame = from_array(data, fs=fs)
    match_results = frame.template_match(template_1d)

    mt, mx = 100, 5
    template_2d = np.zeros((mt, mx), dtype=np.float32)
    for j in range(mx):
        template_2d[j * 10 : j * 10 + 50, j] = 1.0

    data[1000:1100, 3:8] += 3.0 * template_2d
    frame_2d = from_array(data, fs=fs)
    match_2d = frame_2d.template_match(template_2d)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    frame_2d.plot_heatmap(ax=axes[0, 0], title="Raw Data")
    match_results.plot_heatmap(ax=axes[0, 1], title="1D Match (NCC)")
    match_2d.plot_heatmap(ax=axes[1, 0], title="2D Match (Spatio-Temporal)")

    axes[1, 1].plot(match_results.collect()[:, 2], label="CH 2")
    axes[1, 1].plot(match_results.collect()[:, 0], label="CH 0", alpha=0.5)
    axes[1, 1].set_title("Matching Confidence")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
