    # LFM 能量估计算法分析与 C++ 实现

    本文档对 `estimate_lfm_power_stft_known_chirp` 算法进行复杂度分析，探讨优化空间，并提供对应的 C++ 实现方案。

    ## 1. 算法复杂度分析

    假设：
    - 信号长度为 $N$。
    - STFT 窗长为 $M$ (`nperseg`)，通常 $M \ll N$。
    - FFT 点数为 $K$ (`nfft`)，通常 $K \ge M$。
    - 时间帧数为 $T \approx N / \text{hop}$。

    主要步骤复杂度分解：

    1.  **预处理 (带通滤波)**:
        - 复杂度: $O(N)$。
        - 说明: IIR 滤波器（如 Butterworth）是线性的。

    2.  **STFT 计算**:
        - 复杂度: $O(T \cdot K \log K)$。
        - 说明: 即 $O(N \cdot \frac{K}{hop} \log K)$。这是算法中最耗时的部分之一。

    3.  **脊线检测 (Candidate Search)**:
        - 复杂度: $O(T \cdot K_{band})$。
        - 说明: $K_{band}$ 是感兴趣频带内的频点数。需要在每一帧中寻找最大值并计算中位数（或噪声底）。虽然仅仅是遍历，但内存访问此时可能成为瓶颈。

    4.  **$t_0$ 鲁棒估计 (Robust Estimation)**:
        - 假设候选点数量为 $C$ (最坏情况 $C \approx T$)。
        - 中位数计算 (Quick Select/Introselect): $O(C)$。
        - 迭代 MAD: 迭代次数 $I$ (常数, e.g., 3)。总复杂度 $O(I \cdot C)$。
        - 总体: $O(T)$。这一步非常快。

    5.  **掩膜生成与应用**:
        - 复杂度: $O(T \cdot K)$。
        - 说明: 涉及矩阵的点对点操作。

    6.  **逆 STFT (ISTFT)**:
        - 复杂度: $O(T \cdot K \log K)$。
        - 说明: 与 STFT 相同。

    7.  **能量计算**:
        - 复杂度: $O(N)$。

    **总时间复杂度**: $O(N \log K)$ (主导项为 FFT/IFFT)。
    **空间复杂度**: $O(T \cdot K)$ (存储完整的 STFT 频谱矩阵)。

    ---

    ## 2. 优化空间

    ### 2.1 移除 ISTFT (重大优化)
    目前的算法流程是：`STFT -> Mask -> ISTFT -> Time Domain Power`。
    由 **帕塞瓦尔定理 (Parseval's Theorem)**，时域能量等于频域能量（需考虑窗函数归一化因子）。
    我们可以直接在 STFT 域计算掩膜内的能量，从而**完全省去 ISTFT 步骤**。

    优化后复杂度降低约 40%-50%（减少一半的 FFT 运算）。公式如下：
    $$ \sum |x[n]|^2 \propto \sum_{m, k \in Mask} |S[m, k]|^2 $$
    *注意：需要根据窗口类型和重叠率计算修正系数 (NOLA constraint)*。

    ### 2.2 频带限制 (Zoom FFT / Decimation)
    如果是宽带采样（如 16kHz）但只关注窄带信号（如 1-2kHz），可以在 STFT 之前对信号进行**降采样 (Decimation)** 或移频。这将减少 $N$，从而线性地减少所有后续步骤的计算量。

    ### 2.3 内存优化
    目前的 Python 实现存储了整个 `Z` 矩阵 (Complex128)。在 C++ 中，可以采用**流式处理 (Streaming)**：
    - 逐块读取信号。
    - 计算当前块的 FFT。
    - 提取候选点 $(t, f, P)$ 并存储，丢弃频谱数据。
    - 等待所有块处理完，进行 $t_0$ 估计。
    - *难点*：如果需要重构波形或计算精确的 Mask 能量，可能需要二次遍历或保留压缩后的数据。对于“仅估计功率”的任务，我们可以只存储候选点的元数据，而在确认 $t_0$ 后无法回溯应用 Mask（除非保留频谱）。
    - **折中方案**: 若采用 Mask 能量积分法，且 $t_0$ 是全局依赖的，我们必须保留感兴趣频带 magnitude 谱（float），丢弃相位，内存减半。

    ### 2.4 SIMD 向量化
    在“寻找峰值”和“掩膜应用”步骤，可以使用 AVX/NEON 指令集进行加速。

    ---

    ## 3. C++ 实现 (Header-only 风格示例)

    以下是一个基于 C++17 的实现草案。为了保持自包含，假设我们使用 `fftw3` 进行 FFT 计算。

    ### 依赖
    - [FFTW3](http://www.fftw.org/) (或 MKL, KissFFT)
    - C++17 标准库

    ### 代码实现 (`LfmEstimator.hpp`)

    ```cpp
    #include <vector>
    #include <complex>
    #include <cmath>
    #include <algorithm>
    #include <numeric>
    #include <iostream>
    #include <fftw3.h> // 需要链接 -lfftw3 -lm

    // 常量定义
    constexpr double PI = 3.14159265358979323846;

    struct LfmConfig {
        double fs = 16000.0;
        double f_start = 1000.0;
        double f_end = 2000.0;
        double T = 3.0;
        int nperseg = 1024;    // 建议 2^N
        int noverlap = 768;    // 75% overlap
        double gamma = 2.0;
        double delta_f_hz = 150.0;
        double mask_bw_hz = 120.0;
    };

    struct LfmResult {
        double power;
        double duration;
        double t0_detected;
    };

    class LfmEstimator {
    public:
        LfmEstimator(const LfmConfig& cfg) : cfg_(cfg) {
            // 初始化 FFTW 资源
            nfft_ = cfg_.nperseg; // 简单起见 nfft = nperseg
            in_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nfft_);
            out_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nfft_);
            plan_ = fftw_plan_dft_1d(nfft_, in_, out_, FFTW_FORWARD, FFTW_ESTIMATE);

            // 生成 Hann 窗
            window_.resize(nfft_);
            for (int i = 0; i < nfft_; ++i) {
                window_[i] = 0.5 * (1.0 - std::cos(2.0 * PI * i / (nfft_ - 1)));
            }
        }

        ~LfmEstimator() {
            fftw_destroy_plan(plan_);
            fftw_free(in_);
            fftw_free(out_);
        }

        LfmResult process(const std::vector<float>& x) {
            // 0. 参数准备
            int hop = cfg_.nperseg - cfg_.noverlap;
            size_t n_frames = (x.size() - cfg_.nperseg) / hop + 1;
            double df = cfg_.fs / nfft_;
            double dt = (double)hop / cfg_.fs;
            
            // Chirp 斜率
            double slope = (cfg_.f_end - cfg_.f_start) / cfg_.T;

            // 频带索引限制
            int bin_start = (int)(std::min(cfg_.f_start, cfg_.f_end) / df);
            int bin_end = (int)(std::max(cfg_.f_start, cfg_.f_end) / df);
            if (bin_start < 0) bin_start = 0;
            if (bin_end >= nfft_/2) bin_end = nfft_/2 - 1;

            // 存储 STFT 结果 (Magnitude only for power estimation)
            // 优化：只存储 float magnitude，省去 complex
            std::vector<std::vector<float>> spectrogram(n_frames);
            std::vector<double> frame_times(n_frames);

            struct Candidate {
                int frame_idx;
                double t;
                double f_peak;
                double p_peak;
            };
            std::vector<Candidate> candidates;

            // 1. STFT & Candidate Detection Loop
            for (size_t i = 0; i < n_frames; ++i) {
                size_t start_idx = i * hop;
                
                // 加窗 & 复制数据
                for (int k = 0; k < nfft_; ++k) {
                    // 如果 x 长度不足，此处需处理边界，此处假设 x 足够长
                    if (start_idx + k < x.size())
                        in_[k][0] = x[start_idx + k] * window_[k];
                    else
                        in_[k][0] = 0.0;
                    in_[k][1] = 0.0;
                }

                fftw_execute(plan_);

                // 计算 Magnitude Squared (Power)
                std::vector<float>& mag_frame = spectrogram[i];
                mag_frame.resize(bin_end - bin_start + 1);
                
                double max_p = -1.0;
                int max_k = -1;
                std::vector<double> band_powers;
                band_powers.reserve(bin_end - bin_start + 1);

                for (int k = bin_start; k <= bin_end; ++k) {
                    double re = out_[k][0];
                    double im = out_[k][1];
                    double p = re*re + im*im; // |X|^2
                    
                    mag_frame[k - bin_start] = (float)p;
                    band_powers.push_back(p);

                    if (p > max_p) {
                        max_p = p;
                        max_k = k;
                    }
                }

                // 本底噪声估计 (Median)
                double noise_floor = 1e-20;
                if (!band_powers.empty()) {
                    size_t n = band_powers.size();
                    size_t mid = n / 2;
                    std::nth_element(band_powers.begin(), band_powers.begin() + mid, band_powers.end());
                    noise_floor = band_powers[mid];
                }

                double t_curr = i * dt;
                frame_times[i] = t_curr;

                // 阈值判断
                if (max_p > cfg_.gamma * noise_floor) {
                    double f_peak = max_k * df;
                    candidates.push_back({(int)i, t_curr, f_peak, max_p});
                }
            }

            if (candidates.size() < 5) return {0.0, 0.0, 0.0};

            // 2. t0 Robust Estimation
            std::vector<double> t0_estimates;
            t0_estimates.reserve(candidates.size());
            for (const auto& c : candidates) {
                double t0 = c.t - (c.f_peak - cfg_.f_start) / slope;
                t0_estimates.push_back(t0);
            }

            double t0_hat = quick_median(t0_estimates);
            
            // Iterative MAD (Simplified 1 pass)
            // 实际应用可增加迭代
            double current_mad = mad(t0_estimates, t0_hat);
            double threshold = 3.5 * 1.4826 * current_mad;
            
            std::vector<double> final_t0s;
            for (double val : t0_estimates) {
                if (std::abs(val - t0_hat) <= threshold) {
                    final_t0s.push_back(val);
                }
            }
            
            if (final_t0s.empty()) return {0.0, 0.0, 0.0};
            t0_hat = quick_median(final_t0s);

            // 3. Power Estimation (Frequency Domain)
            // 使用优化：直接在 STFT 域求和，避免 ISTFT
            double total_energy = 0.0;
            int active_frame_count = 0;
            
            double bw_bin = cfg_.mask_bw_hz / df;

            for (size_t i = 0; i < n_frames; ++i) {
                double t = frame_times[i];
                double tau = t - t0_hat;
                
                // 时间窗口检查
                if (tau < -0.1 || tau > cfg_.T + 0.1) continue;

                // 预测频率
                double f_pred = cfg_.f_start + slope * tau;
                
                // 检查当前帧是否包含符合轨迹的峰值
                // 这里我们简化：如果当前帧属于 Candidate 且频率接近 f_pred，则算 Active Frame
                bool is_active_frame = false;
                // 简单的 Active 判断：在预测频率附近是否有显著能量？
                // 或者复用 candidate 信息：
                for (const auto& c : candidates) {
                    if (c.frame_idx == (int)i) {
                        if (std::abs(c.f_peak - f_pred) < cfg_.delta_f_hz) {
                            is_active_frame = true;
                            break;
                        }
                    }
                }

                if (is_active_frame) {
                    active_frame_count++;
                    
                    // 积分掩膜内的能量
                    int k_pred = (int)(f_pred / df);
                    int k_min = std::max(bin_start, (int)(k_pred - bw_bin));
                    int k_max = std::min(bin_end, (int)(k_pred + bw_bin));

                    double frame_energy = 0.0;
                    for (int k = k_min; k <= k_max; ++k) {
                        if (k >= bin_start && k <= bin_end) {
                            frame_energy += spectrogram[i][k - bin_start];
                        }
                    }
                    total_energy += frame_energy;
                }
            }

            if (active_frame_count == 0) return {0.0, 0.0, t0_hat};

            // 能量归一化 (Parseval 修正)
            // 简易校准：Energy_time ≈ Energy_freq / (Sum Window^2) * hop ? (需精确推导)
            double win_energy = 0.0;
            for(double w : window_) win_energy += w*w;
            
            // 仅作示意，实际需根据 STFT 定义调整
            double estimated_power = total_energy / (active_frame_count * win_energy); // 假设

            return {estimated_power, active_frame_count * dt, t0_hat};
        }

    private:
        LfmConfig cfg_;
        int nfft_;
        fftw_complex *in_, *out_;
        fftw_plan plan_;
        std::vector<double> window_;

        // Helper: Find median
        double quick_median(std::vector<double>& v) {
            if (v.empty()) return 0.0;
            size_t n = v.size() / 2;
            std::nth_element(v.begin(), v.begin() + n, v.end());
            return v[n];
        }

        // Helper: MAD
        double mad(const std::vector<double>& v, double median) {
            std::vector<double> diffs;
            diffs.reserve(v.size());
            for (double x : v) diffs.push_back(std::abs(x - median));
            return quick_median(diffs);
        }
    };
    ```
