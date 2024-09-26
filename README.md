# Awesome Egocentric Body Motion
A curated list of papers and open-source resources focused on Egocentric Body Motion, intended to keep pace with the anticipated growth and accelerate research in this field.
If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.

## Table of Contents

- [Datasets](#datasets)
- [Motion Capture](#motion-capture)
- [SLAM](#slam)

<details span>
<summary><b>Update Log:</b></summary>
<br>

**Sept 26, 2024**
- Initial commit
</details>
<br>

## Datasets 
## Motion Capture
### [SIGGRAPH ASIA, 2018] Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time
**Authors**: Yinghao Huang, Manuel Kaufmann, Emre Aksan, Michael J. Black, Otmar Hilliges, Gerard Pons-Moll
<details span>
<summary><b>Abstract</b></summary>
We demonstrate a novel deep neural network capable of reconstructing human full body pose in real-time from 6 Inertial Measurement Units (IMUs) worn on the user's body. In doing so, we address several difficult challenges. First, the problem is severely under-constrained as multiple pose parameters produce the same IMU orientations. Second, capturing IMU data in conjunction with ground-truth poses is expensive and difficult to do in many target application scenarios (e.g., outdoors). Third, modeling temporal dependencies through non-linear optimization has proven effective in prior work but makes real-time prediction infeasible. To address this important limitation, we learn the temporal pose priors using deep learning. To learn from sufficient data, we synthesize IMU data from motion capture datasets. A bi-directional RNN architecture leverages past and future information that is available at training time. At test time, we deploy the network in a sliding window fashion, retaining real time capabilities. To evaluate our method, we recorded DIP-IMU, a dataset consisting of 10 subjects wearing 17 IMUs for validation in 64 sequences with 330000 time instants; this constitutes the largest IMU dataset publicly available. We quantitatively evaluate our approach on multiple datasets and show results from a real-time implementation. DIP-IMU and the code are available for research purposes.
</details>

[📄 Paper](https://arxiv.org/abs/1810.04703) | [💻 Code](https://github.com/eth-ait/dip18)
## SLAM



