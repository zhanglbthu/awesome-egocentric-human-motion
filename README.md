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

**Oct 6, 2024**
- Added AMASS, TransPose, PIP, TIP, IMUPoser, DiffusionPoser, EgoLocate, MocapEvery

**Sept 26, 2024**
- Initial commit
</details>
<br>

## Datasets
### [ICCV, 2019] AMASS: Archive of Motion Capture as Surface Shapes
**Authors**: Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, Michael J. Black
<details span>
<summary><b>Abstract</b></summary>
Large datasets are the cornerstone of recent advances in computer vision using deep learning. In contrast, existing human motion capture (mocap) datasets are small and the motions limited, hampering progress on learning models of human motion. While there are many different datasets available, they each use a different parameterization of the body, making it difficult to integrate them into a single meta dataset. To address this, we introduce AMASS, a large and varied database of human motion that unifies 15 different optical marker-based mocap datasets by representing them within a common framework and parameterization. We achieve this using a new method, MoSh++, that converts mocap data into realistic 3D human meshes represented by a rigged body model; here we use SMPL [doi:https://doi.org/10.1145/2816795.2818013], which is widely used and provides a standard skeletal representation as well as a fully rigged surface mesh. The method works for arbitrary marker sets, while recovering soft-tissue dynamics and realistic hand motion. We evaluate MoSh++ and tune its hyperparameters using a new dataset of 4D body scans that are jointly recorded with marker-based mocap. The consistent representation of AMASS makes it readily useful for animation, visualization, and generating training data for deep learning. Our dataset is significantly richer than previous human motion collections, having more than 40 hours of motion data, spanning over 300 subjects, more than 11,000 motions, and will be publicly available to the research community.
</details>

[📄 Paper](https://arxiv.org/abs/1904.03278) | [💻 Code](https://github.com/nghorbani/amass)

## Motion Capture
### [SIGGRAPH ASIA, 2018] Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time
**Authors**: Yinghao Huang, Manuel Kaufmann, Emre Aksan, Michael J. Black, Otmar Hilliges, Gerard Pons-Moll
<details span>
<summary><b>Abstract</b></summary>
We demonstrate a novel deep neural network capable of reconstructing human full body pose in real-time from 6 Inertial Measurement Units (IMUs) worn on the user's body. In doing so, we address several difficult challenges. First, the problem is severely under-constrained as multiple pose parameters produce the same IMU orientations. Second, capturing IMU data in conjunction with ground-truth poses is expensive and difficult to do in many target application scenarios (e.g., outdoors). Third, modeling temporal dependencies through non-linear optimization has proven effective in prior work but makes real-time prediction infeasible. To address this important limitation, we learn the temporal pose priors using deep learning. To learn from sufficient data, we synthesize IMU data from motion capture datasets. A bi-directional RNN architecture leverages past and future information that is available at training time. At test time, we deploy the network in a sliding window fashion, retaining real time capabilities. To evaluate our method, we recorded DIP-IMU, a dataset consisting of 10 subjects wearing 17 IMUs for validation in 64 sequences with 330000 time instants; this constitutes the largest IMU dataset publicly available. We quantitatively evaluate our approach on multiple datasets and show results from a real-time implementation. DIP-IMU and the code are available for research purposes.
</details>

[📄 Paper](https://arxiv.org/abs/1810.04703) | [💻 Code](https://github.com/eth-ait/dip18)

### [SIGGRAPH, 2021] TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors
**Authors**: Xinyu Yi, Yuxiao Zhou, Feng Xu
<details span>
<summary><b>Abstract</b></summary>
Motion capture is facing some new possibilities brought by the inertial sensing technologies which do not suffer from occlusion or wide-range recordings as vision-based solutions do. However, as the recorded signals are sparse and quite noisy, online performance and global translation estimation turn out to be two key difficulties. In this paper, we present TransPose, a DNN-based approach to perform full motion capture (with both global translations and body poses) from only 6 Inertial Measurement Units (IMUs) at over 90 fps. For body pose estimation, we propose a multi-stage network that estimates leaf-to-full joint positions as intermediate results. This design makes the pose estimation much easier, and thus achieves both better accuracy and lower computation cost. For global translation estimation, we propose a supporting-foot-based method and an RNN-based method to robustly solve for the global translations with a confidence-based fusion technique. Quantitative and qualitative comparisons show that our method outperforms the state-of-the-art learning- and optimization-based methods with a large margin in both accuracy and efficiency. As a purely inertial sensor-based approach, our method is not limited by environmental settings (e.g., fixed cameras), making the capture free from common difficulties such as wide-range motion space and strong occlusion.
</details>

[📄 Paper](https://arxiv.org/abs/2105.04605) | [💻 Code](https://github.com/Xinyu-Yi/TransPose)

### [CVPR, 2022] Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors
**Authors**: Xinyu Yi, Yuxiao Zhou, Marc Habermann, Soshi Shimada, Vladislav Golyanik, Christian Theobalt, Feng Xu
<details span>
<summary><b>Abstract</b></summary>
Motion capture from sparse inertial sensors has shown great potential compared to image-based approaches since occlusions do not lead to a reduced tracking quality and the recording space is not restricted to be within the viewing frustum of the camera. However, capturing the motion and global position only from a sparse set of inertial sensors is inherently ambiguous and challenging. In consequence, recent state-of-the-art methods can barely handle very long period motions, and unrealistic artifacts are common due to the unawareness of physical constraints. To this end, we present the first method which combines a neural kinematics estimator and a physics-aware motion optimizer to track body motions with only 6 inertial sensors. The kinematics module first regresses the motion status as a reference, and then the physics module refines the motion to satisfy the physical constraints. Experiments demonstrate a clear improvement over the state of the art in terms of capture accuracy, temporal stability, and physical correctness.
</details>

[📄 Paper](https://arxiv.org/abs/2203.08528) | [💻 Code](https://github.com/Xinyu-Yi/PIP)

### [SIGGRAPH ASIA, 2022] Transformer Inertial Poser: Real-time Human Motion Reconstruction from Sparse IMUs with Simultaneous Terrain Generation
**Authors**: Yifeng Jiang, Yuting Ye, Deepak Gopinath, Jungdam Won, Alexander W. Winkler, C. Karen Liu
<details span>
<summary><b>Abstract</b></summary>
Real-time human motion reconstruction from a sparse set of (e.g. six) wearable IMUs provides a non-intrusive and economic approach to motion capture. Without the ability to acquire position information directly from IMUs, recent works took data-driven approaches that utilize large human motion datasets to tackle this under-determined problem. Still, challenges remain such as temporal consistency, drifting of global and joint motions, and diverse coverage of motion types on various terrains. We propose a novel method to simultaneously estimate full-body motion and generate plausible visited terrain from only six IMU sensors in real-time. Our method incorporates 1. a conditional Transformer decoder model giving consistent predictions by explicitly reasoning prediction history, 2. a simple yet general learning target named "stationary body points" (SBPs) which can be stably predicted by the Transformer model and utilized by analytical routines to correct joint and global drifting, and 3. an algorithm to generate regularized terrain height maps from noisy SBP predictions which can in turn correct noisy global motion estimation. We evaluate our framework extensively on synthesized and real IMU data, and with real-time live demos, and show superior performance over strong baseline methods.
</details>

[📄 Paper](https://arxiv.org/abs/2203.15720) | [💻 Code](https://github.com/jyf588/transformer-inertial-poser)

### [CHI, 2023] IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds
**Authors**: Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, Karan Ahuja
<details span>
<summary><b>Abstract</b></summary>
Tracking body pose on-the-go could have powerful uses in fitness, mobile gaming, context-aware virtual assistants, and rehabilitation. However, users are unlikely to buy and wear special suits or sensor arrays to achieve this end. Instead, in this work, we explore the feasibility of estimating body pose using IMUs already in devices that many users own -- namely smartphones, smartwatches, and earbuds. This approach has several challenges, including noisy data from low-cost commodity IMUs, and the fact that the number of instrumentation points on a users body is both sparse and in flux. Our pipeline receives whatever subset of IMU data is available, potentially from just a single device, and produces a best-guess pose. To evaluate our model, we created the IMUPoser Dataset, collected from 10 participants wearing or holding off-the-shelf consumer devices and across a variety of activity contexts. We provide a comprehensive evaluation of our system, benchmarking it on both our own and existing IMU datasets.
</details>

[📄 Paper](https://arxiv.org/abs/2304.12518) | [💻 Code](https://github.com/FIGLAB/IMUPoser)

### [CVPR, 2024] DiffusionPoser: Real-time Human Motion Reconstruction From Arbitrary Sparse Sensors Using Autoregressive Diffusion
**Authors**: Tom Van Wouwe, Seunghwan Lee, Antoine Falisse, Scott Delp, C. Karen Liu
<details span>
<summary><b>Abstract</b></summary>
Motion capture from a limited number of body-worn sensors, such as inertial measurement units (IMUs) and pressure insoles, has important applications in health, human performance, and entertainment. Recent work has focused on accurately reconstructing whole-body motion from a specific sensor configuration using six IMUs. While a common goal across applications is to use the minimal number of sensors to achieve required accuracy, the optimal arrangement of the sensors might differ from application to application. We propose a single diffusion model, DiffusionPoser, which reconstructs human motion in real-time from an arbitrary combination of sensors, including IMUs placed at specified locations, and, pressure insoles. Unlike existing methods, our model grants users the flexibility to determine the number and arrangement of sensors tailored to the specific activity of interest, without the need for retraining. A novel autoregressive inferencing scheme ensures real-time motion reconstruction that closely aligns with measured sensor signals. The generative nature of DiffusionPoser ensures realistic behavior, even for degrees-of-freedom not directly measured. Qualitative results can be found on our website: this https URL.
</details>

[📄 Paper](https://arxiv.org/abs/2308.16682)

## SLAM
### [SIGGRAPH, 2023] EgoLocate: Real-time Motion Capture, Localization, and Mapping with Sparse Body-mounted Sensors
**Authors**: Xinyu Yi, Yuxiao Zhou, Marc Habermann, Vladislav Golyanik, Shaohua Pan, Christian Theobalt, Feng Xu
<details span>
<summary><b>Abstract</b></summary>
Human and environment sensing are two important topics in Computer Vision and Graphics. Human motion is often captured by inertial sensors, while the environment is mostly reconstructed using cameras. We integrate the two techniques together in EgoLocate, a system that simultaneously performs human motion capture (mocap), localization, and mapping in real time from sparse body-mounted sensors, including 6 inertial measurement units (IMUs) and a monocular phone camera. On one hand, inertial mocap suffers from large translation drift due to the lack of the global positioning signal. EgoLocate leverages image-based simultaneous localization and mapping (SLAM) techniques to locate the human in the reconstructed scene. On the other hand, SLAM often fails when the visual feature is poor. EgoLocate involves inertial mocap to provide a strong prior for the camera motion. Experiments show that localization, a key challenge for both two fields, is largely improved by our technique, compared with the state of the art of the two fields. Our codes are available for research at this https URL.
</details>

[📄 Paper](https://arxiv.org/abs/2305.01599) | [💻 Code](https://github.com/Xinyu-Yi/EgoLocate)

### [CVPR, 2024] Mocap Everyone Everywhere: Lightweight Motion Capture With Smartwatches and a Head-Mounted Camera
**Authors**: Jiye Lee, Hanbyul Joo
<details span>
<summary><b>Abstract</b></summary>
We present a lightweight and affordable motion capture method based on two smartwatches and a head-mounted camera. In contrast to the existing approaches that use six or more expert-level IMU devices, our approach is much more cost-effective and convenient. Our method can make wearable motion capture accessible to everyone everywhere, enabling 3D full-body motion capture in diverse environments. As a key idea to overcome the extreme sparsity and ambiguities of sensor inputs with different modalities, we integrate 6D head poses obtained from the head-mounted cameras for motion estimation. To enable capture in expansive indoor and outdoor scenes, we propose an algorithm to track and update floor level changes to define head poses, coupled with a multi-stage Transformer-based regression module. We also introduce novel strategies leveraging visual cues of egocentric images to further enhance the motion capture quality while reducing ambiguities. We demonstrate the performance of our method on various challenging scenarios, including complex outdoor environments and everyday motions including object interactions and social interactions among multiple individuals.
</details>

[📄 Paper](https://arxiv.org/abs/2401.00847) | [💻 Code](https://github.com/jiyewise/MocapEvery)



