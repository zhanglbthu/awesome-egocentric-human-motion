# Awesome Egocentric Human Motion
A curated list of papers and open-source resources focused on Egocentric Human Motion, intended to keep pace with the anticipated growth and accelerate research in this field.
If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.

## Table of Contents
- [Survey](#survey)
- [Datasets](#datasets)
- [Motion Capture](#motion-capture)
  - [VR](#vr)
  - [IMU](#imu)
  - [Camera](#camera)
  - [IMU + Camera](#imu+camera)

<details span>
<summary><b>Update Log:</b></summary>
<br>
**Oct 24, 2024**
- Add Seeing Invisible Poses...

**Oct 6, 2024**
- Add AMASS, TransPose, PIP, TIP, IMUPoser, DiffusionPoser, EgoLocate, MocapEvery

**Sept 26, 2024**
- Initial commit
</details>
<br>

## Datasets
#### [ICCV, 2019] AMASS: Archive of Motion Capture as Surface Shapes
**Authors**: Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, Michael J. Black
<details span>
<summary><b>Abstract</b></summary>
Large datasets are the cornerstone of recent advances in computer vision using deep learning. In contrast, existing human motion capture (mocap) datasets are small and the motions limited, hampering progress on learning models of human motion. While there are many different datasets available, they each use a different parameterization of the body, making it difficult to integrate them into a single meta dataset. To address this, we introduce AMASS, a large and varied database of human motion that unifies 15 different optical marker-based mocap datasets by representing them within a common framework and parameterization. We achieve this using a new method, MoSh++, that converts mocap data into realistic 3D human meshes represented by a rigged body model; here we use SMPL [doi:https://doi.org/10.1145/2816795.2818013], which is widely used and provides a standard skeletal representation as well as a fully rigged surface mesh. The method works for arbitrary marker sets, while recovering soft-tissue dynamics and realistic hand motion. We evaluate MoSh++ and tune its hyperparameters using a new dataset of 4D body scans that are jointly recorded with marker-based mocap. The consistent representation of AMASS makes it readily useful for animation, visualization, and generating training data for deep learning. Our dataset is significantly richer than previous human motion collections, having more than 40 hours of motion data, spanning over 300 subjects, more than 11,000 motions, and will be publicly available to the research community.
</details>

[📄 Paper](https://arxiv.org/abs/1904.03278) | [💻 Code](https://github.com/nghorbani/amass)

## Motion Capture
### IMU
##### [SIGGRAPH ASIA, 2018] Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time
**Authors**: Yinghao Huang, Manuel Kaufmann, Emre Aksan, Michael J. Black, Otmar Hilliges, Gerard Pons-Moll
<details span>
<summary><b>Abstract</b></summary>
We demonstrate a novel deep neural network capable of reconstructing human full body pose in real-time from 6 Inertial Measurement Units (IMUs) worn on the user's body. In doing so, we address several difficult challenges. First, the problem is severely under-constrained as multiple pose parameters produce the same IMU orientations. Second, capturing IMU data in conjunction with ground-truth poses is expensive and difficult to do in many target application scenarios (e.g., outdoors). Third, modeling temporal dependencies through non-linear optimization has proven effective in prior work but makes real-time prediction infeasible. To address this important limitation, we learn the temporal pose priors using deep learning. To learn from sufficient data, we synthesize IMU data from motion capture datasets. A bi-directional RNN architecture leverages past and future information that is available at training time. At test time, we deploy the network in a sliding window fashion, retaining real time capabilities. To evaluate our method, we recorded DIP-IMU, a dataset consisting of 10 subjects wearing 17 IMUs for validation in 64 sequences with 330000 time instants; this constitutes the largest IMU dataset publicly available. We quantitatively evaluate our approach on multiple datasets and show results from a real-time implementation. DIP-IMU and the code are available for research purposes.
</details>

[📄 Paper](https://arxiv.org/abs/1810.04703) | [💻 Code](https://github.com/eth-ait/dip18)

#### [SIGGRAPH, 2021] TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors
**Authors**: Xinyu Yi, Yuxiao Zhou, Feng Xu
<details span>
<summary><b>Abstract</b></summary>
Motion capture is facing some new possibilities brought by the inertial sensing technologies which do not suffer from occlusion or wide-range recordings as vision-based solutions do. However, as the recorded signals are sparse and quite noisy, online performance and global translation estimation turn out to be two key difficulties. In this paper, we present TransPose, a DNN-based approach to perform full motion capture (with both global translations and body poses) from only 6 Inertial Measurement Units (IMUs) at over 90 fps. For body pose estimation, we propose a multi-stage network that estimates leaf-to-full joint positions as intermediate results. This design makes the pose estimation much easier, and thus achieves both better accuracy and lower computation cost. For global translation estimation, we propose a supporting-foot-based method and an RNN-based method to robustly solve for the global translations with a confidence-based fusion technique. Quantitative and qualitative comparisons show that our method outperforms the state-of-the-art learning- and optimization-based methods with a large margin in both accuracy and efficiency. As a purely inertial sensor-based approach, our method is not limited by environmental settings (e.g., fixed cameras), making the capture free from common difficulties such as wide-range motion space and strong occlusion.
</details>

[📄 Paper](https://arxiv.org/abs/2105.04605) | [💻 Code](https://github.com/Xinyu-Yi/TransPose)

#### [CVPR, 2022] Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors
**Authors**: Xinyu Yi, Yuxiao Zhou, Marc Habermann, Soshi Shimada, Vladislav Golyanik, Christian Theobalt, Feng Xu
<details span>
<summary><b>Abstract</b></summary>
Motion capture from sparse inertial sensors has shown great potential compared to image-based approaches since occlusions do not lead to a reduced tracking quality and the recording space is not restricted to be within the viewing frustum of the camera. However, capturing the motion and global position only from a sparse set of inertial sensors is inherently ambiguous and challenging. In consequence, recent state-of-the-art methods can barely handle very long period motions, and unrealistic artifacts are common due to the unawareness of physical constraints. To this end, we present the first method which combines a neural kinematics estimator and a physics-aware motion optimizer to track body motions with only 6 inertial sensors. The kinematics module first regresses the motion status as a reference, and then the physics module refines the motion to satisfy the physical constraints. Experiments demonstrate a clear improvement over the state of the art in terms of capture accuracy, temporal stability, and physical correctness.
</details>

[📄 Paper](https://arxiv.org/abs/2203.08528) | [💻 Code](https://github.com/Xinyu-Yi/PIP)

#### [SIGGRAPH ASIA, 2022] Transformer Inertial Poser: Real-time Human Motion Reconstruction from Sparse IMUs with Simultaneous Terrain Generation
**Authors**: Yifeng Jiang, Yuting Ye, Deepak Gopinath, Jungdam Won, Alexander W. Winkler, C. Karen Liu
<details span>
<summary><b>Abstract</b></summary>
Real-time human motion reconstruction from a sparse set of (e.g. six) wearable IMUs provides a non-intrusive and economic approach to motion capture. Without the ability to acquire position information directly from IMUs, recent works took data-driven approaches that utilize large human motion datasets to tackle this under-determined problem. Still, challenges remain such as temporal consistency, drifting of global and joint motions, and diverse coverage of motion types on various terrains. We propose a novel method to simultaneously estimate full-body motion and generate plausible visited terrain from only six IMU sensors in real-time. Our method incorporates 1. a conditional Transformer decoder model giving consistent predictions by explicitly reasoning prediction history, 2. a simple yet general learning target named "stationary body points" (SBPs) which can be stably predicted by the Transformer model and utilized by analytical routines to correct joint and global drifting, and 3. an algorithm to generate regularized terrain height maps from noisy SBP predictions which can in turn correct noisy global motion estimation. We evaluate our framework extensively on synthesized and real IMU data, and with real-time live demos, and show superior performance over strong baseline methods.
</details>

[📄 Paper](https://arxiv.org/abs/2203.15720) | [💻 Code](https://github.com/jyf588/transformer-inertial-poser)

#### [CHI, 2023] IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds
**Authors**: Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, Karan Ahuja
<details span>
<summary><b>Abstract</b></summary>
Tracking body pose on-the-go could have powerful uses in fitness, mobile gaming, context-aware virtual assistants, and rehabilitation. However, users are unlikely to buy and wear special suits or sensor arrays to achieve this end. Instead, in this work, we explore the feasibility of estimating body pose using IMUs already in devices that many users own -- namely smartphones, smartwatches, and earbuds. This approach has several challenges, including noisy data from low-cost commodity IMUs, and the fact that the number of instrumentation points on a users body is both sparse and in flux. Our pipeline receives whatever subset of IMU data is available, potentially from just a single device, and produces a best-guess pose. To evaluate our model, we created the IMUPoser Dataset, collected from 10 participants wearing or holding off-the-shelf consumer devices and across a variety of activity contexts. We provide a comprehensive evaluation of our system, benchmarking it on both our own and existing IMU datasets.
</details>

[📄 Paper](https://arxiv.org/abs/2304.12518) | [💻 Code](https://github.com/FIGLAB/IMUPoser)

#### [UIST, 2023] SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data
**Authors**: Nathan DeVrio, Vimal Mollyn, Chris Harrison
<details span>
<summary><b>Abstract</b></summary>
The ability to track a user’s arm pose could be valuable in a wide range of applications, including fitness, rehabilitation, augmented reality input, life logging, and context-aware assistants. Unfortunately, this capability is not readily available to consumers. Systems either require cameras, which carry privacy issues, or utilize multiple worn IMUs or markers. In this work, we describe how an off-the-shelf smartphone and smartwatch can work together to accurately estimate arm pose. Moving beyond prior work, we take advantage of more recent ultra-wideband (UWB) functionality on these devices to capture absolute distance between the two devices. This measurement is the perfect complement to inertial data, which is relative and suffers from drift. We quantify the performance of our software-only approach using off-the-shelf devices, showing it can estimate the wrist and elbow joints with a median positional error of 11.0 cm, without the user having to provide training data.
</details>

[📄 Paper](https://dl.acm.org/doi/10.1145/3586183.3606821)

#### [CVPR, 2024] DiffusionPoser: Real-time Human Motion Reconstruction From Arbitrary Sparse Sensors Using Autoregressive Diffusion
**Authors**: Tom Van Wouwe, Seunghwan Lee, Antoine Falisse, Scott Delp, C. Karen Liu
<details span>
<summary><b>Abstract</b></summary>
Motion capture from a limited number of body-worn sensors, such as inertial measurement units (IMUs) and pressure insoles, has important applications in health, human performance, and entertainment. Recent work has focused on accurately reconstructing whole-body motion from a specific sensor configuration using six IMUs. While a common goal across applications is to use the minimal number of sensors to achieve required accuracy, the optimal arrangement of the sensors might differ from application to application. We propose a single diffusion model, DiffusionPoser, which reconstructs human motion in real-time from an arbitrary combination of sensors, including IMUs placed at specified locations, and, pressure insoles. Unlike existing methods, our model grants users the flexibility to determine the number and arrangement of sensors tailored to the specific activity of interest, without the need for retraining. A novel autoregressive inferencing scheme ensures real-time motion reconstruction that closely aligns with measured sensor signals. The generative nature of DiffusionPoser ensures realistic behavior, even for degrees-of-freedom not directly measured. Qualitative results can be found on our website: this https URL.
</details>

[📄 Paper](https://arxiv.org/abs/2308.16682)

### Camera

#### [CVPR, 2016] Seeing Invisible Poses: Estimating 3D Body Pose from Egocentric Video
**Authors**: Hao Jiang, Kristen Grauman
<details span>
<summary><b>Abstract</b></summary>
Understanding the camera wearer's activity is central to egocentric vision, yet one key facet of that activity is inherently invisible to the camera--the wearer's body pose. Prior work focuses on estimating the pose of hands and arms when they come into view, but this 1) gives an incomplete view of the full body posture, and 2) prevents any pose estimate at all in many frames, since the hands are only visible in a fraction of daily life activities. We propose to infer the "invisible pose" of a person behind the egocentric camera. Given a single video, our efficient learning-based approach returns the full body 3D joint positions for each frame. Our method exploits cues from the dynamic motion signatures of the surrounding scene--which changes predictably as a function of body pose--as well as static scene structures that reveal the viewpoint (e.g., sitting vs. standing). We further introduce a novel energy minimization scheme to infer the pose sequence. It uses soft predictions of the poses per time instant together with a non-parametric model of human pose dynamics over longer windows. Our method outperforms an array of possible alternatives, including deep learning approaches for direct pose regression from images.
</details>

[📄 Paper](https://arxiv.org/abs/1603.07763)

#### [SIGGRAPH, 2016] EgoCap: Egocentric Marker-less Motion Capture with Two Fisheye Cameras
**Authors**: Helge Rhodin, Christian Richardt, Dan Casas, Eldar Insafutdinov, Mohammad Shafiei, Hans-Peter Seidel, Bernt Schiele, Christian Theobalt
<details span>
<summary><b>Abstract</b></summary>
Marker-based and marker-less optical skeletal motion-capture methods use an outside-in arrangement of cameras placed around a scene, with viewpoints converging on the center. They often create discomfort by possibly needed marker suits, and their recording volume is severely restricted and often constrained to indoor scenes with controlled backgrounds. Alternative suit-based systems use several inertial measurement units or an exoskeleton to capture motion. This makes capturing independent of a confined volume, but requires substantial, often constraining, and hard to set up body instrumentation. We therefore propose a new method for real-time, marker-less and egocentric motion capture which estimates the full-body skeleton pose from a lightweight stereo pair of fisheye cameras that are attached to a helmet or virtual reality headset. It combines the strength of a new generative pose estimation framework for fisheye views with a ConvNet-based body-part detector trained on a large new dataset. Our inside-in method captures full-body motion in general indoor and outdoor scenes, and also crowded scenes with many people in close vicinity. The captured user can freely move around, which enables reconstruction of larger-scale activities and is particularly useful in virtual reality to freely roam and interact, while seeing the fully motion-captured virtual body.
</details>

[📄 Paper](https://arxiv.org/abs/1609.07306)
#### [3DV, 2021] Egoglass: Egocentric-view human pose estimation from an eyeglass frame
**Authors**: Dongxu Zhao Zhen Wei Jisan Mahmud Jan-Michael Frahm
<details span>
<summary><b>Abstract</b></summary>
We present a new approach, EgoGlass, towards egocentric motion-capture and human pose estimation. EgoGlass
is a lightweight eyeglass frame with two cameras mounted
on it. Our first contribution is a new egocentric motioncapture device that adds next to no extra burden on the user
and a dataset of real people doing a diverse set of actions
captured by EgoGlass. Second, we propose to utilize body
part information for human pose detection - to help tackle
the problems of limited body coverage and self-occlusions
caused by the egocentric viewpoint and cameras' proximity
to the human body. We also propose a concept of pseudolimb mask as an alternative for segmentation mask when
ground truth segmentation mask is absent for egocentric
images with real subject. We demonstrate that our method
achieves better results than the counterpart method without body part information on our dataset. We also test our
method on two existing egocentric datasets: xR-EgoPose
and EgoCap. Our method achieves state-of-the-art results
on xR-EgoPose and is on par with existing method for EgoCap without requiring temporal information or personalization for each individual user
</details>

[📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9665856)

#### [CVPR, 2022] Estimating Egocentric 3D Human Pose in the Wild with External Weak Supervision
**Authors**: Jian Wang, Lingjie Liu, Weipeng Xu, Kripasindhu Sarkar, Diogo Luvizon, Christian Theobalt
<details span>
<summary><b>Abstract</b></summary>
Egocentric 3D human pose estimation with a single fisheye camera has drawn a significant amount of attention recently. However, existing methods struggle with pose estimation from in-the-wild images, because they can only be trained on synthetic data due to the unavailability of large-scale in-the-wild egocentric datasets. Furthermore, these methods easily fail when the body parts are occluded by or interacting with the surrounding scene. To address the shortage of in-the-wild data, we collect a large-scale in-the-wild egocentric dataset called Egocentric Poses in the Wild (EgoPW). This dataset is captured by a head-mounted fisheye camera and an auxiliary external camera, which provides an additional observation of the human body from a third-person perspective during training. We present a new egocentric pose estimation method, which can be trained on the new dataset with weak external supervision. Specifically, we first generate pseudo labels for the EgoPW dataset with a spatio-temporal optimization method by incorporating the external-view supervision. The pseudo labels are then used to train an egocentric pose estimation network. To facilitate the network training, we propose a novel learning strategy to supervise the egocentric features with the high-quality features extracted by a pretrained external-view pose estimation model. The experiments show that our method predicts accurate 3D poses from a single in-the-wild egocentric image and outperforms the state-of-the-art methods both quantitatively and qualitatively.
</details>

[📄 Paper](https://arxiv.org/abs/2201.07929)

#### [CVPR, 2023] Ego-Body Pose Estimation via Ego-Head Pose Estimation
**Authors**: Jiaman Li, C. Karen Liu, Jiajun Wu
<details span>
<summary><b>Abstract</b></summary>
Estimating 3D human motion from an egocentric video sequence plays a critical role in human behavior understanding and has various applications in VR/AR. However, naively learning a mapping between egocentric videos and human motions is challenging, because the user's body is often unobserved by the front-facing camera placed on the head of the user. In addition, collecting large-scale, high-quality datasets with paired egocentric videos and 3D human motions requires accurate motion capture devices, which often limit the variety of scenes in the videos to lab-like environments. To eliminate the need for paired egocentric video and human motions, we propose a new method, Ego-Body Pose Estimation via Ego-Head Pose Estimation (EgoEgo), which decomposes the problem into two stages, connected by the head motion as an intermediate representation. EgoEgo first integrates SLAM and a learning approach to estimate accurate head motion. Subsequently, leveraging the estimated head pose as input, EgoEgo utilizes conditional diffusion to generate multiple plausible full-body motions. This disentanglement of head and body pose eliminates the need for training datasets with paired egocentric videos and 3D human motion, enabling us to leverage large-scale egocentric video datasets and motion capture datasets separately. Moreover, for systematic benchmarking, we develop a synthetic dataset, AMASS-Replica-Ego-Syn (ARES), with paired egocentric videos and human motion. On both ARES and real data, our EgoEgo model performs significantly better than the current state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/abs/2212.04636) | [💻 Code](https://github.com/lijiaman/egoego_release)

#### [SIGGRAPH ASIA, 2023] Ego3DPose: Capturing 3D Cues from Binocular Egocentric Views
**Authors**: Taeho Kang, Kyungjin Lee, Jinrui Zhang, Youngki Lee
<details span>
<summary><b>Abstract</b></summary>
We present Ego3DPose, a highly accurate binocular egocentric 3D pose reconstruction system. The binocular egocentric setup offers practicality and usefulness in various applications, however, it remains largely under-explored. It has been suffering from low pose estimation accuracy due to viewing distortion, severe self-occlusion, and limited field-of-view of the joints in egocentric 2D images. Here, we notice that two important 3D cues, stereo correspondences, and perspective, contained in the egocentric binocular input are neglected. Current methods heavily rely on 2D image features, implicitly learning 3D information, which introduces biases towards commonly observed motions and leads to low overall accuracy. We observe that they not only fail in challenging occlusion cases but also in estimating visible joint positions. To address these challenges, we propose two novel approaches. First, we design a two-path network architecture with a path that estimates pose per limb independently with its binocular heatmaps. Without full-body information provided, it alleviates bias toward trained full-body distribution. Second, we leverage the egocentric view of body limbs, which exhibits strong perspective variance (e.g., a significantly large-size hand when it is close to the camera). We propose a new perspective-aware representation using trigonometry, enabling the network to estimate the 3D orientation of limbs. Finally, we develop an end-to-end pose reconstruction network that synergizes both techniques. Our comprehensive evaluations demonstrate that Ego3DPose outperforms state-of-the-art models by a pose estimation error (i.e., MPJPE) reduction of 23.1% in the UnrealEgo dataset. Our qualitative results highlight the superiority of our approach across a range of scenarios and challenges.
</details>

[📄 Paper](https://arxiv.org/abs/2309.11962) | [💻 Code](https://github.com/tho-kn/Ego3DPose)

#### [CVPR, 2024] Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting
**Authors**: Taeho Kang, Youngki Lee
<details span>
<summary><b>Abstract</b></summary>
We present EgoTAP, a heatmap-to-3D pose lifting method for highly accurate stereo egocentric 3D pose estimation. Severe self-occlusion and out-of-view limbs in egocentric camera views make accurate pose estimation a challenging problem. To address the challenge, prior methods employ joint heatmaps-probabilistic 2D representations of the body pose, but heatmap-to-3D pose conversion still remains an inaccurate process. We propose a novel heatmap-to-3D lifting method composed of the Grid ViT Encoder and the Propagation Network. The Grid ViT Encoder summarizes joint heatmaps into effective feature embedding using self-attention. Then, the Propagation Network estimates the 3D pose by utilizing skeletal information to better estimate the position of obscure joints. Our method significantly outperforms the previous state-of-the-art qualitatively and quantitatively demonstrated by a 23.9\% reduction of error in an MPJPE metric. Our source code is available in GitHub.
</details>

[📄 Paper](https://arxiv.org/abs/2402.18330) | [💻 Code](https://github.com/tho-kn/EgoTAP)

### VR
#### [ECCV, 2022] AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing
**Authors**: Jiaxi Jiang, Paul Streli, Huajian Qiu, Andreas Fender, Larissa Laich, Patrick Snape, Christian Holz
<details span>
<summary><b>Abstract</b></summary>
Today's Mixed Reality head-mounted displays track the user's head pose in world space as well as the user's hands for interaction in both Augmented Reality and Virtual Reality scenarios. While this is adequate to support user input, it unfortunately limits users' virtual representations to just their upper bodies. Current systems thus resort to floating avatars, whose limitation is particularly evident in collaborative settings. To estimate full-body poses from the sparse input sources, prior work has incorporated additional trackers and sensors at the pelvis or lower body, which increases setup complexity and limits practical application in mobile settings. In this paper, we present AvatarPoser, the first learning-based method that predicts full-body poses in world coordinates using only motion input from the user's head and hands. Our method builds on a Transformer encoder to extract deep features from the input signals and decouples global motion from the learned local joint orientations to guide pose estimation. To obtain accurate full-body motions that resemble motion capture animations, we refine the arm joints' positions using an optimization routine with inverse kinematics to match the original tracking input. In our evaluation, AvatarPoser achieved new state-of-the-art results in evaluations on large motion capture datasets (AMASS). At the same time, our method's inference speed supports real-time operation, providing a practical interface to support holistic avatar control and representation for Metaverse applications.
</details>

[📄 Paper](https://arxiv.org/abs/2207.13784)

### IMU+Camera

#### [CVPR, 2021] Human POSEitioning System (HPS): 3D Human Pose Estimation and Self-localization in Large Scenes from Body-Mounted Sensors
**Authors**: Vladimir Guzov, Aymen Mir, Torsten Sattler, Gerard Pons-Moll
<details span>
<summary><b>Abstract</b></summary>
We introduce (HPS) Human POSEitioning System, a method to recover the full 3D pose of a human registered with a 3D scan of the surrounding environment using wearable sensors. Using IMUs attached at the body limbs and a head mounted camera looking outwards, HPS fuses camera based self-localization with IMU-based human body tracking. The former provides drift-free but noisy position and orientation estimates while the latter is accurate in the short-term but subject to drift over longer periods of time. We show that our optimization-based integration exploits the benefits of the two, resulting in pose accuracy free of drift. Furthermore, we integrate 3D scene constraints into our optimization, such as foot contact with the ground, resulting in physically plausible motion. HPS complements more common third-person-based 3D pose estimation methods. It allows capturing larger recording volumes and longer periods of motion, and could be used for VR/AR applications where humans interact with the scene without requiring direct line of sight with an external camera, or to train agents that navigate and interact with the environment based on first-person visual input, like real humans. With HPS, we recorded a dataset of humans interacting with large 3D scenes (300-1000 sq.m) consisting of 7 subjects and more than 3 hours of diverse motion. The dataset, code and video will be available on the project page: this http URL.
</details>

[📄 Paper](https://arxiv.org/abs/2103.17265) | [💻 Code](https://github.com/miraymen/hps)

#### [SIGGRAPH, 2023] EgoLocate: Real-time Motion Capture, Localization, and Mapping with Sparse Body-mounted Sensors
**Authors**: Xinyu Yi, Yuxiao Zhou, Marc Habermann, Vladislav Golyanik, Shaohua Pan, Christian Theobalt, Feng Xu
<details span>
<summary><b>Abstract</b></summary>
Human and environment sensing are two important topics in Computer Vision and Graphics. Human motion is often captured by inertial sensors, while the environment is mostly reconstructed using cameras. We integrate the two techniques together in EgoLocate, a system that simultaneously performs human motion capture (mocap), localization, and mapping in real time from sparse body-mounted sensors, including 6 inertial measurement units (IMUs) and a monocular phone camera. On one hand, inertial mocap suffers from large translation drift due to the lack of the global positioning signal. EgoLocate leverages image-based simultaneous localization and mapping (SLAM) techniques to locate the human in the reconstructed scene. On the other hand, SLAM often fails when the visual feature is poor. EgoLocate involves inertial mocap to provide a strong prior for the camera motion. Experiments show that localization, a key challenge for both two fields, is largely improved by our technique, compared with the state of the art of the two fields. Our codes are available for research at this https URL.
</details>

[📄 Paper](https://arxiv.org/abs/2305.01599) | [💻 Code](https://github.com/Xinyu-Yi/EgoLocate)

#### [CVPR, 2024] Mocap Everyone Everywhere: Lightweight Motion Capture With Smartwatches and a Head-Mounted Camera
**Authors**: Jiye Lee, Hanbyul Joo
<details span>
<summary><b>Abstract</b></summary>
We present a lightweight and affordable motion capture method based on two smartwatches and a head-mounted camera. In contrast to the existing approaches that use six or more expert-level IMU devices, our approach is much more cost-effective and convenient. Our method can make wearable motion capture accessible to everyone everywhere, enabling 3D full-body motion capture in diverse environments. As a key idea to overcome the extreme sparsity and ambiguities of sensor inputs with different modalities, we integrate 6D head poses obtained from the head-mounted cameras for motion estimation. To enable capture in expansive indoor and outdoor scenes, we propose an algorithm to track and update floor level changes to define head poses, coupled with a multi-stage Transformer-based regression module. We also introduce novel strategies leveraging visual cues of egocentric images to further enhance the motion capture quality while reducing ambiguities. We demonstrate the performance of our method on various challenging scenarios, including complex outdoor environments and everyday motions including object interactions and social interactions among multiple individuals.
</details>

[📄 Paper](https://arxiv.org/abs/2401.00847) | [💻 Code](https://github.com/jiyewise/MocapEvery)



