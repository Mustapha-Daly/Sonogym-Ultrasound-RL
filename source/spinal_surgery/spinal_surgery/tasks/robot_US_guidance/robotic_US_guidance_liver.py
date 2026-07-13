# adapted env from vertebra to liver navigation
# /isaaacsim
# cd ~/IsaacLab
# PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/teleoperation/teleop_se3_agent.py --enable_cameras --task Isaac-robot-US-guidance-v0 --num_envs 1
#

'''
    keyboard name: Isaac Sim 5.1.0
----------------------------------------------
    Toggle gripper (open/close): K
    Move arm along x-axis: W/S
    Move arm along y-axis: A/D
    Move arm along z-axis: Q/E
    Rotate arm along x-axis: Z/X            
    Rotate arm along y-axis: T/G
    Rotate arm along z-axis: C/V

'''

from __future__ import annotations
from isaaclab.utils.math import matrix_from_quat


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    Articulation,
    RigidObject,
)

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
#for new file
from spinal_surgery.lab.sensors.ultrasound.us_clarity import us_clarity_score
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
import nibabel as nib
import cProfile
import time
import numpy as np
from collections.abc import Sequence
import gymnasium as gym

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from spinal_surgery.assets.fr3_US import *
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
)
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from ruamel.yaml import  YAML
from spinal_surgery import PACKAGE_DIR
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTMotionGenerator,
    GTDiscreteMotionGenerator,
)
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
import cProfile
import wandb
import os
import torch.nn.functional as F
#camera 
from isaaclab.sensors import Camera
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg

scene_cfg = YAML().load(
    open(f"{PACKAGE_DIR}/tasks/robot_US_guidance/cfgs/robotic_US_guidance.yaml", "r")
)
# observation scale
if (
    scene_cfg["observation"]["mode"] == "US"
    and scene_cfg["sim"]["us"] == "net"
):
    scene_cfg["observation"]["scale"] = scene_cfg["observation"]["scale_net"]

robot_cfg = scene_cfg["robot"]

# robot
if scene_cfg["robot"]["type"] == "kuka":
    robot_articulation_cfg = KUKA_HIGH_PD_CFG
    INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "lbr_joint_0": robot_cfg["joint_pos"][0],
            "lbr_joint_1": robot_cfg["joint_pos"][1],
            "lbr_joint_2": robot_cfg["joint_pos"][2],
            "lbr_joint_3": robot_cfg["joint_pos"][3],  # -1.2,
            "lbr_joint_4": robot_cfg["joint_pos"][4],
            "lbr_joint_5": robot_cfg["joint_pos"][5],  # 1.5,
            "lbr_joint_6": robot_cfg["joint_pos"][6],
        },
        pos=(
            float(robot_cfg["pos"][0]),
            float(robot_cfg["pos"][1]),
            float(robot_cfg["pos"][2]),
        ),  # ((0.0, -0.75, 0.4))
    )

elif scene_cfg["robot"]["type"] == "fr3":
    robot_articulation_cfg = FR3_HIGH_PD_US_CFG
    INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": robot_cfg["joint_pos"][0],
            "fr3_joint2": robot_cfg["joint_pos"][1],
            "fr3_joint3": robot_cfg["joint_pos"][2],
            "fr3_joint4": robot_cfg["joint_pos"][3],  # -1.2,
            "fr3_joint5": robot_cfg["joint_pos"][4],
            "fr3_joint6": robot_cfg["joint_pos"][5],  # 1.5,
            "fr3_joint7": robot_cfg["joint_pos"][6],
        },
        pos=(
            float(robot_cfg["pos"][0]),
            float(robot_cfg["pos"][1]),
            float(robot_cfg["pos"][2]),
        ),  # ((0.0, -0.75, 0.4))
    )

# patient
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(
        float(patient_cfg["pos"][0]),
        float(patient_cfg["pos"][1]),
        float(patient_cfg["pos"][2]),
    ),  # 0.7
    rot=(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])),
)

# bed
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=(
        float(bed_cfg["pos"][0]),
        float(bed_cfg["pos"][1]),
        float(bed_cfg["pos"][2]),
    ),  # 0.7
    rot=(0.5, 0.5, 0.5, 0.5),
)
scale_bed = bed_cfg["scale"]
# use stl: selected_dataset_body_contact
human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id
    for p_id in patient_cfg["id_list"]
]
human_stl_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id
    for p_id in patient_cfg["id_list"]
]
human_raw_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id
    for p_id in patient_cfg["id_list"]
]

target_anatomy = patient_cfg["target_anatomy"]
target_stl_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + str(target_anatomy)
    + ".stl"
    for p_id in patient_cfg["id_list"]
]
target_traj_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + "standard_right_traj_"
    + str(target_anatomy)[-2:]
    + ".stl"
    for p_id in patient_cfg["id_list"]
]

usd_file_list = [
    human_file + "/combined_wrapwrap/combined_wrapwrap.usd"
    for human_file in human_usd_list
]
label_map_file_list = [
    human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list
]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg["label_res"]
scale = 1 / label_res


@configclass
class roboticUSEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = scene_cfg["sim"]["episode_length"]  # 300
    action_scale = 1
    action_space = 4
    observation_space = {"image": [6, 200, 150], "pose": [24]} # 3 frame image stack + 12 value pose history
    state_space = 0
    observation_scale = scene_cfg["observation"]["scale"]
    #VERTEBRA_LABEL_ID = 7
    LIVER_LABEL_ID = 5 

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120, render_interval=decimation
    )

    robot_cfg: ArticulationCfg = robot_articulation_cfg.replace(
        prim_path="/World/envs/env_.*/Robot_US", init_state=INIT_STATE_ROBOT_US
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, env_spacing=4.0, replicate_physics=False   #I have changed the number of env from 100 to 8
    )


class roboticUSEnv(DirectRLEnv):
    cfg: roboticUSEnvCfg

    def _lock_cmd_to_plane_angle(self, cmd: torch.Tensor) -> torch.Tensor:
        """Keep the probe slice angle fixed when the liver task is locked."""
        if not self.lock_probe_x_angle:
            return cmd
        cmd = cmd.clone()
        cmd[:, 2] = self.coronal_x_angle_rad
        return cmd

    def _warp_label_to_convex(self, img: torch.Tensor, fan_angle_deg: float = 0.0) -> torch.Tensor:
        if fan_angle_deg == 0:
            # Skip warping—keep rectangular slice
            return img.permute(0, 2, 1).contiguous()  # just reorient (B,W,H)→(B,H,W)
        B, C, H, W = img.shape
        device = img.device

        fan_angle = fan_angle_deg * torch.pi / 180.0

        yy, xx = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing="ij"
        )

        # normalize coordinates
        x = (xx - (W - 1) / 2.0) / ((W - 1) / 2.0)     # center horizontally
        y = yy / H                      # depth from top

        # polar coordinates (origin at top-center)
        r = y
        theta = x * (fan_angle / 2.0)

        # convert polar -> rectangular sampling coords
        x_src = torch.tan(theta) * r
        y_src = 2.0 * r - 1.0 

        # normalize to [-1,1] for grid_sample
        x_src = torch.tan(theta) * r
        y_src = 2.0 * y_src - 1.0
        
        x_src = torch.clamp(x_src, -1.0, 1.0)

        grid = torch.stack([x_src, y_src], dim=-1)          # (H,W,2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)         # (B,H,W,2)
        warped = F.grid_sample(
            img,
            grid,
            mode="nearest",          # keep discrete labels (also fine for US if you want)
            padding_mode="zeros",
            align_corners=True,
        )
        valid = (x.abs() <= 1.0)
        warped = warped * valid.unsqueeze(0).unsqueeze(0)
        return warped

    def __init__(self, cfg: roboticUSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # ------------------------------------
        # Ultrasound-based start/end detection
        # ------------------------------------
        self.scan_state = torch.zeros(self.scene.num_envs, device=self.sim.device)
        # 0 = searching
        # 2 = done (end anatomy, liver)

        self.prev_us_clarity = torch.zeros(self.scene.num_envs, device=self.sim.device)
   

        if scene_cfg["robot"]["type"] == "kuka":
            self.robot_entity_cfg = SceneEntityCfg(
                "robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"]
            )
        else:
            self.robot_entity_cfg = SceneEntityCfg(
                "robot_US", joint_names=["fr3_joint.*"], body_names=["fr3_link8"]
            )
        self.robot_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_entity_cfg.body_ids[-1]

        # define ik controllers
        ik_params = {"lambda_val": 0.1}
        pose_diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params,
        )
        self.pose_diff_ik_controller = DifferentialIKController(
            pose_diff_ik_cfg, self.scene.num_envs, device=self.sim.device
        )

        # load label maps
        label_map_list = []
        for label_map_file in label_map_file_list:
            label_map = nib.load(label_map_file).get_fdata()
            label_map_list.append(label_map)
        # load ct maps
        ct_map_list = []
        for ct_map_file in ct_map_file_list:
            ct_map = nib.load(ct_map_file).get_fdata()
            ct_min_max = scene_cfg["sim"]["ct_range"]
            ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
            ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
            ct_map_list.append(ct_map)

        # construct label image slicer
        label_convert_map = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
        )

        # construct US simulator

        us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
        us_generative_cfg = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
        )
        self.sim_cfg = scene_cfg["sim"]
        self.init_cmd_pose_min = (
            torch.tensor(
                self.sim_cfg["patient_xz_init_range"][0], device=self.sim.device
            )
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.init_cmd_pose_max = (
            torch.tensor(
                self.sim_cfg["patient_xz_init_range"][1], device=self.sim.device
            )
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        if scene_cfg["observation"]["3D"]:
            img_thickness = us_cfg["image_3D_thickness"]
        else:
            img_thickness = 1
        self.US_slicer = USSlicer(
            us_cfg,
            label_map_list,
            ct_map_list,
            self.sim_cfg["if_use_ct"],
            human_stl_list,
            self.scene.num_envs,
            self.sim_cfg["patient_xz_range"],
            self.sim_cfg["patient_xz_init_range"][0],
            self.sim.device,
            label_convert_map,
            us_cfg["image_size"],
            us_cfg["resolution"],
            img_thickness=img_thickness,
            visualize=self.sim_cfg["vis_seg_map"],
            sim_mode=scene_cfg["sim"]["us"],
            us_generative_cfg=us_generative_cfg,
            allocate_us_random_maps=scene_cfg["observation"]["mode"] == "US",
        )
        self._inject_target_volume()
        self.coronal_x_angle_rad = float(
            scene_cfg["motion_planning"].get("coronal_x_angle_rad", 0.5 * np.pi)
        )
        self.lock_probe_x_angle = bool(
            scene_cfg["motion_planning"].get("lock_probe_x_angle", False)
        )
        self.US_slicer.current_x_z_x_angle_cmd = self._lock_cmd_to_plane_angle(
            (self.init_cmd_pose_min + self.init_cmd_pose_max) / 2
        )

        self.human_world_poses = (
            self.human.data.root_state_w
        )  # these are already the initial poses

        # construct ground truth motion generator
        motion_plan_cfg = scene_cfg["motion_planning"]
        self.max_roll_adj = motion_plan_cfg.get("max_roll_adj", 0.5)
        self.max_action = torch.tensor(
            scene_cfg["action"]["max_action"], device=self.sim.device
        ).reshape((1, -1))
        self.goal_cmd_pose = (
            torch.tensor(motion_plan_cfg["patient_xz_goal"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.goal_cmd_pose = self._lock_cmd_to_plane_angle(self.goal_cmd_pose)
        self.use_vertebra_goal = motion_plan_cfg["use_vertebra_goal"]
        self.gt_motion_generator = GTDiscreteMotionGenerator(
            goal_cmd_pose=self.goal_cmd_pose,
            scale=torch.tensor(motion_plan_cfg["scale"], device=self.sim.device),
            num_envs=self.scene.num_envs,
            surface_map_list=self.US_slicer.surface_map_list,
            surface_normal_list=self.US_slicer.surface_normal_list,
            label_res=label_res,
            US_height=self.US_slicer.height,
        )

        self.vertebra_viewer = None
        if self.use_vertebra_goal: # new only if task still wants vertebra goals
            self.vertebra_viewer = VertebraViewer(
                self.scene.num_envs,
                len(human_usd_list),
                target_stl_file_list,
                target_traj_file_list,
                False,
                label_res,
                self.sim.device,
            )

        # Added: observation space: image history + pose history as separate branches
        _W, _H = us_cfg["image_size"]
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(6, _W, _H), dtype=np.float32),
            "pose":  gym.spaces.Box(low=0.0, high=1.0, shape=(24,),     dtype=np.float32),
        })
        self.single_observation_space["policy"] = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(6, _W, _H), dtype=np.float32),
            "pose":  gym.spaces.Box(low=0.0, high=1.0, shape=(24,),     dtype=np.float32),
        })

        self.termination_direct = True
        self.observation_mode = scene_cfg["observation"]["mode"]
        self.action_mode = scene_cfg["action"]["mode"]
        self.action_scale = (
            torch.tensor(scene_cfg["action"]["scale"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )

        self.w_pos      = scene_cfg["reward"].get("w_pos", 0.0)
        self.w_liver    = scene_cfg["reward"].get("w_liver", 0.0)
        self.w_bone     = scene_cfg["reward"].get("w_bone", 0.0)
        self.w_progress = scene_cfg["reward"].get("w_progress", 0.0)
        self.w_coverage    = scene_cfg["reward"].get("w_coverage", 5.0)
        self.shadow_thresh = scene_cfg["reward"].get("shadow_thresh", 0.15)
        self.alpha1        = scene_cfg["reward"].get("alpha1", 1.0)
        self.alpha2        = scene_cfg["reward"].get("alpha2", 0.5)
        self.attenuation_Rc = scene_cfg["reward"].get("attenuation_Rc", 30.0)
        self.terminal_bonus_kend = scene_cfg["reward"].get("terminal_bonus_kend", 0.0)


        self.single_action_space = gym.spaces.Box(
            low=-(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            high=(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            shape=(self.cfg.action_space,),
            dtype=np.float32,
        )

        self.num_step = 0

        # Added:  Frame Buffer: holds last 6 Slices for every env
        self.N_frames = 6
        _W, _H = us_cfg["image_size"]   # [W, H] from yaml
        self.frame_buffer = torch.zeros(
            self.scene.num_envs, self.N_frames, _W, _H,
            device=self.sim.device
        )
        # Added: pose buffer: last N poses, each = [x, z, angle, roll]
        self.pose_buffer = torch.zeros(
            self.scene.num_envs, self.N_frames, 4,
            device=self.sim.device
        )

        # normalization ranges [0, 1]
        xz_range = self.sim_cfg["patient_xz_range"]
        self.pose_norm_min = torch.tensor(
            [xz_range[0][0], xz_range[0][1], -3.14, -self.max_roll_adj],
            device=self.sim.device
        )
        self.pose_norm_max = torch.tensor(
            [xz_range[1][0], xz_range[1][1],  3.14,  self.max_roll_adj],
            device=self.sim.device
        )

        self._img_W, self._img_H = us_cfg["image_size"]


    def get_US_target_pose(self):

        if self.vertebra_viewer is None: #new: if no vertebra viewer, just return the original goal pose
            return

        # compute position change
        vertebra_to_US_2d_pos = torch.tensor(
            scene_cfg["motion_planning"]["vertebra_to_US_2d_pos"]
        ).to(self.sim.device)

        vertebra_2d_pos = self.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
        US_target_2d_pos = vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0)

        US_target_2d_angle = self.goal_cmd_pose[:, 2:3] * torch.ones_like(
            vertebra_2d_pos[:, 0:1]
        )

        US_target_2d = torch.cat([US_target_2d_pos, US_target_2d_angle], dim=-1)
        US_target_2d = self._lock_cmd_to_plane_angle(US_target_2d)

        self.goal_cmd_pose = US_target_2d

    def _setup_scene(self):
        """Configuration for a cart-pole scene."""

        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # lights
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        )
        dome_light_cfg.func("/World/Light", dome_light_cfg)

        # articulation
        # kuka US
        self.robot = Articulation(self.cfg.robot_cfg)

        if scene_cfg["sim"]["vis_us"]:
            usd_folder = "usd_colored"
        else:
            usd_folder = "usd_no_contact"
        medical_bed_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bed",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/"
                + usd_folder
                + "/hospital_bed.usd",
                scale=(scale_bed, scale_bed, scale_bed),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),  # Improves a lot of time count=8 0.014-0.013
            ),
            init_state=INIT_STATE_BED,
        )
        medical_bed = RigidObject(medical_bed_cfg)

        # human:
        human_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Human",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_file_list,
                random_choice=False,
                scale=(label_res, label_res, label_res),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=0.000001,
                    max_angular_velocity=0.000001,
                    max_depenetration_velocity=0.00001,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=INIT_STATE_HUMAN,
        )
        self.human = RigidObject(human_cfg)
        # assign members
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human
        # --------------------------------------------------
        # RGB CAMERA (third-person debug camera)
        # --------------------------------------------------
        self.rgb_camera = Camera(
            CameraCfg(
                prim_path="/World/envs/env_.*/ThirdPersonCamera",
                width=640,
                height=480,
                data_types=["rgb"],
                update_period=0.0,  # every sim frame
                spawn=PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=500.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.7, 50.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(-1.8, 0.6, 2.2),
                    rot=(0.4, -0.7071, 0.05, 0.7071),
                    convention="local",
                ),
            )
        )

        # register camera in scene
        self.scene.sensors["third_person_camera"] = self.rgb_camera
        
    def us_clarity_score(us_img: torch.Tensor) -> torch.Tensor:
        """
        us_img: (B, C, H, W), float or uint
        returns: (B,) clarity score
        """
    # Sobel-like gradient magnitude (cheap + stable)
        gx = us_img[:, :, :, 1:] - us_img[:, :, :, :-1]
        gy = us_img[:, :, 1:, :] - us_img[:, :, :-1, :]
        clarity = gx.abs().mean(dim=(1, 2, 3)) + gy.abs().mean(dim=(1, 2, 3))
        return clarity
        cfg: roboticUSEnvCfg

    def _orient_convex_hw(self, img_wh: torch.Tensor) -> torch.Tensor:
        if img_wh.dim() == 3:
            img_hw = img_wh.permute(0, 2, 1).contiguous()
        else:
            img_hw = img_wh.permute(0, 2, 1, 3).contiguous()

        img_hw = torch.flip(img_hw, dims=[2])   # horizontal flip

        return img_hw
    
    #Added: Pose Normalization helper
    def _normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        return (pose - self.pose_norm_min) / (
            self.pose_norm_max - self.pose_norm_min + 1e-6
        )

    def _apply_superficial_bone_shadow(self, label_hw: torch.Tensor) -> torch.Tensor:
        B, H, W = label_hw.shape
        shadowed = label_hw.clone()
        shadow = torch.zeros(B, H, W, dtype=torch.bool, device=label_hw.device)
        top_rows = int(0.4 * H) #only apply shadow in the top 40% of the image, where superficial bones are likely to appear    
        rows = torch.arange(H, device=label_hw.device).view(1, H, 1).expand(B, H, W)

        for lbl in [13, 15]:
            seed = (label_hw == lbl).float()
            seed[:, top_rows:, :] = 0.0

            has_seed = seed.sum(dim=1) > 0
            first_occ = torch.argmax(seed, dim=1)
            first_occ_exp = first_occ.unsqueeze(1).expand(B, H, W)
            has_seed_exp = has_seed.unsqueeze(1).expand(B, H, W)

            col_shadow = (rows > first_occ_exp) & has_seed_exp   # shadow every pixel that is below the first seed row

            shadow |= col_shadow

        shadow &= ~((label_hw == 13) | (label_hw == 15))
        shadowed[shadow] = 0
        return shadowed

    def _inject_target_volume(self):
        """One-time injection of a small spherical target sub-volume into the patient's
        label map, only overwriting liver voxels. Used for the 3D coverage-based reward
        (supervisor's updated task: scan/reconstruct a specific volume inside the liver,
        analogous to a tumor in Bi et al. 2026)."""
        target_cfg = scene_cfg.get("target_volume", {})
        if not target_cfg.get("enabled", False):
            return

        cx, cy, cz = [int(v) for v in target_cfg["center_voxel"]]
        radius_mm = float(target_cfg["radius_mm"])
        radius_voxels = radius_mm / (self.US_slicer.label_res * 1000.0)
        target_label_id = int(target_cfg["label_id"])
        r_int = int(radius_voxels) + 1

        # per-human-type bookkeeping needed for 3D coverage tracking (step 2)
        self.target_bbox_list = []
        self.target_mask_local_list = []
        self.target_total_voxels_list = []

        for i in range(self.US_slicer.n_human_types):
            label_map = self.US_slicer.label_maps[i]
            X, Y, Z = label_map.shape

            x_min, x_max = max(cx - r_int, 0), min(cx + r_int + 1, X)
            y_min, y_max = max(cy - r_int, 0), min(cy + r_int + 1, Y)
            z_min, z_max = max(cz - r_int, 0), min(cz + r_int + 1, Z)

            xs = torch.arange(x_min, x_max, device=label_map.device).view(-1, 1, 1)
            ys = torch.arange(y_min, y_max, device=label_map.device).view(1, -1, 1)
            zs = torch.arange(z_min, z_max, device=label_map.device).view(1, 1, -1)
            dist2 = (xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2
            sphere_mask = dist2 <= radius_voxels ** 2

            local_block = label_map[x_min:x_max, y_min:y_max, z_min:z_max]
            liver_mask = local_block == self.cfg.LIVER_LABEL_ID
            write_mask = sphere_mask & liver_mask
            local_block[write_mask] = target_label_id
            label_map[x_min:x_max, y_min:y_max, z_min:z_max] = local_block

            n_voxels = int(write_mask.sum().item())
            print(f"[TARGET VOLUME] human_type={i}: injected {n_voxels} voxels of label "
                  f"{target_label_id} around center ({cx},{cy},{cz}), "
                  f"radius {radius_voxels:.1f} voxels ({radius_mm}mm)")

            # store for 3D coverage tracking: where the target sits (bbox) and which
            # voxels inside that bbox actually belong to the target (write_mask)
            self.target_bbox_list.append((x_min, x_max, y_min, y_max, z_min, z_max))
            self.target_mask_local_list.append(write_mask.clone())
            self.target_total_voxels_list.append(n_voxels)

        self._init_coverage_buffers()

    def _init_coverage_buffers(self):
        """Per-env boolean buffer tracking which target voxels have been scanned so far
        in the current episode. Reset every episode in _reset_idx."""
        num_envs = self.scene.num_envs
        self.scanned_target_mask = []
        for i in range(self.US_slicer.n_human_types):
            dx, dy, dz = self.target_mask_local_list[i].shape
            self.scanned_target_mask.append(
                torch.zeros((num_envs, dx, dy, dz), dtype=torch.bool, device=self.sim.device)
            )
        # Total target voxels per env
        self.target_total_per_env = torch.zeros(num_envs, device=self.sim.device)
        for b in range(num_envs):
            hi = b % self.US_slicer.n_human_types
            self.target_total_per_env[b] = self.target_total_voxels_list[hi]

        # Cumulative rc sum per env within the current episode (reset each episode)
        self.rc_episode_sum = torch.zeros(num_envs, device=self.sim.device)
        self.reached_95 = torch.zeros(num_envs, dtype=torch.bool, device=self.sim.device)

        # Per-episode accumulators for paper Eq. 6 terminal reward: D and P
        self.episode_dist_sum   = torch.zeros(num_envs, device=self.sim.device)  # Σ dt/Rc
        self.episode_rs_sum     = torch.zeros(num_envs, device=self.sim.device)  # Σ (1-pt)
        self.episode_step_count = torch.zeros(num_envs, device=self.sim.device)  # T

        # Run-wide accumulators for end-of-training summary
        self.run_cov_sum = 0.0
        self.run_term_sum = 0.0
        self.run_ep_reward_sum = 0.0
        self.run_done_count = 0

        # Target center (x, z) in voxel space per env — used for attenuation distance ra
        self.target_center_per_env = torch.zeros(num_envs, 2, device=self.sim.device)
        for b in range(num_envs):
            hi = b % self.US_slicer.n_human_types
            x_min, x_max, _, _, z_min, z_max = self.target_bbox_list[hi]
            self.target_center_per_env[b, 0] = (x_min + x_max) / 2.0  # cx
            self.target_center_per_env[b, 1] = (z_min + z_max) / 2.0  # cz

    def _init_voxel_visualizer(self):
        """Create VisualizationMarkers for live 3D coverage display in Isaac Sim viewport.
        Red spheres = unscanned target voxels, green spheres = scanned. Env 0 only."""
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/CoverageVoxels",
            markers={
                "unscanned": sim_utils.SphereCfg(
                    radius=0.0015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                ),
                "scanned": sim_utils.SphereCfg(
                    radius=0.0015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                ),
            },
        )
        self._voxel_markers = VisualizationMarkers(cfg)

        # Pre-compute patient-space positions of all target voxels (fixed geometry)
        x_min, x_max, y_min, y_max, z_min, z_max = self.target_bbox_list[0]
        mask = self.target_mask_local_list[0]  # (dx, dy, dz) bool
        lx, ly, lz = torch.where(mask)        # (N,) local indices inside bbox
        gx = (lx + x_min).float()
        gy = (ly + y_min).float()
        gz = (lz + z_min).float()
        self._target_voxel_patient_pos = torch.stack([gx, gy, gz], dim=1) * label_res  # (N,3) meters
        self._viz_lx, self._viz_ly, self._viz_lz = lx, ly, lz

    def _update_voxel_viz(self):
        """Update 3D marker positions/colours each visualisation step."""
        R_mat = matrix_from_quat(self.world_to_human_rot[0:1])[0]        # (3,3)
        pts = self._target_voxel_patient_pos.to(self.sim.device)          # (N,3)
        world_pos = (R_mat @ pts.T).T + self.world_to_human_pos[0]       # (N,3)

        scanned = self.scanned_target_mask[0][0]                          # (dx,dy,dz) for env 0
        scanned_flat = scanned[self._viz_lx, self._viz_ly, self._viz_lz] # (N,) bool
        proto_indices = scanned_flat.long()                               # 0=red, 1=green

        self._voxel_markers.visualize(translations=world_pos, marker_indices=proto_indices)

    def _update_coverage(self):
        """
        Every step, we look up which pixels see target-sphere voxels,
        find the unique voxels they map to,count how many of those are new
        never seen before this episode), add the count to new_coverage_count
        and permanently mark them as seen for the rest of this episode
        This is used to compute a 3D coverage reward.
       """
        if not hasattr(self, "scanned_target_mask"):
            return
        if not hasattr(self.US_slicer, "last_sampled_coords_per_type"):
            return

        num_envs = self.scene.num_envs
        n_types = self.US_slicer.n_human_types
        self.new_coverage_count = torch.zeros(num_envs, device=self.sim.device)

        # Gate: shadow fraction using same correct Nt as _get_rewards() (paper Eq. 4).
        # _label_for_task is post-shadow so (!=0) gives only non-shadow scanned pixels.
        # Must add shadow pixels back to get the true full scan area Nt.
        if hasattr(self, "_shadow_mask_for_task") and hasattr(self, "_label_for_task"):
            sh = self._shadow_mask_for_task   # (num_envs, H, W)
            shadow_pix = sh.sum(dim=(1, 2)).float()
            non_shadow_pix = (self._label_for_task != 0).sum(dim=(1, 2)).float()
            Nt = (shadow_pix + non_shadow_pix).clamp(min=1.0)
            shadow_frac_per_env = shadow_pix / Nt
            shadow_ok = shadow_frac_per_env < self.shadow_thresh  # (num_envs,) bool
        else:
            shadow_ok = torch.ones(num_envs, dtype=torch.bool, device=self.sim.device)

        for i in range(n_types):
            env_inds = torch.arange(i, num_envs, n_types, device=self.sim.device)
            B_i = env_inds.numel()

            x_min, x_max, y_min, y_max, z_min, z_max = self.target_bbox_list[i]
            target_mask_local = self.target_mask_local_list[i]  # (dx, dy, dz) bool
            dx, dy, dz = target_mask_local.shape  # dimensions of the bounding box of target sphere

            coords = self.US_slicer.last_sampled_coords_per_type[i]  # (B_i, W, H, E, 3)
            coords_2d = coords[:, :, :, 0, :]            # (B_i, W, H, 3) — elevation 0

            vx = coords_2d[..., 0]  # (B_i, W, H) — volume X axis
            vy = coords_2d[..., 1]  # volume Y axis
            vz = coords_2d[..., 2]  # volume Z axis

            #  pixels  inside the target sphere's bounding box
            in_bbox = (
                (vx >= x_min) & (vx < x_max) &
                (vy >= y_min) & (vy < y_max) &
                (vz >= z_min) & (vz < z_max)
            )  # (B_i, W, H)

            # Convert global voxel coordinates to local coordinates 
            lx = (vx - x_min).clamp(0, dx - 1).long()  # (B_i, W, H)
            ly = (vy - y_min).clamp(0, dy - 1).long()
            lz = (vz - z_min).clamp(0, dz - 1).long()

            # is this local voxel actually inside the sphere 
            is_target = target_mask_local[lx, ly, lz]   # (B_i, W, H) bool
            hit = in_bbox & is_target                    # (B_i, W, H)

            for b_local in range(B_i):
                env_id = env_inds[b_local].item()
                if not shadow_ok[env_id]:
                    continue  # skip env if shadow fraction too high
                hit_b = hit[b_local]  # (W, H)
                if not hit_b.any(): #skip if no pixel hits the sphere
                    continue

                #flatten
                lx_hit = lx[b_local][hit_b]
                ly_hit = ly[b_local][hit_b]
                lz_hit = lz[b_local][hit_b]

                unique_coords = torch.unique(
                    torch.stack([lx_hit, ly_hit, lz_hit], dim=1), dim=0
                )  # (M, 3)
                lx_u, ly_u, lz_u = unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]

                # only voxels not yet seen in this episode are counted
                not_yet = ~self.scanned_target_mask[i][env_id, lx_u, ly_u, lz_u]
                self.new_coverage_count[env_id] += not_yet.sum()

                # permanently mark as scanned for the rest of this episode
                self.scanned_target_mask[i][env_id, lx_u, ly_u, lz_u] = True

    def _get_observations(self) -> dict:
        # -------------------------------------------------
        # Pose extraction
        # -------------------------------------------------
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos = self.human_world_poses[:, 0:3]
        self.world_to_human_rot = self.human_world_poses[:, 3:7]

        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]

        self.num_step += 1

        # -------------------------------------------------
        # Observation modes
        if self.observation_mode == "US":
            self.US_slicer.slice_US(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            self._us_for_vis = self.US_slicer.us_img_tensor[..., 0]

            US_img = (
                self.US_slicer.us_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )

            observations = {"policy": US_img}

        elif self.observation_mode == "CT":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )

            CT_img = (
                self.US_slicer.ct_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )

            observations = {"policy": CT_img}

        elif self.observation_mode == "seg":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            
            #Added: Create task image 
            label_wh = self.US_slicer.label_img_tensor[:, :, :, 0].to(torch.long)   # (B, W, H)
            label_hw = self._orient_convex_hw(label_wh)                              # (B, H, W)
            label_task_hw = self._apply_superficial_bone_shadow(label_hw)            # (B, H, W)
            self._label_for_task = label_task_hw

            # shadow mask
            self._shadow_mask_for_task = (label_hw != 0) & (label_task_hw == 0)

            # Added: Update 3D coverage reward
            self._update_coverage()

            current_frame = (
                label_task_hw.permute(0, 2, 1).unsqueeze(1).float()
                * self.cfg.observation_scale
            )

            # Added: frame buffer: push current frame in, drop oldest
            self.frame_buffer = torch.cat([
                self.frame_buffer[:, 1:, :, :],
                current_frame,
            ], dim=1)  # (B, 3, W, H)

            #  pose buffer: push current pose in, drop oldest 
            current_pose = torch.cat([
                self.US_slicer.current_x_z_x_angle_cmd[:, :3],
                self.US_slicer.roll_adj[:, :1],
            ], dim=-1)  # (B, 4)
            current_pose_norm = self._normalize_pose(current_pose)
            self.pose_buffer = torch.cat([
                self.pose_buffer[:, 1:, :],
                current_pose_norm.unsqueeze(1),
            ], dim=1)  # (B, 3, 4)

            B = self.scene.num_envs
            observations = {
                "policy": {
                    "image": self.frame_buffer.clone(),
                    "pose":  self.pose_buffer.reshape(B, -1).clone(), #(B, 3, 4) → (B, 12) flat vector for MLP
                }
            }
        # -------------------------------------------------
        # Semantic navigation logic
        # -------------------------------------------------
        if self.observation_mode == "seg" and hasattr(self, "_label_for_task"):
            label = self._label_for_task          # (B, H, W) convex
        else:
            label = self.US_slicer.label_img_tensor[:, :, :, 0].permute(0, 2, 1).contiguous()  # (B,H,W)

        liver_visible = (label == self.cfg.LIVER_LABEL_ID).any(dim=(1, 2)) #new: check liver instead of ey
        self._liver_visible = liver_visible
        end_mask = liver_visible
        self.scan_state[end_mask] = 2


        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------
        if self.sim_cfg["vis_us"] and self.num_step % self.sim_cfg["vis_int"] == 0:
            self.US_slicer.visualize("LABEL_RGB")


        # -------------------------------------------------
        # Debug (optional)
        # -------------------------------------------------
        if self.sim.has_gui() and self.num_step % 30 == 0:

            import matplotlib.pyplot as plt

            if self.observation_mode == "US":
                us0 = self._us_for_vis[0].detach().cpu().numpy()

                plt.figure("Ultrasound (US)")
                plt.clf()
                plt.imshow(us0, cmap="gray")
                plt.title(f"US image @ step {self.num_step}")
                plt.colorbar()
                plt.pause(0.001)

            elif self.observation_mode == "seg":
                if self.num_step == 1:
                    labels_all = self.US_slicer.label_img_tensor[..., 0]
                    unique_ids = torch.unique(labels_all)
                    print("All label IDs in segmentation volume:", unique_ids.cpu().tolist())
                
                # labels: (B, W, H)
                label_wh = self.US_slicer.label_img_tensor[:, :, :, 0].to(torch.long)

                # apply SAME orientation as RL
                label_hw = self._orient_convex_hw(label_wh)
                label_hw = self._apply_superficial_bone_shadow(label_hw)

                if hasattr(self.US_slicer, "last_sampled_coords"):
                    # last_sampled_coords is (B, W, H, E, 3) — raw orientation, before _orient_convex_hw's
                    # permute+flip. Apply the SAME transform so indices line up with the displayed image.
                    coords_oriented = self.US_slicer.last_sampled_coords.permute(0, 2, 1, 3, 4)  # (B,H,W,E,3)
                    coords_oriented = torch.flip(coords_oriented, dims=[2])  # match horizontal flip
                    H_img, W_img = label_hw.shape[-2], label_hw.shape[-1]

                    # TARGET_PICK_FRAC: (row_frac, col_frac) as fraction of image height/width, 0.0-1.0
                    # (0.5, 0.5) = dead center. Edit these to probe a specific pixel instead of the center.
                    TARGET_PICK_FRAC = (0.5, 0.5)
                    h_idx = int(TARGET_PICK_FRAC[0] * (H_img - 1))
                    w_idx = int(TARGET_PICK_FRAC[1] * (W_img - 1))

                    voxel_at_pick = coords_oriented[0, h_idx, w_idx, 0]
                    if not os.environ.get("SONOGYM_INFERENCE"):
                        print(f"[TARGET PICK] voxel coords at frac={TARGET_PICK_FRAC} "
                              f"(pixel h={h_idx},w={w_idx} of {H_img}x{W_img}): {voxel_at_pick.tolist()}")

                # pick env 0 for plotting
                label2d = label_hw[0]  # (H,W)
                if self.num_step == 5:
                    unique_ids = torch.unique(label2d)

                    for k in unique_ids.cpu().tolist():
                        mask = (label2d == k).cpu().numpy()

                        import matplotlib.pyplot as plt
                        plt.figure(f"Label ID {k}")
                        plt.imshow(mask, cmap="gray")
                        plt.title(f"Segmentation mask for ID {k}")
                        plt.axis("off")

                if self.num_step % 10 == 0 and not os.environ.get("SONOGYM_INFERENCE"):
                    u = torch.unique(label2d)
                    print("Unique IDs in CURRENT slice:", u.detach().cpu().tolist())
                    
                
                palette = torch.zeros((256,3), dtype=torch.uint8, device=label2d.device)

                palette[0]  = torch.tensor([0,0,0], device=label2d.device)        # background
                palette[1]  = torch.tensor([255,255,0], device=label2d.device)    # spleen
                palette[63] = torch.tensor([160, 160, 255], device=label2d.device)  # IVC
                palette[8]  = torch.tensor([0,255,255], device=label2d.device)    # muscle
                palette[10] = torch.tensor([255, 165, 0], device=label2d.device)  # organ tissue
                palette[12] = torch.tensor([144,238,144], device=label2d.device)      # skin / fat
                palette[13] = torch.tensor([255,0,0], device=label2d.device)      # bone
                palette[5] = torch.tensor([0,0,200], device=label2d.device)  # main organ class
                palette[64] = torch.tensor([0, 100, 255], device=label2d.device)  # Portal Vein
                palette[6] = torch.tensor([148,0,211], device=label2d.device)      # Stomach
                palette[7] = torch.tensor([139, 69, 19], device=label2d.device)  # pancreas
                palette[4] = torch.tensor([255, 255, 255], device=label2d.device)  # gallbladder
                palette[15] = torch.tensor([255, 105, 180], device=label2d.device)  # costal cartilages
                palette[52] = torch.tensor([245, 222, 179], device=label2d.device)  # aorta
                palette[200] = torch.tensor([0, 255, 0], device=label2d.device)  # target volume


                rgb = palette[label2d].detach().cpu().numpy()  # (H,W,3)
                # --- get CT slice for overlay ---
                ct_wh = self.US_slicer.ct_img_tensor[..., 0]
                ct_hw = self._orient_convex_hw(ct_wh)[0].detach().cpu().numpy()
                ct_hw = (ct_hw - ct_hw.min()) / (ct_hw.max() - ct_hw.min() + 1e-6)


                plt.figure("Convex Semantic Label")
                plt.clf()
                plt.imshow(ct_hw , cmap="gray")
                plt.imshow(rgb ,alpha=1.0, interpolation="nearest")
                #plt.imshow(rgb, interpolation="nearest")
                #plt.imshow(rgb[::-1], interpolation="nearest")
                plt.title("Convex Label Map (Semantic RGB)")
                plt.axis("off")
                import matplotlib.patches as mpatches 
                legend_items = [
                    mpatches.Patch(color=(0, 0, 0), label="Background (ID 0)"),
                    mpatches.Patch(color=(1,1,0), label="Spleen (ID 1)"), 
                    mpatches.Patch(color=(160/255, 160/255, 1), label="IVC (ID 63)"),
                    mpatches.Patch(color=(0,1,1), label="Muscle (ID 8)"),
                    mpatches.Patch(color=(1, 165/255, 0), label="Organ tissue (ID 10)"),
                    mpatches.Patch(color=(0.56,0.93,0.56), label="Skin / Fat (ID 12)"),
                    mpatches.Patch(color=(1,0,0), label="Bone / Vertebra (ID 13)"),
                    mpatches.Patch(color=(0,0,0.6), label="Liver (ID 5)"),
                    mpatches.Patch(color=(0, 100/255, 1), label="Portal Vein (ID 64)"),
                    mpatches.Patch(color=(148/255, 0, 211/255), label="Stomach (ID 6)"),
                    mpatches.Patch(color=(139/255, 69/255, 19/255), label="Pancreas (ID 7)"), 
                    mpatches.Patch(color=(1, 1, 1), label="Gallbladder (ID 4)"), 
                    mpatches.Patch(color=(1, 105/255, 180/255), label="Costal Cartilages (ID 15)"),
                    mpatches.Patch(color=(245/255, 222/255, 179/255), label="Aorta (ID 52)"),
                    mpatches.Patch(color=(0, 1, 0), label="Target Volume (ID 200)"),

                ]
                plt.legend(handles=legend_items,loc="center left",bbox_to_anchor=(1.02, 0.5),frameon=True)
                plt.pause(0.001)

            elif self.observation_mode == "CT":
                ct_wh = self.US_slicer.ct_img_tensor[..., 0]          # (B, W, H)
                ct_hw = self._orient_convex_hw(ct_wh)[0].detach().cpu().numpy()  # (H, W)
                ct_hw = (ct_hw - ct_hw.min()) / (ct_hw.max() - ct_hw.min() + 1e-6)

                plt.figure("CT Slice")
                plt.clf()
                plt.imshow(ct_hw, cmap="gray")
                plt.title(f"CT slice @ step {self.num_step}")
                plt.colorbar()
                plt.pause(0.001)
            # in _get_observations:
            x_cmd = self.US_slicer.current_x_z_x_angle_cmd[0, 0].int()
            z_cmd = self.US_slicer.current_x_z_x_angle_cmd[0, 1].int()
            normal_human = self.US_slicer.surface_normal_list[0][x_cmd, z_cmd]

            
            R = matrix_from_quat(self.world_to_human_rot[0:1])[0]   # (3,3)
            normal_world = (R @ normal_human).cpu().numpy()
            robot_pos    = self.robot.data.root_state_w[0, 0:3].cpu().numpy()
            patient_pos  = self.world_to_human_pos[0].cpu().numpy()
            vec_robot_to_patient = patient_pos - robot_pos
            x_cmd = self.US_slicer.current_x_z_x_angle_cmd[0, 0].int()
            z_cmd = self.US_slicer.current_x_z_x_angle_cmd[0, 1].int()
            normal_human = self.US_slicer.surface_normal_list[0][x_cmd, z_cmd]


          

        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Se3Keyboard.advance() returns [x, y, z, rx, ry, rz].
        # This task uses tangential probe motion (local x/y) and in-plane
        # probe rotation (local z / rz) to change the slice angle.
        if actions.shape[-1] == 6:
            actions = actions[:, [0, 1, 3, 5]]
        # update the target command
        # actions = torch.zeros_like(actions).to(self.sim.device)
        # actions[:, 0] = 1
        if self.action_mode == "continuous":
            actions = torch.clamp(
                actions * self.action_scale, -self.max_action, self.max_action
            )
        elif self.action_mode == "discrete":
            actions = torch.sign(actions) * self.action_scale
        else:
            raise ValueError("Invalid action mode")

        if self.lock_probe_x_angle:
            actions = actions.clone()
            actions[:, 2] = 0.0

        self.actions = actions
        # actions: tangential x/y slide in the probe frame
        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        human_to_ee_rot_mat = matrix_from_quat(human_to_ee_quat)
        dx_dz_human = (
            actions[:, 0].unsqueeze(1) * human_to_ee_rot_mat[:, :, 0] # EE X axis in human frame
            + actions[:, 1].unsqueeze(1) * human_to_ee_rot_mat[:, :, 1] # EE Y axis in human frame
        )
        cmd = torch.cat([dx_dz_human[:, [0, 2]], actions[:, 2:3]], dim=-1) #delta angle around EE Z axis (in human frame)
        self.US_slicer.update_cmd(cmd)
        self.US_slicer.roll_adj += actions[:, 3:4]
        self.US_slicer.roll_adj = torch.clamp(
            self.US_slicer.roll_adj,
            min=-self.max_roll_adj,
            max=self.max_roll_adj,
        )
        if self.num_step % 10000 == 0:

            print(
                f"[step {self.num_step}] "
                f"current_x={self.US_slicer.current_x_z_x_angle_cmd[0, 0].item():.2f}, "
                f"current_z={self.US_slicer.current_x_z_x_angle_cmd[0, 1].item():.2f}, "
                f"current_angle={self.US_slicer.current_x_z_x_angle_cmd[0, 2].item():.4f}, "
                f"current_roll={self.US_slicer.roll_adj[0, 0].item():.4f}"
            ) # Current command in human frame (x,z,angle,roll)
        

        # compute desired world to ee pose
        world_to_ee_target_pos, world_to_ee_target_rot = (
            self.US_slicer.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos, self.world_to_human_rot
            )
        )
        # compute desired base to ee pose
        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            world_to_ee_target_rot,
        )
        base_to_ee_target_pose = torch.cat(
            [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
        )

        # set command to robot
        # set new command
        self.pose_diff_ik_controller.set_command(base_to_ee_target_pose)

        # record extras
        self.extras["human_to_ee_pos"] = human_to_ee_pos
        self.extras["human_to_ee_quat"] = human_to_ee_quat

    def _apply_action(self):
        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        # get current ee
        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        # # get joint position targets
        US_jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
        ]
        US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = self.pose_diff_ik_controller.compute(
            self.US_ee_pos_b, self.US_ee_quat_b, US_jacobian, US_joint_pos
        )
        # apply joint position target
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )


    def _get_rewards(self) -> torch.Tensor:
        if self.observation_mode == "seg" and hasattr(self, "_label_for_task"):
            label = self._label_for_task
        else:
            label = self.US_slicer.label_img_tensor[:, :, :, 0]

        B, H, W = label.shape

        if hasattr(self, "_shadow_mask_for_task"):
            shadow_pixels = self._shadow_mask_for_task.sum(dim=(1, 2)).float()   # n_shadow_t
            non_shadow_pixels = (label != 0).sum(dim=(1, 2)).float()
            Nt = (non_shadow_pixels + shadow_pixels).clamp(min=1.0)
            shadow_fraction = shadow_pixels / Nt   # paper's pt = n_shadow_t / Nt
        else:
            shadow_fraction = torch.zeros(B, device=self.sim.device)

        # Reward 1:  coverage level rc = nt / N
        if hasattr(self, "new_coverage_count") and hasattr(self, "target_total_per_env"):
            rc = self.new_coverage_count / self.target_total_per_env.clamp(min=1.0)
        else:
            rc = torch.zeros(B, device=self.sim.device)

        # Reward 2: attenuation minimization: ra = exp(-dt / Rc)
        probe_xz = self.US_slicer.current_x_z_x_angle_cmd[:, :2]  # only [x, z] are considered
        dt = torch.norm(probe_xz - self.target_center_per_env, dim=1) #euclidian distance
        ra = torch.exp(-dt / self.attenuation_Rc)  # Result always in [0, 1]

        # Reward 3: shadow avoidance: rs = 1 - pt
        rs = 1.0 - shadow_fraction  

        #  combined per-step reward
        rt = self.w_coverage * rc + self.alpha1 * ra + self.alpha2 * rs

        # gate: use rt when shadow is acceptable, -0.1 otherwise (paper Eq. 7)
        shadow_ok = shadow_fraction < self.shadow_thresh  # (B,) bool
        reward = torch.where(
            shadow_ok,
            rt,
            torch.full((B,), -0.1, device=self.sim.device),
        )

        # accumulate D and P  
        if hasattr(self, "episode_dist_sum"):
            self.episode_dist_sum   += dt / self.attenuation_Rc
            self.episode_rs_sum     += rs
            self.episode_step_count += 1

        # terminal success bonus
        # D = avg(dt/Rc) over episode,  P = avg(1-pt) over episode
        if self.terminal_bonus_kend > 0.0 and hasattr(self, "scanned_target_mask") and hasattr(self, "target_total_per_env"):
            n_types = self.US_slicer.n_human_types
            scanned_total = torch.zeros(B, device=self.sim.device)
            for i, mask in enumerate(self.scanned_target_mask):
                env_inds = torch.arange(i, B, n_types, device=self.sim.device)
                scanned_total[env_inds] = mask[env_inds].sum(dim=(1, 2, 3)).float()
            cov_frac = scanned_total / self.target_total_per_env.clamp(min=1.0) #episode volume mean
            T = self.episode_step_count.clamp(min=1.0) # number of steps in that episode
            D = (self.episode_dist_sum / T).clamp(min=0.05)  # avg normalised distance; clamp avoids 1/D explosion
            P = self.episode_rs_sum / T                       # avg shadow-free fraction
            r_end = self.terminal_bonus_kend * (1.0 + self.alpha1 / D + self.alpha2 * P)
            # add terminal bonus only on first crossing 0.95 (once per episode)
            just_crossed = (cov_frac >= 0.95) & ~self.reached_95
            self.reached_95 = self.reached_95 | (cov_frac >= 0.95)
            reward = reward + torch.where(
                just_crossed,
                r_end,
                torch.zeros(B, device=self.sim.device),
            )

        # accumulate rc across the episode (reset in _reset_idx)
        if hasattr(self, "rc_episode_sum"):
            self.rc_episode_sum += rc

        # cache for episode-end logging in _get_dones()
        self._wandb_cache = {
            "ra_mean": ra.mean().item(),
            "rs_mean": rs.mean().item(),
            "shadow_ok_frac": shadow_ok.float().mean().item(),
            "shadow_fraction": shadow_fraction.mean().item(),
            "reward_mean": reward.mean().item(),
            "reward_max": reward.max().item(),
        }
        self.total_reward += reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.scene.num_envs
        n_types = self.US_slicer.n_human_types

        # compute cumulative coverage fraction per env from the episode buffer
        if hasattr(self, "scanned_target_mask") and hasattr(self, "target_total_per_env"):
            scanned_total = torch.zeros(num_envs, device=self.sim.device)
            for i, mask in enumerate(self.scanned_target_mask):
                env_inds = torch.arange(i, num_envs, n_types, device=self.sim.device)
                scanned_total[env_inds] = mask[env_inds].sum(dim=(1, 2, 3)).float()
            coverage_fraction = scanned_total / self.target_total_per_env.clamp(min=1.0)
        else:
            coverage_fraction = torch.zeros(num_envs, device=self.sim.device)

        terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.sim.device)
        success = coverage_fraction >= 0.95

        time_outs = self.episode_length_buf >= self.max_episode_length - 1

        episode_done = terminated | time_outs

        if os.environ.get("SONOGYM_INFERENCE") and episode_done.any():
            for env_id in episode_done.nonzero(as_tuple=False).squeeze(-1).tolist():
                outcome = "SUCCESS" if success[env_id] else "TIMEOUT"
                print(f"[EPISODE END] env={env_id} | {outcome} | coverage={coverage_fraction[env_id]:.3f} | steps={self.episode_length_buf[env_id].item()}")

        # rc reset
        if hasattr(self, "rc_episode_sum") and episode_done.any():
            rc_done_vals = self.rc_episode_sum[episode_done].tolist()
            self.rc_episode_sum[episode_done] = 0.0
        else:
            rc_done_vals = []

        if hasattr(self, "total_reward") and episode_done.any():
            ep_reward_done_vals = self.total_reward[episode_done].tolist()
        else:
            ep_reward_done_vals = []

        if episode_done.any():
            done_count = int(episode_done.sum().item())
            self.run_cov_sum += float(coverage_fraction[episode_done].sum().item())
            self.run_term_sum += float(success[episode_done].float().sum().item())
            self.run_ep_reward_sum += float(self.total_reward[episode_done].sum().item())
            self.run_done_count += done_count
            self.total_reward[episode_done] = 0.0

        if wandb.run is not None and episode_done.any():
            if not hasattr(self, "_ep_buf"):
                self._ep_buf = {"cov": [], "term": [], "rc": [], "ep_reward": []}

            self._ep_buf["cov"].extend(coverage_fraction[episode_done].tolist())
            self._ep_buf["term"].extend(success[episode_done].float().tolist())
            self._ep_buf["rc"].extend(rc_done_vals)
            self._ep_buf["ep_reward"].extend(ep_reward_done_vals)

            if len(self._ep_buf["cov"]) >= 8:
                cov  = self._ep_buf["cov"]
                term = self._ep_buf["term"]
                rc   = self._ep_buf["rc"]
                ep_reward = self._ep_buf["ep_reward"]
                log_dict = getattr(self, "_wandb_cache", {}).copy()
                log_dict["episode_volume_fraction_mean"] = sum(cov) / len(cov)
                log_dict["episode_volume_fraction_max"]  = max(cov)
                log_dict["episode_terminated_frac"]      = sum(term) / len(term)
                if rc:
                    log_dict["rc_episode_sum_mean"] = sum(rc) / len(rc)
                    log_dict["rc_episode_sum_max"]  = max(rc)
                if ep_reward:
                    log_dict["episode_reward_mean"] = sum(ep_reward) / len(ep_reward)
                    log_dict["episode_reward_max"]  = max(ep_reward)
                if hasattr(self, "target_total_voxels_list"):
                    log_dict["target_total_voxels"] = self.target_total_voxels_list[0]
                wandb.log(log_dict)
                self._ep_buf["cov"].clear()
                self._ep_buf["term"].clear()
                self._ep_buf["rc"].clear()
                self._ep_buf["ep_reward"].clear()

        return terminated, time_outs

    def get_run_metric_summary(self) -> dict[str, float]:
        """Return exact run-wide episode means accumulated over all completed episodes."""
        if self.run_done_count <= 0:
            return {}
        return {
            "run_episode_volume_fraction_mean": self.run_cov_sum / self.run_done_count,
            "run_episode_terminated_mean": self.run_term_sum / self.run_done_count,
            "run_episode_reward_mean": self.run_ep_reward_sum / self.run_done_count,
            "run_completed_episodes": float(self.run_done_count),
        }
 
    def _move_towards_target(
        self,
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 200,
    ):
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(num_steps):
            self._sim_step_counter += 1
            # set actions into buffers

            # get human frame
            self.human_world_poses = self.human.data.body_link_state_w[
                :, 0, 0:7
            ]  # these are already the initial poses
            # define world to human poses
            self.world_to_human_pos, self.world_to_human_rot = (
                self.human_world_poses[:, 0:3],
                self.human_world_poses[:, 3:7],
            )
            world_ee_target_pos, world_ee_target_quat = combine_frame_transforms(
                self.world_to_human_pos,
                self.world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # get current joint positions
            self.US_ee_pose_w = self.robot.data.body_state_w[
                :, self.robot_entity_cfg.body_ids[-1], 0:7
            ]

            # get current ee
            US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
                self.world_to_base_pose[:, 0:3],
                self.world_to_base_pose[:, 3:7],
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                self.world_to_base_pose[:, 0:3],
                self.world_to_base_pose[:, 3:7],
                world_ee_target_pos,
                world_ee_target_quat,
            )
            base_to_ee_target_pose = torch.cat(
                [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
            )

            # set new command
            self.pose_diff_ik_controller.set_command(base_to_ee_target_pose)

            # get joint position targets
            US_jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
            ]
            US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            # compute the joint commands
            joint_pos_des = self.pose_diff_ik_controller.compute(
                US_ee_pos_b, US_ee_quat_b, US_jacobian, US_joint_pos
            )
            self.robot.set_joint_position_target(
                joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
            )

            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.frame_buffer[env_ids] = 0.0
        self.pose_buffer[env_ids] = 0.0

        if hasattr(self, "scanned_target_mask"):
            for i in range(len(self.scanned_target_mask)):
                self.scanned_target_mask[i][env_ids] = False

        if hasattr(self, "rc_episode_sum"):
            self.rc_episode_sum[env_ids] = 0.0

        if hasattr(self, "reached_95"):
            self.reached_95[env_ids] = False

        if hasattr(self, "episode_dist_sum"):
            self.episode_dist_sum[env_ids]   = 0.0
            self.episode_rs_sum[env_ids]     = 0.0
            self.episode_step_count[env_ids] = 0.0

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()

        self.pose_diff_ik_controller.reset()

        # get ee pose in base frame
        self.US_root_pose_w = self.robot.data.root_state_w[:, 0:7]

        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]
        # compute frame in root frame
        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            self.US_root_pose_w[:, 0:3],
            self.US_root_pose_w[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )

        ik_commands_pose = torch.zeros(
            self.scene.num_envs,
            self.pose_diff_ik_controller.action_dim,
            device=self.sim.device,
        )
        self.pose_diff_ik_controller.set_command(
            ik_commands_pose, self.US_ee_pos_b, self.US_ee_quat_b
        )

        # inverse kinematics?
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        # get human frame  depth: 0.16    
        self.human_world_poses = self.human.data.body_link_state_w[
            :, 0, 0:7
        ]  # these are already the initial poses
        # define world to human poses
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]

        # compute 2d target poses
        cmd_target_poses = torch.rand((self.scene.num_envs, 3), device=self.sim.device)
        min_init = self.init_cmd_pose_min
        max_init = self.init_cmd_pose_max
        cmd_target_poses = cmd_target_poses * (max_init - min_init) + min_init
        cmd_target_poses = self._lock_cmd_to_plane_angle(cmd_target_poses)
        # compute 3d target poses
        self.US_slicer.update_cmd(
            cmd_target_poses - self.US_slicer.current_x_z_x_angle_cmd
        )
        roll_init = float(scene_cfg["motion_planning"].get("roll_init", 0.0))
        self.US_slicer.roll_adj[env_ids] = roll_init

        self.US_slicer.current_x_z_x_angle_cmd = self._lock_cmd_to_plane_angle(
            self.US_slicer.current_x_z_x_angle_cmd
        )
        world_to_ee_init_pos, world_to_ee_init_rot = (
            self.US_slicer.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos, self.world_to_human_rot
            )
        )
        # compute joint positions
        # set joint positions
        self._move_towards_target(
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        # init distance to goal
        if self.use_vertebra_goal:
            self.get_US_target_pose()

        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        self.cur_cmd_pose = self.gt_motion_generator.human_cmd_state_from_ee_pose(
            cur_human_ee_pos, cur_human_ee_quat
        )
        self.distance_to_goal = (
            torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1)
            * self.w_pos
        )
        self.distance_to_goal += torch.norm(
            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1
        )

        self.total_reward = torch.zeros(self.scene.num_envs, device=self.sim.device)

        # record infor
        self.extras["human_to_ee_pos"] = cur_human_ee_pos
        self.extras["human_to_ee_quat"] = cur_human_ee_quat
        self.extras["cur_cmd_pose"] = self.cur_cmd_pose
        self.extras["goal_cmd_pose"] = self.goal_cmd_pose

        # record trajectory
        # tensor: (N, T, 3)
        if scene_cfg["if_record_traj"]:
            record_path = PACKAGE_DIR + scene_cfg["record_path"]
            if hasattr(self, "cmd_pose_trajs"):
                if not os.path.exists(record_path):
                    os.makedirs(record_path)
                self.cmd_pose_trajs = torch.stack(self.cmd_pose_trajs, dim=1)
                torch.save(self.cmd_pose_trajs, record_path + "cmd_pose_trajs.pt")
                torch.save(self.goal_cmd_pose, record_path + "goal_cmd_pose.pt")
            self.cmd_pose_trajs = [self.cur_cmd_pose]
