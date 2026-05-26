# adapted env from vertebra to liver navigation
# /isaaacsim
# cd ~/IsaacLab
# PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/teleoperation/teleop_se3_agent.py --enable_cameras --task Isaac-robot-US-guidance-v0 --num_envs 1


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
    observation_space = [3, 240, 200] #[1, 150, 200]
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
        )
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

        # change observation space to image
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.cfg.observation_space[0],
                self.cfg.observation_space[1],
                self.cfg.observation_space[2],
            ),
            dtype=np.uint8,
        )

        self.cfg.observation_space[0] = self.US_slicer.img_thickness

        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.cfg.observation_space[0],
                self.cfg.observation_space[1],
                self.cfg.observation_space[2],
            ),
            dtype=np.float32,
        )

        self.termination_direct = True
        self.observation_mode = scene_cfg["observation"]["mode"]
        self.action_mode = scene_cfg["action"]["mode"]
        self.action_scale = (
            torch.tensor(scene_cfg["action"]["scale"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )

        self.w_pos = scene_cfg["reward"]["w_pos"]

        self.single_action_space = gym.spaces.Box(
            low=-(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            high=(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            shape=(self.cfg.action_space,),
            dtype=np.float32,
        )

        wandb.init()
        self.num_step = 0

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
        # ADD THE FUNCTION HERE
    def _orient_convex_hw(self, img_wh: torch.Tensor) -> torch.Tensor:
        """
        Convert slicer output (B,W,H) to ultrasound-style orientation (B,H,W)
        """
        if img_wh.dim() == 3:
            img_hw = img_wh.permute(0, 2, 1).contiguous()
        else:
            img_hw = img_wh.permute(0, 2, 1, 3).contiguous()

        # flip vertically so depth increases downward
       # img_hw = torch.flip(img_hw, dims=[1])

        return img_hw
    
    def _apply_superficial_bone_shadow(self, label_hw: torch.Tensor) -> torch.Tensor:
        B, H, W = label_hw.shape
        shadowed = label_hw.clone()
        shadow = torch.zeros(B, H, W, dtype=torch.bool, device=label_hw.device)
        top_rows = int(0.4 * H)
        rows = torch.arange(H, device=label_hw.device).view(1, H, 1).expand(B, H, W)

        for lbl in [13, 15]:
            seed = (label_hw == lbl).float()
            seed[:, top_rows:, :] = 0.0

            has_seed = seed.sum(dim=1) > 0
            first_occ = torch.argmax(seed, dim=1)
            first_occ_exp = first_occ.unsqueeze(1).expand(B, H, W)
            has_seed_exp = has_seed.unsqueeze(1).expand(B, H, W)

            col_shadow = (rows > first_occ_exp) & has_seed_exp   # no upper bound → reaches bottom
            shadow |= col_shadow

        shadow &= ~((label_hw == 13) | (label_hw == 15))
        shadowed[shadow] = 0
        return shadowed

    
  
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

            label_img = (
                self.US_slicer.label_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
            observations = {"policy": label_img}

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
        if self.num_step % 30 == 0:
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

                if self.num_step % 10 == 0:
                    u = torch.unique(label2d)
                    print("Unique IDs in CURRENT slice:", u.detach().cpu().tolist())
                    if self.num_step % 10 == 0:
                        cart_mask = (label2d == 15)
                        liver_mask = (label2d == 5)
                        if cart_mask.any():
                            cart_rows = torch.where(cart_mask)[0]
                            print(f"  Cartilage H rows: min={cart_rows.min().item()}, max={cart_rows.max().item()}")
                        if liver_mask.any():
                            liver_rows = torch.where(liver_mask)[0]
                            print(f"  Liver H rows:     min={liver_rows.min().item()}, max={liver_rows.max().item()}")
                            
                   


                    for k in u.detach().cpu().tolist():
                        cnt = (label2d == k).sum().item()
                        print(f"  id={k:>3} pixels={cnt}")

                palette = torch.zeros((256,3), dtype=torch.uint8, device=label2d.device)

                palette[0]  = torch.tensor([0,0,0], device=label2d.device)        # background
                palette[1]  = torch.tensor([255,255,0], device=label2d.device)    # spleen
                palette[2] = torch.tensor([160, 160, 255], device=label2d.device)  # Right Kidney
                palette[8]  = torch.tensor([0,255,255], device=label2d.device)    # muscle
                palette[10] = torch.tensor([255, 165, 0], device=label2d.device)  # organ tissue
                palette[12] = torch.tensor([144,238,144], device=label2d.device)      # skin / fat
                palette[13] = torch.tensor([255,0,0], device=label2d.device)      # bone
                palette[5] = torch.tensor([0,0,200], device=label2d.device)  # main organ class
                palette[64] = torch.tensor([0, 100, 255], device=label2d.device)  # Portal Vein
                palette[51] = torch.tensor([148,0,211], device=label2d.device)      # IVC
                palette[7] = torch.tensor([139, 69, 19], device=label2d.device)  # pancreas
                palette[4] = torch.tensor([255, 255, 255], device=label2d.device)  # gallbladder
                palette[15] = torch.tensor([255, 105, 180], device=label2d.device)  # costal cartilages


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
                    mpatches.Patch(color=(160/255, 160/255, 1), label="Right Kidney (ID 2)"),
                    mpatches.Patch(color=(0,1,1), label="Muscle (ID 8)"),
                    mpatches.Patch(color=(1, 165/255, 0), label="Organ tissue (ID 10)"),
                    mpatches.Patch(color=(0.56,0.93,0.56), label="Skin / Fat (ID 12)"),
                    mpatches.Patch(color=(1,0,0), label="Bone / Vertebra (ID 13)"),
                    mpatches.Patch(color=(0,0,0.6), label="Liver (ID 5)"),
                    mpatches.Patch(color=(0, 100/255, 1), label="Portal Vein (ID 64)"),
                    mpatches.Patch(color=(148/255, 0, 211/255), label="HEART (ID 51)"),
                    mpatches.Patch(color=(139/255, 69/255, 19/255), label="Pancreas (ID 7)"), 
                    mpatches.Patch(color=(1, 1, 1), label="Gallbladder (ID 4)"), 
                    mpatches.Patch(color=(1, 105/255, 180/255), label="Costal Cartilages (ID 15)"),
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
            print(f"probe x={x_cmd.item()}  z={z_cmd.item()}")   # ← add here
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
        # apply joint oosition target
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )

    def _get_rewards(self) -> torch.Tensor: #new: reward based on liver visibility and progress
        if self.observation_mode == "seg" and hasattr(self, "_label_for_task"):
            label = self._label_for_task      # convex label
        else:
            label = self.US_slicer.label_img_tensor[:, :, :, 0]
        
        liver_mask = (label == self.cfg.LIVER_LABEL_ID)
        B, H, W = liver_mask.shape
        liver_pixels = liver_mask.sum(dim=(1, 2))
        liver_fraction = liver_pixels / (H * W)
        if not hasattr(self, "prev_liver_fraction"):
            self.prev_liver_fraction = liver_fraction.clone()
        progress = liver_fraction - self.prev_liver_fraction
        progress = torch.clamp(progress, min=-0.05, max=0.05)

        reward = (2.0 * liver_fraction + 4.0 * progress)

        self.prev_liver_fraction = liver_fraction.clone()
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # episode ends when liver is sufficiently visible or timeout
        if self.observation_mode == "seg" and hasattr(self, "_label_for_task"):
            label = self._label_for_task
        else:
            label = self.US_slicer.label_img_tensor[:, :, :, 0]

        liver_mask = (label == self.cfg.LIVER_LABEL_ID)
        liver_pixels = liver_mask.sum(dim=(1, 2))
        H, W = label.shape[1], label.shape[2]
        liver_fraction = liver_pixels / (H * W)
        if self.num_step % 50 == 0:
            wandb.log({
                "liver_fraction": liver_fraction.mean().item()
            })
        terminated = (liver_fraction > 0.99)

        time_outs = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_outs

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

        if hasattr(self, "total_reward"):
            wandb.log({"total_reward": self.total_reward.mean().item()})
            wandb.log({"distance_to_goal": self.distance_to_goal.mean().item()})
            wandb.log(
                {
                    "pos err": (
                        torch.norm(
                            self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2],
                            dim=-1,
                        )
                        * self.US_slicer.label_res
                    )
                    .mean()
                    .item()
                }
            )
            wandb.log(
                {
                    "pos err std": (
                        torch.norm(
                            self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2],
                            dim=-1,
                        )
                        * self.US_slicer.label_res
                    )
                    .std()
                    .item()
                }
            )
            wandb.log(
                {
                    "pos err max": (
                        torch.norm(
                            self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2],
                            dim=-1,
                        )
                        * self.US_slicer.label_res
                    )
                    .max()
                    .item()
                }
            )
            wandb.log(
                {
                    "pos err min": (
                        torch.norm(
                            self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2],
                            dim=-1,
                        )
                        * self.US_slicer.label_res
                    )
                    .min()
                    .item()
                }
            )
            wandb.log(
                {
                    "rot err": (
                        torch.norm(
                            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3],
                            dim=-1,
                        )
                        * 180
                        / torch.pi
                    )
                    .mean()
                    .item()
                }
            )
            wandb.log(
                {
                    "rot err std": (
                        torch.norm(
                            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3],
                            dim=-1,
                        )
                        * 180
                        / torch.pi
                    )
                    .std()
                    .item()
                }
            )
            wandb.log(
                {
                    "rot err max": (
                        torch.norm(
                            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3],
                            dim=-1,
                        )
                        * 180
                        / torch.pi
                    )
                    .max()
                    .item()
                }
            )
            wandb.log(
                {
                    "rot err min": (
                        torch.norm(
                            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3],
                            dim=-1,
                        )
                        * 180
                        / torch.pi
                    )
                    .min()
                    .item()
                }
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
        # -------------------------------------------------
