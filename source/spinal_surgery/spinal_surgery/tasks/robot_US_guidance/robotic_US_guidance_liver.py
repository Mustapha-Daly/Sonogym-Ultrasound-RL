# adapted env from vertebra to liver navigation
# /isaaacsim
# cd ~/IsaacLab
# PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/teleoperation/teleop_se3_agent.py --enable_cameras --task Isaac-robot-US-guidance-v0 --num_envs 1
#CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/skrl/train.py --task Isaac-robot-US-guidance-v0 --num_envs 4 --headless --enable_cameras

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

import matplotlib.pyplot as plt

scene_cfg = YAML().load(
    open(f"{PACKAGE_DIR}/tasks/robot_US_guidance/cfgs/robotic_US_guidance.yaml", "r")
)
scene_cfg["per_patient"] = YAML().load(
    open(f"{PACKAGE_DIR}/tasks/robot_US_guidance/cfgs/patient_profiles.yaml", "r")
)
selected_patient_id = os.environ.get("SONOGYM_PATIENT_ID", "").strip()
if selected_patient_id:
    scene_cfg["patient"]["id_list"] = [selected_patient_id]

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
if not patient_cfg.get("id_list"):
    raise ValueError("patient.id_list must contain at least one patient id")
primary_patient_id = patient_cfg["id_list"][0]


def get_patient_param(patient_id, key, default=None):
    """Per-patient config lookup: scene_cfg['per_patient'][patient_id][key], else `default`.
    Lets each patient carry its own pos / liver ranges / center_voxel, so switching
    id_list needs no other edits (omitted fields fall back to the globals)."""
    return scene_cfg.get("per_patient", {}).get(patient_id, {}).get(key, default)


# DO NOT set globally for single patient — per-env wiring in __init__ handles this for all patients
# scene_cfg["sim"]["patient_xz_range"] = get_patient_param(...)
# scene_cfg["sim"]["patient_xz_init_range"] = get_patient_param(...)
_patient_euler = get_patient_param(
    primary_patient_id, "euler_yxz", patient_cfg.get("euler_yxz", [-90.0, 90.0, 0.0])
)
quat = R.from_euler("yxz", _patient_euler, degrees=True).as_quat()
# per-patient body placement (primary patient in id_list); falls back to global patient.pos
_patient_pos = get_patient_param(primary_patient_id, "pos", patient_cfg.get("pos", [0.35, 0.15, 0.6]))
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(
        float(_patient_pos[0]),
        float(_patient_pos[1]),
        float(_patient_pos[2]),
    ),
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
        num_envs=10, env_spacing=4.0, replicate_physics=False   # 10 patients, 1 env per patient (s0030, s0004, ..., s0038)
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
        # PER-ENV init ranges — one per patient
        n_envs = self.scene.num_envs
        n_patients = len(patient_cfg["id_list"])
        self.init_cmd_pose_min = torch.zeros((n_envs, 3), device=self.sim.device)
        self.init_cmd_pose_max = torch.zeros((n_envs, 3), device=self.sim.device)

        for i, patient_id in enumerate(patient_cfg["id_list"]):
            init_range = get_patient_param(patient_id, "patient_xz_init_range", self.sim_cfg["patient_xz_init_range"])
            min_val = torch.tensor(init_range[0], device=self.sim.device)
            max_val = torch.tensor(init_range[1], device=self.sim.device)
            # Assign to all envs using this patient: env_idx % n_patients == i
            for env_idx in range(n_envs):
                if env_idx % n_patients == i:
                    self.init_cmd_pose_min[env_idx] = min_val
                    self.init_cmd_pose_max[env_idx] = max_val
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
        # PER-ENV physical x/z/angle clamp — USSlicer.update_cmd() clamps every probe
        # command to a single global self.x_z_range (was built from patient_xz_range,
        # i.e. patient #0/s0030's box only). With 10 different patients that silently
        # capped the probe into s0030's box on every other env, making most of those
        # patients' liver/target areas physically unreachable.
        # Patch update_cmd itself (not self.x_z_range) with a per-env [min,max] clamp:
        # x_z_range is also read by construct_T_maps/construct_Vl_maps (US_slicer.py),
        # which assume its old (2,3) shape — dormant today (only runs when
        # observation.mode=="US"; current cfg is "seg") but reshaping x_z_range in
        # place would silently break them the day that mode is turned on. Overriding
        # only this instance's update_cmd leaves x_z_range itself untouched.
        n_patients = len(patient_cfg["id_list"])
        _xz_min = torch.zeros((self.scene.num_envs, 3), device=self.sim.device)
        _xz_max = torch.zeros((self.scene.num_envs, 3), device=self.sim.device)
        for i, patient_id in enumerate(patient_cfg["id_list"]):
            xz_range = get_patient_param(patient_id, "patient_xz_range", self.sim_cfg["patient_xz_range"])
            min_val = torch.tensor(xz_range[0], device=self.sim.device)
            max_val = torch.tensor(xz_range[1], device=self.sim.device)
            for env_idx in range(self.scene.num_envs):
                if env_idx % n_patients == i:
                    _xz_min[env_idx] = min_val
                    _xz_max[env_idx] = max_val

        def _update_cmd_per_env(d_x_z_x_angle, _slicer=self.US_slicer, _min=_xz_min, _max=_xz_max):
            _slicer.current_x_z_x_angle_cmd += d_x_z_x_angle
            _slicer.current_x_z_x_angle_cmd = torch.clamp(_slicer.current_x_z_x_angle_cmd, _min, _max)

        self.US_slicer.update_cmd = _update_cmd_per_env
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
        self.time_penalty  = scene_cfg["reward"].get("time_penalty", 0.0)
        self.alpha_vis     = scene_cfg["reward"].get("alpha_vis", 0.0)
        self.liver_frac_thresh = scene_cfg["reward"].get("liver_frac_thresh", 0.0)  # min liver fraction in view for gate
        self.liver_penalty_k = scene_cfg["reward"].get("liver_penalty_k", 0.1)  # graded liver-penalty scale
        self.w_explore = scene_cfg["reward"].get("w_explore", 0.0)  # visitation bonus per NEW (x,z) cell/episode → drives deliberate sweep
        self.explore_cell = scene_cfg["reward"].get("explore_cell_size", 5.0)  # voxels per exploration grid cell
        self.explore_decay = int(scene_cfg["reward"].get("explore_decay", 50))  # steps before a visited cell is rewardable again; small=more re-sweeping, huge≈once/episode


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

        # PER-ENV normalization ranges [0, 1] — one per patient
        # env i uses patient i % n_human_types, so build tensors for all patients
        n_envs = self.scene.num_envs
        n_patients = len(patient_cfg["id_list"])

        self.pose_norm_min_per_env = torch.zeros((n_envs, 4), device=self.sim.device)
        self.pose_norm_max_per_env = torch.zeros((n_envs, 4), device=self.sim.device)

        for i, patient_id in enumerate(patient_cfg["id_list"]):
            xz_range = get_patient_param(patient_id, "patient_xz_range", self.sim_cfg["patient_xz_range"])
            pose_min = torch.tensor(
                [xz_range[0][0], xz_range[0][1], -3.14, -self.max_roll_adj],
                device=self.sim.device
            )
            pose_max = torch.tensor(
                [xz_range[1][0], xz_range[1][1],  3.14,  self.max_roll_adj],
                device=self.sim.device
            )
            # Assign to all envs that use this patient: env i uses patient i % n_patients
            for env_idx in range(n_envs):
                if env_idx % n_patients == i:
                    self.pose_norm_min_per_env[env_idx] = pose_min
                    self.pose_norm_max_per_env[env_idx] = pose_max

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

        # Clone Robot_US/Bed (identical across every env — safe and correct to
        # replicate from env_0) BEFORE spawning Human. copy_from_source=False is fine
        # here (fast path, lightweight inherit) since Robot_US/Bed really are meant to
        # be identical everywhere.
        #
        # Human is spawned AFTER clone_environments() on purpose — this used to run
        # BEFORE it, and that was the actual root cause of "every env shows the same
        # patient": Cloner.clone()'s source_prim_path is /World/envs/env_0, the WHOLE
        # env container, not just its Human child. Whichever copy_from_source value is
        # used, that call re-touches env_0's *entire* subtree onto every other env
        # afterward (Sdf.CopySpec full overwrite when True; an inherits arc — which
        # outranks references in USD composition strength — when False). Either way it
        # clobbered the per-patient MultiUsdFileCfg spawn that had already put a
        # DIFFERENT patient under each env's own Human prim. Confirmed empirically:
        # swapping which patient is id_list[0] made every env follow it. Spawning
        # Human after clone_environments() means there's no later clone call left to
        # overwrite/inherit over it — MultiUsdFileCfg's own regex-matched per-env spawn
        # (usd_file_list[index % len(usd_file_list)]) is the last and only word for
        # that specific prim.
        # copy_from_source=True: confirmed via Isaac Sim's own cloner test suite
        # (test_grid_cloner_inherit_addition vs test_grid_cloner_copy_addition) that
        # copy_from_source=False's "inherits" arc is LIVE — env_1 dynamically picks up
        # anything added to env_0 later, regardless of when, which is exactly why
        # reordering alone (Human spawned after this call) didn't fix it: env_1 kept
        # inheriting whatever showed up under env_0/Human afterward. copy_from_source=
        # True is a static snapshot at the moment this line runs — Human doesn't exist
        # on ANY env yet at this point (spawned below, after this call), so there is
        # nothing Human-related to snapshot/propagate here at all.
        self.scene.clone_environments(copy_from_source=True)

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

        # add articulation to scene
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human
        # --------------------------------------------------
        # RGB CAMERA (third-person debug camera) — only when visualizing.
        # Skipped during headless training so we can drop --enable_cameras and
        # free the RTX rendering memory (this camera is NOT used by obs/reward).
        # --------------------------------------------------
        if scene_cfg["sim"]["vis_us"]:
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
        """Normalize pose [x, z, angle, roll] using per-env patient ranges.
        pose: (N, 4), where N = num_envs
        """
        return (pose - self.pose_norm_min_per_env) / (
            self.pose_norm_max_per_env - self.pose_norm_min_per_env + 1e-6
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

    def _construct_target_valid_xyz(self, patient_id, human_stl_dir, label_map, kernel, n_sphere, r_int,
                                     skin_y_array, max_visible_depth_voxels, radius_voxels):
        """Precompute every (cx, cy, cz) — x, z, AND depth, all three free — where a target
        sphere

        (a) fits the liver with >=99% coverage — same tolerance _randomize_target used
        to check one random guess at a time, NOT a strict 100%-pure erosion (verified offline
        against this project's own logged targets: a strict version rejects spots the old
        200-attempt loop already found and used successfully — livers have vessels),
        (b) sits within the probe's actual visible imaging depth from the skin AT THAT (x,z),
        with a radius_voxels margin subtracted so the WHOLE sphere stays within the nominal
        depth budget, not just its center — a center placed right at max_visible_depth_voxels
        would have its deepest surface point radius_voxels further down, past the nominal
        imaging depth. This is now the ONLY depth restriction — full liver depth is allowed.
        (Was previously also capped to the shallow half [mean_y, skin]; removed per
        supervisor's actual spec — depth should vary across the whole reachable liver, capped
        only by probe visibility, not artificially confined to the shallow half.)
        """

   
        # extra margin beyond the physically-derived (100 nominal - radius) figure: this
        # check only reasons about straight-line distance from skin along Y, but the real
        # probe can be tilted/rolled, so the actual sampled depth at extreme poses can
        # differ from this straight-line estimate. Not correcting a direction bug — the
        # skin/deep axis convention itself is verified correct (see construct_highest_y_array
        # and the visible-mask discussion) — this is purely a precautionary buffer against
        # the pose-tilt approximation.
        pose_tilt_margin_voxels = scene_cfg.get("target_volume", {}).get("pose_tilt_margin_voxels", 0)
        effective_max_depth_voxels = max_visible_depth_voxels - radius_voxels - pose_tilt_margin_voxels
        cache_path = f"{human_stl_dir}/target_erosion_xyz_r{r_int}_depth{int(round(effective_max_depth_voxels))}.pt"
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location=self.sim.device)

        from scipy.signal import fftconvolve 
        liver_np = (label_map.detach().cpu().numpy() == self.cfg.LIVER_LABEL_ID).astype(np.float32)
        kernel_np = kernel.detach().cpu().numpy().astype(np.float32)
        liver_fraction = fftconvolve(liver_np, kernel_np, mode="same") / n_sphere
        fits_liver = liver_fraction >= 0.99                                # (X, Y, Z) bool, full volume

        X, Y, Z = label_map.shape
        y_idx = np.arange(Y).reshape(1, Y, 1)

        skin_y_np = skin_y_array.detach().cpu().numpy()                    # (X, Z)
        visible = (skin_y_np[:, None, :] - y_idx) <= effective_max_depth_voxels   # (X, Y, Z) bool

        valid_xyz_np = fits_liver & visible
        coords = torch.from_numpy(np.argwhere(valid_xyz_np)).to(device=self.sim.device, dtype=torch.long)  # (N,3)

        y_span = f"y in [{coords[:,1].min().item()},{coords[:,1].max().item()}]" if coords.shape[0] > 0 else "empty"
        if coords.shape[0] > 0:
            # depth-from-skin (not raw y) for every valid center, for curriculum-cutoff
            # decisions: how much of this patient's valid target pool survives at
            # progressively shallower depth caps.
            coords_np = coords.cpu().numpy()
            depth_np = skin_y_np[coords_np[:, 0], coords_np[:, 2]] - coords_np[:, 1]
            q25, q50, q75 = np.percentile(depth_np, [25, 50, 75])
            counts = ", ".join(
                f"<={c}vox:{(depth_np <= c).mean()*100:.0f}%" for c in (20, 30, 40, 50, 60, 75, 90)
            )
            print(f"[EROSION-DEPTH] {patient_id}: depth-from-skin quartiles "
                  f"25%={q25:.0f} 50%={q50:.0f} 75%={q75:.0f} vox | fraction of pool retained at cutoff: {counts}")
        print(f"[EROSION] {patient_id}: {coords.shape[0]} valid (x,y,z) target centers "
              f"(fit the liver >=99% + WHOLE sphere within {effective_max_depth_voxels:.1f} vox of skin "
              f"[{max_visible_depth_voxels:.1f} nominal - {radius_voxels:.1f} radius margin], full liver depth) "
              f"— depth range: {y_span}")
        torch.save(coords.cpu(), cache_path)
        return coords

    def _inject_target_volume(self):
        """One-time injection of a small spherical target sub-volume into the patient's
        label map, only overwriting liver voxels. Used for the 3D coverage-based reward
        (supervisor's updated task: scan/reconstruct a specific volume inside the liver,
        analogous to a tumor in Bi et al. 2026)."""
        target_cfg = scene_cfg.get("target_volume", {})
        if not target_cfg.get("enabled", False):
            return

        radius_mm = float(target_cfg["radius_mm"])
        radius_voxels = radius_mm / (self.US_slicer.label_res * 1000.0)  # e.g. 15mm / 1.5mm/vox = 10 voxels (NOT 15 — 15 is the mm value)
        target_label_id = int(target_cfg["label_id"])
        r_int = int(radius_voxels) + 1

        # position-independent sphere kernel 
        
        offsets = torch.arange(-r_int, r_int + 1, device=self.sim.device)
        dist2_kernel = offsets.view(-1, 1, 1) ** 2 + offsets.view(1, -1, 1) ** 2 + offsets.view(1, 1, -1) ** 2
        kernel = dist2_kernel <= radius_voxels ** 2
        n_sphere = int(kernel.sum().item())

        # probe's actual visible imaging depth, converted from the US image's physical
        # size into label-map voxels: image_size[1] px * resolution m/px = physical depth
        # the rendered image spans; / label_res = that same depth in label-map voxels.
        us_cfg_depth = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
        max_visible_depth_voxels = (us_cfg_depth["image_size"][1] * us_cfg_depth["resolution"]) / self.US_slicer.label_res

        # per-human-type bookkeeping needed for 3D coverage tracking
        self.target_bbox_list = []
        self.target_mask_local_list = []  # FULL sphere mask — coverage/success tracking AND restore-to-liver
        self.target_total_voxels_list = []
        self.target_valid_xyz_list = []  # NEW: per-patient (N,3) erosion-+-depth-valid (cx,cy,cz) coords

        for i in range(self.US_slicer.n_human_types):
            # per-patient target center — only used as the ONE-TIME seed injection below,
            # before the first _randomize_target call; the actual random picks used for
            # every episode after that come entirely from target_valid_xyz_list.
            patient_id = patient_cfg["id_list"][i]
            cx, cy, cz = [int(v) for v in get_patient_param(patient_id, "center_voxel", target_cfg["center_voxel"])]
            label_map = self.US_slicer.label_maps[i]
            X, Y, Z = label_map.shape
            cy = min(max(cy, r_int), Y - r_int - 1)

            valid_xyz = self._construct_target_valid_xyz(
                patient_id, human_stl_list[i], label_map, kernel, n_sphere, r_int,
                self.US_slicer.surface_map_list[i], max_visible_depth_voxels, radius_voxels,
            )
            self.target_valid_xyz_list.append(valid_xyz)

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
            write_mask = sphere_mask & liver_mask  # full sphere — injected/visible AND what coverage is measured on
            local_block[write_mask] = target_label_id
            label_map[x_min:x_max, y_min:y_max, z_min:z_max] = local_block

            n_voxels = int(write_mask.sum().item())
            print(f"[TARGET VOLUME] human_type={i}: injected {n_voxels} voxels of label "
                  f"{target_label_id} around center ({cx},{cy},{cz}), "
                  f"radius {radius_voxels:.1f} voxels ({radius_mm}mm)")

            # store for 3D coverage tracking: where the target sits (bbox) and the full
            # injected sphere mask — used both for coverage/success measurement AND to
            # restore the whole region back to liver when the target moves (no separate
            # "core" subset anymore — the whole sphere counts).
            self.target_bbox_list.append((x_min, x_max, y_min, y_max, z_min, z_max))
            self.target_mask_local_list.append(write_mask.clone())
            self.target_total_voxels_list.append(n_voxels)

        self._init_coverage_buffers()

    def _randomize_target(self, env_ids):
        """Pick a new random (cx, cz) within liver bounds, restore old sphere, inject new one.
        Scoped to the patient type(s) touched by env_ids only — each env's target is now
        re-randomized independently, the moment THAT env finishes (see _reset_idx). Was:
        looped over every patient type unconditionally on every call, so any single env
        finishing re-shuffled ALL 10 patients' targets and zeroed episode_length_buf for
        every env, not just the one that actually finished."""
        target_cfg = scene_cfg.get("target_volume", {})
        if not target_cfg.get("randomize", False):
            return

        radius_mm = float(target_cfg["radius_mm"])
        radius_voxels = radius_mm / (self.US_slicer.label_res * 1000.0)
        target_label_id = int(target_cfg["label_id"])
        r_int = int(radius_voxels) + 1
        # sphere_mask is position-independent: only relative offsets from center matter
        offsets = torch.arange(-r_int, r_int + 1, device=self.sim.device)
        xs_rel = offsets.view(-1, 1, 1)
        ys_rel = offsets.view(1, -1, 1)
        zs_rel = offsets.view(1, 1, -1)
        dist2_rel = xs_rel ** 2 + ys_rel ** 2 + zs_rel ** 2
        sphere_mask = dist2_rel <= radius_voxels ** 2
        n_sphere = int(sphere_mask.sum().item())       # theoretical max (e.g. 4169)
        # (99%-liver-fit + probe-visible-depth check now lives entirely in the precomputed
        # target_valid_xyz_list — see _construct_target_valid_xyz — not checked here anymore)

        n_types = self.US_slicer.n_human_types
        env_ids_t = torch.as_tensor(env_ids, device=self.sim.device)
        affected_types = torch.unique(env_ids_t % n_types).tolist()

        for i in affected_types:
            patient_id = patient_cfg["id_list"][i]
            label_map = self.US_slicer.label_maps[i]
            X, Y, Z = label_map.shape

            # restore old sphere voxels back to liver — the full sphere that was actually
            # injected (target_mask_local_list is the full sphere again, not a core subset)
            ox0, ox1, oy0, oy1, oz0, oz1 = self.target_bbox_list[i]
            old_block = label_map[ox0:ox1, oy0:oy1, oz0:oz1]
            old_block[self.target_mask_local_list[i]] = self.cfg.LIVER_LABEL_ID
            label_map[ox0:ox1, oy0:oy1, oz0:oz1] = old_block

            # sample a GUARANTEED-valid (x, y, z) from the precomputed erosion+depth-valid list for this patient type
            valid_xyz = self.target_valid_xyz_list[i]

            # depth-curriculum diagnostic: ALL 10 envs share ONE depth cutoff per ROUND (not
            # a fresh random pick per env-call) — the same cutoff stays active for every
            # env until the round's 10-slot buffer fills, at which point _get_dones picks a
            # new one for the next round. This lets a round cleanly answer "out of 10 envs,
            # how many succeeded at depth=X" as one exact percentage, instead of the earlier
            # per-env-independent design's confusing slowly-converging running average.
            depth_buckets = target_cfg.get("depth_curriculum_voxels", [90])
            if not hasattr(self, "_current_depth_cutoff"):
                self._current_depth_cutoff = depth_buckets[int(torch.randint(0, len(depth_buckets), (1,)).item())]
            chosen_depth_cutoff = self._current_depth_cutoff
            if not hasattr(self, "target_depth_cutoff_per_env"):
                self.target_depth_cutoff_per_env = torch.zeros(self.scene.num_envs, device=self.sim.device)

            if valid_xyz.shape[0] == 0:
                # defensive fallback only — shouldn't happen for any sanely-tuned patient
                # (s0030 alone had ~37% of its liver volume valid in offline testing)
                print(f"[TARGET WARNING] type={i}: erosion mask is empty, "
                      f"falling back to the seed center_voxel — check this patient's tuning")
                seed_cx, seed_cy, seed_cz = get_patient_param(patient_id, "center_voxel", target_cfg["center_voxel"])
                cx, cy, cz = int(seed_cx), int(seed_cy), int(seed_cz)
            else:
                skin_y_this = self.US_slicer.surface_map_list[i]  # (X, Z)
                depth_from_skin = skin_y_this[valid_xyz[:, 0], valid_xyz[:, 2]] - valid_xyz[:, 1]
                within_cutoff = depth_from_skin <= chosen_depth_cutoff
                pool = valid_xyz[within_cutoff] if within_cutoff.any() else valid_xyz  # fall back to full range if this patient has nothing that shallow
                pick = pool[torch.randint(0, pool.shape[0], (1,), device=self.sim.device)][0]
                cx, cy, cz = int(pick[0].item()), int(pick[1].item()), int(pick[2].item())

            for b in range(self.scene.num_envs):
                if b % n_types == i:
                    self.target_depth_cutoff_per_env[b] = chosen_depth_cutoff

            # clamp so the fixed-size sphere_mask kernel (2*r_int+1 in
            # every dim, built once above) always ANDs against a same-shaped liver_mask
            # slice. Erosion only guarantees >=99% of the sphere is LIVER near a boundary
            # — it does NOT guarantee the full bbox stays inside the array. A sliver
            # poking past the edge can be well under 1% of the sphere (easily clears the
            # 99% threshold) while still silently truncating the slice below 2*r_int+1,
            # which is exactly what crashed here (23 vs 22).
            cx = min(max(cx, r_int), X - r_int - 1)
            cy = min(max(cy, r_int), Y - r_int - 1)
            cz = min(max(cz, r_int), Z - r_int - 1)

            x_min, x_max = cx - r_int, cx + r_int + 1
            y_min, y_max = cy - r_int, cy + r_int + 1
            z_min, z_max = cz - r_int, cz + r_int + 1

            local_block = label_map[x_min:x_max, y_min:y_max, z_min:z_max]
            liver_mask = local_block == self.cfg.LIVER_LABEL_ID
            write_mask = sphere_mask & liver_mask  # full sphere — injected/visible AND counts toward coverage
            n_voxels = int(write_mask.sum().item())

            # gated like [EPISODE END]/[PROGRESS]/[POSE] — this fires on every single
            # per-env target reset (frequent, esp. with fast patients), so it's log spam
            # during a real training run; still useful when you actually want to watch it
            if os.environ.get("SONOGYM_INFERENCE"):
                print(f"[TARGET VOLUME] type={i}: cx={cx} cy={cy} cz={cz} → {n_voxels}/{n_sphere} voxels "
                      f"(erosion-guaranteed fit)")

            # write confirmed position into label map
            local_block = label_map[x_min:x_max, y_min:y_max, z_min:z_max]
            local_block[write_mask] = target_label_id
            label_map[x_min:x_max, y_min:y_max, z_min:z_max] = local_block

            self.target_bbox_list[i] = (x_min, x_max, y_min, y_max, z_min, z_max)
            self.target_mask_local_list[i] = write_mask.clone()
            self.target_total_voxels_list[i] = n_voxels

        # update only the envs belonging to the affected patient type(s) — not every env.
        # (today num_envs == n_patients, so this is exactly the one env in env_ids; if
        # num_envs > n_patients later, every env sharing an affected type gets the new
        # target here since they physically share the same label_map voxels — but only
        # the ones actually in env_ids also get scanned-mask/episode-budget resets below,
        # so multiple envs per patient would need that reset scoped per-type too at that
        # point, mirroring what used to be global here.)
        num_envs = self.scene.num_envs
        for i in affected_types:
            for b in range(num_envs):
                if b % n_types == i:
                    self.target_total_per_env[b] = self.target_total_voxels_list[i]
                    x_min, x_max, y_min, y_max, z_min, z_max = self.target_bbox_list[i]
                    self.target_center_per_env[b, 0] = (x_min + x_max) / 2.0
                    self.target_center_per_env[b, 1] = (z_min + z_max) / 2.0
                    self.target_depth_per_env[b] = (y_min + y_max) / 2.0

        # clear scanned masks only for the affected type(s) — was clearing every type's
        # mask on every call, wiping OTHER unrelated patients' mid-episode scan progress
        if hasattr(self, "scanned_target_mask"):
            for i in affected_types:
                self.scanned_target_mask[i][:] = False
        # fresh episode budget only for the env(s) that actually got a new target (the
        # IsaacLab base _reset_idx already zeroes episode_length_buf for these same
        # env_ids too — this is a harmless redundant write, kept for clarity/safety)
        self.episode_length_buf[env_ids_t] = 0

        if hasattr(self, "_prev_coords_per_type"):
            for i in affected_types:
                self._prev_coords_per_type.pop(i, None)

        # _target_round_log is intentionally NOT reset here anymore — it's a separate,
        # purely-bookkeeping wandb accumulator that fills in one env's slot per completed
        # episode (see _get_dones) and is reset only once ALL envs have contributed a
        # sample. Resetting it here on every single-env target change used to wipe out
        # other envs' already-recorded samples before they could ever be flushed.

        if hasattr(self, "_voxel_markers"):
            self._refresh_voxel_visualizer_geometry()

    def _reset_target_round_log(self):
        """Keep only one completed-episode sample per env for the current shared target."""
        num_envs = self.scene.num_envs
        self._target_round_log = {
            "seen": torch.zeros(num_envs, dtype=torch.bool, device=self.sim.device),
            "cov": torch.zeros(num_envs, device=self.sim.device),
            "term": torch.zeros(num_envs, device=self.sim.device),
            "rc": torch.zeros(num_envs, device=self.sim.device),
            "ep_reward": torch.zeros(num_envs, device=self.sim.device),
            "depth": torch.zeros(num_envs, device=self.sim.device),  # NEW: target depth (cy) that episode used
        }

    def _reset_round_step_stats(self):
        """Running per-STEP statistics for the current round window, reset when the
        round's 10-slot buffer flushes (see _reset_target_round_log). Replaces the old
        _wandb_cache, which was overwritten every step and only ever reported whatever
        single step happened to trigger the flush — not an aggregate over the round at
        all, which is why reward_max rarely showed the terminal bonus even though it
        was firing constantly. This accumulates sum/count/max/min across EVERY step of
        EVERY env since the last flush, so the reduction at flush time is a true
        mean/max/min over the whole round, not a one-step snapshot."""
        self._round_step_stats = {
            "reward_sum": 0.0, "reward_count": 0, "reward_max": -float("inf"), "reward_min": float("inf"),
            "rv_sum": 0.0, "rv_count": 0,
            "rs_sum": 0.0, "rs_count": 0,  # alpha2 is active (nonzero) — rs feeds training, log it
            "liver_frac_sum": 0.0, "liver_frac_count": 0,
            "shadow_ok_sum": 0.0, "shadow_ok_count": 0,
            "shadow_fraction_sum": 0.0, "shadow_fraction_count": 0,
            "r_explore_sum": 0.0, "r_explore_count": 0,
            # rv averaged ONLY over steps where the target is actually visible — rv_mean
            # (above) is diluted by the many steps target isn't in view at all (mostly
            # search/sweep), so it understates the per-step reward actually competing
            # against r_explore/w_explore at the moment that matters. This is the number
            # to check before deciding whether alpha_vis needs raising.
            "rv_on_target_sum": 0.0, "rv_on_target_count": 0,
        }

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

        # Per-episode accumulators terminal reward: D and P
        self.episode_dist_sum   = torch.zeros(num_envs, device=self.sim.device)  # Σ dt/Rc
        self.episode_rs_sum     = torch.zeros(num_envs, device=self.sim.device)  # Σ (1-pt)
        self.episode_step_count = torch.zeros(num_envs, device=self.sim.device)  # T

        # Run-wide accumulators for end-of-training summary
        self.run_cov_sum = 0.0
        self.run_term_sum = 0.0
        self.run_ep_reward_sum = 0.0
        self.run_done_count = 0
        self._reset_target_round_log()
        self._reset_round_step_stats()

        # Target center (x, z) in voxel space per env — used for attenuation distance ra
        self.target_center_per_env = torch.zeros(num_envs, 2, device=self.sim.device)
        # Target depth (y) per env — NEW, informational (wandb) now that depth varies;
        # not used by any reward term, just logged so depth randomization is observable.
        self.target_depth_per_env = torch.zeros(num_envs, device=self.sim.device)
        for b in range(num_envs):
            hi = b % self.US_slicer.n_human_types
            x_min, x_max, y_min, y_max, z_min, z_max = self.target_bbox_list[hi]
            self.target_center_per_env[b, 0] = (x_min + x_max) / 2.0  # cx
            self.target_center_per_env[b, 1] = (z_min + z_max) / 2.0  # cz
            self.target_depth_per_env[b] = (y_min + y_max) / 2.0      # cy
     
    # Added: 3D coverage visualizer for Isaac Sim viewport
    def _init_voxel_visualizer(self):
        """Create VisualizationMarkers for live 3D coverage display in Isaac Sim viewport.
        Red spheres = unscanned target voxels, green spheres = scanned. Env 0 only."""
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/CoverageVoxels",
            markers={
                "unscanned": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                ),
                "scanned": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                ),
            },
        )
        self._voxel_markers = VisualizationMarkers(cfg)
        self._refresh_voxel_visualizer_geometry()

    def _refresh_voxel_visualizer_geometry(self):
        """Refresh cached target voxel geometry for the current target."""
        x_min, x_max, y_min, y_max, z_min, z_max = self.target_bbox_list[0]
        mask = self.target_mask_local_list[0]  # (dx, dy, dz) bool
        lx, ly, lz = torch.where(mask)        # (N,) local indices inside bbox
        gx = (lx + x_min).float()
        gy = (ly + y_min).float()
        gz = (lz + z_min).float()
        self._target_voxel_patient_pos = torch.stack([gx, gy, gz], dim=1) * label_res  # (N,3) meters
        self._viz_lx, self._viz_ly, self._viz_lz = lx, ly, lz
        print(f"[VIZ] Coverage voxel markers initialized: {self._target_voxel_patient_pos.shape[0]} voxels at prim /World/Visuals/CoverageVoxels")

    def _update_voxel_viz(self):
        """Update 3D marker positions/colours each visualisation step."""
        if not hasattr(self, "_target_voxel_patient_pos"):
            self._init_voxel_visualizer()
        R_mat = matrix_from_quat(self.world_to_human_rot[0:1])[0]        # (3,3)
        pts = self._target_voxel_patient_pos.to(self.sim.device)          # (N,3)
        world_pos = (R_mat @ pts.T).T + self.world_to_human_pos[0]       # (N,3)

        scanned = self.scanned_target_mask[0][0]                          # (dx,dy,dz) for env 0
        scanned_flat = scanned[self._viz_lx, self._viz_ly, self._viz_lz] # (N,) bool
        proto_indices = scanned_flat.long()                               # 0=red, 1=green

        self._voxel_markers.visualize(translations=world_pos, marker_indices=proto_indices)

    def _update_coverage_plot(self):
        """Matplotlib 3D scatter showing red/green target voxels live """

        if not hasattr(self, "_viz_lx"):
            self._init_voxel_visualizer()

        if not hasattr(self, "_cov_fig"):
            self._cov_fig = plt.figure("3D Coverage")
            self._cov_ax = self._cov_fig.add_subplot(111, projection="3d")

        ax = self._cov_ax
        ax.cla()

        lx = self._viz_lx.cpu().numpy()
        ly = self._viz_ly.cpu().numpy()
        lz = self._viz_lz.cpu().numpy()
        
        #check which voxels have been scanned in env 0, type 0 (boolean output)
        scanned = self.scanned_target_mask[0][0][self._viz_lx, self._viz_ly, self._viz_lz].cpu().numpy()

        # X,Z = probe position axes; Y = depth axis — matches the patient orientation
        if (~scanned).any():
            ax.scatter(lx[~scanned], lz[~scanned], ly[~scanned], c="red",  s=1, alpha=0.3)
        if scanned.any():
            ax.scatter(lx[scanned],  lz[scanned],  ly[scanned],  c="lime", s=2, alpha=0.9)

        n_s = int(scanned.sum())
        n_t = len(scanned)
        pct = (n_s / n_t * 100.0) if n_t > 0 else 0.0   # guard: n_t=0 when target not placed (wrong per-patient ranges)
        ax.set_title(f"Coverage  {n_s}/{n_t} = {pct:.1f}%")
        ax.set_xlabel("X vox")
        ax.set_ylabel("Z vox")
        ax.set_zlabel("Y (depth)")
        self._cov_fig.canvas.draw_idle()
        plt.pause(0.001)

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
        # target voxels visible in the CURRENT slice (scanned or not) — same voxel
        # units as coverage; used for the visibility reward rv
        self.visible_target_count = torch.zeros(num_envs, device=self.sim.device)

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

            curr_coords = self.US_slicer.last_sampled_coords_per_type[i]  #  the exact voxels that generated the 2d image(B_i, W, H, E, 3)

            #  Added interpolation: sample 4 intermediate poses between previous and current
            N_INTERP = 4
            if hasattr(self, "_prev_coords_per_type") and i in self._prev_coords_per_type:
                prev_c = self._prev_coords_per_type[i]
                coords_list = (
                    [prev_c + (k / (N_INTERP + 1)) * (curr_coords - prev_c) for k in range(1, N_INTERP + 1)]
                    + [curr_coords]
                )
            else:
                coords_list = [curr_coords]

            for c_idx, coords in enumerate(coords_list):
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

                    # visibility: count target voxels seen in the CURRENT (last) frame
                    # only — this is the current slice, not the swept interp path
                    if c_idx == len(coords_list) - 1:
                        self.visible_target_count[env_id] = float(unique_coords.shape[0])

                    # only voxels not yet seen in this episode are counted
                    not_yet = ~self.scanned_target_mask[i][env_id, lx_u, ly_u, lz_u]
                    self.new_coverage_count[env_id] += not_yet.sum()

                    # permanently mark as scanned for the rest of this episode
                    self.scanned_target_mask[i][env_id, lx_u, ly_u, lz_u] = True

            if not hasattr(self, "_prev_coords_per_type"):
                self._prev_coords_per_type = {}
            self._prev_coords_per_type[i] = curr_coords.detach()

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
        if self.sim_cfg["vis_us"]:
            self.US_slicer.visualize("LABEL_RGB")


        # -------------------------------------------------
        # Debug (optional)
        # -------------------------------------------------
        if self.sim.has_gui():

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
                    #if not os.environ.get("SONOGYM_INFERENCE"):
                        #print(f"[TARGET PICK] voxel coords at frac={TARGET_PICK_FRAC} "
                              #f"(pixel h={h_idx},w={w_idx} of {H_img}x{W_img}): {voxel_at_pick.tolist()}")
                if not hasattr(self, "_viz_step"):
                    self._viz_step = 0
                self._viz_step += 1

                # Isaac viewport coverage spheres: update EVERY step so they track
                # the probe (cheap — just marker translations/colours, no mpl redraw)
                if hasattr(self, "target_bbox_list"):
                    self._update_voxel_viz()

                VIZ_EVERY = 3 # matplotlib in lock-step with the probe (slower). Try 2-3 if too slow.
                if self._viz_step % VIZ_EVERY == 0:
                    # pick env 0 for plotting
                    label2d = label_hw[0]  # (H,W)
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
                    ct_wh = self.US_slicer.ct_img_tensor[..., 0]
                    ct_hw = self._orient_convex_hw(ct_wh)[0].detach().cpu().numpy()
                    ct_hw = (ct_hw - ct_hw.min()) / (ct_hw.max() - ct_hw.min() + 1e-6)

                    plt.figure("Convex Semantic Label")
                    plt.clf()
                    plt.imshow(ct_hw, cmap="gray")
                    plt.imshow(rgb, alpha=1.0, interpolation="nearest")
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
                    plt.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
                    plt.pause(0.001)
                    if hasattr(self, "target_bbox_list"):
                        # _update_voxel_viz() now runs every step above (outside this gate)
                        self._update_coverage_plot()

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
            max_x = self.US_slicer.surface_normal_list[0].shape[0] - 1
            max_z = self.US_slicer.surface_normal_list[0].shape[1] - 1
            x_cmd = x_cmd.clamp(0, max_x)
            z_cmd = z_cmd.clamp(0, max_z)
            normal_human = self.US_slicer.surface_normal_list[0][x_cmd, z_cmd]

            
            R = matrix_from_quat(self.world_to_human_rot[0:1])[0]   # (3,3)
            normal_world = (R @ normal_human).cpu().numpy()
            robot_pos    = self.robot.data.root_state_w[0, 0:3].cpu().numpy()
            patient_pos  = self.world_to_human_pos[0].cpu().numpy()
            vec_robot_to_patient = patient_pos - robot_pos
            x_cmd = self.US_slicer.current_x_z_x_angle_cmd[0, 0].int()
            z_cmd = self.US_slicer.current_x_z_x_angle_cmd[0, 1].int()
            x_cmd = x_cmd.clamp(0, max_x)
            z_cmd = z_cmd.clamp(0, max_z)
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

        # Reward 4: target visibility rv = fraction of target VOXELS currently imaged.
        # how many of the target voxels are visible in the current slice (computed in update coverage)
        # Difference from rc: rc says covers whole sphere and rv says keep it in view while you do
        if hasattr(self, "visible_target_count") and hasattr(self, "target_total_per_env"):
            rv = self.visible_target_count / self.target_total_per_env.clamp(min=1.0)
        else:
            rv = torch.zeros(B, device=self.sim.device)

        # Reward 5: exploration (visitation) reward: bonus for entering a NEW (x,z) cell 
        if self.w_explore > 0.0:
            probe_xz = self.US_slicer.current_x_z_x_angle_cmd[:, :2]  # (B, 2) voxel coords
            if not hasattr(self, "_visited_step"):
                # PER-ENV grid origin (each env's own patient_xz_range) — was a single
                # global origin/size from patient_xz_range (s0030's box), which aliased
                # every other patient's wider-ranging probe positions onto s0030-sized
                # grid cells (clamped), collapsing the re-sweep exploration bonus for them.
                # Grid is sized to the widest patient span so every patient's full range
                # of cells fits; narrower patients simply use a subset of the grid.
                n_patients = len(patient_cfg["id_list"])
                self._ex_x0 = torch.zeros(B, device=self.sim.device)
                self._ex_z0 = torch.zeros(B, device=self.sim.device)
                max_span_x = max_span_z = 0.0
                for i, patient_id in enumerate(patient_cfg["id_list"]):
                    xzr = get_patient_param(patient_id, "patient_xz_range", self.sim_cfg["patient_xz_range"])
                    x0, z0 = xzr[0][0], xzr[0][1]
                    span_x, span_z = xzr[1][0] - xzr[0][0], xzr[1][1] - xzr[0][1]
                    max_span_x, max_span_z = max(max_span_x, span_x), max(max_span_z, span_z)
                    for env_idx in range(B):
                        if env_idx % n_patients == i:
                            self._ex_x0[env_idx] = x0
                            self._ex_z0[env_idx] = z0
                self._ex_nx = int(max_span_x / self.explore_cell) + 2
                self._ex_nz = int(max_span_z / self.explore_cell) + 2
                # episode-step each cell was last visited; -large = never visited → rewardable
                self._visited_step = torch.full((B, self._ex_nx, self._ex_nz), -(10**9), dtype=torch.long, device=self.sim.device)
            cx = ((probe_xz[:, 0] - self._ex_x0) / self.explore_cell).long().clamp(0, self._ex_nx - 1)
            cz = ((probe_xz[:, 1] - self._ex_z0) / self.explore_cell).long().clamp(0, self._ex_nz - 1)
            eidx = torch.arange(B, device=self.sim.device)
            now = self.episode_length_buf                       # (B,) step within each env's episode
            age = now - self._visited_step[eidx, cx, cz]        # steps since this cell was last visited
            new_cell = age >= self.explore_decay                # rewardable again once it has "decayed"
            self._visited_step[eidx, cx, cz] = now              # stamp current visit
            r_explore = self.w_explore * new_cell.float()  # decaying visitation → forces continual re-sweeping of the whole liver
        else:
            r_explore = torch.zeros(B, device=self.sim.device)

        # while the target is in view, switch OFF exploration → dwell and scan instead of
        # being pulled away to sweep fresh cells (rc still drives the on-target scan motion)
        if hasattr(self, "visible_target_count"):
            r_explore = torch.where(
                self.visible_target_count > 0,
                torch.zeros_like(r_explore),
                r_explore,
            )

        #  combined per-step reward
        rt = self.w_coverage * rc + self.alpha1 * ra + self.alpha2 * rs + self.alpha_vis * rv + r_explore

        # shadow penalty + liver penatly, graded by how far below threshold 
        shadow_ok = shadow_fraction < self.shadow_thresh  # (B,) bool
        liver_frac = (label == self.cfg.LIVER_LABEL_ID).float().mean(dim=(1, 2))  # (B,)
        liver_deficit = (self.liver_frac_thresh - liver_frac).clamp(min=0.0)
        liver_penalty = self.liver_penalty_k * (liver_deficit / max(self.liver_frac_thresh, 1e-6))  # 0 in-liver → k fully outside
        shadow_penalty = torch.where(
            shadow_ok,
            torch.zeros(B, device=self.sim.device),
            torch.full((B,), 0.1, device=self.sim.device),
        )
        reward = rt - liver_penalty - self.time_penalty

        # per-step living cost: each extra step lowers the return, encourages finishing
        # fast. Was dead code (self.time_penalty loaded from config but never subtracted
        # anywhere) — restored.

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
            # add terminal bonus only on first crossing 0.85 (once per episode)
            just_crossed = (cov_frac >= 0.85) & ~self.reached_95
            self.reached_95 = self.reached_95 | (cov_frac >= 0.85)
            reward = reward + torch.where(
                just_crossed,
                r_end,
                torch.zeros(B, device=self.sim.device),
            )

        # accumulate rc across the episode (reset in _reset_idx)
        if hasattr(self, "rc_episode_sum"):
            self.rc_episode_sum += rc

        # accumulate this step into the round-window stats — flushed/reduced (and reset)
        # in _get_dones() once the round's 10-slot buffer fills. See _reset_round_step_stats
        # for why this replaces the old per-step-snapshot _wandb_cache.
        if not hasattr(self, "_round_step_stats"):
            self._reset_round_step_stats()
        s = self._round_step_stats
        s["reward_sum"]   += float(reward.sum().item())
        s["reward_count"] += reward.numel()
        s["reward_max"]    = max(s["reward_max"], float(reward.max().item()))
        s["reward_min"]    = min(s["reward_min"], float(reward.min().item()))
        s["rv_sum"]        += float(rv.sum().item())
        s["rv_count"]      += rv.numel()
        s["rs_sum"]        += float(rs.sum().item())
        s["rs_count"]      += rs.numel()
        s["r_explore_sum"]   += float(r_explore.sum().item())
        s["r_explore_count"] += r_explore.numel()
        if hasattr(self, "visible_target_count"):
            on_target = self.visible_target_count > 0
            if on_target.any():
                s["rv_on_target_sum"]   += float(rv[on_target].sum().item())
                s["rv_on_target_count"] += int(on_target.sum().item())
        s["liver_frac_sum"]   += float(liver_frac.sum().item())
        s["liver_frac_count"] += liver_frac.numel()
        s["shadow_ok_sum"]     += float(shadow_ok.float().sum().item())
        s["shadow_ok_count"]   += shadow_ok.numel()
        s["shadow_fraction_sum"]   += float(shadow_fraction.sum().item())
        s["shadow_fraction_count"] += shadow_fraction.numel()

        self.total_reward += reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.scene.num_envs
        n_types = self.US_slicer.n_human_types

        # vestigial: _pending_sync_reset is never set to True anymore (target
        # randomization is per-env/immediate now, see _reset_idx / _randomize_target) —
        # left in place as a harmless no-op safety net.
        if getattr(self, "_pending_sync_reset", False):
            terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.sim.device)
            time_outs  = torch.ones(num_envs,  dtype=torch.bool, device=self.sim.device)
            return terminated, time_outs

        # compute cumulative coverage fraction per env 
        if hasattr(self, "scanned_target_mask") and hasattr(self, "target_total_per_env"):
            scanned_total = torch.zeros(num_envs, device=self.sim.device)
            for i, mask in enumerate(self.scanned_target_mask):
                env_inds = torch.arange(i, num_envs, n_types, device=self.sim.device)
                scanned_total[env_inds] = mask[env_inds].sum(dim=(1, 2, 3)).float()
            coverage_fraction = (scanned_total / self.target_total_per_env.clamp(min=1.0)).clamp(max=1.0)
        else:
            coverage_fraction = torch.zeros(num_envs, device=self.sim.device)

        terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.sim.device)
        terminated |= (coverage_fraction >= 0.85)
        # success =  85% coverage  400 steps
        success = (coverage_fraction >= 0.85) & (self.episode_length_buf <= 360)

        time_outs = self.episode_length_buf >= self.max_episode_length - 1

        episode_done = terminated | time_outs

        if os.environ.get("SONOGYM_INFERENCE") and episode_done.any():
            for env_id in episode_done.nonzero(as_tuple=False).squeeze(-1).tolist():
                outcome = "SUCCESS" if success[env_id] else "TIMEOUT"
                expected_patient = patient_cfg["id_list"][env_id % n_types]
                print(f"[EPISODE END] env={env_id} (patient={expected_patient}) | {outcome} | coverage={coverage_fraction[env_id]:.3f} | steps={self.episode_length_buf[env_id].item()}")
                # proof, not theory: query the LIVE USD stage for whichever mesh is
                # actually resolved for this env's Human prim right now, and flag it if
                # it doesn't match the patient env_id is supposed to be running.
                
        # rc reset
        rc_done_tensor = torch.zeros(num_envs, device=self.sim.device)
        if hasattr(self, "rc_episode_sum") and episode_done.any():
            rc_done_tensor[episode_done] = self.rc_episode_sum[episode_done]
            self.rc_episode_sum[episode_done] = 0.0
       
        ep_reward_done_tensor = torch.zeros(num_envs, device=self.sim.device)
        if hasattr(self, "total_reward") and episode_done.any():
            ep_reward_done_tensor[episode_done] = self.total_reward[episode_done]

        if episode_done.any():
            done_count = int(episode_done.sum().item())
            self.run_cov_sum += float(coverage_fraction[episode_done].sum().item())
            self.run_term_sum += float(success[episode_done].float().sum().item())
            self.run_ep_reward_sum += float(self.total_reward[episode_done].sum().item())
            self.run_done_count += done_count
            self.total_reward[episode_done] = 0.0
        #wandb logging: only when an episode finishes
        if wandb.run is not None and episode_done.any():
            if not hasattr(self, "_target_round_log"):
                self._reset_target_round_log()
            if not hasattr(self, "_round_step_stats"):
                self._reset_round_step_stats()

            record_mask = episode_done & ~self._target_round_log["seen"] #seen means slot is locked until the whole buffer resets
            if record_mask.any():
                self._target_round_log["seen"][record_mask] = True
                self._target_round_log["cov"][record_mask] = coverage_fraction[record_mask]
                self._target_round_log["term"][record_mask] = terminated[record_mask].float()  # any-time 85% (real success rate), not just ≤400 steps
                self._target_round_log["rc"][record_mask] = rc_done_tensor[record_mask]
                self._target_round_log["ep_reward"][record_mask] = ep_reward_done_tensor[record_mask]
                self._target_round_log["depth"][record_mask] = self.target_depth_per_env[record_mask]

            if self._target_round_log["seen"].all(): # all slots have been filled
                # true round-window aggregates — accumulated across every step of every
                # env since the last flush (see _reset_round_step_stats), not a single-step
                # snapshot. ra_mean still dropped: alpha1=0, so it doesn't affect training.
                # rs_mean kept: alpha2 is active (nonzero) — re-add ra_mean here too if
                # alpha1 ever gets re-enabled.
                st = self._round_step_stats
                log_dict = {}
                if st["reward_count"] > 0:
                    log_dict["reward_mean"] = st["reward_sum"] / st["reward_count"]
                    log_dict["reward_max"]  = st["reward_max"]
                    log_dict["reward_min"]  = st["reward_min"]
                if st["rv_count"] > 0:
                    log_dict["rv_mean"] = st["rv_sum"] / st["rv_count"]
                if st["rs_count"] > 0:
                    log_dict["rs_mean"] = st["rs_sum"] / st["rs_count"]
                if st["r_explore_count"] > 0:
                    log_dict["r_explore_mean"] = st["r_explore_sum"] / st["r_explore_count"]
                if st["rv_on_target_count"] > 0:
                    log_dict["rv_mean_on_target"] = st["rv_on_target_sum"] / st["rv_on_target_count"]
                if st["liver_frac_count"] > 0:
                    log_dict["liver_frac_mean"] = st["liver_frac_sum"] / st["liver_frac_count"]
                if st["shadow_ok_count"] > 0:
                    log_dict["shadow_ok_frac"] = st["shadow_ok_sum"] / st["shadow_ok_count"]
                if st["shadow_fraction_count"] > 0:
                    log_dict["shadow_fraction"] = st["shadow_fraction_sum"] / st["shadow_fraction_count"]

                log_dict["episode_volume_fraction_mean"] = self._target_round_log["cov"].mean().item()
                log_dict["episode_volume_fraction_max"]  = self._target_round_log["cov"].max().item()
                log_dict["episode_terminated_frac"]      = self._target_round_log["term"].mean().item()
                log_dict["episode_success_count"] = int(self._target_round_log["term"].sum().item())
                log_dict["rc_episode_sum_mean"] = self._target_round_log["rc"].mean().item()
                log_dict["rc_episode_sum_max"]  = self._target_round_log["rc"].max().item()
                log_dict["episode_reward_mean"] = self._target_round_log["ep_reward"].mean().item()
                log_dict["episode_reward_max"]  = self._target_round_log["ep_reward"].max().item()
                
                for i, patient_id in enumerate(patient_cfg["id_list"]):
                    env_inds = torch.arange(i, num_envs, n_types, device=self.sim.device)
                    log_dict[f"episode_volume_fraction/{patient_id}"] = self._target_round_log["cov"][env_inds].mean().item()
        
                    log_dict[f"target_depth/{patient_id}"] = self._target_round_log["depth"][env_inds].mean().item()
                if hasattr(self, "target_total_voxels_list"):
                    # was hardcoded to target_total_voxels_list[0] (patient #0/s0030 only,
                    # stale/misleading now that each patient has its own target voxel count)
                    log_dict["target_total_voxels_mean"] = float(np.mean(self.target_total_voxels_list))
                    for i, patient_id in enumerate(patient_cfg["id_list"]):
                        log_dict[f"target_total_voxels/{patient_id}"] = self.target_total_voxels_list[i]

                # depth-curriculum diagnostic: this ROUND's exact result — all 10 envs used
                # the SAME depth cutoff (self._current_depth_cutoff, set in _randomize_target
                # and held fixed until this flush), so "term" here is literally X out of 10
                # envs that succeeded AT THAT DEPTH, this round. One clean data point per
                # round, not a slowly-converging running average.
                if hasattr(self, "_current_depth_cutoff"):
                    success_pct_this_round = self._target_round_log["term"].float().mean().item() * 100.0
                    log_dict[f"success_pct/depth_{int(self._current_depth_cutoff)}"] = success_pct_this_round
                    # pick the NEXT round's cutoff now, so every env's following resets use it
                    depth_buckets = scene_cfg.get("target_volume", {}).get("depth_curriculum_voxels", [90])
                    self._current_depth_cutoff = depth_buckets[int(torch.randint(0, len(depth_buckets), (1,)).item())]

                wandb.log(log_dict)
                self._reset_target_round_log()
                self._reset_round_step_stats()

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

        if hasattr(self, "target_bbox_list"):
            # Randomize immediately for whichever env(s) just finished, scoped to THEIR
            # OWN patient type only (_randomize_target scopes itself via env_ids).
            # Was: wait until every one of the 10 patients had individually finished at
            # least once, THEN force every env to a synchronized new round together. With
            # 10 different patients of very different difficulty/speed, a fast/easy one
            # (e.g. a fixed deterministic init pose landing right next to an easy target)
            # would finish in ~10 steps and then just replay the SAME un-randomized target
            # from the SAME start pose over and over while waiting for the slowest patient
            # to also finish once — zero learning signal for that patient meanwhile, and
            # it starves every other env of a fresh target too. The forced synchronized
            # flush (_pending_sync_reset in _get_dones) also clipped whichever envs it
            # caught mid-episode to a spurious 1-step episode — this is why episode length
            # (min) in wandb was pinned at 1.
            self._randomize_target(env_ids)

        if hasattr(self, "scanned_target_mask"):
            for i in range(len(self.scanned_target_mask)):
                self.scanned_target_mask[i][env_ids] = False

        # NOTE: _prev_coords_per_type is no longer blanket-cleared here — _randomize_target
        # (called above, now unconditional) already pops just the affected type(s). A
        # blanket .clear() here would undo that scoping and wipe OTHER active patients'
        # cached coords every time any single env resets.

        if hasattr(self, "rc_episode_sum"):
            self.rc_episode_sum[env_ids] = 0.0

        if hasattr(self, "reached_95"):
            self.reached_95[env_ids] = False

        if hasattr(self, "_visited_step"):
            self._visited_step[env_ids] = -(10**9)  # fresh sweep each episode (all cells rewardable)

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

        # PER-ENV human body placement — INIT_STATE_HUMAN (used at spawn) only ever
        # carried patient #0's (id_list[0], i.e. s0030's) pos/euler_yxz, applied to every
        # env even though each env's mesh is a DIFFERENT patient (MultiUsdFileCfg cycles
        # usd_file_list per env via index % len(usd_file_list) — confirmed against
        # IsaacLab's spawn_multi_asset). Every non-#0 patient's individually teleop-tuned
        # pos/euler_yxz was silently unused. Overwrite here (computed once, cached) with
        # each env's own patient's tuned transform. Placed here (not _setup_scene) because
        # root_physx_view isn't alive yet at spawn time; by _reset_idx the sim has already
        # stepped once and human.data reads below already depend on it being live.
        if not hasattr(self, "_human_pos_per_env"):
            n_patients = len(patient_cfg["id_list"])
            self._human_pos_per_env = torch.zeros((self.scene.num_envs, 3), device=self.sim.device)
            self._human_quat_per_env = torch.zeros((self.scene.num_envs, 4), device=self.sim.device)  # (w,x,y,z)
            for i, patient_id in enumerate(patient_cfg["id_list"]):
                pos = get_patient_param(patient_id, "pos", patient_cfg.get("pos", [0.35, 0.15, 0.6]))
                euler = get_patient_param(patient_id, "euler_yxz", patient_cfg.get("euler_yxz", [-90.0, 90.0, 0.0]))
                q = R.from_euler("yxz", euler, degrees=True).as_quat()  # scipy order [x,y,z,w]
                pos_t = torch.tensor(pos, device=self.sim.device)
                quat_t = torch.tensor([q[3], q[0], q[1], q[2]], device=self.sim.device)  # -> (w,x,y,z)
                for env_idx in range(self.scene.num_envs):
                    if env_idx % n_patients == i:
                        self._human_pos_per_env[env_idx] = pos_t
                        self._human_quat_per_env[env_idx] = quat_t
        _human_root_pose = torch.cat(
            [
                self._human_pos_per_env[env_ids] + self.scene.env_origins[env_ids],
                self._human_quat_per_env[env_ids],
            ],
            dim=-1,
        )
        self.human.write_root_pose_to_sim(_human_root_pose, env_ids=env_ids)

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
