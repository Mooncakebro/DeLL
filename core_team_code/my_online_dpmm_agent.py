"""
Sensor agent for closed-loop evaluation using core_team_code models.

Usage (Bench2Drive eval):
    TEAM_AGENT=core_team_code/my_online_dpmm_agent.py
    TEAM_CONFIG=/path/to/ability_dir (contains config.json + model_0030.pth + latest.pth)
"""

import os
import sys
from copy import deepcopy
from collections import deque

import cv2
import carla
import numpy as np
import math
import torch
import torch.nn.functional as F

# --- Path setup so imports resolve to core_team_code/ first ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_TEAM_CODE = _THIS_DIR  # we are in core_team_code/
_TEAM_CODE = os.path.join(_CORE_TEAM_CODE, '..', 'team_code')
for _p in [_TEAM_CODE, _CORE_TEAM_CODE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import GlobalConfig
from my_model_wTFFdeQtd import LidarCenterNet
from data import CARLA_Data
from nav_planner import RoutePlanner, extrapolate_waypoint_route

import transfuser_utils as t_u
from leaderboard.autoagents import autonomous_agent

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.optimize import fsolve

from scenario_logger import ScenarioLogger
from utils import print_data_info

import pathlib
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson
import gzip

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():
    return 'OnlineDPMMSensorAgent'


def strtobool(v):
    return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


# ---------------------------------------------------------------------------
# Ego model & filter helpers (same as team_code agent)
# ---------------------------------------------------------------------------

def bicycle_model_forward(x, dt, steer, throttle, brake):
    front_wb = -0.090769015
    rear_wb = 1.4178275
    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837
    locs_0, locs_1, yaw, speed = x[0], x[1], x[2], x[3]
    accel = brake_accel if brake else throt_accel * throttle
    wheel = steer_gain * steer
    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    return np.array([
        locs_0.item() + speed * math.cos(yaw + beta) * dt,
        locs_1.item() + speed * math.sin(yaw + beta) * dt,
        yaw + speed / rear_wb * math.sin(beta) * dt,
        (speed + accel * dt) * (speed + accel * dt > 0.0),
    ])


def measurement_function_hx(vehicle_state):
    return vehicle_state


def state_mean(state, wm):
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x


def measurement_mean(state, wm):
    return state_mean(state, wm)


def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OnlineDPMMSensorAgent(autonomous_agent.AutonomousAgent):
    """Agent that loads a model from an ability dir and runs closed-loop eval."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        print('[OnlineDPMMAgent] setup start.')
        torch.cuda.empty_cache()
        self.IS_BENCH2DRIVE = strtobool(os.environ.get('IS_BENCH2DRIVE', 'False'))
        self.track = (autonomous_agent.Track.MAP
                      if os.environ.get('CHALLENGE_TRACK_CODENAME') == 'MAP'
                      else autonomous_agent.Track.SENSORS)
        self.config_path = path_to_conf_file.split('+')[0] if self.IS_BENCH2DRIVE else path_to_conf_file
        self.step = -1
        self.initialized = False
        self.device = torch.device('cuda:0')

        # Load saved config
        with open(os.path.join(self.config_path, 'config.json'), 'rt', encoding='utf-8') as f:
            saved_config = jsonpickle.decode(f.read())
        self.config = GlobalConfig()
        self.config.__dict__.update(saved_config.__dict__)

        # Eval overrides
        self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
        self.config.inference_direct_controller = int(os.environ.get('DIRECT', 1))
        self.config.brake_uncertainty_threshold = float(
            os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
        self.compile = int(os.environ.get('COMPILE', 0))
        self.stop_after_meter = int(os.environ.get('STOP_AFTER_METER', -1))
        self.stop_sign_controller = int(os.environ.get('STOP_CONTROL', 1))
        if int(os.environ.get('SLOWER', 0)):
            self.inference_target_speeds = [self.config.slower_factor * s for s in self.config.target_speeds]
        else:
            self.inference_target_speeds = self.config.target_speeds

        # Load model(s)
        self.nets = []
        self.model_count = 0
        for fname in sorted(os.listdir(self.config_path)):
            if fname.endswith('.pth') and fname.startswith('model'):
                self.model_count += 1
                path = os.path.join(self.config_path, fname)
                print(f'[OnlineDPMMAgent] Loading {path}')
                net = LidarCenterNet(self.config)
                if self.config.sync_batch_norm:
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                state_dict = torch.load(path, map_location=self.device)
                net.load_state_dict(state_dict, strict=False)
                net.cuda(device=self.device)
                net.eval()
                if self.compile:
                    net = torch.compile(net, mode=self.config.compile_mode)
                self.nets.append(net)

        if self.model_count == 0:
            raise FileNotFoundError(f'No model_*.pth found in {self.config_path}')

        self.stop_sign_buffer = deque(maxlen=1) if self.stop_sign_controller else None
        self.clear_stop_sign = 0
        self.stuck_detector = 0
        self.force_move = 0
        self.bb_buffer = deque(maxlen=1)
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # UKF filter
        self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
        self.ukf = UKF(dim_x=4, dim_z=4, fx=bicycle_model_forward, hx=measurement_function_hx,
                        dt=self.config.carla_frame_rate, points=self.points,
                        x_mean_fn=state_mean, z_mean_fn=measurement_mean,
                        residual_x=residual_state_x,
                        residual_z=lambda a, b: a - b)
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False
        self.state_log = deque(maxlen=max(self.config.lidar_seq_len * self.config.data_save_freq, 2))
        self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)
        self.lidar_last = None
        self.meters_travelled = 0 if self.stop_after_meter > 0 else 0

        self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)
        self.save_path = os.environ.get('SAVE_PATH', None)
        if route_index is not None:
            route_index = str(route_index)
        if self.save_path and route_index is not None:
            self.save_path = pathlib.Path(self.save_path) / route_index
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
            self.lon_logger = ScenarioLogger(
                save_path=self.save_path, route_index=route_index,
                logging_freq=self.config.logging_freq, log_only=True, route_only=False,
                roi=self.config.logger_region_of_interest)
        else:
            self.save_path = None
        self.metric_info = {}
        print('[OnlineDPMMAgent] setup done.')

    # ------------------------------------------------------------------
    # sensors / _init / tick / run_step (same as team_code agent, adapted
    # for core_team_code model)
    # ------------------------------------------------------------------

    def _init(self):
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]['lon']
            lat = self._global_plan[0][0]['lat']
            earth_radius_equa = 6378137.0

            def equations(v):
                x, y = v
                eq1 = (lon * math.cos(x * math.pi / 180.0) -
                       (locx * x * 180.0) / (math.pi * earth_radius_equa) -
                       math.cos(x * math.pi / 180.0) * y)
                eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) *
                       earth_radius_equa * math.cos(x * math.pi / 180.0) +
                       locy - math.cos(x * math.pi / 180.0) * earth_radius_equa *
                       math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                return [eq1, eq2]
            solution = fsolve(equations, [0.0, 0.0])
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0.0, 0.0

        if self.save_path is not None:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            from nav_planner import interpolate_trajectory
            self.world_map = CarlaDataProvider.get_map()
            trajectory = [item[0].location for item in self._global_plan_world_coord]
            self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)
            self._waypoint_planner = RoutePlanner(
                self.config.log_route_planner_min_distance,
                self.config.route_planner_max_distance, self.lat_ref, self.lon_ref)
            self._waypoint_planner.set_route(self.dense_route, True)
            vehicle = CarlaDataProvider.get_hero_actor()
            self.lon_logger.ego_vehicle = vehicle
            self.lon_logger.world = vehicle.get_world()
            self.nets[0].init_visualization()

        self._route_planner = RoutePlanner(
            self.config.route_planner_min_distance,
            self.config.route_planner_max_distance, self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': self.config.camera_pos[0],
             'y': self.config.camera_pos[1], 'z': self.config.camera_pos[2],
             'roll': self.config.camera_rot_0[0], 'pitch': self.config.camera_rot_0[1],
             'yaw': self.config.camera_rot_0[2], 'width': self.config.camera_width,
             'height': self.config.camera_height, 'fov': self.config.camera_fov, 'id': 'rgb_front'},
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': self.config.carla_frame_rate, 'id': 'imu'},
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'gps'},
            {'type': 'sensor.speedometer', 'reading_frequency': self.config.carla_fps, 'id': 'speed'},
        ]
        if self.config.backbone not in ('aim',):
            sensors.append({
                'type': 'sensor.lidar.ray_cast', 'x': self.config.lidar_pos[0],
                'y': self.config.lidar_pos[1], 'z': self.config.lidar_pos[2],
                'roll': self.config.lidar_rot[0], 'pitch': self.config.lidar_rot[1],
                'yaw': self.config.lidar_rot[2], 'id': 'lidar'})
        return sensors

    @torch.inference_mode()
    def tick(self, input_data):
        rgb = []
        for cam in ['front']:
            camera = input_data['rgb_' + cam][1][:, :, :3]
            _, compressed = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
            rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_pos = t_u.crop_array(self.config, rgb_pos)
            rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

        gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
        speed = input_data['speed'][1]['speed']
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        result = {'rgb': rgb, 'compass': compass}
        if self.config.backbone not in ('aim',):
            result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

        if not self.filter_initialized:
            self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
            self.filter_initialized = True

        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle,
                         brake=self.control.brake)
        self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)
        result['gps'] = filtered_state[0:2]

        waypoint_route = self._route_planner.run_step(np.append(filtered_state[0:2], gps_pos[2]))
        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
            target_point_next, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
            target_point_next = target_point
        else:
            target_point, far_command = waypoint_route[0]
            target_point_next = target_point

        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)

        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)
        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
        result['target_point'] = ego_target_point
        if self.config.two_tp_input:
            ego_tp_next = t_u.inverse_conversion_2d(target_point_next[:2], result['gps'], result['compass'])
            result['target_point_next'] = torch.from_numpy(ego_tp_next[np.newaxis]).to(
                self.device, dtype=torch.float32)
        result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

        if self.save_path is not None:
            waypoint_route = self._waypoint_planner.run_step(np.append(result['gps'], gps_pos[2]))
            waypoint_route = extrapolate_waypoint_route(waypoint_route, self.config.route_points)
            route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])[:self.config.route_points]
            self.lon_logger.log_step(route)

        return result

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            tick_data = self.tick(input_data)
            if self.config.backbone not in ('aim',):
                self.lidar_last = deepcopy(tick_data['lidar'])
            return self.control

        tick_data = self.tick(input_data)
        lidar_indices = [i * self.config.data_save_freq for i in range(self.config.lidar_seq_len)]
        ego_x, ego_y, ego_theta = self.state_log[-1][0], self.state_log[-1][1], self.state_log[-1][2]
        ego_xl, ego_yl, ego_thetal = self.state_log[-2][0], self.state_log[-2][1], self.state_log[-2][2]

        if self.config.backbone not in ('aim',):
            lidar_last = self.align_lidar(
                self.lidar_last, ego_xl, ego_yl, ego_thetal, ego_x, ego_y, ego_theta)
        if self.stop_sign_controller:
            self.update_stop_box(self.stop_sign_buffer, ego_xl, ego_yl, ego_thetal, ego_x, ego_y, ego_theta)

        if self.config.backbone not in ('aim',):
            lidar_full = np.concatenate((deepcopy(tick_data['lidar']), lidar_last), axis=0)
            self.lidar_buffer.append(lidar_full)
            if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
                self.lidar_last = deepcopy(tick_data['lidar'])
                self.control = carla.VehicleControl(0.0, 0.0, 1.0)
                return self.control

        if self.config.backbone in ('aim',):
            lidar_bev = torch.zeros((1, 1 + int(self.config.use_ground_plane),
                                     self.config.lidar_resolution_height,
                                     self.config.lidar_resolution_width),
                                    device=self.device, dtype=torch.float32)
        else:
            lidar_bev = []
            for i in lidar_indices:
                pt_cloud = deepcopy(self.lidar_buffer[-(i + 1)])
                if self.config.realign_lidar and self.config.lidar_seq_len > 1:
                    cx, cy, ct = self.state_log[i][0], self.state_log[i][1], self.state_log[i][2]
                    pt_cloud = self.align_lidar(pt_cloud, cx, cy, ct, ego_x, ego_y, ego_theta)
                hist = self.data.lidar_to_histogram_features(pt_cloud, use_ground_plane=self.config.use_ground_plane)
                lidar_bev.append(torch.from_numpy(hist).unsqueeze(0).to(self.device, dtype=torch.float32))
            lidar_bev = torch.cat(lidar_bev, dim=1)

        if self.config.backbone not in ('aim',):
            self.lidar_last = deepcopy(tick_data['lidar'])

        velocity = tick_data['speed'].reshape(1, 1)
        speed = tick_data['speed'].item()
        if self.stop_after_meter > 0:
            self.meters_travelled += speed * self.config.carla_frame_rate

        # Ensemble forward
        pred_target_speeds = []
        pred_checkpoints = []
        for net in self.nets:
            _, pred_target_speed, pred_trajectories, pred_traj_probs, _, _, _, _, _, _, _ = net(
                rgb=tick_data['rgb'], lidar_bev=lidar_bev,
                target_point=tick_data['target_point'],
                ego_vel=velocity, command=tick_data['command'],
                target_point_next=tick_data.get('target_point_next'))

            best_idx = torch.argmax(pred_traj_probs, dim=0)
            batch_idx = torch.arange(pred_traj_probs.size(1), device=pred_trajectories.device)
            pred_ckpt = pred_trajectories[best_idx, batch_idx]  # (B, 10, 2)
            pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
            pred_checkpoints.append(pred_ckpt[0])

        pred_spd_ens = torch.stack(pred_target_speeds, dim=0).mean(dim=0)
        if self.uncertainty_weight:
            unc = pred_spd_ens.detach().cpu().numpy()
            if unc[0] > self.config.brake_uncertainty_threshold:
                pred_speed_scalar = self.inference_target_speeds[0]
            else:
                pred_speed_scalar = sum(unc * self.inference_target_speeds)
        else:
            pred_speed_scalar = self.inference_target_speeds[torch.argmax(pred_spd_ens).item()]

        pred_ckpts = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()
        steer, throttle, brake = self.nets[0].control_pid_direct(pred_ckpts, pred_speed_scalar, speed)

        # Stuck detector
        if speed < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration
        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
            throttle, brake = 0.0, True

        self.control = carla.VehicleControl(0.0, 0.0, 1.0) if self.step < self.config.inital_frames_delay else \
                       carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        return self.control

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
        pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
        rot_diff = t_u.normalize_angle(orientation_target - orientation)
        rm = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                       [np.sin(orientation_target), np.cos(orientation_target), 0.0],
                       [0.0, 0.0, 1.0]])
        return t_u.algin_lidar(lidar, rm.T @ pos_diff, rot_diff)

    def update_stop_box(self, boxes, x, y, orientation, x_target, y_target, orientation_target):
        pos_diff = np.array([x_target, y_target]) - np.array([x, y])
        rot_diff = t_u.normalize_angle(orientation_target - orientation)
        rm = np.array([[np.cos(orientation_target), -np.sin(orientation_target)],
                       [np.sin(orientation_target), np.cos(orientation_target)]])
        pos_diff = rm.T @ pos_diff
        local_rot = np.array([[np.cos(rot_diff), -np.sin(rot_diff)],
                              [np.sin(rot_diff), np.cos(rot_diff)]])
        for box in boxes:
            box[:2] = (local_rot.T @ (box[:2] - pos_diff).T).T
            box[4] = t_u.normalize_angle(box[4] - rot_diff)

    def destroy(self, results=None):
        if self.save_path is not None:
            self.lon_logger.dump_to_json()
            if len(self.nets[0].speed_histogram) > 0:
                with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
                    ujson.dump(self.nets[0].speed_histogram, f, indent=4)
        del self.nets
        del self.config
        del self.metric_info