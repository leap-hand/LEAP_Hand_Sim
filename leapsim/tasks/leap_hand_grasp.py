# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: 
# https://github.com/HaozhiQi/hora/blob/main/hora/tasks/leap_hand_grasp.py
# --------------------------------------------------------

import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis, quat_mul, tensor_clamp, to_torch
from leapsim.tasks.leap_hand_rot import LeapHandRot


class LeapHandGrasp(LeapHandRot):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless)
        self.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=self.device)


        if "canonical_pose" in cfg["env"]:
            self.canonical_pose = cfg["env"]["canonical_pose"]
        else:
            self.canonical_pose = [
                0.082, 1.244, 0.265, 0.298, 1.104, 1.163, 0.953, -0.138,
                0.005, 1.096, 0.080, 0.150, 0.029, 1.337, 0.285, 0.317,
            ]

        if "num_contact_fingers" in cfg["env"]:
            self.num_contact_fingers = cfg["env"]["num_contact_fingers"]
        else:
            self.num_contact_fingers = 2
        
        if "finger_dist_threshold" in cfg["env"]:
            self.finger_dist_threshold = cfg["env"]["finger_dist_threshold"]
        else:
            self.finger_dist_threshold = 0.1
            
        if "grasp_cache_len" not in self.cfg["env"]:
            self.cfg["env"]["grasp_cache_len"] = 5e4
        
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def reset_idx(self, env_ids):
        if self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
        else:
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_leap_hand_dofs * 2 + 5), device=self.device)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        success = self.progress_buf[env_ids] == self.max_episode_length
        all_states = torch.cat([
            self.leap_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
        print('current cache size:', self.saved_grasping_states.shape[0])
        if len(self.saved_grasping_states) >= self.cfg["env"]["grasp_cache_len"]:
            name = f'cache/{self.grasp_cache_name}_grasp_50k_s{str(self.base_obj_scale).replace(".", "")}.npy'
            np.save(name, self.saved_grasping_states[:self.cfg["env"]["grasp_cache_len"]].cpu().numpy())
            exit()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx]
        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        new_object_rot[:] = 0
        new_object_rot[:, -1] = 1
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        pos = to_torch(self.canonical_pose, device=self.device)[None].repeat(len(env_ids), 1)
        pos += self.cfg["env"]["grasp_dof_search_radius"] * rand_floats[:, 5:5 + self.num_leap_hand_dofs]
        pos = tensor_clamp(pos, self.leap_hand_dof_lower_limits[env_ids], self.leap_hand_dof_upper_limits[env_ids])

        self.leap_hand_dof_pos[env_ids, :] = pos
        self.leap_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :self.num_leap_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_leap_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_reward(self, actions):
        def list_intersect(li, hash_num):
            # 17 is the object index
            # 4, 8, 12, 16 are fingertip index
            # return number of contact with obj_id
            obj_id = 17
            query_list = [obj_id * hash_num + 4, obj_id * hash_num + 8, obj_id * hash_num + 12, obj_id * hash_num + 16]
            return len(np.intersect1d(query_list, li))
        assert self.device == 'cpu'
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        contact_list = [list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000) for contact in contacts]
        contact_condition = to_torch(contact_list, device=self.device)

        
        obj_pos = self.rigid_body_states[:, [-1], :3]
        finger_pos = self.rigid_body_states[:, [4, 8, 12, 16], :3]
        # the sampled pose need to satisfy (check 1 here):
        # 1) all fingertips is nearby objects
        cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < self.finger_dist_threshold).all(-1)
        # 2) at least two fingers are in contact with object
        cond2 = contact_condition >= self.num_contact_fingers
        # 3) object does not fall after a few iterations
        # 0.645 for internal leap
        # 0.625 for public leap
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        cond = cond1.float() * cond2.float() * cond3.float()
        # reset if any of the above condition does not hold
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
