# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/algo/deploy/robots/leap.py
# --------------------------------------------------------

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
import numpy as np
from leap_hand.srv import *

class LeapHand(object):
    def __init__(self):
        """ Simple python interface to the leap Hand.

        The leapClient is a simple python interface to an leap
        robot hand.  It enables you to command the hand directly through
        python library calls (joint positions, joint torques, or 'named'
        grasps).

        The constructors sets up publishers and subscribes to the joint states
        topic for the hand.
        """

        # Topics (that can be remapped) for named graps
        # (ready/envelop/grasp/etc.), joint commands (position and
        # velocity), joint state (subscribing), and envelop torque. Note that
        # we can change the hand topic prefix (for example, to leapHand_0)
        # instead of remapping it at the command line.
        topic_joint_command = '/leaphand_node/cmd_ones'

        # Publishers for above topics.
        self.pub_joint = rospy.Publisher(topic_joint_command, JointState, queue_size=10)
        #rospy.Subscriber(topic_joint_state, JointState, self._joint_state_callback)
        self._joint_state = None
        self.leap_position = rospy.ServiceProxy('/leap_position', leap_position)
        self.leap_effort = rospy.ServiceProxy('/leap_effort', leap_effort)

    #def _joint_state_callback(self, data):
    #self._joint_state = data

    def sim_to_real(self, values):
        return values[self.sim_to_real_indices]

    def command_joint_position(self, desired_pose):
        """
            Takes as input self.cur_targets (raw joints) in numpy array
        """

        # Check that the desired pose can have len() applied to it, and that
        # the number of dimensions is the same as the number of hand joints.
        if (not hasattr(desired_pose, '__len__') or
                len(desired_pose) != 16):
            rospy.logwarn('Desired pose must be a {}-d array: got {}.'
                          .format(16, desired_pose))
            return False

        msg = JointState()  # Create and publish

        # convert desired_pose to ros_targets
        desired_pose = (2 * desired_pose - self.leap_dof_lower - self.leap_dof_upper) / (self.leap_dof_upper - self.leap_dof_lower)
        desired_pose = self.sim_to_real(desired_pose) 

        try:
            msg.position = desired_pose
            self.pub_joint.publish(msg)
            rospy.logdebug('Published desired pose.')
            return True
        except rospy.exceptions.ROSSerializationException:
            rospy.logwarn('Incorrect type for desired pose: {}.'.format(
                desired_pose))
            return False

    def real_to_sim(self, values):
        return values[self.real_to_sim_indices]

    def poll_joint_position(self):
        """ Get the current joint positions of the hand.

        :param wait: If true, waits for a 'fresh' state reading.
        :return: Joint positions, or None if none have been received.
        """
        joint_position = np.array(self.leap_position().position)
        #joint_effort = np.array(self.leap_effort().effort)

        joint_position = self.LEAPhand_to_sim_ones(joint_position)
        joint_position = self.real_to_sim(joint_position)
        joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower

        return (joint_position, None)

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        
        return sim_min, sim_max

    def LEAPhand_to_LEAPsim(self, joints):
        joints = np.array(joints)
        ret_joints = joints - 3.14
        return ret_joints

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)
        
        return joints

def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower)/(upper - lower)


