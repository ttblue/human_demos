#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

import roslib
import rospy

roslib.load_manifest("tf")
import tf

import os
import os.path as osp

import sensor_msgs.msg as sm
import visualization_msgs.msg as vm
import geometry_msgs.msg as gm
import std_msgs.msg as stdm
from visualization_msgs.msg import Marker, MarkerArray

from StringIO import StringIO
import traceback
from func_utils import once
import conversions
import transformations
from colorize import colorize

import urdf
import time
import weakref
from copy import deepcopy

from hd_utils.conversions import hmats_to_transs_rots
import thread


@once
def get_tf_listener():
    return tf.TransformListener()


def pc2xyzrgb(pc):
    arr = np.fromstring(pc.data, dtype='float32').reshape(pc.height,
            pc.width, pc.point_step / 4)
    xyz = arr[:, :, 0:3]

    rgb0 = np.ndarray(buffer=arr[:, :, 4].copy(), shape=(pc.height,
                      pc.width, 4), dtype='uint8')
    
    rgb = rgb0[:, :, 0:3]

    return (xyz, rgb)


def pc2xyz(pc):
    arr = np.fromstring(pc.data, dtype='float32').reshape(pc.height,
            pc.width, pc.point_step / 4)
    
    xyz = arr[:, :, 0:3]
    return xyz


def xyz2pc(xyz, frame_id, use_time_now=True):
    bgr = np.zeros_like(xyz)
    return xyzrgb2pc(xyz, bgr, frame_id, use_time_now)


def xyzrgb2pc(
    xyz,
    bgr,
    frame_id='',
    use_time_now=True,
    ):
    
    xyz = np.asarray(xyz)
    bgr = np.asarray(bgr)

    assert xyz.shape == bgr.shape
    if xyz.ndim == 2:
        xyz = xyz[None, :, :]
        bgr = bgr[None, :, :]

    height = xyz.shape[0]
    width = xyz.shape[1]

    arr = np.empty((height, width, 8), dtype='float32')
    arr[:, :, 0:3] = xyz
    bgr1 = np.empty((height, width, 4), dtype='uint8')
    bgr1[:, :, 0:3] = bgr
    arr[:, :, 4] = bgr1.view(dtype='float32').reshape(height, width)
    data = arr.tostring()
    msg = sm.PointCloud2()
    msg.data = data
    msg.header.frame_id = frame_id
    msg.fields = [sm.PointField(name='x', offset=0, datatype=7,
                  count=1), sm.PointField(name='y', offset=4,
                  datatype=7, count=1), sm.PointField(name='z',
                  offset=8, datatype=7, count=1),
                  sm.PointField(name='rgb', offset=16, datatype=7,
                  count=1)]
    msg.is_dense = False
    msg.width = width
    msg.height = height
    if use_time_now:
        msg.header.stamp = rospy.Time.now()

    msg.point_step = 32
    msg.row_step = 32 * width
    msg.is_bigendian = False

    return msg


def transformPointCloud2(
    cloud,
    listener,
    to_frame,
    from_frame,
    ):
    
    (xyz, rgb) = pc2xyzrgb(cloud)
    xyz_tf = transform_points(xyz, listener, to_frame, from_frame)
    pc = xyzrgb2pc(xyz_tf, rgb, to_frame)
    return pc


def transform_points(
    xyz,
    listener,
    to_frame,
    from_frame,
    n_tries=10,
    ):

    # listener.waitForTransform(to_frame, from_frame, rospy.Time.now(), rospy.Duration(1))

    for i in xrange(n_tries):
        try:
            (trans, rot) = listener.lookupTransform(to_frame,
                    from_frame, rospy.Time(0))
            break
        except Exception:
            print 'tf exception:'
            print colorize(traceback.format_exc(), 'yellow')
            rospy.sleep(.1)
            time.sleep(.05)
    if i == n_tries - 1:
        raise Exception('fuck tf')
    hmat = conversions.trans_rot_to_hmat(trans, rot)

    xyz1 = np.c_[xyz.reshape(-1, 3), np.ones((xyz.size / 3, 1))]
    xyz1_tf = np.dot(xyz1, hmat.T)
    xyz_tf = xyz1_tf[:, :3].reshape(xyz.shape)
    return xyz_tf


def clone_msg(msg):
    sio = StringIO()
    msg.serialize(sio)
    newmsg = type(msg)()
    newmsg.deserialize(sio.getvalue())
    return newmsg


def msg2arr(msg):
    """convert a msg to an array of floats
    don't use on a stamped msg"""

    s = StringIO()
    msg.serialize_numpy(s, np)
    return np.fromstring(s.getvalue())


def arr2msg(arr, msg_type):
    """convert array of floats to msg
    don't use on a stamped msg type"""

    msg = msg_type()
    msg.deserialize_numpy(np.asarray(arr).tostring(), np)
    return msg


class TopicListener(object):

    """
    stores last message of the topic
    """

    last_msg = None

    def __init__(self, topic_name, msg_type):
        self.sub = rospy.Subscriber(topic_name, msg_type, self.callback)

        rospy.loginfo('waiting for the first message: %s' % topic_name)
        while self.last_msg is None:
            rospy.sleep(.1)
        rospy.loginfo('ok: %s' % topic_name)

    def callback(self, msg):
        self.last_msg = msg


def wait_for_service_verbose(service, timeout=None):
    rospy.loginfo('waiting for service: %s' % service)
    rospy.wait_for_service(service, timeout=timeout)
    rospy.loginfo('ok: %s' % service)


def wait_for_message_verbose(topic, msgtype, timeout=None):
    rospy.loginfo('waiting for message: %s' % topic)
    msg = rospy.wait_for_message(topic, msgtype, timeout=timeout)
    rospy.loginfo('ok: %s' % topic)
    return msg


def call_service(service, srv_class, req=None):
    wait_for_service_verbose(service)
    client = rospy.ServiceProxy(service, srv_class)
    if req is None:
        return client()
    else:
        return client(req)


def image2numpy(image):
    if image.encoding == 'rgb8':
        return np.fromstring(image.data,
                             dtype=np.uint8).reshape(image.height,
                image.width, 3)[:, :, ::-1]
    if image.encoding == 'bgr8':
        return np.fromstring(image.data,
                             dtype=np.uint8).reshape(image.height,
                image.width, 3)
    elif image.encoding == 'mono8':
        return np.fromstring(image.data,
                             dtype=np.uint8).reshape(image.height,
                image.width)
    elif image.encoding == '32FC1':
        return np.fromstring(image.data,
                             dtype=np.float32).reshape(image.height,
                image.width)
    else:
        raise Exception


class MarkerHandle(object):

    disable_delete = False

    def __init__(self, marker, pub):
        self.pub = pub
        self.marker = marker
        HANDLES[np.random.randint(0, 2147483647)] = self

    def __del__(self):
        try:
            if not self.disable_delete:
                self.marker.action = 2  # Marker.DELETE
                self.pub.publish(self.marker)
        except Exception:
            pass


class MarkerListHandle(MarkerHandle):

    def __init__(self, marker_handles):
        self.marker_handles = marker_handles
        HANDLES[np.random.randint(0, 2147483647)] = self

    def __del__(self):
        for marker_handle in self.marker_handles:
            marker_handle.__del__()


HANDLES = weakref.WeakValueDictionary()


def delete_all_handles():
    print 'deleting handles'
    for (_, val) in HANDLES.items():
        val.__del__()
    time.sleep(.1)


@once
def register_deletion():
    rospy.on_shutdown(delete_all_handles)


class RvizWrapper:

    @once
    def create():
        return RvizWrapper()

    def __init__(self):
        self.pub = rospy.Publisher('visualization_marker', Marker,
                                   latch=True)
        self.array_pub = rospy.Publisher('visualization_marker_array',
                MarkerArray)
        self.ids = set([])
        register_deletion()

    def draw_traj_points(
        self,
        pose_array,
        rgba=(0, 1, 0, 1),
        arrow_scale=.05,
        ns='default_ns',
        duration=0,
        ):
        marker_array = MarkerArray()
        for pose in pose_array.poses:
            marker = Marker(type=Marker.ARROW, action=Marker.ADD)
            marker.header = pose_array.header
            marker.pose = pose
            marker.lifetime = rospy.Duration(0)
            marker.color = stdm.ColorRGBA(*rgba)
            marker.scale = gm.Vector3(arrow_scale, arrow_scale,
                    arrow_scale)
            marker.id = self.get_unused_id()
            marker.ns = ns
            marker_array.markers.append(marker)
            self.ids.add(marker.id)
        self.array_pub.publish(marker_array)
        return MarkerListHandle([MarkerHandle(marker, self.pub)
                                for marker in marker_array.markers])

    def draw_curve(
        self,
        pose_array,
        id=None,
        rgba=(0, 1, 0, 1),
        width=.01,
        ns='default_ns',
        duration=0,
        type=Marker.LINE_STRIP,
        ):
        if id is None:
            id = self.get_unused_id()
        marker = Marker(type=type, action=Marker.ADD)
        marker.header = pose_array.header
        marker.points = [pose.position for pose in pose_array.poses]
        marker.lifetime = rospy.Duration(0)
        if isinstance(rgba, list):
            assert len(rgba) == len(pose_array.poses)
            marker.colors = [stdm.ColorRGBA(*c) for c in rgba]
        else:
            marker.color = stdm.ColorRGBA(*rgba)
        marker.scale = gm.Vector3(width, width, width)
        marker.id = id
        marker.ns = ns
        self.pub.publish(marker)
        self.ids.add(id)
        return MarkerHandle(marker, self.pub)

    def draw_marker(
        self,
        ps,
        id=None,
        type=Marker.CUBE,
        ns='default_ns',
        rgba=(0, 1, 0, 1),
        scale=(.03, .03, .03),
        text='',
        duration=0,
        ):
        """
        If markers aren't showing up, it's probably because the id's are being overwritten. Make sure to set
        the ids and namespace.
        """

        if id is None:
            id = self.get_unused_id()
        marker = Marker(type=type, action=Marker.ADD)
        marker.ns = ns
        marker.header = ps.header
        marker.pose = ps.pose
        marker.scale = gm.Vector3(*scale)
        marker.color = stdm.ColorRGBA(*rgba)
        marker.id = id
        marker.text = text
        marker.lifetime = rospy.Duration(duration)
        self.pub.publish(marker)
        self.ids.add(id)
        return MarkerHandle(marker, self.pub)

    def get_unused_id(self):
        while True:
            id = np.random.randint(0, 2147483647)
            if id not in self.ids:
                return id

    def place_kin_tree_from_link(
        self,
        ps,
        linkname,
        valuedict=None,
        color=(1, 1, 0, 0.5),
        ns='default_ns',
        ):
        markers = make_kin_tree_from_link(ps, linkname,
                valuedict=valuedict, color=color)
        marker_array = MarkerArray()
        marker_array.markers = markers
        handles = []
        for marker in markers:
            id = self.get_unused_id()
            self.ids.add(id)
            marker.id = id
            marker.ns = ns
            handles.append(MarkerHandle(marker, self.pub))
            self.pub.publish(marker)
        return handles

    def draw_trajectory(
        self,
        pose_array,
        open_fracs,
        color=(1, 1, 0, 0.5),
        ns='default_ns',
        ):
        decimation = 1  # max(len(pose_array.poses)//6, 1)
        ps = gm.PoseStamped()
        ps.header.frame_id = pose_array.header.frame_id
        ps.header.stamp = rospy.Time.now()
        handles = []

        for (pose, open_frac) in zip(pose_array.poses, open_fracs)[::
                decimation]:

            ps.pose = deepcopy(pose)
            pointing_axis = \
                transformations.quaternion_matrix([pose.orientation.x,
                    pose.orientation.y, pose.orientation.z,
                    pose.orientation.w])[:3, 0]
            ps.pose.position.x -= .18 * pointing_axis[0]
            ps.pose.position.y -= .18 * pointing_axis[1]
            ps.pose.position.z -= .18 * pointing_axis[2]

            joint = .0166 + open_frac * (.463 - .0166)
            root_link = 'r_wrist_roll_link'
            valuedict = {
                'r_gripper_l_finger_joint': joint,
                'r_gripper_r_finger_joint': joint,
                'r_gripper_l_finger_tip_joint': joint,
                'r_gripper_r_finger_tip_joint': joint,
                'r_gripper_joint': joint,
                }
            handles.extend(self.place_kin_tree_from_link(ps, root_link,
                           valuedict, color=color, ns=ns))
        return handles


@once
def get_pr2_urdf():
    U = urdf.URDF()
    urdf_path = osp.join('/opt/ros', os.getenv('ROS_DISTRO'),
                         'stacks/pr2_mechanism/pr2_mechanism_model/pr2.urdf'
                         )
    U.load(urdf_path)
    return U


def link_filter(names):
    return [name for name in names if name.endswith('link')]


def joint_filter(names):
    return [name for name in names if name.endswith('joint')]


def make_kin_tree_from_link(
    ps,
    linkname,
    ns='default_ns',
    valuedict=None,
    color=(1, 1, 0, 0.5),
    ):
    markers = []
    U = get_pr2_urdf()
    link = U.links[linkname]

    if link.visual and link.visual.geometry \
        and isinstance(link.visual.geometry, urdf.Mesh):
        rospy.logdebug('%s is a mesh. drawing it.', linkname)
        marker = Marker(type=Marker.MESH_RESOURCE, action=Marker.ADD)
        marker.ns = ns
        marker.header = ps.header

        linkFromGeom = \
            conversions.trans_rot_to_hmat(link.visual.origin.position,
                transformations.quaternion_from_euler(*link.visual.origin.rotation))
        origFromLink = conversions.pose_to_hmat(ps.pose)
        origFromGeom = np.dot(origFromLink, linkFromGeom)

        marker.pose = conversions.hmat_to_pose(origFromGeom)
        marker.scale = gm.Vector3(1, 1, 1)
        marker.color = stdm.ColorRGBA(*color)
        marker.id = 0
        marker.lifetime = rospy.Duration()
        marker.mesh_resource = str(link.visual.geometry.filename)
        marker.mesh_use_embedded_materials = False
        markers.append(marker)
    else:
        rospy.logdebug('%s is not a mesh', linkname)

    if U.child_map.has_key(linkname):
        for (cjoint, clink) in U.child_map[linkname]:
            markers.extend(make_kin_tree_from_joint(ps, cjoint, ns=ns,
                           valuedict=valuedict, color=color))

    return markers


def make_kin_tree_from_joint(
    ps,
    jointname,
    ns='default_ns',
    valuedict=None,
    color=(1, 1, 0, 0.5),
    ):
    rospy.logdebug('joint: %s' % jointname)
    U = get_pr2_urdf()

    joint = U.joints[jointname]
    joint.origin = joint.origin or urdf.Pose()
    if not joint.origin.position:
        joint.origin.position = [0, 0, 0]
    if not joint.origin.rotation:
        joint.origin.rotation = [0, 0, 0]

    parentFromChild = \
        conversions.trans_rot_to_hmat(joint.origin.position,
            transformations.quaternion_from_euler(*joint.origin.rotation))

    childFromRotated = np.eye(4)
    if valuedict and jointname in valuedict:

        if joint.joint_type == 'revolute' or joint.joint_type \
            == 'continuous':
            childFromRotated = \
                transformations.rotation_matrix(valuedict[jointname],
                    joint.axis)
        elif joint.joint_type == 'prismatic':
            childFromRotated = \
                transformations.translation_matrix(np.array(joint.axis)
                    * valuedict[jointname])

    parentFromRotated = np.dot(parentFromChild, childFromRotated)
    originFromParent = conversions.pose_to_hmat(ps.pose)
    originFromRotated = np.dot(originFromParent, parentFromRotated)

    newps = gm.PoseStamped()
    newps.header = ps.header
    newps.pose = conversions.hmat_to_pose(originFromRotated)

    return make_kin_tree_from_link(newps, joint.child, ns=ns,
                                   valuedict=valuedict, color=color)


def publish_static_tfm(parent_frame, child_frame, tfm):
    publish_static_tfms([parent_frame], [child_frame], [tfm])

def publish_static_tfms(parent_frames, child_frames, tfms):
    """
    Publish multiple static frames on the TF.
    """
    tf_broadcaster = tf.TransformBroadcaster()
    transs, rots   = hmats_to_transs_rots(tfms)
    N = len(tfms)
    def spin_pub():
        sleeper = rospy.Rate(100)
        while not rospy.is_shutdown():
            for i in xrange(N):
                tf_broadcaster.sendTransform(transs[i], rots[i],
                                             rospy.Time.now(),
                                             child_frames[i], parent_frames[i])
            sleeper.sleep()
    thread.start_new_thread(spin_pub, ())
    
