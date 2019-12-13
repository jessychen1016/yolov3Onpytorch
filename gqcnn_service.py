#!/usr/bin/env	python3
# coding=utf-8
import python3_in_ros
import sys
import os
import queue
import threading
import cv2
import rospy
import random
import numpy as np
from easygqcnn import GraspingPolicy, GraspCloseWidth
from ruamel.yaml import YAML
from mb_grasp.srv import easyGQService, easyGQServiceResponse

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/policy.yaml')


def load_config(file):
    """ 加载配置文件 """
    yaml = YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(file, 'r', encoding="utf-8") as f:
        config = yaml.load(f)
    return config


def image_msg_to_cv2(img_msg):
    """ 由于cv_bridge在Python3中无法使用
    这里简单的实现一下由sensor_msgs/Image到np.ndarray的转换
    """
    if 'C' in img_msg.encoding:
        map_dtype = {'U': 'uint', 'S': 'int', 'F': 'float'}
        dtype_str, n_channels_str = img_msg.encoding.split('C')
        n_channels = int(n_channels_str)
        dtype = np.dtype(map_dtype[dtype_str[-1]] + dtype_str[:-1])
    elif img_msg.encoding == 'bgr8':
        n_channels = 3
        dtype = np.dtype('uint8')
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                    dtype=dtype, buffer=img_msg.data)
    im = np.squeeze(im)
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        im = im.byteswap().newbyteorder()
    return im


def plot_grasp_cv(img, g, offset=[0, 0]):
    """ 使用plt在图像上展示一个夹爪 """
    red = (0, 0, 255)
    bule = (255, 0, 0)

    def plot_2p(img, p0, p1, color=red, width=1):
        p0 -= offset
        p1 -= offset
        cv2.line(img, tuple(p0.astype('int')), tuple(p1.astype('int')), color, width)

    def plot_center(img, center, axis, length, color=red, width=2):
        axis = axis / np.linalg.norm(axis)
        p0 = center - axis * length / 2
        p1 = center + axis * length / 2
        plot_2p(img, p0, p1, color, width)

    p0, p1 = g.endpoints
    axis = [g.axis[1], -g.axis[0]]
    plot_2p(img, p0, p1)
    plot_center(img, p0, axis, g.width_px/2.5, width=3)
    plot_center(img, p1, axis, g.width_px/2.5, width=3)
    cv2.circle(img, tuple(g.center.astype('int')), 3, bule, -1)


class EasygqcnnService(object):
    def __init__(self, config, q):
        self._config = config
        self._queue = q
        self._policy = GraspingPolicy(self._config)

    def easy_gqcnn_handle(self, req):
        depth_raw = req.depth_image
        depth = image_msg_to_cv2(depth_raw)
        depth_BGR = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        self._queue.put(depth_BGR)
        # 进行gqcnn的计算
        random.seed(15)
        np.random.seed(15)
        g, q = self._policy.action(depth, None)
        print('抓取深度:', g.depth, '抓取中心:', g.center, '抓取角度:', np.rad2deg(g.angle))
        # 计算抓取宽度
        grasp_width = GraspCloseWidth(depth, None)
        width_px = list(grasp_width.action(g))
        if width_px[0] is not None and width_px[0] > 45:
            width_px[0] = None
        print('抓取宽度为:', width_px)
        plot_grasp_cv(depth_BGR, g)
        self._queue.put(depth_BGR)
        # 发布服务的响应消息
        res = self.grenerate_Response(g, *width_px)
        return res

    def grenerate_Response(self, g, w, c0, c1):
        res = easyGQServiceResponse()
        res.grasp_pose_2d.x = g.center_float[0]
        res.grasp_pose_2d.y = g.center_float[1]
        res.grasp_pose_2d.theta = g.angle
        res.depth = g.depth
        if w is not None:
            res.width = w
            res.contact0.x = c0[0]
            res.contact0.y = c0[1]
            res.contact0.z = g.depth
            res.contact1.x = c1[0]
            res.contact1.y = c1[1]
            res.contact1.z = g.depth
        else:
            res.width = 0
        return res


class ImageThread (threading.Thread):
    def __init__(self, q, name, image_size):
        threading.Thread.__init__(self)
        self._queue = q
        self._name = name
        self._size = image_size

    def run(self):
        print("开启线程:" + self._name)
        self.process_data()
        print("退出线程:" + self._name)

    def process_data(self):
        cv2.namedWindow(self._name, 0)
        data = np.zeros(self._size)
        while not rospy.is_shutdown():
            if not self._queue.empty():
                data = self._queue.get()
            cv2.imshow(self._name, data)
            cv2.waitKey(10)


def main():
    rospy.init_node('gqcnn_node')
    config = load_config(TEST_CFG_FILE)
    # 创建一个显示的深度图的线程
    depthQueue = queue.Queue(10)
    # 开启线程，用于显示图片
    depth_thread = ImageThread(depthQueue, 'depth', (480, 640, 1))
    depth_thread.start()
    service = EasygqcnnService(config, depthQueue)
    rospy.Service('/gqcnn', easyGQService, service.easy_gqcnn_handle)
    rospy.spin()


if __name__ == "__main__":
    # config = load_config(TEST_CFG_FILE)
    main()
