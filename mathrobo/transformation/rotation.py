#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.08.17 Created by T.Ishigaki

import numpy as np

from ..basic import *
from ..lie import *

def euler_x(theta):
    m = zeros((3,3))
    
    m[0,0] = 1
    m[1,1] = cos(theta)
    m[1,2] = -sin(theta)
    m[2,1] = sin(theta)
    m[2,2] = cos(theta)
    
    return m
    
def euler_y(theta):
    m = zeros((3,3))
    
    m[0,0] = cos(theta)
    m[0,2] = sin(theta)
    m[1,1] = 1
    m[2,0] = -sin(theta)
    m[2,2] = cos(theta)

    return m
    
def euler_z(theta):
    m = zeros((3,3))
    
    m[0,0] = cos(theta)
    m[0,1] = -sin(theta)
    m[1,0] = sin(theta)
    m[1,1] = cos(theta)
    m[2,2] = 1
    
    return m
    
def euler_rpy(vec):
    m = euler_z(vec[2]) @ euler_y(vec[1]) @ euler_x(vec[0])
    return m
