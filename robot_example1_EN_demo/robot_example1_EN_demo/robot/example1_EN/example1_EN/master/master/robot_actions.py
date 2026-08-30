#!/usr/bin/env python3
"""
Robot action definition module
Define joint trajectories for actions such as wave, expand, agree
"""

import math
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    """Action type enumeration"""
    WAVE = "wave"           # Wave
    EXPAND = "expand"       # Expand
    AGREE = "agree"         # Agree

@dataclass
class JointTrajectory:
    """Joint trajectory data structure"""
    positions: List[float]      # Joint positions
    time_from_start: float      # Time from start
    description: str            # Description

class RobotActions:
    """Robot action class"""
    
    def __init__(self):
        # Joint name list (consistent with order in URDF file)
        self.joint_names = [
            'wheel_left_joint', 'wheel_right_joint', 'lift_joint', 'waist_joint', 'head_joint',
            'J1_left_joint', 'J2_left_joint', 'J3_left_joint', 'J4_left_joint', 'J5_left_joint', 
            'J6_left_joint', 'J7_left_joint',
            'J1_right_joint', 'J2_right_joint', 'J3_right_joint', 'J4_right_joint', 'J5_right_joint', 
            'J6_right_joint', 'J7_right_joint'
        ]
        
        # Initial positions (all zero)
        self.initial_positions = [0.0] * len(self.joint_names)
        
        # Joint index mapping
        self.joint_indices = {name: idx for idx, name in enumerate(self.joint_names)}
    
    def get_wave_action(self) -> List[JointTrajectory]:
        """Get wave action trajectory
        Robot raises right arm and waves left and right for 3 seconds
        """
        trajectories = []
        
        # Initial pose
        trajectories.append(JointTrajectory(
            positions=self.initial_positions.copy(),
            time_from_start=0.0,
            description="Initial pose"
        ))
        
        # Raise right arm
        wave_start = self.initial_positions.copy()
        wave_start[self.joint_indices['J1_right_joint']] = 1.57  # Right shoulder raise
        wave_start[self.joint_indices['J2_right_joint']] = -0.5  # Right elbow bend
        wave_start[self.joint_indices['J3_right_joint']] = -1.0  # Right wrist rotation
        
        trajectories.append(JointTrajectory(
            positions=wave_start,
            time_from_start=1.0,
            description="Raise right arm"
        ))
        
        # Wave left
        wave_left = wave_start.copy()
        wave_left[self.joint_indices['J3_right_joint']] = -1.57  # Wave left
        
        trajectories.append(JointTrajectory(
            positions=wave_left,
            time_from_start=2.0,
            description="Wave left"
        ))
        
        # Wave right
        wave_right = wave_start.copy()
        wave_right[self.joint_indices['J3_right_joint']] = -0.5  # Wave right
        
        trajectories.append(JointTrajectory(
            positions=wave_right,
            time_from_start=3.0,
            description="Wave right"
        ))
        
        # Wave left again
        trajectories.append(JointTrajectory(
            positions=wave_left,
            time_from_start=4.0,
            description="Wave left again"
        ))
        
        # Return to initial pose
        trajectories.append(JointTrajectory(
            positions=self.initial_positions.copy(),
            time_from_start=5.0,
            description="Return to initial pose"
        ))
        
        return trajectories
    
    def get_expand_action(self) -> List[JointTrajectory]:
        """Get expand action trajectory
        Robot extends both arms outward to shoulder level, returns to initial pose after 2s, repeats in loop
        """
        trajectories = []
        
        # Initial pose
        trajectories.append(JointTrajectory(
            positions=self.initial_positions.copy(),
            time_from_start=0.0,
            description="Initial pose"
        ))
        
        # Raise both arms to shoulder level
        expand_start = self.initial_positions.copy()
        expand_start[self.joint_indices['J1_left_joint']] = 1.57   # Left shoulder raise
        expand_start[self.joint_indices['J2_left_joint']] = -0.5   # Left elbow bend
        expand_start[self.joint_indices['J3_left_joint']] = 0.0    # Left wrist horizontal
        
        expand_start[self.joint_indices['J1_right_joint']] = 1.57  # Right shoulder raise
        expand_start[self.joint_indices['J2_right_joint']] = -0.5  # Right elbow bend
        expand_start[self.joint_indices['J3_right_joint']] = 0.0   # Right wrist horizontal
        
        trajectories.append(JointTrajectory(
            positions=expand_start,
            time_from_start=1.5,
            description="Raise both arms to shoulder level"
        ))
        
        # Extend both arms outward
        expand_full = expand_start.copy()
        expand_full[self.joint_indices['J4_left_joint']] = 1.57    # Left elbow extend
        expand_full[self.joint_indices['J4_right_joint']] = 1.57   # Right elbow extend
        
        trajectories.append(JointTrajectory(
            positions=expand_full,
            time_from_start=3.0,
            description="Extend both arms outward"
        ))
        
        # Hold extended pose for 2 seconds
        trajectories.append(JointTrajectory(
            positions=expand_full,
            time_from_start=5.0,
            description="Hold extended pose"
        ))
        
        # Return to shoulder level
        trajectories.append(JointTrajectory(
            positions=expand_start,
            time_from_start=6.0,
            description="Return to shoulder level"
        ))
        
        # Return to initial pose
        trajectories.append(JointTrajectory(
            positions=self.initial_positions.copy(),
            time_from_start=7.5,
            description="Return to initial pose"
        ))
        
        return trajectories
    
    def get_agree_action(self) -> List[JointTrajectory]:
        """Get agree action trajectory
        Robot body slightly rises (waist joint rises), then nods 3 times, returns to initial pose after 2 seconds
        """
        trajectories = []
        
        # Initial pose
        trajectories.append(JointTrajectory(
            positions=self.initial_positions.copy(),
            time_from_start=0.0,
            description="Initial pose"
        ))
        
        # Body slightly rises
        agree_start = self.initial_positions.copy()
        agree_start[self.joint_indices['lift_joint']] = 0.1  # Body rises
        
        trajectories.append(JointTrajectory(
            positions=agree_start,
            time_from_start=1.0,
            description="Body rises"
        ))
        
        # Nod 1st time
        agree_nod1 = agree_start.copy()
        agree_nod1[self.joint_indices['head_joint']] = 0.3  # Nod
        
        trajectories.append(JointTrajectory(
            positions=agree_nod1,
            time_from_start=2.0,
            description="Nod 1st time"
        ))
        
        # Return to middle
        trajectories.append(JointTrajectory(
            positions=agree_start,
            time_from_start=2.5,
            description="Return to middle"
        ))
        
        # Nod 2nd time
        agree_nod2 = agree_start.copy()
        agree_nod2[self.joint_indices['head_joint']] = 0.3  # Nod
        
        trajectories.append(JointTrajectory(
            positions=agree_nod2,
            time_from_start=3.0,
            description="Nod 2nd time"
        ))
        
        # Return to middle
        trajectories.append(JointTrajectory(
            positions=agree_start,
            time_from_start=3.5,
            description="Return to middle"
        ))
        
        # Nod 3rd time
        agree_nod3 = agree_start.copy()
        agree_nod3[self.joint_indices['head_joint']] = 0.3  # Nod
        
        trajectories.append(JointTrajectory(
            positions=agree_nod3,
            time_from_start=4.0,
            description="Nod 3rd time"
        ))
        
        # Return to middle
        trajectories.append(JointTrajectory(
            positions=agree_start,
            time_from_start=4.5,
            description="Return to middle"
        ))
        
        # Return to initial pose
        trajectories.append(JointTrajectory(
            positions=self.initial_positions.copy(),
            time_from_start=6.0,
            description="Return to initial pose"
        ))
        
        return trajectories
    
    def get_action_trajectories(self, action_type: ActionType) -> List[JointTrajectory]:
        """Get trajectories based on action type"""
        if action_type == ActionType.WAVE:
            return self.get_wave_action()
        elif action_type == ActionType.EXPAND:
            return self.get_expand_action()
        elif action_type == ActionType.AGREE:
            return self.get_agree_action()
        else:
            return []
    
    def get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        return [action.value for action in ActionType]
