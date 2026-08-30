# 🤖 R2 Humanoid Robot Embodied AI Navigation System

An end-to-end embodied visual navigation, 3D semantic mapping, and reinforcement learning motion control framework for the **R2 wheeled humanoid robot** in photorealistic 3D indoor environments (GRScenes & SAPIEN).

---

## 🌟 Key Architecture & Highlights

1. **Layer 1: Global Semantic Topology Planning (SayPlan / Hydra paradigm)**
   - Room-to-room topological connectivity routing and doorway waypoint generation via [`maniskill_r2/scene_semantic_graph.py`](maniskill_r2/scene_semantic_graph.py).
2. **Layer 2: Zero-Shot VLM 3D Grounding (L3MVN / ConceptGraphs paradigm)**
   - Open-vocabulary object detection and camera 3D back-projection via [`maniskill_r2/vlm_active_search.py`](maniskill_r2/vlm_active_search.py).
3. **Layer 3: Local Reactive Obstacle Avoidance (PPO + 48D LiDAR + APF)**
   - High-frequency planar non-holonomic controller with Artificial Potential Field (APF) doorway centering and trained PPO policy network via [`maniskill_r2/ppo_nav_agent.py`](maniskill_r2/ppo_nav_agent.py).
4. **100% Real 3D CAD Mesh Environment Simulation**
   - High-fidelity USD mesh extraction and PBR material rendering in SAPIEN 3.0.3 Vulkan engine via [`maniskill_r2/render_3d_topdown_navigation.py`](maniskill_r2/render_3d_topdown_navigation.py).

---

## 📂 Repository Structure

- `maniskill_r2/`: Core embodied AI navigation pipeline (VLM, PPO agent, semantic graphs, simulators, streamer).
- `InternUtopia/`: R2 humanoid robot URDF definitions, kinematic models, sensor configurations, and controllers.
- `robot_example1_EN_demo/`: R2 real-robot ROS 2 driver and hardware executor reference implementation.
- `genesis_smoke/`: Physical simulation smoke tests and diagnostics.
- `doc/`: Milestone technical handover documentation and stage reports.
- `output/`: Benchmark evaluation metrics, zero-oracle grounding JSONs, and HD render results.
- `交接文档.md`: Detailed engineering handover documentation.
