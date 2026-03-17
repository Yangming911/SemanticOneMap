from typing import Optional
from spock import spock

@spock
class MappingConf:
    n_points: int
    size: int
    agent_radius: float
    blur_kernel_size: float
    obstacle_map_threshold: float
    fully_explored_threshold: float
    checked_map_threshold: float

    depth_factor: float
    gradient_factor: float

    optimal_object_distance: float
    optimal_object_factor: float

    obstacle_min: float
    obstacle_max: float

    filter_stairs: bool
    floor_level: float
    floor_threshold: float
    use_clip_semantic_nav_map: bool
    clip_semantic_sim_threshold: float
    use_yolo_obstacle_map: bool
    yolo_window_size: int
    use_clip_cp_obstacle_map: bool
    clip_cp_threshold: float
    clip_cp_use_oacp: bool
    clip_cp_target_coverage: float
    clip_cp_window_size: int
    use_yolo_cp_obstacle_map: Optional[bool] = False
    yolo_cp_threshold: Optional[float] = 0.5
    yolo_cp_target_coverage: Optional[float] = 0.9
    yolo_cp_window_size: Optional[int] = 200
    yolo_cp_frustum_max_depth: Optional[float] = 5.0

