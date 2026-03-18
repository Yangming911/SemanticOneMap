"""
Diagnostic: for GT obstacle cells, compute argmax GCLIP label prediction.
Checks whether stored map features can correctly classify obstacle cells.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

if "-c" not in sys.argv:
    sys.argv += ["-c", "config/mon/eval_conf_gclip_vis.yaml"]

from config import load_eval_config
from eval.actor import MONActor
from eval.habitat_evaluator import HabitatEvaluator
from visualize_single_scene import resolve_eval_paths
import habitat_sim

cfg = load_eval_config()
resolve_eval_paths(cfg.EvalConf)
evaluator = HabitatEvaluator(cfg.EvalConf, MONActor(cfg.EvalConf))

OBSTACLE_LABELS = ["chair", "potted plant", "toilet", "sofa", "bed", "tv monitor"]

for ep_id in [6, 7, 9, 16]:  # chair episodes
    episode = [ep for ep in evaluator.episodes if ep.episode_id == ep_id]
    if not episode:
        continue
    episode = episode[0]

    evaluator.load_scene(episode.scene_id)
    evaluator.sim.initialize_agent(
        0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
    evaluator.actor.reset()
    evaluator.actor.set_query(episode.obj_sequence[0])
    gt_cd = evaluator.get_semantic_collision_data(episode.scene_id, episode.obj_sequence[0])
    evaluator.actor.mapper.set_gt_label_map(gt_cd.label_map, gt_cd.labels)

    # Run full episode
    mapper = evaluator.actor.mapper
    for step in range(500):
        observations = evaluator.sim.get_sensor_observations()
        observations["state"] = evaluator.sim.get_agent(0).get_state()
        action, _ = evaluator.actor.act(observations)
        evaluator.execute_action(action)

    # Evaluate: for GT obstacle cells with GCLIP features, check argmax label
    gclip_feat = mapper.one_map.feature_map_gclip     # [n,n,512]
    gclip_conf = mapper.one_map.confidence_map_gclip  # [n,n]
    gclip_model = mapper.gclip_model

    text_feats = gclip_model.get_text_features(
        [f"a {l}" for l in OBSTACLE_LABELS]
    )  # [6, 512]

    label_map = gt_cd.label_map  # [n,n], 0=bg, 1+=obstacle
    labels = gt_cd.labels

    n_cells = mapper.one_map.n_cells
    observed = gclip_conf > 0

    print(f"\n=== Episode {ep_id} | query={episode.obj_sequence[0]} ===")
    print(f"  GT labels in scene: {labels}")
    print(f"  GT obstacle cells: {(label_map > 0).sum()}")
    print(f"  Observed cells: {observed.sum().item():.0f}")

    # For each GT obstacle cell that has been observed
    for lbl_idx, lbl in enumerate(labels):
        gt_mask = (label_map == lbl_idx + 1)
        gt_and_obs = torch.tensor(gt_mask).to(observed.device) & observed
        n = int(gt_and_obs.sum().item())
        if n == 0:
            print(f"  {lbl}: 0 GT cells observed — skipped")
            continue

        feats = gclip_feat[gt_and_obs]  # [n, 512]
        feats_norm = F.normalize(feats.float(), dim=1)
        sims = feats_norm @ text_feats.T.to(feats_norm.device)  # [n, 6]
        pred_idx = sims.argmax(dim=1).cpu().numpy()
        correct_idx = OBSTACLE_LABELS.index(lbl) if lbl in OBSTACLE_LABELS else -1

        if correct_idx < 0:
            print(f"  {lbl}: not in OBSTACLE_LABELS list")
            continue

        acc = float((pred_idx == correct_idx).mean())
        sim_to_true = sims[:, correct_idx].cpu().numpy()
        sim_to_others = sims.cpu().numpy()
        sim_to_others = np.delete(sim_to_others, correct_idx, axis=1)

        print(f"  {lbl}: n={n}, acc={acc:.2%}, "
              f"sim_true={sim_to_true.mean():.4f}±{sim_to_true.std():.4f}, "
              f"sim_others_max={sim_to_others.max(axis=1).mean():.4f}")

    evaluator.sim.close()
