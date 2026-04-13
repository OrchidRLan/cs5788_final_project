"""
Compute GT SDF values from a mesh file.
Usage: python data/scripts/compute_sdf.py \
           --mesh_path data/laukry.glb \
           --output data/cache/laukry_sdf.npy \
           --n_points 100000
"""
import argparse
import numpy as np
import trimesh
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_points", type=int, default=100000)
    p.add_argument("--bbox_scale", type=float, default=1.2)
    return p.parse_args()

def load_and_normalize_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    # Normalize to unit sphere
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(1.0 / scale)
    print(f"Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh

def compute_sdf(mesh, n_points, bbox_scale):
    try:
        from pysdf import SDF
        sdf_fn = SDF(mesh.vertices, mesh.faces)

        # Sample points: 70% near surface, 30% random in bbox
        n_surface = int(n_points * 0.7)
        n_random  = n_points - n_surface

        # Near-surface points (perturbed)
        surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
        noise = np.random.randn(*surface_pts.shape) * 0.02
        surface_pts = surface_pts + noise

        # Random bbox points
        random_pts = np.random.uniform(-bbox_scale, bbox_scale, (n_random, 3))

        points = np.vstack([surface_pts, random_pts]).astype(np.float32)
        sdf_vals = sdf_fn(points).astype(np.float32)

        return points, sdf_vals

    except ImportError:
        raise ImportError("pysdf not installed. Run: pip install pysdf")

def run(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mesh = load_and_normalize_mesh(args.mesh_path)
    points, sdf_vals = compute_sdf(mesh, args.n_points, args.bbox_scale)

    # Compute normals at surface points
    n_surface = int(args.n_points * 0.7)
    _, _, normals = trimesh.proximity.closest_point(mesh, points[:n_surface])

    np.save(args.output, {
        "points": points,
        "sdf":    sdf_vals,
        "normals_surface_idx": n_surface,
        "normals": normals.astype(np.float32)
    })
    print(f"Saved SDF cache: {args.output}")
    print(f"  Points: {len(points)}, SDF range: [{sdf_vals.min():.4f}, {sdf_vals.max():.4f}]")

if __name__ == "__main__":
    run(parse_args())
