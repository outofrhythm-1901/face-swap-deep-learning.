"""Landmark-based face swap demo. Not a custom trained CNN/GAN model."""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# FaceMesh landmark subset around the face outline and inner features.
LANDMARK_IDS = sorted(set([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109, 151, 108, 69, 104, 68, 71, 139,
    34, 227, 137, 177, 215, 116, 111, 117, 118, 119, 120, 121, 128, 245,
    188, 174, 196, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14,
    17, 18, 200, 199, 175, 267, 269, 270, 409, 291, 375, 321, 405, 314,
    84, 181, 91, 146, 61, 185, 40, 39, 37
]))


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}. Check the path and format.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a color image: {path}")
    return image


def get_landmarks(image):
    h, w = image.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:
        result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        raise ValueError("No face detected. Try a clear, front-facing portrait with good lighting.")
    # Choose largest detected face.
    faces = []
    for face in result.multi_face_landmarks:
        pts = np.array([(p.x * w, p.y * h) for p in face.landmark], dtype=np.float32)
        extent = np.ptp(pts, axis=0)
        faces.append((float(extent[0] * extent[1]), pts))
    points = max(faces, key=lambda item: item[0])[1][LANDMARK_IDS]
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    return points


def warp_triangle(src, dst, src_tri, dst_tri):
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))
    sx, sy, sw, sh = sr
    dx, dy, dw, dh = dr
    if min(sw, sh, dw, dh) <= 0:
        return
    crop = src[sy:sy + sh, sx:sx + sw]
    if crop.size == 0:
        return
    src_local = np.float32([[p[0] - sx, p[1] - sy] for p in src_tri])
    dst_local = np.float32([[p[0] - dx, p[1] - dy] for p in dst_tri])
    matrix = cv2.getAffineTransform(src_local, dst_local)
    warped = cv2.warpAffine(crop, matrix, (dw, dh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((dh, dw, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_local), (1, 1, 1), 16, 0)
    region = dst[dy:dy + dh, dx:dx + dw]
    if region.shape == warped.shape:
        dst[dy:dy + dh, dx:dx + dw] = region * (1 - mask) + warped * mask


def face_swap(source, target):
    src_points, dst_points = get_landmarks(source), get_landmarks(target)
    h, w = target.shape[:2]
    warped = np.zeros_like(target, dtype=np.float32)

    # Build a Delaunay triangulation over target landmarks.
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for x, y in dst_points:
        try:
            subdiv.insert((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))
        except cv2.error:
            pass  # near-duplicate points can occur
    triangles = subdiv.getTriangleList()

    def nearest_index(x, y):
        d = np.sum((dst_points - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
        i = int(np.argmin(d))
        return i if float(d[i]) <= 4.0 else None

    for tri in triangles:
        idx = [nearest_index(tri[i], tri[i + 1]) for i in (0, 2, 4)]
        if any(i is None for i in idx) or len(set(idx)) != 3:
            continue
        dt, st = dst_points[idx], src_points[idx]
        if (np.any(st[:, 0] < 0) or np.any(st[:, 0] >= source.shape[1]) or
                np.any(st[:, 1] < 0) or np.any(st[:, 1] >= source.shape[0])):
            continue
        warp_triangle(source, warped, st, dt)

    hull = cv2.convexHull(np.int32(dst_points))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    x, y, bw, bh = cv2.boundingRect(hull)
    if bw < 2 or bh < 2:
        raise ValueError("Could not build a valid face mask. Use larger portrait photos.")
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    try:
        return cv2.seamlessClone(warped, target, mask, (x + bw // 2, y + bh // 2), cv2.NORMAL_CLONE)
    except cv2.error as exc:
        raise ValueError("Blending failed. Try larger images with similar face angles and lighting.") from exc


def main():
    parser = argparse.ArgumentParser(description="Swap a face between two consented portrait images.")
    parser.add_argument("--source", required=True, help="Image containing the face to transfer")
    parser.add_argument("--target", required=True, help="Image receiving the face")
    parser.add_argument("--output", default="swapped.jpg", help="Output path (default: swapped.jpg)")
    args = parser.parse_args()
    try:
        result = face_swap(load_image(args.source), load_image(args.target))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), result):
            raise ValueError(f"Could not write output file: {out}")
        print(f"Face swap saved to: {out.resolve()}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
