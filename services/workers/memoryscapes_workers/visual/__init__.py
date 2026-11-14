"""Visual pipeline helpers (video I/O, keyframes, SfM, alignment, splat I/O, training).

Modules that need heavy dependencies (cv2, pycolmap, torch, gsplat) import them lazily,
so the light modules (alignment, splat_io) work with numpy only.
"""
