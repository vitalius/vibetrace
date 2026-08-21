import numpy as np

# ---------- Scene setup ----------
WIDTH, HEIGHT = 800, 600
FOV = 60.0  # vertical field of view in degrees

cam_pos = np.array([0.0, 0.0, 3.0])          # camera position
sphere_center = np.array([0.0, 0.0, 0.0])    # sphere in the middle of the viewport
sphere_radius = 1.0

light_dir = np.array([0.5, 0.8, 0.6])        # direction *toward* the light
light_dir = light_dir / np.linalg.norm(light_dir)

sphere_color = np.array([0.9, 0.35, 0.3])


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ray_sphere_intersect(origins, directions, center, radius):
    """Vectorized ray-sphere test. Returns (hit_mask, t)."""
    oc = origins - center
    a = np.einsum("ij,ij->i", directions, directions)
    b = 2.0 * np.einsum("ij,ij->i", oc, directions)
    c = np.einsum("ij,ij->i", oc, oc) - radius * radius

    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

    t_near = (-b - sqrt_disc) / (2.0 * a)
    t_far = (-b + sqrt_disc) / (2.0 * a)
    # If the near root is behind the camera, fall back to the far root
    t = np.where((t_near < 0) & (t_far > 0), t_far, t_near)

    hit = (disc > 0) & (t > 0)
    return hit, t


# ---------- Generate one ray per pixel ----------
px = (np.arange(WIDTH) + 0.5) / WIDTH
py = (np.arange(HEIGHT) + 0.5) / HEIGHT
x, y = np.meshgrid(px, py)  # each (H, W)

aspect = WIDTH / HEIGHT
scale = np.tan(np.radians(FOV) / 2.0)

dirs = np.stack([
    (2.0 * x - 1.0) * aspect * scale,   # x offset on image plane
    (1.0 - 2.0 * y) * scale,            # y offset (flipped so +y is up)
    -np.ones_like(x),                   # forward (-z)
], axis=-1)
dirs = normalize(dirs)

origins = np.empty_like(dirs)
origins[:] = cam_pos

# ---------- Trace ----------
hit, t = ray_sphere_intersect(origins.reshape(-1, 3), dirs.reshape(-1, 3),
                              sphere_center, sphere_radius)
hit = hit.reshape(HEIGHT, WIDTH)
t = t.reshape(HEIGHT, WIDTH)

# Background: simple vertical gradient
top = np.array([0.25, 0.45, 0.85])
bottom = np.array([0.85, 0.90, 0.95])
gradient = top[None, None, :] + (bottom - top)[None, None, :] * \
    np.linspace(0, 1, HEIGHT)[:, None, None]
img = np.repeat(gradient, WIDTH, axis=1)

# ---------- Shade the sphere ----------
if hit.any():
    hit_pts = origins[hit] + t[hit][:, None] * dirs[hit]
    normals = normalize(hit_pts - sphere_center)

    # Ambient + diffuse
    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)

    # Specular (Blinn-Phong)
    view = -normalize(dirs[hit])
    half = normalize(light_dir[None, :] + view)
    spec = np.clip(np.einsum("ij,ij->i", normals, half), 0.0, 1.0) ** 32.0

    color = (0.15 * sphere_color + diffuse[:, None] * sphere_color
             + spec[:, None] * np.array([1.0, 1.0, 1.0]))
    img[hit] = np.clip(color, 0.0, 1.0)

# ---------- Save as PPM (no PIL required) ----------
out = "sphere.ppm"
with open(out, "wb") as f:
    f.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode())
    f.write((img * 255).astype(np.uint8).tobytes())
print(f"Saved {out}")

# Optional: display with matplotlib if you have it
# import matplotlib.pyplot as plt
# plt.imshow(img); plt.axis("off"); plt.show()

