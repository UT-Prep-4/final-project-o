#Name(s): Killian Murphy
#Final Project - Build Something Worth Showing Off
'''
This is the big one. At the end of camp you will demo this project at the
SHOWCASE, and it should be good enough to put on a resume or mention in a
college application. That means it is not just "code that works." It is a
project you designed, built, polished, and can explain.

WHAT MAKES IT SHOWCASE-WORTHY (the autograder checks for these):
  1. ORGANIZED: your code is split into clear, purposeful segments (functions optional), not one
     giant blob. (Aim for at least 3-4 functions with real jobs.)
  2. SUBSTANTIAL: this is a multi-day build, bigger than the mini-project.
  3. REAL LOGIC: decisions (if/elif/else) and repetition (loops) working together.
  4. DOCUMENTED: fill out PROJECT.md so a stranger (or a college admissions
     reader!) can understand what you built and how to run it.

Whether it is impressive, creative, and demo-ready is judged by humans at
showcase, not by the autograder.

============================= PICK YOUR TRACK =================================

TRACK A: IMAGE PROCESSING PROGRAM
  Build a program that opens an image and transforms it with a special
  function you write yourself: brightness adjustment, a color filter overlay,
  grayscale, mirror, pixelate, or invent your own effect.
  The Pillow library is preinstalled. The core moves:

      from PIL import Image
      img = Image.open("photo.png")
      width, height = img.size
      pixel = img.getpixel((x, y))          # (red, green, blue), each 0-255
      img.putpixel((x, y), (r, g, b))       # set a pixel
      img.save("output.png")                # then click it in VS Code to view!

  Brightness is a for loop over every pixel that multiplies r, g, b by a
  factor the user chooses (careful: values must stay between 0 and 255).
  A filter overlay nudges every pixel toward a color (add red, drop blue...).
  Level up: ask the user which effect to apply with input(), show a menu,
  process any image file they name, draw the result with turtle or pygame.

TRACK B: ADVENTURE GAME
  Build a text adventure where the player explores, makes choices, and wins
  or loses based on decisions and luck. Use random for surprises: treasure,
  traps, enemy encounters, dice rolls, critical hits.
  The shape of it: one function per location or scene, input() for choices,
  an inventory list, health or gold as numbers, and random.randint() for
  the unexpected. Level up: turn-based combat, a map, multiple endings,
  ASCII art title screens, a save-your-score high score.

TRACK C: YOUR OWN IDEA
  A bigger game (pygame or turtle), a quiz app, a tool that solves a real
  problem you have, a simulation, generative turtle art... Pitch it to your
  instructor FIRST, then build it. The four requirements above still apply.

=============================== PLAN FIRST ====================================
Before you write code, fill this in (it will keep you honest all week):

  MY PROJECT: 3D aerodynamic (fluid) model using LBM
  THE PIECES I NEED TO BUILD: Setup dimensions, define fluid directions and weight (Lattice Boltzmann Method),
  
  WHAT I WILL DEMO AT SHOWCASE: (the 60-second version)

==============================================================================
Build your project below (and split it into more .py files if it gets big;
the grader reads all of them). Delete this line and start!
'''

import torch
import numpy as np
import pyvista as pv

# 1. Setup Stable Grid Dimensions & Flow Velocity
NX, NY, NZ = 60, 30, 30
STEPS = 300
tau = 0.8
init_velocity = 0.02
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define D3Q19 Lattice Directions and Weights
v_np = np.array([
    0, 0, 0,
    1, 0, 0,   -1, 0, 0,    0, 1, 0,    0, -1, 0,    0, 0, 1,    0, 0, -1,
    1, 1, 0,   -1, -1, 0,   1, -1, 0,   -1, 1, 0,
    1, 0, 1,   -1, 0, -1,   1, 0, -1,   -1, 0, 1,
    0, 1, 1,   0, -1, -1,   0, 1, -1,   0, -1, 1
]).reshape(19, 3)

w_np = np.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36
])

opposite = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]

v = torch.tensor(v_np, dtype=torch.float32, device=device)
w = torch.tensor(w_np, dtype=torch.float32, device=device)

# 3. Initialize populations (f) at equilibrium
f = torch.zeros((19, NX, NY, NZ), dtype=torch.float32, device=device)
for i in range(19):
    f[i] = w[i] * (1 + 3 * init_velocity * v[i, 0])

# 4. Create a 3D Obstacle (Sphere)
X, Y, Z = torch.meshgrid(torch.arange(NX), torch.arange(NY), torch.arange(NZ), indexing='ij')
cx, cy, cz, r = NX // 4, NY // 2, NZ // 2, NY // 5
obstacle = ((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) < r**2
obstacle = obstacle.to(device)

# 5. Main Simulation Loop
print(f"Running 3D fluid simulation on {device}...")
for step in range(STEPS):
    for i in range(19):
        f[i] = torch.roll(f[i], shifts=(int(v_np[i,0]), int(v_np[i,1]), int(v_np[i,2])), dims=(0, 1, 2))
        
    rho = torch.sum(f, dim=0)
    ux = torch.sum(f * v[:, 0, None, None, None], dim=0) / rho
    uy = torch.sum(f * v[:, 1, None, None, None], dim=0) / rho
    uz = torch.sum(f * v[:, 2, None, None, None], dim=0) / rho
    
    rho_inlet = rho[0, :, :]
    for i in range(19):
        vu = v[i, 0] * init_velocity
        u2 = init_velocity**2
        feq_inlet = w[i] * rho_inlet * (1 + 3*vu + 4.5*(vu**2) - 1.5*u2)
        f[i, 0, :, :] = feq_inlet

    rho_outlet = rho[-1, :, :]
    ux_outlet = ux[-1, :, :]
    uy_outlet = uy[-1, :, :]
    uz_outlet = uz[-1, :, :]
    for i in range(19):
        vu = v[i, 0] * ux_outlet + v[i, 1] * uy_outlet + v[i, 2] * uz_outlet
        u2 = ux_outlet**2 + uy_outlet**2 + uz_outlet**2
        feq_outlet = w[i] * rho_outlet * (1 + 3*vu + 4.5*(vu**2) - 1.5*u2)
        f[i, -1, :, :] = feq_outlet

    f_old = f.clone()
    for i in range(19):
        f[i, obstacle] = f_old[opposite[i], obstacle]
        
    rho = torch.sum(f, dim=0)
    ux = torch.sum(f * v[:, 0, None, None, None], dim=0) / rho
    uy = torch.sum(f * v[:, 1, None, None, None], dim=0) / rho
    uz = torch.sum(f * v[:, 2, None, None, None], dim=0) / rho
    
    for i in range(19):
        vu = v[i, 0] * ux + v[i, 1] * uy + v[i, 2] * uz
        u2 = ux**2 + uy**2 + uz**2
        feq = w[i] * rho * (1 + 3*vu + 4.5*(vu**2) - 1.5*u2)
        f[i] += -(1.0 / tau) * (f[i] - feq)

# 6. Post-Processing & PyVista Data Prep (Type-Casted for VTK Core Compatibility)
ux_3d = np.nan_to_num(ux.cpu().numpy(), nan=0.0)
uy_3d = np.nan_to_num(uy.cpu().numpy(), nan=0.0)
uz_3d = np.nan_to_num(uz.cpu().numpy(), nan=0.0)

raw_velocity_mag = np.sqrt(ux_3d**2 + uy_3d**2 + uz_3d**2)

conversion_factor = 1250.0 
velocity_mag = raw_velocity_mag * conversion_factor

ux_3d *= conversion_factor
uy_3d *= conversion_factor
uz_3d *= conversion_factor

obstacle_3d = obstacle.cpu().numpy().astype(np.uint8)

ux_f = np.asfortranarray(ux_3d).astype(np.float64)
uy_f = np.asfortranarray(uy_3d).astype(np.float64)
uz_f = np.asfortranarray(uz_3d).astype(np.float64)
vel_mag_f = np.asfortranarray(velocity_mag).astype(np.float64)
obstacle_f = np.asfortranarray(obstacle_3d)

x = np.arange(NX, dtype=np.float64)
y = np.arange(NY, dtype=np.float64)
z = np.arange(NZ, dtype=np.float64)

grid = pv.RectilinearGrid(x, y, z)
grid.point_data["Velocity Magnitude"] = vel_mag_f.flatten(order="F")
grid.point_data["Vectors"] = np.c_[ux_f.flatten(order="F"), 
                                   uy_f.flatten(order="F"), 
                                   uz_f.flatten(order="F")]

grid.set_active_scalars("Velocity Magnitude")
grid.set_active_vectors("Vectors")

obstacle_grid = pv.RectilinearGrid(x, y, z)
obstacle_grid.point_data["Obstacle"] = obstacle_f.flatten(order="F")
contour_obstacle = obstacle_grid.contour([0.5], scalars="Obstacle")

streamlines = grid.streamlines(
    vectors="Vectors",
    source_center=(0.5, float(NY // 2), float(NZ // 2)), 
    source_radius=float(NY // 4.0),    
    n_points=100,                                        
    max_length=150.0                                     
)

# 7. Interactive Graphics Rendering Canvas (Slice + Streamline Fusion)
plotter = pv.Plotter()

color_bar_styles = {
    "title": "Velocity Magnitude",
    "color": "white",
    "shadow": True,
    "font_family": "arial",
    "title_font_size": 16,
    "label_font_size": 14
}

plotter.add_mesh(contour_obstacle, color="#E0E0E0", opacity=1.0, label="Sphere Obstacle")

slice_plane = grid.slice(normal='z', origin=(NX // 4, NY // 2, NZ // 2))
plotter.add_mesh(
    slice_plane, 
    scalars="Velocity Magnitude", 
    cmap="turbo", 
    opacity=0.7,
    scalar_bar_args={
        "title": "Velocity (m/s)", 
        "color": "white",
        "shadow": True,
        "font_family": "arial",
        "title_font_size": 16,
        "label_font_size": 14
    }
)


if streamlines is not None and streamlines.n_points > 0:
    stream_tubes = streamlines.tube(radius=0.2)
    plotter.add_mesh(stream_tubes, color="white", opacity=0.9, label="Fluid Streamlines")
else:
    print("[!] Adjusting streamline source bounds to catch active flow...")

plotter.add_axes()

plotter.show_grid(color = "white")
plotter.set_background("#0B0C10")

print("Launching complete aerodynamic window!")
plotter.show()