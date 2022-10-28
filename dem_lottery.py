import taichi as ti
import math
import os
import random

ti.init(arch=ti.cpu)
vec = ti.math.vec2

SAVE_FRAMES = True

window_size = 1024  # Number of pixels of the window
n = 33  # Number of grains

density = 100.0
stiffness = 8e3
restitution_coef = 0.001
gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60
maxbf=3000
bf=maxbf
time=0
pt=1


@ti.dataclass
class Grain:
    n: int # Name
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force


gf = Grain.field(shape=(n, ))

grid_n = 8
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.02
grain_r_max = 0.02

cen=vec(0.5, 0.6)
rcb=0.3   # radius of boundary circle
bounce_coef = 0.5  # Velocity damping
elas=100000

assert grain_r_max * 2 < grid_size

@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        l = i * grid_size
        padding = 0.1
        region_width = 1.0 - padding * 2
        pos = vec(l % region_width + padding + grid_size * ti.random() * 0.2,
                  l // region_width * grid_size + 0.3)
        gf[i].n = i+1
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].m = density * math.pi * gf[i].r**2


@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a


@ti.kernel
def apply_bc_old():
    for i in gf:
        rel_pos = gf[i].p - cen
        dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        normal = rel_pos  / dist

        overdis = dist+gf[i].r - rcb # how much grain outside bound
        if overdis > 0: 
#            gf[i].p = cen + normal*(rcb-2*gf[i].r) 
            gf[i].f += -normal*overdis*elas 
            a = gf[i].f / gf[i].m
            gf[i].v += (gf[i].a + a) * dt / 2.0
            gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
            gf[i].a = a
            
@ti.kernel
def apply_bc():
    for i in gf:
        rel_pos = gf[i].p - cen
        dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        normal = rel_pos  / dist

        overdis = dist+gf[i].r - rcb # how much grain outside bound
        if overdis > 0: 
#            gf[i].p = cen + normal*(rcb-2*gf[i].r) 
            gf[i].f += -normal*overdis*elas 
            a = gf[i].f / gf[i].m
            gf[i].v += (gf[i].a + a) * dt / 2.0
            gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
            gf[i].a = a
            
        

@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2 - f1
        gf[j].f -= f2 - f1


list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.kernel
def apply_force(pt: int):
    for i in gf:
      if pt%2 == 1:
         bf = gravity
         gf[i].f = vec(0., bf* gf[i].m)  
      else:
         rn = 2 * ti.random() - 1 
         bf = rn*maxbf
         gf[i].f = vec(0., bf* gf[i].m)  
  
@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
 
    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                   list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i < j:
                        resolve(i, j)


init()
gui = ti.GUI('Taichi DEM', (window_size, window_size))
step = 0
tol=1e-1

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while gui.running:
    gui.circle(cen.to_numpy(), radius=(rcb+0.05)*window_size, color=0x068587)
    gui.circle(cen.to_numpy(), radius=rcb*window_size, color=0xFFFFFF)
    
    gui.rect((0.2,0.1), (0.8,0.115), radius=0.03*window_size, color=0xED5538)
    gui.text('Press SPACE to switch between play and draw.',(0.19,0.12), font_size=35, color=0xFFFFFF)

    gui.text('Lucky 6 of 33:',(0.19,0.22), font_size=35, color=0xFFFFFF)
        

  
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.SPACE:
            pt += 1

    if pt%2 == 0 :
       substeps = 60
       sort_id = [i+2 for i in range(n)]
    
    for s in range(substeps):
        update()
        apply_bc()
        apply_force(pt)
        contact(gf)
        if pt%2 == 1:
            if gf[0].v[1]**2<tol:
               gy = [gf[i].p[1] for i in range(n)]
               li=[]
               for i in range(len(gy)):
                   li.append([gy[i],i])
               li.sort()
               sort_id = [ y[1] for y in li ]
               substeps=0
               break;   

    if substeps == 0:
       for i in range(6):
           gui.circle((0.42+i*0.06, 0.2), radius=0.02*window_size, color=0xFF0000)
           gui.text(str(sort_id[i]+1), (0.42+i*0.06-0.012, 0.2+0.012), font_size=25)

    pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    gui.circles(pos, radius=r,color=0xFF0000)
    for i in range(n):
        gui.text(str(i+1), gf[i].p.to_numpy()-(0.012,-0.012), font_size=25)
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()
    step += 1
