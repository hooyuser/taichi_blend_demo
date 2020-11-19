import bpy
import bmesh
import taichi as ti
import numblend as nb
import numpy as np
from math import sqrt

nb.init()
ti.init(arch=ti.cpu, debug=False)


# Get the active mesh
obj = bpy.data.objects["Plane"]
me = obj.data

# Get a BMesh representation
bm = bmesh.new()   # create an empty BMesh
bm.from_mesh(me)   # fill it in from a Mesh
vertex_num = len(bm.verts)
edge_num = len(bm.edges)
face_num = len(bm.faces)
#link_num = edge_num
link_num = edge_num + face_num * 2
substep_num = 15
solver_num = 30
dt = 1e-3
drag_damping = 1
#k = 0.9
coll_r = 1.01


vgroup = obj.vertex_groups['pin']
c_obj = bpy.data.objects["Sphere"]


x = ti.Vector.field(3, dtype=ti.f32, shape=vertex_num)
p = ti.Vector.field(3, dtype=ti.f32, shape=vertex_num)
v = ti.Vector.field(3, dtype=ti.f32, shape=vertex_num)
w = ti.field(dtype=ti.f32, shape=vertex_num)
link = ti.Vector.field(2, dtype=ti.i32, shape=link_num)
link_len = ti.field(dtype=ti.f32, shape=link_num)
link_k = ti.field(dtype=ti.f32, shape=link_num)
link_idx = ti.field(dtype=ti.i32, shape=())
coll_origin = ti.Vector.field(3, dtype=ti.f32, shape=1)


def calc_dist(first, second):

    locx = second[0] - first[0]
    locy = second[1] - first[1]
    locz = second[2] - first[2]

    distance = sqrt((locx)**2 + (locy)**2 + (locz)**2) 
    return distance


def set_faces():   
    for i in range(face_num):
        v = bm.faces[i].verts
        
        for j in ti.static(range(2)):
            link[link_idx[None]] = ti.Vector([v[0+j].index, v[1+j].index])
            link_len[link_idx[None]] = calc_dist(v[0+j].co, v[1+j].co)
            link_k[link_idx[None]] = 0.7
            link_idx[None] += 1
        
        
def set_edges():
    for i in range(edge_num):
        e = bm.edges[link_idx[None]]
        link[link_idx[None]] = ti.Vector([e.verts[0].index, e.verts[1].index])
        link_len[link_idx[None]] = e.calc_length()
        link_k[link_idx[None]] = 0.9
        link_idx[None] += 1


def init():
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for i in range(vertex_num):
        x[i] = ti.Vector(list(bm.verts[i].co))
        w[i] = 1
        for group in me.vertices[i].groups:
            w[i] = 0  # vgroup.weight(i)

    link_idx[None] = 0
    set_edges()
    set_faces()
    # print(link_num,link_idx[None])

#@ti.func
#def sdf(px:ti.f32,py:ti.f32,pz:ti.f32,ox:ti.f32,oy:ti.f32,oz:ti.f32,r:ti.f32) -> ti.f32:
#    return ti.sqrt((px-ox)**2+(py-oy)**2+(pz-oz)**2)-r
    

@ti.kernel
def substep():
    for i in range(vertex_num):
        v[i] += dt * ti.Vector([0, 0, -9.8]) * w[i]
        v[i] *= ti.exp(-dt * drag_damping)
        p[i] = x[i] + dt * v[i]

    for n in range(solver_num):
        for l in range(link_num):
            p0 = link[l][0]
            p1 = link[l][1]
            p_01 = p[p0] - p[p1]
            
            kp = 1 - pow((1 - link_k[l]), 1 / (n + 1))
            
            length = link_len[l]
            dp0 = -0.5 * w[p0] * (p_01.norm() - length) * p_01.normalized()
            p[link[l][0]] += kp * dp0

            dp1 = 0.5 * w[p1] * (p_01.norm() - length) * p_01.normalized()
            p[link[l][1]] += kp * dp1
            
        for vi in range(vertex_num):
            dist = ti.sqrt((p[vi].x-coll_origin[0].x)**2+(p[vi].y-coll_origin[0].y)**2+(p[vi].z-coll_origin[0].z)**2)
            if w[vi] != 0 and dist - coll_r < 0:
                dp = - (dist - coll_r) / dist * (p[vi] - coll_origin[0])
                p[vi] += dp
                
            
    for i in range(vertex_num):
        # print('p[',i,']:', p[i])
        v[i] = (p[i] - x[i]) / dt
        x[i] = p[i]
        

init()

@nb.add_animation
def main():  
    for frame in range(1, 2500):
        yield nb.mesh_update(me, x.to_numpy().reshape(vertex_num,3))   
        #s = 1
        for step in range(substep_num):
            #print('frame:',frame,', substep:',s)
            #s += 1
            coll_origin[0].x = c_obj.location[0]
            coll_origin[0].y = c_obj.location[1]
            coll_origin[0].z = c_obj.location[2]
            substep()

