#!/usr/bin/env python
"""补充 6 指手爪到 R2 URDF (参考 BrainCo Revo2)"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

# 先删测试关节
for j in root.findall('joint'):
    if j.get('name') == 'test_joint':
        root.remove(j)

FINGERS = [
    ('thumb_flex',   0.0, 1.03, 0.02, 0.01, 0.06),
    ('thumb_abduct', 0.0, 1.57, 0.02, 0.01, 0.05),
    ('index',        0.0, 1.41, 0.015, 0.01, 0.08),
    ('middle',       0.0, 1.41, 0.015, 0.01, 0.085),
    ('ring',         0.0, 1.41, 0.015, 0.01, 0.08),
    ('pinky',        0.0, 1.41, 0.012, 0.01, 0.07),
]

def add_fingers(side):
    palm = f'hand_{side}_Link'
    for i, (fname, lo, up, w, h, ln) in enumerate(FINGERS):
        jname = f'{fname}_{side}_joint'
        lname = f'{fname}_{side}_Link'
        j = ET.SubElement(root, 'joint')
        j.set('name', jname); j.set('type', 'revolute')
        ET.SubElement(j, 'origin').set('xyz', f'0 {0.02*(i-2.5)} 0.02')
        ET.SubElement(j, 'parent').set('link', palm)
        ET.SubElement(j, 'child').set('link', lname)
        ET.SubElement(j, 'axis').set('xyz', '0 1 0')
        lim = ET.SubElement(j, 'limit')
        lim.set('lower', str(lo)); lim.set('upper', str(up))
        lim.set('effort', '10'); lim.set('velocity', '5')
        l = ET.SubElement(root, 'link')
        l.set('name', lname)
        inert = ET.SubElement(l, 'inertial')
        ET.SubElement(inert, 'origin').set('xyz', f'0 0 {ln/2}')
        ET.SubElement(inert, 'mass').set('value', '0.05')
        ixx = ET.SubElement(inert, 'inertia')
        ixx.set('ixx', '1e-5'); ixx.set('iyy', '1e-5'); ixx.set('izz', '1e-5')
        ixx.set('ixy', '0'); ixx.set('ixz', '0'); ixx.set('iyz', '0')
        vis = ET.SubElement(l, 'visual')
        ET.SubElement(vis, 'origin').set('xyz', f'0 0 {ln/2}')
        geom = ET.SubElement(vis, 'geometry')
        box = ET.SubElement(geom, 'box')
        box.set('size', f'{w} {h} {ln}')
        mat = ET.SubElement(vis, 'material')
        ET.SubElement(mat, 'color').set('rgba', '0.72 0.78 0.72 1')
        col = ET.SubElement(l, 'collision')
        ET.SubElement(col, 'origin').set('xyz', f'0 0 {ln/2}')
        cgeom = ET.SubElement(col, 'geometry')
        cbox = ET.SubElement(cgeom, 'box')
        cbox.set('size', f'{w} {h} {ln}')

add_fingers('left')
add_fingers('right')
tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('已补充 6 指手爪 (左右各 6 关节)')
