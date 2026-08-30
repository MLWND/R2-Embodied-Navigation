#!/usr/bin/env python
"""修正手指排列: y 排开 -> x 排开 (拇指在手掌一侧), 手掌朝内"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

# 手指沿 x 排开的位置 (y=0, 拇指在 -x 侧)
FINGER_X = {
    'thumb_flex': -0.04, 'thumb_abduct': -0.04,
    'index': -0.02, 'middle': 0.0, 'ring': 0.02, 'pinky': 0.04,
}

# 1. 手掌 box: y 0.1 -> 0.06 (手指沿 x 排开, 手掌深 0.06)
for l in root.findall('link'):
    if l.get('name') in ('hand_left_Link', 'hand_right_Link'):
        for tag in ('visual', 'collision'):
            g = l.find(f'{tag}/geometry/box')
            if g is not None:
                size = g.get('size').split()
                size[1] = '0.06'
                g.set('size', ' '.join(size))
                print(f"{l.get('name')}.{tag} box -> {' '.join(size)}")

# 2. 手指 joint origin: (0, y, -0.14) -> (x, 0, -0.14)
for j in root.findall('joint'):
    n = j.get('name')
    for key, x in FINGER_X.items():
        if n.startswith(key) and n.endswith(('_left_joint', '_right_joint')):
            o = j.find('origin')
            xyz = o.get('xyz').split()
            xyz[0] = str(x)
            xyz[1] = '0'
            xyz[2] = '-0.14'
            o.set('xyz', ' '.join(xyz))
            print(f"{n}: origin=({xyz[0]}, 0, -0.14)")

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('手指排列修正: 沿 x 排开, 手掌朝内')
