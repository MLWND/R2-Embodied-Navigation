#!/usr/bin/env python
"""加大手指尺寸/间距, 让 5 指清晰可见 (thumb 分开明显)"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

# 手指沿 x 排开位置 (间距 0.025)
FINGER_X = {
    'thumb_flex': -0.05, 'thumb_abduct': -0.05,
    'index': -0.025, 'middle': 0.0, 'ring': 0.025, 'pinky': 0.05,
}
# 手指 box 尺寸和 origin (link 坐标系, 向下伸出)
FINGER_GEOM = {
    'thumb_flex': (0.03, 0.018, 0.08, -0.04),
    'thumb_abduct': (0.03, 0.018, 0.06, -0.03),
    'index': (0.022, 0.016, 0.10, -0.05),
    'middle': (0.022, 0.016, 0.10, -0.05),
    'ring': (0.022, 0.016, 0.10, -0.05),
    'pinky': (0.018, 0.016, 0.09, -0.045),
}

# 1. joint origin x 位置
for j in root.findall('joint'):
    n = j.get('name')
    for key, x in FINGER_X.items():
        if n.startswith(key) and n.endswith(('_left_joint', '_right_joint')):
            o = j.find('origin')
            xyz = o.get('xyz').split()
            xyz[0] = str(x)
            o.set('xyz', ' '.join(xyz))

# 2. finger link box 尺寸 + origin
for l in root.findall('link'):
    n = l.get('name')
    for key, (sx, sy, sz, oz) in FINGER_GEOM.items():
        if n.startswith(key) and n.endswith(('_left_Link', '_right_Link')):
            for tag in ('visual', 'collision'):
                g = l.find(f'{tag}/geometry/box')
                if g is not None:
                    g.set('size', f'{sx} {sy} {sz}')
                o = l.find(f'{tag}/origin')
                if o is not None:
                    xyz = o.get('xyz').split()
                    xyz[2] = str(oz)
                    o.set('xyz', ' '.join(xyz))
            oi = l.find('inertial/origin')
            if oi is not None:
                xyz = oi.get('xyz').split()
                xyz[2] = str(oz)
                oi.set('xyz', ' '.join(xyz))
            print(f"{n}: box={sx}x{sy}x{sz}, origin z={oz}")

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('手指加大完成')
