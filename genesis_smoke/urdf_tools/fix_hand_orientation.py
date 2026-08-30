#!/usr/bin/env python
"""按 R1 真机修正灵巧手: 手指从手掌底部(-z)向下伸出, 拇指旋转绕 z 轴"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

FINGER_KEYS = ['thumb', 'index', 'middle', 'ring', 'pinky']

# 1. 手掌 box 加宽 (y 0.06 -> 0.1, 容纳 5 指排列)
for l in root.findall('link'):
    if l.get('name') in ('hand_left_Link', 'hand_right_Link'):
        for tag in ('visual', 'collision'):
            g = l.find(f'{tag}/geometry/box')
            if g is not None:
                size = g.get('size').split()
                size[1] = '0.1'
                g.set('size', ' '.join(size))
                print(f"{l.get('name')}.{tag} box -> {' '.join(size)}")

# 2. 手指关节: origin z 0.02 -> -0.14 (手掌底部), axis 修正
for j in root.findall('joint'):
    n = j.get('name')
    if any(k in n for k in FINGER_KEYS):
        o = j.find('origin')
        xyz = o.get('xyz').split()
        xyz[2] = '-0.14'
        o.set('xyz', ' '.join(xyz))
        ax = j.find('axis')
        if 'abduct' in n:
            ax.set('xyz', '0 0 1')   # 拇指旋转: 绕 z 轴
        else:
            ax.set('xyz', '0 -1 0')  # 手指弯曲: 朝下 -> 朝前
        print(f"{n}: origin z=-0.14, axis={ax.get('xyz')}")

# 3. 手指 link: box/inertial origin z 取负 (向下伸出)
for l in root.findall('link'):
    n = l.get('name')
    if any(k in n for k in FINGER_KEYS):
        for tag in ('visual', 'collision', 'inertial'):
            o = l.find(f'{tag}/origin')
            if o is not None:
                xyz = o.get('xyz').split()
                xyz[2] = str(-float(xyz[2]))
                o.set('xyz', ' '.join(xyz))
        print(f"{n}: origin z 取负完成")

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('灵巧手修正完成: 手指朝下')
