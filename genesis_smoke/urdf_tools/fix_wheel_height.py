#!/usr/bin/env python
"""修复驱动轮高度: 0.075 -> 0.085 (轮底从 -0.01 插地归零, 与 caster 同高)"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

for j in root.findall('joint'):
    if j.get('name') in ('wheel_left_joint', 'wheel_right_joint'):
        o = j.find('origin')
        xyz = o.get('xyz').split()
        xyz[2] = '0.085'
        o.set('xyz', ' '.join(xyz))
        print(f"{j.get('name')}: origin z -> {xyz[2]}")

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('驱动轮 z: 0.075 -> 0.085 完成')
