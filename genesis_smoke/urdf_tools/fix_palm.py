#!/usr/bin/env python
"""手掌仿真修正: 手指微弯 + 拇指内收 + 手腕圆柱过渡"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

# 1. 手指段间角度加大 (真机静止弯曲 15-30°): 0.15/0.3 -> 0.3/0.5 rad
for l in root.findall('link'):
    n = l.get('name')
    if any(k in n for k in ['index', 'middle', 'ring', 'pinky', 'thumb_flex']) and n.endswith(('_left_Link', '_right_Link')):
        for v in l.findall('visual'):
            o = v.find('origin')
            if o is not None and o.get('rpy'):
                rpy = [float(x) for x in o.get('rpy').split()]
                if rpy[1] == 0.15:
                    rpy[1] = 0.3
                    o.set('rpy', ' '.join(str(x) for x in rpy))
                elif rpy[1] == 0.3:
                    rpy[1] = 0.5
                    o.set('rpy', ' '.join(str(x) for x in rpy))
        print(f"{n}: 段间角度加大完成")

# 2. 拇指内收: thumb_flex origin 加 rpy 旋转 (朝掌心)
for j in root.findall('joint'):
    n = j.get('name')
    if n.startswith('thumb_flex') and n.endswith(('_left_joint', '_right_joint')):
        o = j.find('origin')
        o.set('rpy', '-0.8 0 0')  # 绕 x 轴 -46°, 拇指朝掌心内收
        print(f"{n}: 拇指内收 rpy=-0.8")

# 3. 手腕圆柱过渡: hand link 加圆柱 visual
for l in root.findall('link'):
    n = l.get('name')
    if n in ('hand_left_Link', 'hand_right_Link'):
        v = ET.Element('visual')
        o = ET.SubElement(v, 'origin')
        o.set('xyz', '0 0 -0.02')
        o.set('rpy', '1.5708 0 0')  # 圆柱沿 z
        g = ET.SubElement(v, 'geometry')
        cyl = ET.SubElement(g, 'cylinder')
        cyl.set('radius', '0.03')
        cyl.set('length', '0.05')
        m = ET.SubElement(v, 'material')
        m.set('name', 'mat_wrist')
        c = ET.SubElement(m, 'color')
        c.set('rgba', '0.2 0.2 0.2 1')  # 深灰手腕
        l.append(v)
        print(f"{n}: 手腕圆柱添加")

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('手掌修正完成')
