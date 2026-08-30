#!/usr/bin/env python
"""手指视觉多段化: 每指 3 段 box (近端/中间/远端), 段间小角度偏移像指节
保持 1 个弯曲关节 + 1 个碰撞 box, 不增加 DOF"""
import xml.etree.ElementTree as ET

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
tree = ET.parse(URDF)
root = tree.getroot()

# 每指 3 段: (近端box, 近端origin_z, 中间box, 中间origin_z, 中间rpy_y, 远端box, 远端origin_z, 远端rpy_y)
SEGMENTS = {
    'thumb_flex': ((0.03, 0.018, 0.03), -0.015, (0.027, 0.017, 0.03), -0.045, 0.15, (0.024, 0.016, 0.025), -0.07, 0.3),
    'index':      ((0.022, 0.016, 0.04), -0.02, (0.02, 0.015, 0.04), -0.06, 0.15, (0.018, 0.014, 0.03), -0.09, 0.3),
    'middle':     ((0.022, 0.016, 0.04), -0.02, (0.02, 0.015, 0.04), -0.06, 0.15, (0.018, 0.014, 0.03), -0.09, 0.3),
    'ring':       ((0.022, 0.016, 0.04), -0.02, (0.02, 0.015, 0.04), -0.06, 0.15, (0.018, 0.014, 0.03), -0.09, 0.3),
    'pinky':      ((0.018, 0.016, 0.035), -0.0175, (0.016, 0.015, 0.035), -0.0525, 0.15, (0.014, 0.014, 0.025), -0.08, 0.3),
}

def make_visual(box, oz, rpy_y=0.0):
    v = ET.SubElement(None, 'visual')  # placeholder, will re-parent
    o = ET.SubElement(v, 'origin')
    o.set('xyz', f'0 0 {oz}')
    o.set('rpy', f'0 {rpy_y} 0')
    g = ET.SubElement(v, 'geometry')
    b = ET.SubElement(g, 'box')
    b.set('size', f'{box[0]} {box[1]} {box[2]}')
    m = ET.SubElement(v, 'material')
    c = ET.SubElement(m, 'color')
    c.set('rgba', '0.72 0.78 0.72 1')
    return v

for l in root.findall('link'):
    n = l.get('name')
    for key, seg in SEGMENTS.items():
        if n.startswith(key) and n.endswith(('_left_Link', '_right_Link')):
            # 删除现有 visual
            for v in l.findall('visual'):
                l.remove(v)
            # 添加 3 段 visual
            b1, z1, b2, z2, r2, b3, z3, r3 = seg
            for box, oz, ry in [(b1, z1, 0.0), (b2, z2, r2), (b3, z3, r3)]:
                v = ET.Element('visual')
                o = ET.SubElement(v, 'origin')
                o.set('xyz', f'0 0 {oz}')
                o.set('rpy', f'0 {ry} 0')
                g = ET.SubElement(v, 'geometry')
                b = ET.SubElement(g, 'box')
                b.set('size', f'{box[0]} {box[1]} {box[2]}')
                m = ET.SubElement(v, 'material')
                m.set('name', 'mat_finger')
                c = ET.SubElement(m, 'color')
                c.set('rgba', '0.72 0.78 0.72 1')
                l.append(v)
            print(f"{n}: 3 段视觉完成")

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('手指视觉多段化完成')
