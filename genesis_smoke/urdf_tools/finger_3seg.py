#!/usr/bin/env python
"""参考 GR1 inspire_hand: 每指 3 段 (proximal/intermediate/distal), intermediate/distal 用 mimic 联动
高层控制保持 R1 的 6 位置量: thumb_flex, thumb_abduct, index, middle, ring, pinky"""
import xml.etree.ElementTree as ET
import shutil

URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
shutil.copy(URDF, URDF + '.bak_palm3seg')
tree = ET.parse(URDF)
root = tree.getroot()

def make_link(name, box, oz, color='0.75 0.75 0.75 1'):
    l = ET.Element('link')
    l.set('name', name)
    inert = ET.SubElement(l, 'inertial')
    io = ET.SubElement(inert, 'origin')
    io.set('xyz', f'0 0 {oz}')
    m = ET.SubElement(inert, 'mass')
    m.set('value', '0.02')
    ii = ET.SubElement(inert, 'inertia')
    ii.set('ixx', '1e-5'); ii.set('iyy', '1e-5'); ii.set('izz', '1e-5')
    ii.set('ixy', '0'); ii.set('ixz', '0'); ii.set('iyz', '0')
    for tag in ('visual', 'collision'):
        v = ET.SubElement(l, tag)
        o = ET.SubElement(v, 'origin')
        o.set('xyz', f'0 0 {oz}')
        g = ET.SubElement(v, 'geometry')
        b = ET.SubElement(g, 'box')
        b.set('size', f'{box[0]} {box[1]} {box[2]}')
        if tag == 'visual':
            mat = ET.SubElement(v, 'material')
            mat.set('name', f'mat_{name}')
            c = ET.SubElement(mat, 'color')
            c.set('rgba', color)
    return l

def make_joint(name, parent, child, origin, axis, mimic=None, limit=(0.0, 1.41)):
    j = ET.Element('joint')
    j.set('name', name)
    j.set('type', 'revolute')
    o = ET.SubElement(j, 'origin')
    o.set('xyz', origin)
    p = ET.SubElement(j, 'parent')
    p.set('link', parent)
    c = ET.SubElement(j, 'child')
    c.set('link', child)
    a = ET.SubElement(j, 'axis')
    a.set('xyz', axis)
    lim = ET.SubElement(j, 'limit')
    lim.set('lower', str(limit[0])); lim.set('upper', str(limit[1]))
    lim.set('effort', '10'); lim.set('velocity', '5')
    if mimic:
        mc = ET.SubElement(j, 'mimic')
        mc.set('joint', mimic)
        mc.set('multiplier', '1')
        mc.set('offset', '0')
    return j

# 4 指: 每指 3 段 (proximal 控制, intermediate/distal mimic)
FINGERS = ['index', 'middle', 'ring', 'pinky']
FINGER_X = {'index': -0.025, 'middle': 0.0, 'ring': 0.025, 'pinky': 0.05}
# 段几何: (proximal box, intermediate box, distal box)
SEG = {
    'index': ((0.022, 0.016, 0.04), (0.02, 0.015, 0.04), (0.018, 0.014, 0.03)),
    'middle': ((0.022, 0.016, 0.04), (0.02, 0.015, 0.04), (0.018, 0.014, 0.03)),
    'ring': ((0.022, 0.016, 0.04), (0.02, 0.015, 0.04), (0.018, 0.014, 0.03)),
    'pinky': ((0.018, 0.016, 0.035), (0.016, 0.015, 0.035), (0.014, 0.014, 0.025)),
}

for side in ['left', 'right']:
    for f in FINGERS:
        ctrl = f'{f}_{side}_joint'
        # 1. 控制关节的 child link 改名 + 改几何 (近端段)
        for l in root.findall('link'):
            if l.get('name') == f'{f}_{side}_Link':
                l.set('name', f'{f}_proximal_{side}_Link')
                # 删除所有 visual/collision, 重建近端段
                for v in l.findall('visual'):
                    l.remove(v)
                for v in l.findall('collision'):
                    l.remove(v)
                pb = SEG[f][0]
                for tag in ('visual', 'collision'):
                    v = ET.SubElement(l, tag)
                    o = ET.SubElement(v, 'origin')
                    o.set('xyz', f'0 0 {-pb[2]/2}')
                    g = ET.SubElement(v, 'geometry')
                    b = ET.SubElement(g, 'box')
                    b.set('size', f'{pb[0]} {pb[1]} {pb[2]}')
                    if tag == 'visual':
                        mat = ET.SubElement(v, 'material')
                        mat.set('name', f'mat_{f}_proximal_{side}')
                        c = ET.SubElement(mat, 'color')
                        c.set('rgba', '0.75 0.75 0.75 1')
                # 更新 inertial
                io = l.find('inertial/origin')
                if io is not None:
                    io.set('xyz', f'0 0 {-pb[2]/2}')
        # 2. 控制关节 child 引用更新
        for j in root.findall('joint'):
            if j.get('name') == ctrl:
                j.find('child').set('link', f'{f}_proximal_{side}_Link')
        # 3. 添加 intermediate/distal 关节 + link
        ib, db = SEG[f][1], SEG[f][2]
        root.append(make_joint(f'{f}_intermediate_{side}_joint', f'{f}_proximal_{side}_Link',
                               f'{f}_intermediate_{side}_Link', f'0 0 {-pb[2]}', '0 -1 0', mimic=ctrl))
        root.append(make_link(f'{f}_intermediate_{side}_Link', ib, -ib[2]/2))
        root.append(make_joint(f'{f}_distal_{side}_joint', f'{f}_intermediate_{side}_Link',
                               f'{f}_distal_{side}_Link', f'0 0 {-ib[2]}', '0 -1 0', mimic=ctrl))
        root.append(make_link(f'{f}_distal_{side}_Link', db, -db[2]/2))
        print(f'{f}_{side}: 3 段完成')

# 拇指: thumb_abduct(基座) -> thumb_flex(控制) -> intermediate -> distal
for side in ['left', 'right']:
    # thumb_abduct link 改几何 (基座)
    for l in root.findall('link'):
        if l.get('name') == f'thumb_abduct_{side}_Link':
            for v in l.findall('visual'):
                l.remove(v)
            for v in l.findall('collision'):
                l.remove(v)
            for tag in ('visual', 'collision'):
                v = ET.SubElement(l, tag)
                o = ET.SubElement(v, 'origin')
                o.set('xyz', '0 0 -0.015')
                g = ET.SubElement(v, 'geometry')
                b = ET.SubElement(g, 'box')
                b.set('size', '0.03 0.018 0.03')
                if tag == 'visual':
                    mat = ET.SubElement(v, 'material')
                    mat.set('name', f'mat_thumb_abduct_{side}')
                    c = ET.SubElement(mat, 'color')
                    c.set('rgba', '0.75 0.75 0.75 1')
            io = l.find('inertial/origin')
            if io is not None:
                io.set('xyz', '0 0 -0.015')
    # thumb_flex: parent 改 thumb_abduct, child 改 thumb_proximal
    for j in root.findall('joint'):
        if j.get('name') == f'thumb_flex_{side}_joint':
            j.find('parent').set('link', f'thumb_abduct_{side}_Link')
            j.find('child').set('link', f'thumb_proximal_{side}_Link')
            o = j.find('origin')
            o.set('xyz', '0 0 -0.03')
    # thumb_flex link 改名 + 改几何 (近端段)
    for l in root.findall('link'):
        if l.get('name') == f'thumb_flex_{side}_Link':
            l.set('name', f'thumb_proximal_{side}_Link')
            for v in l.findall('visual'):
                l.remove(v)
            for v in l.findall('collision'):
                l.remove(v)
            for tag in ('visual', 'collision'):
                v = ET.SubElement(l, tag)
                o = ET.SubElement(v, 'origin')
                o.set('xyz', '0 0 -0.015')
                g = ET.SubElement(v, 'geometry')
                b = ET.SubElement(g, 'box')
                b.set('size', '0.028 0.017 0.03')
                if tag == 'visual':
                    mat = ET.SubElement(v, 'material')
                    mat.set('name', f'mat_thumb_proximal_{side}')
                    c = ET.SubElement(mat, 'color')
                    c.set('rgba', '0.75 0.75 0.75 1')
            io = l.find('inertial/origin')
            if io is not None:
                io.set('xyz', '0 0 -0.015')
    # 添加 thumb intermediate/distal
    root.append(make_joint(f'thumb_intermediate_{side}_joint', f'thumb_proximal_{side}_Link',
                           f'thumb_intermediate_{side}_Link', '0 0 -0.03', '0 -1 0', mimic=f'thumb_flex_{side}_joint', limit=(0.0, 1.03)))
    root.append(make_link(f'thumb_intermediate_{side}_Link', (0.026, 0.016, 0.03), -0.015))
    root.append(make_joint(f'thumb_distal_{side}_joint', f'thumb_intermediate_{side}_Link',
                           f'thumb_distal_{side}_Link', '0 0 -0.03', '0 -1 0', mimic=f'thumb_flex_{side}_joint', limit=(0.0, 1.03)))
    root.append(make_link(f'thumb_distal_{side}_Link', (0.024, 0.015, 0.025), -0.0125))
    print(f'thumb_{side}: 3 段完成')

tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('手指 3 段化完成 (mimic 联动, 控制量保持 6)')
