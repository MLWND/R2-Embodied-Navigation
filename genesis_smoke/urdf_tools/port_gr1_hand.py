#!/usr/bin/env python
"""完全移植 GR1 inspire_hand 到 R1:
1. 提取 GR1 手部 (hand_base + 手指)
2. 控制关节改名映射到 R1 6 位置量 (thumb_flex/abduct + 4 指)
3. 挂到 R1 J7, 绕 x 轴 90° 旋转 (手指从 -y 转到 -z 朝下)
4. mesh 路径更新"""
import xml.etree.ElementTree as ET
import shutil

R2_URDF = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/r2/urdf/r2_urdf_v01_colored.urdf'
GR1_L = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/gr1/inspire_hand/inspire_hand_left.urdf'
GR1_R = '/home/xujinlong/test/InternUtopia/internutopia/assets/robots/gr1/inspire_hand/inspire_hand_right.urdf'
shutil.copy(R2_URDF, R2_URDF + '.bak_gr1hand')

# 控制关节映射: GR1 关节名 -> R1 控制关节名
CTRL_MAP = {
    'thumb_proximal_yaw': 'thumb_abduct',   # 拇指旋转
    'thumb_proximal_pitch': 'thumb_flex',   # 拇指弯曲
    'index_proximal': 'index',
    'middle_proximal': 'middle',
    'ring_proximal': 'ring',
    'pinky_proximal': 'pinky',
}

def extract_hand(gr1_file, side):
    """提取 GR1 手部, 改名, 返回 (links, joints)"""
    tree = ET.parse(gr1_file)
    root = tree.getroot()
    links, joints = [], []
    for l in root.findall('link'):
        n = l.get('name')
        if n.startswith(f'{side}_hand_base') or n.startswith(f'{side}_thumb') or \
           n.startswith(f'{side}_index') or n.startswith(f'{side}_middle') or \
           n.startswith(f'{side}_ring') or n.startswith(f'{side}_pinky'):
            links.append(l)
    for j in root.findall('joint'):
        n = j.get('name')
        if n.startswith(f'{side}_thumb') or n.startswith(f'{side}_index') or \
           n.startswith(f'{side}_middle') or n.startswith(f'{side}_ring') or \
           n.startswith(f'{side}_pinky'):
            joints.append(j)
    return links, joints

def rename_hand(links, joints, side):
    """改名: 控制关节 -> R1 名, 其他加 gr1_ 前缀; mesh 路径更新"""
    r1_side = 'left' if side == 'L' else 'right'
    # 1. 控制关节改名
    ctrl_old_new = {}
    for j in joints:
        n = j.get('name')
        for gr1_key, r1_key in CTRL_MAP.items():
            if n == f'{side}_{gr1_key}_joint':
                new = f'{r1_key}_{r1_side}_joint'
                ctrl_old_new[n] = new
                j.set('name', new)
                break
    # 2. 其他关节加 gr1_ 前缀 (跳过已改名的控制关节)
    for j in joints:
        n = j.get('name')
        if n not in ctrl_old_new.values() and not n.startswith('gr1_'):
            j.set('name', f'gr1_{n}')
    # 3. link 加 gr1_ 前缀
    link_old_new = {}
    for l in links:
        old = l.get('name')
        new = f'gr1_{old}'
        link_old_new[old] = new
        l.set('name', new)
    # 4. 更新 parent/child 引用
    for j in joints:
        p = j.find('parent')
        c = j.find('child')
        if p is not None and p.get('link') in link_old_new:
            p.set('link', link_old_new[p.get('link')])
        if c is not None and c.get('link') in link_old_new:
            c.set('link', link_old_new[c.get('link')])
    # 5. mimic 引用更新 (控制关节名)
    for j in joints:
        mc = j.find('mimic')
        if mc is not None:
            old = mc.get('joint')
            if old in ctrl_old_new:
                mc.set('joint', ctrl_old_new[old])
            else:
                mc.set('joint', f'gr1_{old}')
    # 6. mesh 路径更新
    for l in links:
        for m in l.iter('mesh'):
            f = m.get('filename')
            if f and f.startswith('./meshes/'):
                m.set('filename', f'../meshes/gr1_hand/{f.split("/")[-1]}')
    return links, joints

def attach_to_r2(r2_root, links, joints, side, gr1_side):
    """挂到 R1 J7: 删除原 hand+手指, 添加 hand joint + GR1 手"""
    # 1. 删除 R1 原 hand + 手指 (joint + link)
    for j in r2_root.findall('joint'):
        n = j.get('name')
        if n in (f'hand_{side}_joint',) or \
           any(k in n for k in ['thumb', 'index', 'middle', 'ring', 'pinky']) and n.endswith(f'_{side}_joint'):
            r2_root.remove(j)
    for l in r2_root.findall('link'):
        n = l.get('name')
        if n == f'hand_{side}_Link' or \
           any(k in n for k in ['thumb', 'index', 'middle', 'ring', 'pinky']) and n.endswith(f'_{side}_Link'):
            r2_root.remove(l)
    # 2. 添加 hand joint (fixed, 挂 GR1 手, 绕 x 90°)
    hj = ET.Element('joint')
    hj.set('name', f'hand_{side}_joint')
    hj.set('type', 'fixed')
    ho = ET.SubElement(hj, 'origin')
    ho.set('xyz', '0 0 -0.014')
    ho.set('rpy', '1.5708 0 0')  # 绕 x 轴 90°, 手指从 -y 转到 -z
    hp = ET.SubElement(hj, 'parent')
    hp.set('link', f'J7_{side}_Link')
    hc = ET.SubElement(hj, 'child')
    hc.set('link', f'gr1_{gr1_side}_hand_base_link')
    r2_root.append(hj)
    # 3. 添加 GR1 手部
    for l in links:
        r2_root.append(l)
    for j in joints:
        r2_root.append(j)
    print(f'{side}: GR1 手挂载完成')

# 主流程
r2_tree = ET.parse(R2_URDF)
r2_root = r2_tree.getroot()

for side, gr1_file in [('L', GR1_L), ('R', GR1_R)]:
    links, joints = extract_hand(gr1_file, side)
    links, joints = rename_hand(links, joints, side)
    attach_to_r2(r2_root, links, joints, 'left' if side == 'L' else 'right', side)

r2_tree.write(R2_URDF, encoding='utf-8', xml_declaration=True)
print('GR1 手部完全移植完成')
