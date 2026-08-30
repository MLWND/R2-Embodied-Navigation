#!/usr/bin/env python
"""GRScenes 材质转换 v5: MDL -> UsdPreviewSurface

问题: GRScenes 的材质全部用 MDL 描述 (shader 上的 info:mdl:sourceAsset +
材质的 outputs:mdl:surface). Genesis 导入 USD 时只解析标准的
UsdPreviewSurface (genesis/utils/usd/usd_material.py), MDL 烘焙又依赖
omniverse-kit (未安装), 所以直接加载时所有 mesh 都渲染成灰色.
另外 Genesis 不读取"已绑定材质"的 mesh 上的 displayColor, 旧方案无效.

材质分三类, 按优先级提取颜色:
1. UE4 参数模板 (Num*): 颜色在 USD 的 inputs:BaseColor_Color /
   BaseColor_Tex / IsBaseColorTex 开关里, 每个材质实例各自一份;
2. KooPbr::KooMtl 模板: 参数在 MDL 文件里, diffuse 是 color(...) 或
   texture_2d("./textures/...") 贴图 (pbrjson2mdl 生成, 每参数一行);
3. WorldGridMaterial (房间墙体/天花板): 程序化网格材质, 用固定灰色近似;
   其余自由格式 MDL (DayMaterial 天空球等) 跳过.

每个材质生成标准网络写回 USD (Genesis 原生可解析):
    Material
      +- GenesisUVReader   (UsdPrimvarReader_float2, varname=st)
      +- GenesisDiffuseTex (UsdUVTexture, file=贴图)      -- 仅贴图材质
      +- GenesisSurface    (UsdPreviewSurface)
    Material.outputs:surface -> GenesisSurface.outputs:surface
outputs:mdl:surface 原样保留, 不影响其他工具加载.

用法: python grscene_extract_color.py [scene.usd] [输出.usd]
"""
import os
import re
import sys
from pxr import Gf, Sdf, Usd, UsdShade

SCENE = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/xujinlong/test/GRScenes/scenes/GRScenes-100/home_scenes/scenes/MWBGLKQKTKJZ2AABAAAAABA8_usd/start_result_raw.usd"
OUT = sys.argv[2] if len(sys.argv) > 2 else SCENE.replace('.usd', '_colored.usd')
scene_dir = os.path.dirname(os.path.abspath(SCENE))

# Genesis 只支持 PIL 能打开的贴图格式
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tif', '.tiff')

# 解析不到时的已知特殊材质兜底 (程序化材质 -> 近似色)
SPECIAL_MDLS = {
    'WorldGridMaterial.mdl': {'diffuse': [0.35, 0.35, 0.35]},
    # 场景自带天空球 (半径~86m 包裹全建筑), 原设计=白色自发光背景
    'DayMaterial.mdl': {'diffuse': [1.0, 1.0, 1.0], 'emissive': [1.0, 1.0, 1.0]},
}


def resolve_materials_dir():
    """定位共享 Materials 目录. 下载包里的软链接可能被打平成文本文件,
    内容是原始链接目标 (如 '../../Materials')."""
    d = os.path.join(scene_dir, 'Materials')
    if os.path.isdir(d):
        return d
    if os.path.isfile(d):
        try:
            with open(d) as f:
                target = f.read().strip()
        except OSError:
            return None
        target = os.path.normpath(os.path.join(scene_dir, target))
        if os.path.isdir(target):
            return target
    return None


def resolve_asset(v, mat_dir, require_image=True):
    """@./Materials/xxx@ / Sdf.AssetPath / 相对路径 -> 存在的文件绝对路径."""
    if v is None:
        return None
    p = getattr(v, 'path', None) or str(v)
    p = p.strip().strip('@')
    while p.startswith('./'):
        p = p[2:]
    if p.startswith('Materials/'):
        cand = os.path.join(mat_dir, p[len('Materials/'):])
    else:
        cand = os.path.join(scene_dir, p)
    cand = os.path.normpath(cand)
    if os.path.isfile(cand) and (not require_image or cand.lower().endswith(IMG_EXTS)):
        return cand
    return None


_COLOR_RE = re.compile(
    r'color\(\s*([-0-9.eE+]+)f?\s*,\s*([-0-9.eE+]+)f?\s*,\s*([-0-9.eE+]+)f?\s*\)')
_FLOAT_RE = re.compile(r'^([-0-9.eE+]+)f?$')
_TEX_RE = re.compile(r'texture_2d\(\s*"([^"]+)"')
_PARAM_RE = re.compile(r'^\s+([a-zA-Z_]+):\s*(.*?),?\s*$')


def _parse_color(val):
    m = _COLOR_RE.search(val)
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else None


def _parse_float(val):
    m = _FLOAT_RE.match(val.strip())
    return float(m.group(1)) if m else None


def _parse_diffuse(val, mdl_dir):
    """diffuse 值可能是 color(...) 或 KooPbr_bitmap(...texture_2d("path")...).tint"""
    m = _COLOR_RE.search(val)
    if m:
        return {'diffuse': [float(m.group(i)) for i in (1, 2, 3)]}
    m = _TEX_RE.search(val)
    if m:
        tex = os.path.normpath(os.path.join(mdl_dir, m.group(1)))
        if os.path.isfile(tex) and tex.lower().endswith(IMG_EXTS):
            return {'diffuse_tex': tex}
    return {}


def parse_mdl(mdl_file, cache={}):
    """解析 KooPbr::KooMtl 模板参数. 每个参数独占一行 (自动生成格式),
    KooPbr_bitmap(...) 整体也是单行, 按行解析不会碰到嵌套括号问题."""
    if mdl_file in cache:
        return cache[mdl_file]
    props = {}
    try:
        with open(mdl_file, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except OSError:
        cache[mdl_file] = props
        return props
    in_mtl = False
    for line in lines:
        if not in_mtl:
            if 'KooMtl(' in line:
                in_mtl = True
            continue
        m = _PARAM_RE.match(line)
        if not m:
            if props:  # 参数块结束
                break
            continue
        name, val = m.group(1), m.group(2)
        if name == 'diffuse':
            props.update(_parse_diffuse(val, os.path.dirname(mdl_file)))
        elif name == 'opacity':
            v = _parse_float(val)
            if v is not None:
                props['opacity'] = v
        elif name == 'reflection_metalness':
            v = _parse_float(val)
            if v is not None:
                props['metallic'] = v
        elif name == 'reflect_glossiness':
            v = _parse_float(val)
            if v is not None:  # V-Ray 系 glossiness -> roughness
                props['roughness'] = min(max(1.0 - v, 0.0), 1.0)
        elif name == 'self_illumination':
            c = _parse_color(val)
            if c and any(x > 0.01 for x in c):
                props['emissive'] = c
    cache[mdl_file] = props
    return props


def props_from_inputs(inputs, mat_dir):
    """UE4 参数模板 (Num*): 从 USD 的 inputs:* 提取, 覆盖 MDL 文件默认值."""
    if 'BaseColor_Color' not in inputs and 'IsBaseColorTex' not in inputs:
        return None
    props = {}

    def flt(name, default=0.0):
        v = inputs.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    if flt('IsBaseColorTex') >= 0.5:
        tex = resolve_asset(inputs.get('BaseColor_Tex'), mat_dir)
        if tex:
            props['diffuse_tex'] = tex
    if 'diffuse_tex' not in props:
        c = inputs.get('BaseColor_Color')
        if c is not None and len(c) >= 3:
            props['diffuse'] = [float(c[0]), float(c[1]), float(c[2])]

    m = inputs.get('Metallic_Color')
    if m is not None and flt('IsMetallicTex') < 0.5:
        props['metallic'] = max(0.0, float(m[0]))
    g = inputs.get('Gloss_Color')
    if g is not None and flt('IsGlossTex') < 0.5:  # gloss -> roughness
        props['roughness'] = min(max(1.0 - float(g[0]), 0.0), 1.0)
    e = inputs.get('Emissive_Color')
    if e is not None and flt('EmissiveIntensity') > 0.01 and flt('IsEmissiveTex') < 0.5:
        k = flt('EmissiveIntensity')
        props['emissive'] = [float(e[i]) * k for i in range(3)]
    return props if ('diffuse' in props or 'diffuse_tex' in props) else None


def build_preview_surface(stage, mat_prim, props):
    """在 Material 下生成 Genesis 能解析的 UsdPreviewSurface 网络."""
    if mat_prim.GetChild('GenesisSurface'):
        return 'skip'
    mat = UsdShade.Material(mat_prim)

    surf = UsdShade.Shader.Define(stage, mat_prim.GetPath().AppendChild('GenesisSurface'))
    surf.CreateIdAttr('UsdPreviewSurface')

    tex = props.get('diffuse_tex')
    if tex:
        reader = UsdShade.Shader.Define(
            stage, mat_prim.GetPath().AppendChild('GenesisUVReader'))
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')

        texshader = UsdShade.Shader.Define(
            stage, mat_prim.GetPath().AppendChild('GenesisDiffuseTex'))
        texshader.CreateIdAttr('UsdUVTexture')
        # 相对导出层所在目录, 随 GRScenes 目录树整体迁移仍然有效
        rel = os.path.relpath(tex, scene_dir)
        texshader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel))
        texshader.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        texshader.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.GetPath().AppendProperty('outputs:result'))

        surf.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texshader.GetPath().AppendProperty('outputs:rgb'))
    elif props.get('diffuse'):
        surf.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*props['diffuse']))
    else:
        return 'noprops'

    if 'opacity' in props and props['opacity'] < 1.0:
        surf.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(props['opacity'])
    if 'metallic' in props:
        surf.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(props['metallic'])
    if 'roughness' in props:
        surf.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(props['roughness'])
    if props.get('emissive'):
        surf.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*props['emissive']))

    mat.CreateSurfaceOutput().ConnectToSource(
        surf.GetPath().AppendProperty('outputs:surface'))
    return 'tex' if tex else 'flat'


def main():
    mat_dir = resolve_materials_dir()
    if not mat_dir:
        sys.exit(f'[err] 找不到 Materials 目录 (scene_dir={scene_dir})')

    stage = Usd.Stage.Open(SCENE)
    print(f'打开场景: {os.path.basename(SCENE)}')

    # 1. 收集材质 -> (shader inputs, MDL 路径)
    materials = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        for child in prim.GetChildren():
            ref = child.GetAttribute('info:mdl:sourceAsset').Get()
            if not ref:
                continue
            ref = str(getattr(ref, 'path', ref)).strip().strip('@')
            inputs = {a.GetName()[len('inputs:'):]: a.Get()
                      for a in child.GetAttributes()
                      if a.GetName().startswith('inputs:') and a.HasAuthoredValue()}
            mdl = resolve_asset(ref, mat_dir, require_image=False) \
                if ref.endswith('.mdl') else None
            materials.append((prim, inputs, mdl, os.path.basename(ref)))
            break
    print(f'材质: {len(materials)} 个 (引用 MDL)')

    # 2+3. 逐个提取颜色并生成 UsdPreviewSurface
    n_tex = n_flat = n_noprops = 0
    for i, (mat_prim, inputs, mdl, name) in enumerate(materials):
        props = props_from_inputs(inputs, mat_dir)
        if props is None and mdl:
            props = parse_mdl(mdl)
        if not props:
            props = SPECIAL_MDLS.get(name, {})
        kind = build_preview_surface(stage, mat_prim, props)
        if kind == 'tex':
            n_tex += 1
        elif kind == 'flat':
            n_flat += 1
        elif kind == 'noprops':
            n_noprops += 1
        if (i + 1) % 1000 == 0:
            print(f'  ... {i + 1}/{len(materials)}')
    print(f'转换完成: 贴图材质 {n_tex}, 纯色材质 {n_flat}, '
          f'跳过(无法解析) {n_noprops}, 已存在跳过 '
          f'{len(materials) - n_tex - n_flat - n_noprops}')

    stage.GetRootLayer().Export(OUT)
    print(f'保存到: {OUT}')


if __name__ == '__main__':
    main()
