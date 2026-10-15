import bpy, json
from bpy.props import EnumProperty

# 角色 UV 部位选择工具（Origami 版）
# 用法：在文本编辑器中运行一次本脚本 -> UV 编辑器右侧 N 面板“角色 UV”页 -> 选择部位 / 选择整组。
# 数据来源：网格的面属性 UV_panel（部位编号）与对象属性 UV_panel_names / UV_panel_names_cn。


def _names(o):
    en = json.loads(o.get('UV_panel_names', '[]'))
    cn = json.loads(o.get('UV_panel_names_cn', '[]')) or en
    return en, cn


def _items(self, context):
    o = context.object
    if not o or o.type != 'MESH' or 'UV_panel' not in o.data.attributes:
        return [('NONE', '先选择角色网格', '')]
    en, cn = _names(o)
    return [(str(i), f'{i:02d} {cn[i]}', en[i]) for i in range(len(en))]


def _select(context, test):
    o = context.object
    if o.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    a = o.data.attributes['UV_panel']
    vals = [0] * len(o.data.polygons)
    a.data.foreach_get('value', vals)
    for f, v in zip(o.data.polygons, vals):
        f.select = test(v)
    bpy.ops.object.mode_set(mode='EDIT')
    context.tool_settings.mesh_select_mode = (False, False, True)
    if hasattr(context.tool_settings, 'use_uv_select_sync'):
        context.tool_settings.use_uv_select_sync = True


class UV_OT_select_origami_region(bpy.types.Operator):
    bl_idname = 'uv.select_origami_region'; bl_label = '选择部位'; bl_options = {'REGISTER', 'UNDO'}
    region: EnumProperty(name='部位', items=_items)
    mirror: bpy.props.BoolProperty(name='同时选择左右对称部位', default=False)

    def execute(self, context):
        o = context.object
        if not o or 'UV_panel' not in o.data.attributes or self.region == 'NONE':
            return {'CANCELLED'}
        en, _ = _names(o)
        target = int(self.region)
        keep = {target}
        if self.mirror:
            n = en[target]
            for tag_a, tag_b in (('_L', '_R'), ('_R', '_L')):
                if tag_a in n:
                    twin = n.replace(tag_a, tag_b)
                    keep |= {i for i, x in enumerate(en) if x == twin}
        _select(context, lambda v: v in keep)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=380)


class UV_PT_origami_regions(bpy.types.Panel):
    bl_label = '角色 UV 部位'; bl_space_type = 'IMAGE_EDITOR'; bl_region_type = 'UI'; bl_category = '角色 UV'

    def draw(self, context):
        col = self.layout.column()
        col.label(text='先选中角色网格，再选择部位')
        col.operator('uv.select_origami_region', text='选择部位…')


for cls in (UV_OT_select_origami_region, UV_PT_origami_regions):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        pass
