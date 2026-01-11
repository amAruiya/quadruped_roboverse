import genesis as gs

# 初始化 Genesis
gs.init(backend=gs.gpu)
scene = gs.Scene()

# 加载 Leap 机器人
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/home/ubuntu/RoboVerse/roboverse_data/robots/Leap/urdf/Leap.urdf",
        fixed=False,
    )
)

scene.build()

# 测试 1: 检查 links 属性
print("=== 测试 1: robot 对象的属性 ===")
print(f"robot 类型: {type(robot)}")
print(f"是否有 links 属性: {hasattr(robot, 'links')}")
if hasattr(robot, "links"):
    print(f"links 数量: {len(robot.links)}")
    print(f"第一个 link: {robot.links[0]}")

# 测试 2: 获取 link 名称
print("\n=== 测试 2: link 名称列表 ===")
if hasattr(robot, "links"):
    for i, link in enumerate(robot.links):
        print(f"[{i}] {link.name if hasattr(link, 'name') else '无 name 属性'}")

# 测试 3: 其他可能的属性
print("\n=== 测试 3: 其他可能的属性 ===")
for attr in ["bodies", "rigid_bodies", "link_names", "body_names"]:
    if hasattr(robot, attr):
        val = getattr(robot, attr)
        print(f"✓ robot.{attr} 存在: {type(val)}")
        if isinstance(val, list) and len(val) > 0:
            print(f"  示例: {val[:3]}")