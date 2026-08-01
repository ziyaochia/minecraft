# Minecraft 复刻 (Python)

用 Python 复刻的 Minecraft，包含三种渲染后端：OpenGL 光栅化、Vulkan (wgpu) 光栅化、以及基于 GLSL 体素 DDA 的光线追踪模式。

| OpenGL | Vulkan | 光线追踪 |
|--------|--------|----------|
| ![gl](docs/gl_day.png) | ![vk](docs/vk_day.png) | ![rt](docs/rt_day.png) |

## 运行

```bash
pip install -r requirements.txt
python -m minecraft                    # OpenGL 渲染 (默认)
python -m minecraft --renderer vk      # Vulkan 渲染 (wgpu)
python -m minecraft --renderer rt      # 光线追踪 (GLSL 体素 DDA)
```

需要支持 OpenGL 3.3 Core 的显卡；Vulkan 模式需要 Vulkan 驱动（经 wgpu 支持，Windows 上走 DX12/Vulkan 均可，优先高性能适配器）。

## 操作

| 按键 | 功能 |
|------|------|
| W A S D | 移动 |
| 鼠标 | 视角 |
| 空格 | 跳跃 / 飞行时上升 / 水中上浮 |
| Shift | 飞行时下降 |
| Ctrl | 疾跑 |
| F | 飞行开关 |
| 左键 | 挖掘方块 (可长按) |
| 右键 | 放置方块 (可长按) |
| 中键 | 选取准星所指方块 |
| 1-9 / 滚轮 | 切换物品栏 |
| Esc | 释放鼠标 / 退出 |

退出时自动保存世界到 `saves/world_<seed>.npz`，下次以同一种子启动时自动读取（玩家位置、视角、物品栏、游戏时间、全部已生成区块及方块改动均会保留）。`--fresh` 忽略存档重新开始。

## 命令行参数

```
--seed N        世界种子 (默认 20260731)
--renderer      gl | rt | vk
--pos "x,y,z"   指定出生点
--yaw --pitch   初始视角 (弧度)
--time T        初始时间 [0,1)，0.32 为上午，0.55 为夜晚
--rd N          渲染距离 (区块，默认 8)
--fresh         忽略存档
--save          截图模式下也启用存读档
--screenshot P  无头截图模式：固定步长运行 --frames 帧后保存截图退出
--frames N      截图模式帧数 (默认 60)
--demo "F:act,…" 脚本化演示：在指定帧执行 place / break / shot:路径
```

## 已实现机制

- **世界生成**：基于值噪声/分形的高度场地形（海洋、沙滩、丘陵、雪山）、湖泊与海洋水体、树木、矿脉（煤/铁/金/钻石）、基岩层；无限世界按 16×256×16 区块流式生成（双线程池：生成 + 网格化），种子确定性。
- **方块**：18 种可放置方块，程序化生成的 16×16 纹理图集（草方块、泥土、石头、圆石、木板、原木、树叶、沙子、砂岩、玻璃、砖块、沙砾、雪、四种矿石、基岩、水）。
- **网格化**：numpy 向量化区块网格器，逐面环境光遮蔽 (AO)，半透明（水/玻璃）独立批次，区块邻接面剔除。
- **物理**：AABB 逐轴碰撞、子步进防穿透、重力/跳跃/疾跑/飞行、水中阻力与上浮。
- **交互**：Amanatides-Woo DDA 准星射线检测（6 米），挖掘/放置（含玩家重叠拒绝、基岩不可破坏、跨区块重网格）、中键选取。
- **昼夜循环**：太阳/月亮/星星/动态云层的天空着色器，雾色随时间变化，20 分钟一天。
- **HUD**：准星、9 格物品栏（3D 方块图标）。
- **存档**：退出时保存全部区块与玩家状态 (npz 压缩)，启动自动读取。

## 渲染架构

- **gl** — moderngl (OpenGL 3.3 Core)：纹理图集 + AO 着色 + 距离雾。
- **vk** — wgpu (Vulkan)：同一套世界/网格/交互代码，WGSL 着色器复刻 GL 管线；窗口经 rendercanvas/glfw 呈现。
- **rt** — 光线追踪：GLSL 全屏体素 DDA。世界以 R8UI 3D 纹理 (176×256×176) 跟随玩家重中心；逐像素主射线 + 朝太阳的阴影射线（硬阴影）、水面单次反弹反射（云/天空/地形倒影）、水与玻璃透射染色、调色板纹理取方块均色。

三种模式共享同一个游戏循环基类 (`game.py`)，仅覆盖窗口创建、绘制与脏区块处理钩子。

## 未实现（诚实范围）

生物/怪物、合成与熔炼、完整物品栏与背包界面、饥饿/生命值、音效、红石、流体物理（水为静态方块）、生物群系细分、多人游戏、成就/菜单界面。树叶/水的动画纹理未做。光线追踪为单次反弹 + 硬阴影，非全局光照。

## 测试

```bash
python tests/test_player.py     # 物理/碰撞/射线/交互精确数值断言
python tests/test_world.py      # 网格化面数断言
python tests/test_render_gl.py  # 无头渲染截图冒烟测试
```

另可用 `--screenshot` + `--demo` 做脚本化端到端验证（三种渲染器均支持），例如：

```bash
python -m minecraft --renderer rt --screenshot out.png --frames 200 --rd 2 \
  --pos "30,83.5,30" --yaw -0.62 --pitch -1.0 \
  --demo "60:place,120:shot:placed.png,130:break,190:shot:broken.png"
```

## 代码结构

```
minecraft/
  config.py     全局常量 (物理、窗口、渲染距离等)
  noise.py      值噪声 / 分形噪声
  worldgen.py   地形生成器
  chunk.py      区块 (16×256×16 uint8)
  world.py      区块字典、方块读写、邻域取样
  blocks.py     方块注册表
  textures.py   程序化纹理图集 / RT 调色板
  mesher.py     numpy 区块网格器 (含 AO)
  mat4.py       矩阵/视锥工具
  raycast.py    准星 DDA
  player.py     玩家物理
  interact.py   挖掘/放置规则
  sky.py        昼夜状态
  render_gl.py  OpenGL 渲染器
  render_vk.py  Vulkan (wgpu) 渲染器
  render_rt.py  光线追踪渲染器
  game.py       游戏循环基类 (输入、流式加载、HUD)
  game_rt.py    RT 模式 (体素体积流式维护)
  game_vk.py    VK 模式
  saveload.py   世界存取
```
