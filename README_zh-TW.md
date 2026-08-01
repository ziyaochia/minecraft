# Minecraft 復刻 (Python)

用 Python 復刻的 Minecraft，包含三種渲染後端：OpenGL 光柵化、Vulkan (wgpu) 光柵化，以及基於 GLSL 體素 DDA 的光線追蹤模式。

| OpenGL | Vulkan | 光線追蹤 |
|--------|--------|----------|
| ![gl](docs/gl_day.png) | ![vk](docs/vk_day.png) | ![rt](docs/rt_day.png) |

## 執行

```bash
pip install -r requirements.txt
python -m minecraft                    # OpenGL 渲染（預設）
python -m minecraft --renderer vk      # Vulkan 渲染（wgpu）
python -m minecraft --renderer rt      # 光線追蹤（GLSL 體素 DDA）
```

需要支援 OpenGL 3.3 Core 的顯示卡；Vulkan 模式需要 Vulkan 驅動程式（經 wgpu 支援，Windows 上走 DX12/Vulkan 均可，優先高效能適配器）。

## 操作

| 按鍵 | 功能 |
|------|------|
| W A S D | 移動 |
| 滑鼠 | 視角 |
| 空白鍵 | 跳躍 / 飛行時上升 / 水中上浮 |
| Shift | 飛行時下降 |
| Ctrl | 疾跑 |
| F | 飛行開關 |
| 左鍵 | 挖掘方塊（可長按） |
| 右鍵 | 放置方塊（可長按） |
| 中鍵 | 選取準星所指方塊 |
| 1-9 / 滾輪 | 切換物品欄 |
| Esc | 釋放滑鼠 / 離開 |

離開時自動儲存世界到 `saves/world_<seed>.npz`，下次以同一種子啟動時自動讀取（玩家位置、視角、物品欄、遊戲時間、全部已生成區塊及方塊改動均會保留）。`--fresh` 忽略存檔重新開始。

## 命令列參數

```
--seed N        世界種子（預設 20260731）
--renderer      gl | rt | vk
--pos "x,y,z"   指定重生點
--yaw --pitch   初始視角（弧度）
--time T        初始時間 [0,1)，0.32 為上午，0.55 為夜晚
--rd N          渲染距離（區塊，預設 8）
--fresh         忽略存檔
--save          截圖模式下也啟用存讀檔
--screenshot P  無頭截圖模式：固定步長執行 --frames 幀後儲存截圖離開
--frames N      截圖模式幀數（預設 60）
--demo "F:act,…" 腳本化演示：在指定幀執行 place / break / shot:路徑
```

## 已實作機制

- **世界生成**：基於值雜訊/分形的高度場地形（海洋、沙灘、丘陵、雪山）、湖泊與海洋水體、樹木、礦脈（煤/鐵/金/鑽石）、基岩層；無限世界按 16×256×16 區塊流式生成（雙執行緒池：生成 + 網格化），種子確定性。
- **方塊**：18 種可放置方塊，程序化生成的 16×16 紋理圖集（草地塊、泥土、石頭、鵝卵石、木板、原木、樹葉、沙子、砂岩、玻璃、磚塊、礫石、雪、四種礦石、基岩、水）。
- **網格化**：numpy 向量化區塊網格器，逐面環境光遮蔽（AO），半透明（水/玻璃）獨立批次，區塊鄰接面剔除。
- **物理**：AABB 逐軸碰撞、子步進防穿透、重力/跳躍/疾跑/飛行、水中阻力與上浮。
- **互動**：Amanatides-Woo DDA 準星射線檢測（6 公尺），挖掘/放置（含玩家重疊拒絕、基岩不可破壞、跨區塊重網格）、中鍵選取。
- **晝夜循環**：太陽/月亮/星星/動態雲層的天空著色器，霧色隨時間變化，20 分鐘一天。
- **HUD**：準星、9 格物品欄（3D 方塊圖示）。
- **存檔**：離開時儲存全部區塊與玩家狀態（npz 壓縮），啟動自動讀取。

## 渲染架構

- **gl** — moderngl（OpenGL 3.3 Core）：紋理圖集 + AO 著色 + 距離霧。
- **vk** — wgpu（Vulkan）：同一套世界/網格/互動程式碼，WGSL 著色器復刻 GL 管線；視窗經 rendercanvas/glfw 呈現。
- **rt** — 光線追蹤：GLSL 全螢幕體素 DDA。世界以 R8UI 3D 紋理（176×256×176）跟隨玩家重中心；逐像素主射線 + 朝太陽的陰影射線（硬陰影）、水面單次反彈反射（雲/天空/地形倒影）、水與玻璃透射染色、調色盤紋理取方塊均色。

三種模式共享同一個遊戲迴路基類（`game.py`），僅覆寫視窗建立、繪製與髒區塊處理掛鉤。

## 未實作（誠實範圍）

生物/怪物、合成與熔煉、完整物品欄與背包介面、飢餓/生命值、音效、紅石、流體物理（水為靜態方塊）、生物群系細分、多人遊戲、成就/選單介面。樹葉/水的動畫紋理未做。光線追蹤為單次反彈 + 硬陰影，非全域光照。

## 測試

```bash
python tests/test_player.py     # 物理/碰撞/射線/互動精確數值斷言
python tests/test_world.py      # 網格化面數斷言
python tests/test_render_gl.py  # 無頭渲染截圖冒煙測試
```

另可用 `--screenshot` + `--demo` 做腳本化端到端驗證（三種渲染器均支援），例如：

```bash
python -m minecraft --renderer rt --screenshot out.png --frames 200 --rd 2 \
  --pos "30,83.5,30" --yaw -0.62 --pitch -1.0 \
  --demo "60:place,120:shot:placed.png,130:break,190:shot:broken.png"
```

## 程式碼結構

```
minecraft/
  config.py     全域常數（物理、視窗、渲染距離等）
  noise.py      值雜訊 / 分形雜訊
  worldgen.py   地形生成器
  chunk.py      區塊（16×256×16 uint8）
  world.py      區塊字典、方塊讀寫、鄰域取樣
  blocks.py     方塊註冊表
  textures.py   程序化紋理圖集 / RT 調色盤
  mesher.py     numpy 區塊網格器（含 AO）
  mat4.py       矩陣/視錐工具
  raycast.py    準星 DDA
  player.py     玩家物理
  interact.py   挖掘/放置規則
  sky.py        晝夜狀態
  render_gl.py  OpenGL 渲染器
  render_vk.py  Vulkan (wgpu) 渲染器
  render_rt.py  光線追蹤渲染器
  game.py       遊戲迴路基類（輸入、流式載入、HUD）
  game_rt.py    RT 模式（體素體積流式維護）
  game_vk.py    VK 模式
  saveload.py   世界存取
```
