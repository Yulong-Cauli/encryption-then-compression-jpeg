# 抗 JPEG 压缩的图片混淆系统 (EtC)

主题: Encryption-then-Compression (EtC) / JPEG 抗压缩混淆

**参考文献**: *An Encryption-then-Compression System for JPEG Standard*

K. Kurihara, S. Shiota and H. Kiya, "An encryption-then-compression system for JPEG standard," 2015 Picture Coding Symposium (PCS), Cairns, QLD, Australia, 2015, pp. 119-123, doi: 10.1109/PCS.2015.7170059. keywords: {Cryptography;Image coding;Standards},



这是一种 **"先加密后压缩" (Encryption-then-Compression)** 的方案。

加密后的图片看起来是乱码，但依然符合 JPEG 标准，能被标准的 JPEG 算法高效压缩，且解密后画质无损。

我读下来，**发现关键点就一句话**：

> 论文明确指出：为了适应色度下采样，必须将 $16 \times 16$ 作为最小分割单元。这是算法能“抗压缩”的**物理前提**。

其他部分，我觉得是不是换任何一种随机方案都可以（口胡）。



## 原理

JPEG 压缩的核心机制是对图像进行分块（通常是 $8\times8$），利用 DCT 变换去除高频信息 4。

**传统的混淆**，如像素级打乱，破坏了图像的局部相关性，产生大量高频噪声，导致 JPEG 压缩效率极低且解压后产生严重伪影。

**本方案的逻辑**：只要混淆操作是**基于块 (Block-based)** 进行的，且不破坏块内部的像素平滑度，JPEG 编码器就会认为它依然是一张“正常”的图片进行压缩。



## 具体步骤

为了适配 JPEG 的色彩采样（如 4:2:0），块的大小建议设为 **$16\times16$** 。算法包含以下四个独立步骤：

![论文框架](assets\Structure.png)
<center style="font-size:14px;color:#000000;"><b>图1.</b> 论文中提到的框架图，重点在 Fig.3 ，Fig.2 与 Fig.4 是他的具体打乱方法，对应蓝色框和黄色框</center>

### Step 1: 块分割与置乱 (Block Scrambling)

- 将图像切分为 $16\times16$ 的块。
- 利用密钥生成随机序列，打乱这些块的排列位置 。

### Step 2: 块旋转与翻转 (Block Rotation and Inversion)

- **旋转**：随机旋转 $0^\circ, 90^\circ, 180^\circ, 270^\circ$ 8。
- **翻转**：随机进行水平或垂直镜像翻转 9。
- *作用*：改变块的方向特性，增加破解难度。

### Step 3: 负片转换 (Negative-Positive Transformation)

- 随机反转块内所有像素的值。
- 公式：$p' = 255 - p$。

### Step 4: 颜色分量洗牌 (Color Component Shuffling)

- 在块内部，随机交换 R, G, B 三个通道的顺序（例如 R变B，B变G）。
- *作用*：破坏色彩的视觉可读性，但保留色彩统计特性。

------



## 代码实现

基于论文逻辑复现的完整工具脚本。

**环境依赖**:

```bash
pip install numpy Pillow
```

**脚本 (`etc_scramble.py`):**

```Python
import numpy as np
from PIL import Image
import random


def encryption_then_compression(image_path, output_path, seed, block_size=16, mode='encrypt'):
    """
    基于论文 "An Encryption-then-Compression System for JPEG Standard" 的实现。
    包含四个步骤：块置乱、块旋转/翻转、负片转换、颜色分量洗牌。

    :param block_size: 论文建议设为 16 以适应 JPEG 的 chroma subsampling [cite: 186]
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_array = np.array(img)
    h, w, c = img_array.shape

    # 1. 补齐边缘（Padding）
    # 论文提到必须分割为非重叠块，JPEG通常基于MCU(16x16) [cite: 185]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h != 0 or pad_w != 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w, c = img_array.shape

    # 2. 分块 (Block Division)
    rows = h // block_size
    cols = w // block_size
    num_blocks = rows * cols

    # 提取所有块
    blocks = []
    for r in range(rows):
        for c in range(cols):
            blocks.append(img_array[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size])

    # 初始化随机数生成器 (Key Generation) [cite: 189]
    rng = random.Random(seed)

    # 生成各步骤的随机参数 (保证解密时能生成同样的参数)
    # A. 置乱索引 (Permutation Index)
    perm_indices = list(range(num_blocks))
    rng.shuffle(perm_indices)

    # B. 旋转/翻转参数 (Rotation/Inversion Params) [cite: 148, 150]
    # 0:0deg, 1:90deg, 2:180deg, 3:270deg
    # 翻转: 0:None, 1:Horizontal, 2:Vertical, 3:Both
    rot_params = [rng.randint(0, 3) for _ in range(num_blocks)]
    inv_params = [rng.randint(0, 3) for _ in range(num_blocks)]

    # C. 负片转换参数 (Negative-Positive Params) [cite: 169]
    neg_params = [rng.randint(0, 1) for _ in range(num_blocks)]

    # D. 颜色洗牌参数 (Color Shuffling Params) [cite: 176]
    # RGB 的 6 种排列组合: (0,1,2), (0,2,1), ...
    import itertools
    color_perms = list(itertools.permutations([0, 1, 2]))
    shuffle_params = [rng.choice(color_perms) for _ in range(num_blocks)]

    # --- 执行处理 ---
    processed_blocks = [None] * num_blocks

    if mode == 'encrypt':
        # 加密流程
        for i in range(num_blocks):
            # 获取原始块（根据置乱索引）
            # Step 2: Block Scrambling (位置置乱)
            # 我们要放置到第 i 个位置的块，是原图中的 perm_indices[i] 块
            # (注意：逻辑可以是把 i 放到 perm[i]，也可以是取 perm[i] 放到 i，这里取后者方便构建流)
            curr_block = blocks[perm_indices[i]].copy()

            # Step 3: Rotation and Inversion
            # 旋转
            curr_block = np.rot90(curr_block, k=rot_params[i])
            # 翻转
            if inv_params[i] == 1:  # Horizontal
                curr_block = np.fliplr(curr_block)
            elif inv_params[i] == 2:  # Vertical
                curr_block = np.flipud(curr_block)
            elif inv_params[i] == 3:  # Both
                curr_block = np.flipud(np.fliplr(curr_block))

            # Step 4: Negative-Positive Transformation
            if neg_params[i] == 1:
                curr_block = 255 - curr_block

            # Step 5: Color Component Shuffling
            p = shuffle_params[i]  # e.g., (2, 0, 1) -> B, R, G
            curr_block = curr_block[:, :, p]

            processed_blocks[i] = curr_block

    elif mode == 'decrypt':
        # 解密流程 (逆序操作)

        # 创建一个临时列表来存放处理完内容的块，注意顺序问题
        # 加密时: target[i] = Source[perm[i]] (把乱序的取过来)
        # 解密时: 我们手头是 target[i]，需要先把它的内容还原，然后放回 Source[perm[i]]

        temp_blocks = [None] * num_blocks

        for i in range(num_blocks):
            curr_block = blocks[i].copy()  # 这里的 blocks 其实是加密图的块

            # 逆操作 Step 5: Color (找回原来的 RGB)
            p = shuffle_params[i]
            # 如果 p 是 (2, 0, 1) 即 R->2, G->0, B->1
            # 逆置换:
            rev_p = [0, 0, 0]
            for idx, val in enumerate(p):
                rev_p[val] = idx
            curr_block = curr_block[:, :, rev_p]

            # 逆操作 Step 4: Negative
            if neg_params[i] == 1:
                curr_block = 255 - curr_block

            # 逆操作 Step 3: Inversion & Rotation
            # 翻转是自逆的 (flip 两次等于没 flip)，但要注意顺序
            # 加密: rot -> flip. 解密: flip -> rot_inverse
            if inv_params[i] == 3:
                curr_block = np.fliplr(np.flipud(curr_block))
            elif inv_params[i] == 2:
                curr_block = np.flipud(curr_block)
            elif inv_params[i] == 1:
                curr_block = np.fliplr(curr_block)

            # 旋转的逆操作: rot90(k=1) 的逆是 rot90(k=-1 或 3)
            curr_block = np.rot90(curr_block, k=-rot_params[i])

            temp_blocks[i] = curr_block

        # 逆操作 Step 2: Unscramble (位置还原)
        # 加密: Dest[i] = Src[perm[i]]
        # 解密: Src[perm[i]] = Dest[i]
        for i in range(num_blocks):
            processed_blocks[perm_indices[i]] = temp_blocks[i]

    # 4. 拼合图像 (Integration) [cite: 180]
    final_img_array = np.zeros((rows * block_size, cols * block_size, 3), dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            final_img_array[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size] = \
            processed_blocks[idx]
            idx += 1

    result_img = Image.fromarray(final_img_array)

    # 如果是解密，裁剪掉 Padding
    if mode == 'decrypt':
        orig_w = w - pad_w
        orig_h = h - pad_h
        result_img = result_img.crop((0, 0, orig_w, orig_h))

    result_img.save(output_path, quality=95, subsampling=0)  # 尽量保持质量
    print(f"Done: {mode} -> {output_path}")


# --- 使用方法 ---
# 1. 加密
encryption_then_compression("input.jpg", "encrypted.jpg", seed=123456, mode='encrypt')

# 2. 解密 (必须用相同的 seed)
encryption_then_compression("encrypted.jpg", "restored.jpg", seed=123456, mode='decrypt')
```

**结果**：

![](assets\result.png)
