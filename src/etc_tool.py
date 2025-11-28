#!/usr/bin/env python3
import numpy as np
from PIL import Image
import random
import argparse
import sys
import os


def encryption_then_compression(image_path, output_path, seed, block_size=16, mode='encrypt'):
    """
    Implementation based on "An Encryption-then-Compression System for JPEG Standard".
    """

    # Check input file
    if not os.path.exists(image_path):
        print(f"Error: Input file '{image_path}' not found. / 错误: 输入文件不存在")
        sys.exit(1)

    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        h, w, c = img_array.shape
    except Exception as e:
        print(f"Error: Failed to open image. / 错误: 无法读取图片 ({e})")
        sys.exit(1)

    # 1. Padding
    # [cite_start]Paper recommends block size 16 or integer multiple for JPEG 4:2:0 compatibility [cite: 186]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    if pad_h != 0 or pad_w != 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w, c = img_array.shape

    # 2. Block Division
    rows = h // block_size
    cols = w // block_size
    num_blocks = rows * cols

    blocks = []
    for r in range(rows):
        for c in range(cols):
            blocks.append(img_array[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size])

    # [cite_start]Key Generation [cite: 189]
    rng = random.Random(seed)

    # Generate Parameters
    # [cite_start]A. Permutation Index [cite: 135]
    perm_indices = list(range(num_blocks))
    rng.shuffle(perm_indices)

    # [cite_start]B. Rotation/Inversion [cite: 148, 150]
    rot_params = [rng.randint(0, 3) for _ in range(num_blocks)]
    inv_params = [rng.randint(0, 3) for _ in range(num_blocks)]

    # [cite_start]C. Negative-Positive [cite: 169]
    neg_params = [rng.randint(0, 1) for _ in range(num_blocks)]

    # [cite_start]D. Color Shuffling [cite: 176]
    import itertools
    color_perms = list(itertools.permutations([0, 1, 2]))
    shuffle_params = [rng.choice(color_perms) for _ in range(num_blocks)]

    # --- Processing ---
    processed_blocks = [None] * num_blocks

    action_msg = "Encrypting" if mode == 'encrypt' else "Decrypting"
    action_msg_cn = "正在加密" if mode == 'encrypt' else "正在解密"
    print(f"[{action_msg} / {action_msg_cn}] {image_path} -> {output_path} (Seed: {seed})")

    if mode == 'encrypt':
        for i in range(num_blocks):
            # Step 2: Block Scrambling
            curr_block = blocks[perm_indices[i]].copy()

            # Step 3: Rotation and Inversion
            curr_block = np.rot90(curr_block, k=rot_params[i])
            if inv_params[i] == 1:  # Horizontal
                curr_block = np.fliplr(curr_block)
            elif inv_params[i] == 2:  # Vertical
                curr_block = np.flipud(curr_block)
            elif inv_params[i] == 3:  # Both
                curr_block = np.flipud(np.fliplr(curr_block))

            # Step 4: Negative-Positive
            if neg_params[i] == 1:
                curr_block = 255 - curr_block

            # Step 5: Color Component Shuffling
            p = shuffle_params[i]
            curr_block = curr_block[:, :, p]

            processed_blocks[i] = curr_block

    elif mode == 'decrypt':
        temp_blocks = [None] * num_blocks

        for i in range(num_blocks):
            curr_block = blocks[i].copy()

            # Inverse Step 5: Color
            p = shuffle_params[i]
            rev_p = [0, 0, 0]
            for idx, val in enumerate(p):
                rev_p[val] = idx
            curr_block = curr_block[:, :, rev_p]

            # Inverse Step 4: Negative
            if neg_params[i] == 1:
                curr_block = 255 - curr_block

            # Inverse Step 3: Inversion & Rotation
            if inv_params[i] == 3:
                curr_block = np.fliplr(np.flipud(curr_block))
            elif inv_params[i] == 2:
                curr_block = np.flipud(curr_block)
            elif inv_params[i] == 1:
                curr_block = np.fliplr(curr_block)
            curr_block = np.rot90(curr_block, k=-rot_params[i])

            temp_blocks[i] = curr_block

        # Inverse Step 2: Unscramble
        for i in range(num_blocks):
            processed_blocks[perm_indices[i]] = temp_blocks[i]

    # Integration
    final_img_array = np.zeros((rows * block_size, cols * block_size, 3), dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            final_img_array[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size] = \
                processed_blocks[idx]
            idx += 1

    result_img = Image.fromarray(final_img_array)

    if mode == 'decrypt':
        orig_w = w - pad_w
        orig_h = h - pad_h
        result_img = result_img.crop((0, 0, orig_w, orig_h))

    try:
        # subsampling=0 ensures best color quality (4:4:4), helping preservation
        result_img.save(output_path, quality=95, subsampling=0)
        print(f"Success! Output saved to: {output_path} / 处理完成")
    except Exception as e:
        print(f"Error: Failed to save file. / 错误: 保存失败 ({e})")


def main():
    parser = argparse.ArgumentParser(
        description="JPEG Encryption-then-Compression Tool (EtC) / JPEG 抗压缩混淆工具",
        epilog="Examples / 示例:\n"
               "  python etc_tool.py input.jpg enc.jpg --mode encrypt --seed 114514\n"
               "  python etc_tool.py enc.jpg dec.jpg --mode decrypt --seed 114514",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input_file",
                        help="Path to input image / 输入图片路径")

    parser.add_argument("output_file",
                        help="Path to output image / 输出图片路径")

    parser.add_argument("-m", "--mode", choices=['encrypt', 'decrypt'], default='encrypt',
                        help="Operation mode: 'encrypt' or 'decrypt' (Default: encrypt) / "
                             "操作模式: encrypt(加密) 或 decrypt(解密)")

    parser.add_argument("-s", "--seed", default=114514,
                        help="Seed for random number generator (Default: 114514). "
                             "Must match for encryption and decryption. / "
                             "密钥(Seed)，加解密必须一致")

    parser.add_argument("-b", "--block-size", type=int, default=16,
                        help="Block size (Default: 16). "
                             "Must be a multiple of 16 for JPEG compatibility. / "
                             "分块大小，建议为16以适配JPEG标准")

    args = parser.parse_args()

    if args.block_size < 8 or args.block_size % 8 != 0:
        print("Warning: Recommended block size is 16 or multiple of 8 for best JPEG resilience. / "
              "警告: 建议 block-size 为 16 或 8 的倍数以获得最佳抗压缩性。")

    encryption_then_compression(
        image_path=args.input_file,
        output_path=args.output_file,
        seed=args.seed,
        block_size=args.block_size,
        mode=args.mode
    )


if __name__ == "__main__":
    main()