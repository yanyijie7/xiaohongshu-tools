import os
import configparser
import numpy as np
from PIL import Image


def load_config(config_path='config.ini'):
    """加载并验证配置文件，兼容无target_index的情况"""
    config = configparser.ConfigParser()
    if not config.read(config_path, encoding='utf-8'):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 路径配置
    path_config = {
        'input_dir': config['PATH'].get('input_dir', 'input_images'),
        'base_image': config['PATH']['base_image'],
        'output_dir': config['PATH'].get('output_dir', 'output_images'),
        'overlay_image': config['PATH'].get('overlay_image', '').strip()
    }

    # 处理配置（关键修复：target_index改为可选，且仅在有overlay时生效）
    process_config = {
        'margin': int(config['PROCESS'].get('margin', 0)),
        'output_format': config['PROCESS'].get('output_format', 'png'),
        'target_index': None,  # 默认None
        'multiply_blend': config['PROCESS'].getboolean('multiply_blend', False)
    }

    # 仅当配置了overlay_image时，才尝试读取target_index
    if path_config['overlay_image']:
        # 检查是否存在target_index配置
        if 'target_index' not in config['PROCESS']:
            raise ValueError("配置了overlay_image时，必须同时设置target_index")
        # 尝试转换为整数
        try:
            process_config['target_index'] = int(config['PROCESS']['target_index'])
        except ValueError:
            raise ValueError(f"target_index必须是整数，当前值: {config['PROCESS']['target_index']}")

    # 验证底图
    if not os.path.exists(path_config['base_image']):
        raise FileNotFoundError(f"底图不存在: {path_config['base_image']}")

    # 验证覆盖图（若配置）
    if path_config['overlay_image'] and not os.path.exists(path_config['overlay_image']):
        raise FileNotFoundError(f"覆盖图不存在: {path_config['overlay_image']}")

    return {'path': path_config, 'process': process_config}


def get_all_image_paths(root_dir):
    """递归获取所有图片的输入输出路径对"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_pairs = []

    for dirpath, _, filenames in os.walk(root_dir):
        # 计算相对路径用于构建输出目录
        rel_path = os.path.relpath(dirpath, root_dir)
        for filename in filenames:
            if (filename.lower().endswith(image_extensions) 
                and not filename.startswith('.')):  # 过滤隐藏文件
                input_path = os.path.join(dirpath, filename)
                output_subdir = os.path.join(config['path']['output_dir'], rel_path)
                output_path = os.path.join(
                    output_subdir, 
                    f"{os.path.splitext(filename)[0]}.{config['process']['output_format']}"
                )
                image_pairs.append((input_path, output_path))

    # 按路径排序确保处理顺序一致
    return sorted(image_pairs, key=lambda x: x[0])


def apply_multiply_blend(base_img, overlay_img, position):
    """应用正片叠底混合模式
    
    Args:
        base_img: 底图 (RGBA)
        overlay_img: 叠加图片 (RGBA)
        position: 叠加位置 (x, y)
    
    Returns:
        处理后的图片
    """
    # 创建结果图片副本
    result = base_img.copy()
    
    # 获取图片尺寸
    base_width, base_height = base_img.size
    overlay_width, overlay_height = overlay_img.size
    pos_x, pos_y = position
    
    # 确保叠加区域在底图范围内
    crop_x_start = max(0, pos_x)
    crop_y_start = max(0, pos_y)
    crop_x_end = min(base_width, pos_x + overlay_width)
    crop_y_end = min(base_height, pos_y + overlay_height)
    
    if crop_x_start >= crop_x_end or crop_y_start >= crop_y_end:
        return result
    
    # 计算在叠加图片中的对应区域
    overlay_x_start = crop_x_start - pos_x
    overlay_y_start = crop_y_start - pos_y
    overlay_x_end = crop_x_end - pos_x
    overlay_y_end = crop_y_end - pos_y
    
    # 获取需要处理的区域
    base_region = result.crop((crop_x_start, crop_y_start, crop_x_end, crop_y_end))
    overlay_region = overlay_img.crop((overlay_x_start, overlay_y_start, overlay_x_end, overlay_y_end))
    
    # 转换为numpy数组进行像素级操作
    base_array = np.array(base_region)
    overlay_array = np.array(overlay_region)
    
    # 确保两个数组形状相同
    if base_array.shape != overlay_array.shape:
        return result
    
    # 应用正片叠底公式: result = (base * overlay) / 255
    # 只对RGB通道应用，Alpha通道保持不变
    result_rgb = (base_array[:, :, :3].astype(np.float32) * 
                 overlay_array[:, :, :3].astype(np.float32)) / 255.0
    
    # 确保值在0-255范围内
    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    
    # 处理Alpha通道：使用叠加图片的Alpha通道
    alpha = overlay_array[:, :, 3:4]
    
    # 根据Alpha通道混合原图和结果
    base_alpha = (255 - alpha).astype(np.float32) / 255.0
    overlay_alpha = alpha.astype(np.float32) / 255.0
    
    final_rgb = (base_array[:, :, :3].astype(np.float32) * base_alpha + 
                result_rgb.astype(np.float32) * overlay_alpha)
    
    final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)
    
    # 合并RGB和Alpha通道
    result_alpha = np.minimum(base_array[:, :, 3:4] + alpha, 255)
    final_array = np.concatenate([final_rgb, result_alpha], axis=2)
    
    # 将结果转换回PIL图片
    blended_region = Image.fromarray(final_array, 'RGBA')
    
    # 将处理后的区域粘贴回原图
    result.paste(blended_region, (crop_x_start, crop_y_start))
    
    return result


def process_single_image(base_img, input_img_path, margin, overlay_img=None, multiply_blend=False):
    """处理单张图片：缩放、居中叠加到底图，可选叠加覆盖图，支持正片叠底混合模式"""
    # 打开输入图片
    with Image.open(input_img_path).convert('RGBA') as input_img:
        # 计算可用区域（底图尺寸减去边距）
        base_width, base_height = base_img.size
        available_width = base_width - 2 * margin
        available_height = base_height - 2 * margin

        # 确保可用区域为正
        available_width = max(1, available_width)
        available_height = max(1, available_height)

        # 计算缩放比例
        img_width, img_height = input_img.size
        scale = min(available_width / img_width, available_height / img_height)

        # 缩放图片
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_img = input_img.resize(new_size, Image.Resampling.LANCZOS)

        # 计算居中位置
        x = margin + (available_width - new_size[0]) // 2
        y = margin + (available_height - new_size[1]) // 2

        # 创建底图副本
        result_img = base_img.copy()
        
        # 根据混合模式处理图片叠加
        if multiply_blend:
            # 正片叠底混合模式
            result_img = apply_multiply_blend(result_img, resized_img, (x, y))
        else:
            # 普通叠加模式
            result_img.paste(resized_img, (x, y), resized_img)

        # 叠加覆盖图（如果指定）
        if overlay_img:
            # 缩放覆盖图以匹配底图尺寸
            overlay_resized = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
            result_img.paste(overlay_resized, (0, 0), overlay_resized)

        return result_img


def batch_process_images(config):
    """批量处理所有图片"""
    path_cfg = config['path']
    process_cfg = config['process']

    # 获取所有图片路径对
    image_pairs = get_all_image_paths(path_cfg['input_dir'])
    if not image_pairs:
        print("未找到任何图片文件，程序退出")
        return

    # 加载底图
    with Image.open(path_cfg['base_image']).convert('RGBA') as base_img:
        # 加载覆盖图（若配置）
        overlay_img = None
        if path_cfg['overlay_image']:
            overlay_img = Image.open(path_cfg['overlay_image']).convert('RGBA')
            # 验证目标序号有效性
            if (process_cfg['target_index'] < 1 
                or process_cfg['target_index'] > len(image_pairs)):
                raise ValueError(
                    f"target_index超出范围，有效范围: 1-{len(image_pairs)}"
                )

        # 遍历处理图片
        for idx, (input_path, output_path) in enumerate(image_pairs, start=1):
            try:
                # 创建输出目录
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 判断是否需要叠加覆盖图
                current_overlay = overlay_img if (
                    overlay_img and idx == process_cfg['target_index']
                ) else None

                # 处理图片
                result_img = process_single_image(
                    base_img=base_img,
                    input_img_path=input_path,
                    margin=process_cfg['margin'],
                    overlay_img=current_overlay,
                    multiply_blend=process_cfg['multiply_blend']
                )

                # 保存结果
                result_img.save(output_path)
                print(f"[{idx}/{len(image_pairs)}] 已保存: {output_path}")

                # 处理到目标序号且有覆盖图时停止
                if current_overlay:
                    print(f"已处理目标序号 {idx}，停止处理")
                    break

            except Exception as e:
                print(f"[{idx}] 处理失败 {input_path}: {str(e)}")
                # 目标图片处理失败时终止程序
                if overlay_img and idx == process_cfg['target_index']:
                    print("目标图片处理失败，程序终止")
                    return

    # 关闭覆盖图
    if overlay_img:
        overlay_img.close()


if __name__ == "__main__":
    try:
        config = load_config()
        print("配置加载成功，开始处理图片...")
        batch_process_images(config)
        print("处理完成！")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        exit(1)