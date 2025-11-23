import cv2
import numpy as np
import logging
import os
import argparse # 引入命令行接口模块

# 【配置】设置日志系统
# 作用：代替 print，能在控制台输出带时间戳的运行记录，方便排查故障
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ==========================================
# Part 1: 视觉内核
# ==========================================
def process_single_image(file_path, output_dir):
    """
    核心处理函数：输入一张图片的路径，完成 识别 -> 画图 -> 保存 的全过程
    """
    # 1. 【动作】读取图片
    # 工业现场注意：路径中不能包含中文，否则 cv2.imread 会读取失败返回 None
    img = cv2.imread(file_path)

    # 防呆检查：如果图片损坏或路径错误，直接跳过，防止程序崩溃
    if img is None:
        logging.warning(f"无法读取图片: {file_path}，跳过。")
        return

    # 2. 【原理】图像预处理 (彩色转灰度)
    # 原因：彩色图有RGB三个通道，数据量大；做轮廓识别只需要亮度信息(灰度)，处理速度快3倍
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 【核心】二值化 (图像分割)
    # 逻辑：设定阈值 80。比 80 暗的(耳机盒)变成白色，比 80 亮的(背景)变成黑色
    # 参数说明：
    #   - 80: 经验值，根据现场光照调整 (光强则调高，光弱则调低)
    #   - THRESH_BINARY_INV: 反向阈值，因为 findContours 找的是白色区域，而耳机盒是黑色的
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 4. 【动作】查找轮廓 (Blob 分析)
    # 作用：把二值化图像中的白色块边缘找出来，存到 contours 列表里
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_count = 0  # 计数器，记录这张图找到了几个合格产品

    # 5. 【逻辑】筛选与结果绘制 (遍历每一个找到的轮廓)
    for cnt in contours:
        # 计算当前轮廓的面积 (像素单位)
        area = cv2.contourArea(cnt)

        # --- 工业级过滤 (重点) ---
        # 目的：剔除干扰项
        # MIN_AREA (500): 过滤灰尘、噪点
        # MAX_AREA (200000): 过滤背景误判、阴影
        if area < 500 or area > 200000:
            continue  # 如果不满足条件，直接跳过，看下一个轮廓

        # --- 计算几何特征 ---
        # 使用几何矩 (Moments) 计算轮廓的重心 (cx, cy)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 1. 画外接矩形 (红色框, 线宽15) - 标记识别范围
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 15)

            # 2. 画中心点 (绿色实心圆) - 标记抓取坐标
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

            # 3. 写文字 (亮黄色) - 显示机器视觉坐标
            # 参数：图片, 内容, 坐标, 字体, 字号(3), 颜色(黄), 线宽(5)
            text = f"Pos: ({cx}, {cy})"
            cv2.putText(img, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

            found_count += 1
            logging.info(f"  -> 找到目标，位置: ({cx}, {cy})，面积: {area}")

    # ==========================================
    # Part 2: 结果保存 (相当于自动存图功能)
    # ==========================================

    # 获取原始文件名 (例如: input/test.jpg -> test.jpg)
    filename = os.path.basename(file_path)

    # 拼接保存路径 (例如: output_result/processed_test.jpg)
    save_path = os.path.join(output_dir, f"processed_{filename}")

    # 写入硬盘
    cv2.imwrite(save_path, img)
    logging.info(f"处理完成: {filename} -> 保存至 {save_path}")


# ==========================================
# Part 3: 自动化外壳 (主程序入口)
# ==========================================
def main():
    # 1. 定义命令行接口 (相当于软件的设置界面)
    parser = argparse.ArgumentParser(description="工业视觉批量处理器")
    parser.add_argument('--input', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', default='./output_result', help='处理结果保存路径')

    args = parser.parse_args()

    # 2. 环境准备
    # 如果输出文件夹不存在，自动创建一个，免得报错
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"新建输出目录: {args.output}")

    logging.info(f">>> 开始批量处理: {args.input}")

    # 3. 批量循环 (相当于传送带，把文件一个个送进去)
    files = os.listdir(args.input)
    for file in files:
        # 过滤非图片文件 (防止读到 .txt 或文件夹报错)
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            # 拼接绝对路径
            full_path = os.path.join(args.input, file)

            # 【关键】调用上面的视觉内核进行处理
            process_single_image(full_path, args.output)

    logging.info(">>> 所有图片处理完毕！去 output_result 文件夹收货吧！")


if __name__ == '__main__':
    main()