import os
import random
import sys
import carla
import numpy as np
import time
import pygame
import queue
import math
from PIL import Image
import json

from PIL import Image, ImageDraw, ImageFont


def init_random():
    random.seed(0)  # 设置 random 模块的随机数种子为 0
    np.random.seed(0)  # 设置 NumPy 模块的随机数种子为 0
    return None
def pygame_show_settings():
    # **---  指南针参数设置 ---**
    compass_radius = 50  # 指南针半径
    compass_center_x = compass_radius + 10  # 指南针中心 X 坐标 (左上角偏移)
    compass_center_y = compass_radius + 10  # 指南针中心 Y 坐标 (左上角偏移)
    compass_color = (200, 200, 200)  # 指南针颜色 (浅灰色)
    arrow_color = (255, 0, 0)  # 箭头颜色 (红色)
    north_direction = (0, -1)  # 北方向向量 (Pygame 坐标系，Y 轴向上为负)
    # **--- 指南针参数设置完成 ---**
    # **---  绘制方向文本 ---**
    pygame.font.init()  # 确保字体模块已初始化 (虽然在代码开头已经初始化，但为了代码片段的完整性保留)
    font = pygame.font.Font(None, 20)  # 创建字体对象，None 表示使用默认字体，20 为字体大小 (您可以调整字体大小)
    text_color = (0, 0, 0)  # 文本颜色 (黑色)
    n_text = font.render("N", True, text_color)  # 创建北方文本对象
    s_text = font.render("S", True, text_color)  # 创建南方文本对象
    w_text = font.render("W", True, text_color)  # 创建西方文本对象
    e_text = font.render("E", True, text_color)  # 创建东方文本对象
    return compass_radius, compass_center_x, compass_center_y,compass_color,arrow_color, n_text, s_text, w_text, e_text
def pygame_draw(screen,pedestrian,compass_radius, compass_center_x, compass_center_y,compass_color,arrow_color, n_text, s_text, w_text, e_text):
    # **---  绘制指南针 ---**
    pedestrian_yaw = pedestrian.get_transform().rotation.yaw  # 获取行人 Yaw 角
    compass_rotation_degrees = -pedestrian_yaw  # 反转 Yaw 角，并转换为 Pygame 逆时针角度
    compass_rotation_radians = math.radians(compass_rotation_degrees)  # 转换为弧度
    arrow_end_x = compass_center_x + compass_radius * math.cos(compass_rotation_radians)
    arrow_end_y = compass_center_y + compass_radius * math.sin(compass_rotation_radians)
    pygame.draw.circle(screen, compass_color, (compass_center_x, compass_center_y), compass_radius, 2)
    pygame.draw.line(screen, arrow_color, (compass_center_x, compass_center_y), (arrow_end_x, arrow_end_y), 3)
    screen.blit(n_text, (compass_center_x - n_text.get_width() / 2, compass_center_y - compass_radius - n_text.get_height() / 2 - 5))
    screen.blit(s_text, (compass_center_x - s_text.get_width() / 2, compass_center_y + compass_radius + s_text.get_height() / 2 + 5))
    screen.blit(w_text, (compass_center_x - compass_radius - w_text.get_width() / 2 - 5, compass_center_y - w_text.get_height() / 2))
    screen.blit(e_text, (compass_center_x + compass_radius + e_text.get_width() / 2 + 5, compass_center_y - e_text.get_height() / 2))
    # **---  绘制完成 ---**
    return screen
def spawn_pedestrian(world, max_attempts=5):
    ped_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    if not ped_blueprints:
        print("Error: No pedestrian blueprints found.")
        return None
    pedestrian_bp = random.choice(ped_blueprints)
    for attempt in range(max_attempts):
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()
        if spawn_point.location is None:
            print(f"Attempt {attempt + 1}: Unable to find a valid spawn point.")
            continue
        pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
        if pedestrian is not None:
            print(f"Pedestrian spawned successfully after {attempt + 1} attempts!")
            return pedestrian
        else:
            print(f"Attempt {attempt + 1}: Failed to spawn pedestrian.")
    print(f"Error: Failed to spawn pedestrian after {max_attempts} attempts.")
    return None
def update_spectator_view(spectator, pedestrian):
    """更新观察者视角"""
    pedestrian_transform = pedestrian.get_transform()
    spectator_transform = carla.Transform(
        pedestrian_transform.location - pedestrian_transform.rotation.get_forward_vector() * 3,
        pedestrian_transform.rotation)
    spectator_transform.location.z += 2.0
    spectator_rotation = spectator_transform.rotation
    spectator_rotation.pitch -= 15.0
    spectator_transform.rotation = spectator_rotation
    spectator.set_transform(spectator_transform)
def cleanup(world, original_settings, camera, camera_front, pedestrian):
    world.apply_settings(original_settings)
    if camera:
        camera.destroy()
    if camera_front:
        camera_front.destroy()
    if pedestrian:
        pedestrian.destroy()
    pygame.quit()

    # Destroy vehicles (This part is often necessary)
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.destroy()
def control_walker_with_inference(pedestrian, res, move_increment):
    """
    根据推理结果控制行人移动。

    Args:
        pedestrian: CARLA 行人对象。
        res: 推理函数返回的结果，包含 x_dist 和 y_dist。
        move_increment: 移动增量。
    """
    x_dist, y_dist = res  # 从推理结果中获取 x_dist 和 y_dist
    control_walker = carla.WalkerControl()
    control_walker.speed = math.sqrt(x_dist**2 + y_dist**2) * move_increment # 根据距离计算速度
    control_walker.direction.x = x_dist
    control_walker.direction.y = y_dist
    control_walker.direction.z = 0  # z 方向设置为 0
    pedestrian.apply_control(control_walker)

def inference(img_front):
    """
    固定返回往东方向的指令。
    """
    x_dist = 1.0  # 东方向的 x 分量
    y_dist = 0.0  # 东方向的 y 分量
    return [x_dist, y_dist]

def get_random_destination(world):
    """获取一个随机的目标位置。"""
    # 这里可以根据你的需求来生成随机位置
    # 例如，你可以使用 world.get_map().get_spawn_points() 获取一些随机的生成点
    # 然后选择其中一个作为目标位置
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        return random.choice(spawn_points).location
    else:
        return carla.Location(x=0, y=0, z=0)  # 返回一个默认位置

def draw_compass_pil(draw, width, height, pedestrian_yaw):
    """使用 PIL 在图像上绘制指南针。"""

    # **---  指南针参数设置 ---**
    compass_radius = 50  # 指南针半径
    compass_center_x = compass_radius + 10  # 指南针中心 X 坐标 (左上角偏移)
    compass_center_y = compass_radius + 10  # 指南针中心 Y 坐标 (左上角偏移)
    compass_color = (200, 200, 200)  # 指南针颜色 (浅灰色)
    arrow_color = (255, 0, 0)  # 箭头颜色 (红色)
    # **--- 指南针参数设置完成 ---**

    # **---  绘制方向文本 ---**
    font = ImageFont.truetype('arial.ttf', 24)  # 创建字体对象，并设置字号
    text_color = (0, 0, 0)  # 文本颜色 (黑色)
    # **---  绘制方向文本完成 ---**

    # **---  绘制指南针 ---**
    yaw_rad = math.radians(-pedestrian_yaw)  # 将偏航角转换为弧度，并取负值
    arrow_end_x = compass_center_x + compass_radius * math.cos(yaw_rad)
    arrow_end_y = compass_center_y + compass_radius * math.sin(yaw_rad)

    # 绘制圆形
    draw.ellipse((compass_center_x - compass_radius, compass_center_y - compass_radius, compass_center_x + compass_radius, compass_center_y + compass_radius), outline=compass_color, width=2)
    # 绘制箭头
    draw.line((compass_center_x, compass_center_y, arrow_end_x, arrow_end_y), fill=arrow_color, width=3)

    # 绘制文字
    w_box = font.getbbox("W")
    w_width = w_box[2] - w_box[0]
    draw.text((compass_center_x - compass_radius - 5 - w_width / 2, compass_center_y - 15), "W", fill=text_color, font=font)  # 西

    e_box = font.getbbox("E")
    e_width = e_box[2] - e_box[0]
    draw.text((compass_center_x + compass_radius + 5 - e_width / 2, compass_center_y - 15), "E", fill=text_color, font=font)  # 东

    n_box = font.getbbox("N")
    n_width = n_box[2] - n_box[0]
    draw.text((compass_center_x - n_width / 2, compass_center_y - compass_radius - 15), "N", fill=text_color, font=font)  # 北

    s_box = font.getbbox("S")
    s_width = s_box[2] - s_box[0]
    draw.text((compass_center_x - s_width / 2, compass_center_y + compass_radius + 5), "S", fill=text_color, font=font)  # 南

def calculate_relative_info(current_location, destination, walker_rotation):
    """计算目标位置相对于当前位置的信息 (仅考虑 x 和 y 分量)，
       并返回未归一化和归一化的方向向量，以及行人的当前朝向。
    """

    # 1. 未归一化的方向向量
    unnormalized_direction_vector = destination - current_location
    unnormalized_direction_vector = carla.Vector3D(unnormalized_direction_vector.x, unnormalized_direction_vector.y, 0)  # 仅保留 x 和 y 分量

    # 2. 归一化的方向向量
    distance = math.sqrt(unnormalized_direction_vector.x**2 + unnormalized_direction_vector.y**2)
    if distance > 0:  # 避免除以0
        normalized_direction_vector = carla.Vector3D(unnormalized_direction_vector.x / distance, unnormalized_direction_vector.y / distance, 0)  # 仅考虑 x 和 y 分量
    else:
        normalized_direction_vector = carla.Vector3D(0, 0, 0)  # 如果距离为0，则方向向量为(0, 0, 0)

    # 3. 行人当前朝向 (假设 walker_rotation 是一个 carla.Rotation 对象)
    walker_yaw = walker_rotation.yaw  # yaw 角表示行人朝向

    return unnormalized_direction_vector, normalized_direction_vector, walker_yaw

def save_data_to_json(output_dir, frame_count, image_filename, direction_vec, norm_direction_vec, walker_yaw, walker_location, destination):
    """构建 JSON 数据并保存到文件。"""
    json_data = {
        "image_path": image_filename,
        "walker_location": {
            "x": walker_location.x,
            "y": walker_location.y,
            "z": walker_location.z
        },
        "destination": {
            "x": destination.x,
            "y": destination.y,
            "z": destination.z
        },
        "direction_vector": {
            "x": direction_vec.x,
            "y": direction_vec.y,
            "z": direction_vec.z  # 添加 z 分量
        },
        "normalized_direction_vector": {
            "x": norm_direction_vec.x,
            "y": norm_direction_vec.y,
            "z": norm_direction_vec.z  # 添加 z 分量
        },
        "walker_yaw": walker_yaw
    }

    json_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.json")
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=4)

def main():
    try:
        init_random()
        #init pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("CARLA 场景渲染")
        clock = pygame.time.Clock()
        #init client & world
        client = carla.Client('localhost', 2000)
        client.load_world('Town05')
        client.set_timeout(20.0)
        world = client.get_world()
        #settings
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds,settings.actor_active_distance,settings.hybrid_physics_mode= True, 0.05, 2000, True
        world.apply_settings(settings)
        #spectator
        spectator = world.get_spectator()
        #camera
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(z=1.5))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=spectator)
        image_queue = queue.Queue() # 创建队列，用于存储相机图像数据
        camera.listen(image_queue.put)  # 设置相机监听器，将图像数据放入队列
        #walker
        pedestrian=spawn_pedestrian(world, max_attempts=5)
        world.tick() # 更新 CARLA 世界
        print("行人已生成！")
        # 创建 "camera front" 相机
        camera_front_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_front_transform = carla.Transform(carla.Location(x=0.2, z=1.5)) # 初始位置，后续会更新
        camera_front = world.spawn_actor(camera_front_bp, camera_front_transform, attach_to=pedestrian)
        image_queue_front = queue.Queue()
        camera_front.listen(image_queue_front.put)
        move_increment = 2.1 # 设置行人移动速度增量
        #pygame settings
        compass_radius, compass_center_x, compass_center_y, compass_color, arrow_color, n_text, s_text, w_text, e_text=pygame_show_settings()
        # 设置保存图像的帧率间隔

        # 1. 创建带时间后缀的输出目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
        output_dir = f"Front_View_{timestamp}"  # 添加时间后缀
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0 #帧计数器

        destination = get_random_destination(world)
        while True:
            clock.tick_busy_loop(20)
            #1. 显示图像
            # *————————show in pygame window -----
            try:
                image = image_queue.get_nowait()
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                # *————————show in pygame window -----
                screen=pygame_draw(screen, pedestrian, compass_radius, compass_center_x, compass_center_y, compass_color,
                            arrow_color, n_text, s_text, w_text, e_text)
                update_spectator_view(spectator, pedestrian)
                pygame.display.flip() # 更新 Pygame 窗口显示
            except queue.Empty:
                continue


            # 2. 计算目标位置相对于行人当前位置的信息
            walker_location=pedestrian.get_location()
            walker_rotation = pedestrian.get_transform().rotation  # 获取行人的旋转信息
            direction_vec, norm_direction_vec, walker_yaw = calculate_relative_info(walker_location, destination, walker_rotation)
            print(f"行人位置{pedestrian.get_location()}, 目标位置{destination}, 行人距离目标位置的方向向量为：{direction_vec}")
            # 2. 保存当前帧的 image_front
            try:
                image_front = image_queue_front.get_nowait()
                array_front = np.frombuffer(image_front.raw_data, dtype=np.dtype("uint8"))
                array_front = np.reshape(array_front, (image_front.height, image_front.width, 4))
                array_front = array_front[:, :, :3]  # 去除 alpha 通道
                array_front = array_front[:, :, ::-1]  # BGR to RGB
                img_front_np = array_front  # 转换为 NumPy 数组

                # 保存为 PNG 图像文件
                image = Image.fromarray(img_front_np)  # 将 NumPy 数组转换为 PIL Image 对象
                draw = ImageDraw.Draw(image)  # 创建 ImageDraw 对象
                draw_compass_pil(draw, image.size[0], image.size[1], walker_yaw)  # 绘制指南针
                image_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
                image.save(image_filename)  # 保存图像

                save_data_to_json(output_dir, frame_count, image_filename, direction_vec, norm_direction_vec, walker_yaw, walker_location, destination)

                frame_count += 1  # 增加帧计数器

            except queue.Empty:
                img_front_np = None
            except Exception as e:
                print(f"Error getting or processing image_front: {e}")
                img_front_np = None



            # 3. 推理  <--- Corrected line
            res = inference(img_front_np)

            # 4. 控制行人移动
            if res is not None:
                control_walker_with_inference(pedestrian, res, move_increment)
            else:
                control_walker = carla.WalkerControl()
                control_walker.speed = 0.0
                pedestrian.apply_control(control_walker)

            # 判断是否到达目标位置
            if pedestrian.get_location().distance(destination) < 1.0:  # 设置一个阈值
                print("到达目标位置！")
                break  # 结束循环

            #5.更新世界状态
            world.tick()
    finally:
        cleanup(world, original_settings, camera, camera_front, pedestrian)
if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    main()