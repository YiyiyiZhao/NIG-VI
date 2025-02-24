import carla
import time
import random
import queue
import math
import os
import sys
import datetime
import matplotlib.pyplot as plt
import pygame
import mss
import numpy as np
import cv2
import pyautogui

module_dir = os.path.abspath("/home/yi/Projects/Carla_0.9/PythonAPI/carla")
sys.path.append(module_dir)

def init_random():
    random.seed(0)
    return None
def init_world(town='Town05'):
    # 连接到 CARLA 客户端
    client = carla.Client('localhost', 2000)
    client.load_world(town)  # 载入 Town05 地图，您可以根据需要更改
    client.set_timeout(20.0)
    world = client.get_world()
    return world, client
def set_syn(world):
    # 启动同步模式
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.actor_active_distance = 2000
    settings.hybrid_physics_mode = 0.05
    world.apply_settings(settings)
    return original_settings
def spawn_pedestrian_and_set_destination(world, min_distance=5):
    ped_blueprints = world.get_blueprint_library().filter('*pedestrian*')
    pedestrian_bp = random.choice(ped_blueprints)
    spawn_point = carla.Transform()
    spawn_point.location = world.get_random_location_from_navigation()
    max_spawn_attempts = 5
    spawn_attempt = 0
    pedestrian=None
    while spawn_attempt < max_spawn_attempts:
        pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
        if pedestrian:
            print("行人生成成功！")
            break
        else:
            print("Retrying ...")
            spawn_attempt += 1
            spawn_point.location = world.get_random_location_from_navigation()
    world.tick()

    start_location = pedestrian.get_location()
    while True:
        destination_location = world.get_random_location_from_navigation()
        distance_x = destination_location.x - start_location.x
        distance_y = destination_location.y - start_location.y
        distance_xy = math.sqrt(distance_x**2 + distance_y**2)

        if distance_xy >= min_distance:
            break
    print(f"Starting location: {start_location}")
    print(f"Destination location: {destination_location}")
    return pedestrian, start_location, destination_location

def setup_camera(world, pedestrian, town, image_folder_front="camera_images_front", image_folder_semantic="camera_images_semantic_front"):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    semantic_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_transform = carla.Transform(carla.Location(x=0.2, z=1.65))

    camera_front = world.spawn_actor(camera_bp, camera_transform, attach_to=pedestrian)
    semantic_camera = world.spawn_actor(semantic_bp, camera_transform, attach_to=pedestrian)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_folder_rgb = f"{timestamp}_{town}/{image_folder_front}"
    image_folder_sem = f"{timestamp}_{town}/{image_folder_semantic}"

    if not os.path.exists(image_folder_rgb):
        os.makedirs(image_folder_rgb)
    if not os.path.exists(image_folder_sem):
        os.makedirs(image_folder_sem)

    def save_rgb_image(image):
        image.save_to_disk(os.path.join(image_folder_rgb, f"{image.frame_number}.png"))

    def save_semantic_image(image):
        image.save_to_disk(os.path.join(image_folder_sem, f"{image.frame_number}.png"), cc_segmentation)

    camera_front.listen(save_rgb_image)
    cc_segmentation = carla.ColorConverter.CityScapesPalette
    semantic_camera.listen(save_semantic_image)
    return camera_front, semantic_camera,timestamp
def set_ai_control_walker(world, pedestrian, destination,speed=4.5):
    ai_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    ai_controller = world.spawn_actor(ai_controller_bp, carla.Transform(), pedestrian)
    ai_controller.start()
    ai_controller.go_to_location(destination)
    ai_controller.set_max_speed(speed)
    return ai_controller
def set_spectator_transform(spectator, pedestrian):
    """设置观察者视角跟随行人，位于行人后上方."""
    pedestrian_transform = pedestrian.get_transform()
    spectator_location = pedestrian_transform.location - pedestrian_transform.rotation.get_forward_vector() * 3.0
    spectator_location.z += 2.0
    spectator_rotation = pedestrian_transform.rotation
    spectator_rotation.pitch -= 15.0
    spectator_transform = carla.Transform(spectator_location, spectator_rotation)
    spectator.set_transform(spectator_transform)
def process_distance(current_location, destination_location, destination_reached_threshold=2):
    stop_flag = False
    distance_to_destination_x = destination_location.x - current_location.x  # 计算 X 轴方向距离分量
    distance_to_destination_y = destination_location.y - current_location.y  # 计算 Y 轴方向距离分量

    # 只计算 x 和 y 轴的距离
    distance_to_destination_xy = math.sqrt(distance_to_destination_x**2 + distance_to_destination_y**2)

    # 仍然保留总距离的计算，用于输出信息
    distance_to_destination = current_location.distance(destination_location)

    print(f"行人到目的地的距离:  "
          f"目的地[X:{distance_to_destination_x:.2f}米, Y:{distance_to_destination_y:.2f}米, XY:{distance_to_destination_xy:.2f}米, 总:{distance_to_destination:.2f}米],  ")

    if distance_to_destination_xy <= destination_reached_threshold:  # 检测是否到达目的地附近 (仅基于 x 和 y 轴)
        print("行人已到达目的地附近！停止移动。")
        stop_flag = True
    return stop_flag

def draw_trajectory_birdseye(world, start_location, destination_location, trajectory, timestamp, town, image_size_x=1280, image_size_y=720, fov=90, filename="walker_trajectory_birdseye.png"):
    filepath=os.path.join(f"{timestamp}_{town}", filename)

    # 初始化 pygame
    pygame.init() # 添加这一行

    # 初始化 pygame.font
    pygame.font.init()
    # 创建鸟瞰相机并保存图像
    center_location_x = (start_location.x + destination_location.x) / 2.0
    center_location_y = (start_location.y + destination_location.y) / 2.0

    camera_location = carla.Location(x=center_location_x, y=center_location_y, z=100)
    camera_rotation = carla.Rotation(pitch=-90)
    camera_transform = carla.Transform(camera_location, camera_rotation)

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_size_x))
    camera_bp.set_attribute('image_size_y', str(image_size_y))
    camera_bp.set_attribute('fov', str(fov))
    camera = world.spawn_actor(camera_bp, camera_transform)

    #save image
    image = None

    def get_image(pygame_image):
        nonlocal image
        image = pygame_image

    camera.listen(get_image)

    world.tick()
    time.sleep(1)
    camera.stop()
    camera.destroy()

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    # 绘制起点和终点标记
    start_pixel = (int(image_size_x / 2.0 - (center_location_x - start_location.x) * (image_size_x / 200.0)),
                   int(image_size_y / 2.0 - (center_location_y - start_location.y) * (image_size_y / 200.0)))
    destination_pixel = (int(image_size_x / 2.0 - (center_location_x - destination_location.x) * (image_size_x / 200.0)),
                         int(image_size_y / 2.0 - (center_location_y - destination_location.y) * (image_size_y / 200.0)))

    # 绘制起点标记和文本
    if 0 <= start_pixel[0] < image_size_x and 0 <= start_pixel[1] < image_size_y:
        pygame.draw.circle(surface, pygame.Color('green'), start_pixel, 5)
        # 绘制文本背景
        text_bg = pygame.Surface((50, 20))
        text_bg.fill(pygame.Color('white'))
        surface.blit(text_bg, (start_pixel[0] + 10, start_pixel[1] - 20))
        # 绘制文本
        font = pygame.font.Font(None, 24)
        text_surface = font.render("Start", True, pygame.Color('green'))
        surface.blit(text_surface, (start_pixel[0] + 10, start_pixel[1] - 20))

    # 绘制终点标记和文本
    if 0 <= destination_pixel[0] < image_size_x and 0 <= destination_pixel[1] < image_size_y:
        pygame.draw.rect(surface, pygame.Color('red'), (destination_pixel[0] - 5, destination_pixel[1] - 5, 10, 10))
        # 绘制文本背景
        text_bg = pygame.Surface((80, 20))
        text_bg.fill(pygame.Color('white'))
        surface.blit(text_bg, (destination_pixel[0] + 10, destination_pixel[1] - 20))
        # 绘制文本
        font = pygame.font.Font(None, 24)
        text_surface = font.render("Destination", True, pygame.Color('red'))
        surface.blit(text_surface, (destination_pixel[0] + 10, destination_pixel[1] - 20))

    # 绘制轨迹点
    for loc in trajectory:
        loc_pixel = (int(image_size_x / 2.0 - (center_location_x - loc.x) * (image_size_x / 200.0)), # 使用相同的比例因子
                     int(image_size_y / 2.0 - (center_location_y - loc.y) * (image_size_y / 200.0))) # 使用相同的比例因子

        if 0 <= loc_pixel[0] < image_size_x and 0 <= loc_pixel[1] < image_size_y:
            # 绘制文本背景
            text_bg = pygame.Surface((10, 10))
            text_bg.fill(pygame.Color('white'))
            surface.blit(text_bg, loc_pixel)
            # 绘制文本
            font = pygame.font.Font(None, 24)
            text_surface = font.render("*", True, pygame.Color('blue'))
            surface.blit(text_surface, loc_pixel)

    pygame.image.save(surface, filepath)
    print(f"Bird's-eye view with Pygame saved to: {filepath}")
def cleanup(world, original_settings, camera_front=None, semantic_camera=None):
    world.apply_settings(original_settings)  # **恢复原始世界设置 (异步模式)**

    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith('walker') or actor.type_id.startswith(
                'controller.ai.walker'):  # 检查 actor 的 type_id 是否以 'walker' 开头 (判断是否是行人)
            actor.destroy()  # 如果是行人，则销毁
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.destroy()

    if camera_front:
        camera_front.destroy()
    if  semantic_camera:
        semantic_camera.destroy()

def get_map_bounds(world):
    """获取 Carla 0.9 地图的边界."""
    topology = world.get_map().get_topology()
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for waypoint_pair in topology:
        waypoint1, waypoint2 = waypoint_pair
        location1 = waypoint1.transform.location
        location2 = waypoint2.transform.location

        min_x = min(min_x, location1.x, location2.x)
        min_y = min(min_y, location1.y, location2.y)
        max_x = max(max_x, location1.x, location2.x)
        max_y = max(max_y, location1.y, location2.y)

    return min_x, min_y, max_x, max_y

def main(town='Town05'):
    try:
        world, client=init_world(town)
        original_settings=set_syn(world)
        spectator = world.get_spectator()
        pedestrian,start_location,destination_location=spawn_pedestrian_and_set_destination(world)
        camera_front, semantic_camera, timestamp = setup_camera(world, pedestrian,town)
        ai_controller=set_ai_control_walker(world, pedestrian, destination_location,speed=10)

        trajectory = [start_location]
        frame_count=0

        while True:
            world.tick()
            set_spectator_transform(spectator, pedestrian)
            current_location = pedestrian.get_location()
            trajectory.append(current_location)
            stop_flag=process_distance(current_location, destination_location)
            if stop_flag:
                ai_controller.stop()
                cleanup(world, original_settings)
                break

            time.sleep(0.05)
            frame_count+=1

        print(trajectory)
        draw_trajectory_birdseye(world, start_location, destination_location, trajectory, timestamp, town)

    finally:
        cleanup(world, original_settings,camera_front, semantic_camera)

if __name__ == '__main__':
    for i in range(1,5):
        main(f'Town0{i}')