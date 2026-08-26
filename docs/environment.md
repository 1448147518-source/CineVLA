# 相机环境接口

CineVLA 的闭环推理只通过 `CameraEnvironment` 获取 RGB 观测：

```python
observation = environment.reset()
observation, terminated = environment.step(camera_pose)  # camera_pose: [7]
```

`camera_pose` 的格式是四元数 `wxyz` 加平移 `xyz`。环境应在物理执行或渲染完成后返回该动作对应的真实 RGB 图像，而不是缓存的未来视频帧。

## 离线回放

`OfflineReplayEnv` 按时序逐帧提供现有视频/帧目录，适合检查因果 rollout、模型日志和回归测试。它不根据动作渲染新图像，因此只是一种确定性调试环境，不能用于证明闭环控制效果。

## 接入渲染器或真实相机

实现一个同步回调 `render_fn(pose)`，输入 CPU 上的 `[7]` tensor，返回 RGB 的 `HWC` 或 `CHW` NumPy/Torch 数组。`RendererCameraEnv` 负责尺寸转换、范围归一化和 episode 计数：

```python
from envs.renderer import RendererCameraEnv

environment = RendererCameraEnv(
    render_fn=blender_bridge.render_pose,
    initial_pose=initial_pose,
    image_size=224,
    max_steps=29,
)
result = engine.run_environment(environment, text='环绕主体')
```

Blender/Unity/3DGS 或真实相机桥接层的唯一职责是：执行 pose，并返回该 pose 下新拍摄或新渲染的帧。不要把整个序列预先传给模型。
