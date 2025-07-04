#!/bin/bash

# 设置基本路径
ISAAC_LAB_ROOT="/home/shi/IsaacLab"
SCRIPT_PATH="${ISAAC_LAB_ROOT}/source/standalone/sky/fyp_sky/visual_pushing_grasping.py"

# 默认参数
NUM_ENVS=1                           # 环境数量
CAMERA_ID=0                          # 相机ID (0=顶部相机, 1=侧面相机)
DEBUG=false                          # 是否开启调试输出
SAVE=false                           # 是否保存数据
SAVE_PLY=false                       # 是否保存点云为PLY格式
# 更新默认工作空间范围以适应点云坐标变换后的情况
# 特别是将Y轴范围调整为 -0.2 到 0.2 变为 0.0 到 0.4，以匹配Y轴偏移修正
WORKSPACE_LIMITS="0.3 0.7 -0.2 0.2 0.0 0.3"  # 工作空间范围 [x_min x_max y_min y_max z_min z_max]
HEIGHTMAP_RESOLUTION=0.002           # 高度图分辨率（米/像素）

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --num_envs)
            NUM_ENVS="$2"
            shift
            shift
            ;;
        --camera_id)
            CAMERA_ID="$2"
            shift
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --save)
            SAVE=true
            shift
            ;;
        --save_ply)
            SAVE_PLY=true
            shift
            ;;
        --workspace_limits)
            WORKSPACE_LIMITS="$2"
            shift
            shift
            ;;
        --heightmap_resolution)
            HEIGHTMAP_RESOLUTION="$2"
            shift
            shift
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# 构建命令
CMD="${ISAAC_LAB_ROOT}/isaaclab.sh -p ${SCRIPT_PATH} --num_envs ${NUM_ENVS} --camera_id ${CAMERA_ID}"

# 添加可选参数
if [ "$DEBUG" = true ]; then
    CMD="${CMD} --debug"
fi

if [ "$SAVE" = true ]; then
    CMD="${CMD} --save"
fi

if [ "$SAVE_PLY" = true ]; then
    CMD="${CMD} --save_ply"
fi

# 添加工作空间限制和高度图分辨率
CMD="${CMD} --workspace_limits ${WORKSPACE_LIMITS} --heightmap_resolution ${HEIGHTMAP_RESOLUTION}"

# 打印并执行命令
echo "执行命令: ${CMD}"
echo "相机ID: ${CAMERA_ID} ($([ $CAMERA_ID -eq 0 ] && echo '顶部相机' || echo '侧面相机'))"
echo "工作空间范围: ${WORKSPACE_LIMITS}"
echo "说明: 工作空间范围格式为 'x_min x_max y_min y_max z_min z_max'"
echo "      当前Y轴范围已调整为适应点云坐标变换"
echo "高度图分辨率: ${HEIGHTMAP_RESOLUTION}米/像素"
echo "调试模式: $([ "$DEBUG" = true ] && echo '开启' || echo '关闭')"
echo "保存数据: $([ "$SAVE" = true ] && echo '是' || echo '否')"
echo "保存PLY点云: $([ "$SAVE_PLY" = true ] && echo '是' || echo '否')"

# 执行命令
eval ${CMD} 