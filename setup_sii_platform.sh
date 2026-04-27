#!/bin/bash
# ============================================================
# OneMap 环境搭建脚本 - SII 启智平台
#
# 使用流程：
#   1. 在可上网区(CPU/4090)开一个实例，选基础镜像 ngc-pytorch:24.05-cuda12.4-py3
#   2. 运行本脚本的 Phase 1 和 Phase 2
#   3. 保存镜像（保存前先清cache：conda clean --all && pip cache purge）
#   4. 用保存的镜像在 H100 区启动实例
#   5. H100 实例中运行 Phase 3 验证
#
# 注意：
#   - 个人存储路径需要替换为你的实际路径
#   - Matterport3D 数据需要单独上传到个人存储
#   - 创建实例时记得开启共享内存（建议设为总内存的2/3）
# ============================================================

PERSONAL_STORAGE="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/YOUR_NAME"  # 替换为你的个人存储路径

# ======================== Phase 1: 基础环境 ========================
# 在可上网区执行

phase1_base() {
    echo "===== Phase 1: 安装基础环境 ====="

    # 1. 配置 apt 源（可上网区可能不需要，但保险起见）
    cat > /etc/apt/sources.list << EOF
deb http://nexus.sii.shaipower.online/repository/ubuntu/ jammy main restricted universe multiverse
deb-src http://nexus.sii.shaipower.online/repository/ubuntu/ jammy main restricted universe multiverse
deb http://nexus.sii.shaipower.online/repository/ubuntu/ jammy-security main restricted universe multiverse
deb-src http://nexus.sii.shaipower.online/repository/ubuntu/ jammy-security main restricted universe multiverse
deb http://nexus.sii.shaipower.online/repository/ubuntu/ jammy-updates main restricted universe multiverse
deb-src http://nexus.sii.shaipower.online/repository/ubuntu/ jammy-updates main restricted universe multiverse
deb http://nexus.sii.shaipower.online/repository/ubuntu/ jammy-backports main restricted universe multiverse
deb-src http://nexus.sii.shaipower.online/repository/ubuntu/ jammy-backports main restricted universe multiverse
EOF

    # 2. 安装系统依赖
    apt-get update
    apt-get install -y xvfb zip unzip tmux htop git-lfs

    # 3. 安装 H100 图形驱动（habitat-sim EGL 渲染需要）
    # H100 驱动版本 570.124.06
    wget http://archive.ubuntu.com/ubuntu/pool/multiverse/n/nvidia-graphics-drivers-570-server/libnvidia-gl-570-server_570.124.06-0ubuntu1_amd64.deb
    dpkg -i libnvidia-gl-570-server_570.124.06-0ubuntu1_amd64.deb || apt -f install -y
    rm -f libnvidia-gl-570-server_570.124.06-0ubuntu1_amd64.deb

    # 4. 安装 miniconda（如果镜像没自带）
    if ! command -v conda &> /dev/null; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p /opt/miniconda3
        rm /tmp/miniconda.sh
        echo 'source /opt/miniconda3/bin/activate' >> ~/.bashrc
        source /opt/miniconda3/bin/activate
    fi

    echo "===== Phase 1 完成 ====="
}

# ======================== Phase 2: Python 环境 ========================
# 在可上网区执行

phase2_python_env() {
    echo "===== Phase 2: 创建 Python 环境 ====="

    # 1. 创建 conda 环境
    conda create -n onemap python=3.8 -y
    conda activate onemap || source activate onemap

    # 2. 安装 PyTorch (CUDA 12.1)
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

    # 3. 安装 habitat-sim（通过 conda，最稳定的方式）
    conda install habitat-sim=0.2.2 withbullet -c conda-forge -c aihabitat -y

    # 4. 安装标准 pip 包（从 requirements_pip.txt）
    pip install -r requirements_pip.txt

    # 5. 安装 GitHub 源码包（可上网区可以直接 git clone）
    pip install "git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2"
    pip install "git+https://github.com/ChaoningZhang/MobileSAM.git@b01a9ccef3b9e10b099b544efe004d0871802c3b"
    pip install "git+https://github.com/naokiyokoyama/depth_camera_filtering@39d6e2f391c8b2198a67ad96f94bf6da0acd48a0"
    pip install "git+https://github.com/xb534/SED/@db20d93c0ec8b0ce7fae267a10f826dfc7da15f5#subdirectory=open_clip"

    # 6. 克隆项目代码到个人存储
    cd ${PERSONAL_STORAGE}
    git clone https://github.com/Yangming911/SemanticOneMap.git
    cd SemanticOneMap

    # 7. 安装项目本地包（planning_cpp）
    cd planning_cpp && pip install . && cd ..

    echo "===== Phase 2 完成 ====="
}

# ======================== Phase 3: H100 验证 ========================
# 在 H100 实例中执行

phase3_verify() {
    echo "===== Phase 3: H100 环境验证 ====="

    conda activate onemap || source activate onemap

    # 验证 CUDA
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

    # 验证 habitat-sim
    python -c "import habitat_sim; print(f'Habitat-sim: {habitat_sim.__version__}')"

    # 验证关键包
    python -c "
import open_clip
import ultralytics
import mobile_sam
import detectron2
print('All key packages imported successfully!')
"

    # 验证 EGL 渲染（habitat 需要）
    xvfb-run -a python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = 'NONE'
print('EGL rendering OK')
"

    echo "===== Phase 3 完成 ====="
    echo ""
    echo "下一步："
    echo "  1. 将 Matterport3D 数据放到个人存储: ${PERSONAL_STORAGE}/data/"
    echo "  2. 修改 config/mon/eval_conf.yaml 中的数据路径"
    echo "  3. 运行: xvfb-run -a python -u eval_habitat.py -c config/mon/eval_conf.yaml"
}

# ======================== 保存镜像前清理 ========================
pre_save_cleanup() {
    echo "===== 清理缓存（保存镜像前必做）====="
    conda clean --all -y
    pip cache purge
    rm -rf /tmp/* /root/.cache/pip
    echo "清理完成，现在可以保存镜像了"
}

# ======================== 执行入口 ========================
echo "OneMap SII Platform Setup"
echo "========================="
echo "用法："
echo "  source setup_sii_platform.sh"
echo "  phase1_base          # 安装系统依赖"
echo "  phase2_python_env    # 安装 Python 环境"
echo "  phase3_verify        # H100 上验证"
echo "  pre_save_cleanup     # 保存镜像前清理"
echo ""
echo "建议逐步执行，每步确认无误后再继续"
