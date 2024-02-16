#python3 setup.py build develop #--no-deps # for building d2
export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=0

ID="cs"

config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet_r101.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet-regnet.yaml"
# config_path="configs/COCOA-Base-RCNN-FPN-Fast-BCNet.yaml"
# config_path="configs/D2SA-Base-RCNN-FPN-Fast-BCNet.yaml"

weight_path="weights/r50.pth"

python3 tools/train_net.py --num-gpus 1 \
        --config-file ${config_path} \
        --eval-only MODEL.WEIGHTS ${weight_path} 2>&1 | tee log/${ID}.txt