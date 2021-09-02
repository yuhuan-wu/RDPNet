
cd ../
SAVE_ROOT=training_dir/
SAVE_DIR=come_r50_try2/
SAVE_PATH=$SAVE_ROOT/$SAVE_DIR
CONFIG_NAME=configs/rdpnet/r50-come.yaml

# rsync codes to SAVE_PATH
mkdir $SAVE_ROOT
mkdir $SAVE_ROOT/$SAVE_DIR

rsync -auv maskrcnn_benchmark $SAVE_PATH
rsync -auv scripts/$0 $SAVE_PATH
rsync -auv $CONFIG_NAME $SAVE_PATH

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file $CONFIG_NAME \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR $SAVE_PATH
cd scripts
