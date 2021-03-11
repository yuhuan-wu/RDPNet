
cd ../
SAVE_ROOT=training_dir/
SAVE_DIR=soc_r50_try2/
LOOP_IDX=$1
SAVE_PATH=$SAVE_ROOT/$SAVE_DIR/$LOOP_IDX
CONFIG_NAME=configs/rdpnet/r50-soc.yaml

# rsync codes to SAVE_PATH
mkdir $SAVE_ROOT
mkdir $SAVE_ROOT/$SAVE_DIR
mkdir $SAVE_ROOT/$SAVE_DIR/$LOOP_IDX

rsync -auv maskrcnn_benchmark $SAVE_PATH
rsync -auv scripts/train_mask.sh $SAVE_PATH
rsync -auv $CONFIG_NAME $SAVE_PATH

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file $CONFIG_NAME \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR $SAVE_PATH
cd scripts
