CUDA_VISIBLE_DEVICES=0 python tools/predict.py \
    --input_file /home/PaddleVideo/IPC3.mp4 \
    --config configs/recognition/pptsm/pptsm_fight_frames_dense.yaml \
    --model_file inference/ppTSM_fight/ppTSM/model.pdmodel \
    --params_file inference/ppTSM_fight/ppTSM/model.pdiparams \
    --use_gpu=True \
    --use_tensorrt=False