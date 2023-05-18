CUDA_VISIBLE_DEVICES=0 python tools/predict.py \
    --input_file time-test/file.list \
    --config configs/recognition/pptsm/pptsm_fight_frames_dense.yaml \
    --model_file inference/ppTSM_fight/ppTSM/model.pdmodel \
    --params_file inference/ppTSM_fight/ppTSM/model.pdiparams \
    --use_gpu=True \
    --use_tensorrt=False \
    --time_test_file=True \
    --enable_mkldnn=False \
    --enable_benchmark=True \
    --disable_glog True

    # input_file:       指定测试文件/文件列表, 示例使用1.1小节提供的测试数据
    # time_test_file:   是否进行时间测试，请设为True
    # config:           指定模型配置文件
    # model_file:       指定推理文件pdmodel路径
    # params_file:      指定推理文件pdiparams路径
    # use_gpu:          是否使用GPU预测, False则使用CPU预测
    # use_tensorrt:     是否开启TensorRT预测
    # enable_benchmark: 开启benchmark时间测试，默认设为True
    # disable_glog:     是否关闭推理时的日志，请设为True