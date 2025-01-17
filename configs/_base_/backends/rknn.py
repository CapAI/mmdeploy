backend_config = dict(
    type="rknn",
    common_config=dict(target_platform="rv1126", optimization_level=1),  # 'rk3588'
    quantization_config=dict(do_quantization=True, dataset=None, pre_compile=False, rknn_batch_size=-1),
)
