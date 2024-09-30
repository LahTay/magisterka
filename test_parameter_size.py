import math
from model.attentionMIC.spatial_groupwise_enhance_attentionMIC import create_attentionMIC_SGE_model
from model.attentionMIC.efficient_channel_attentionMIC import create_attentionMIC_ECA_model
from model.attentionMIC.time_frequency_attentionMIC import (create_attentionMIC_TFQ_model)
from model.attentionMIC.attentionMIC import create_attentionMIC_model, create_attentionMIC_multihead_model
from model.attentionMIC.no_attention import create_no_attentionMIC_model
from model.attentionMIC.ECA_TFQ_hybrid import create_attentionMIC_ECA_TFQ_hybrid_model
from model.attentionMIC.new_models import *



def main():
    input_shape = (40, 216, 1)
    num_outputs = 11

    model1 = create_attentionMIC_ECA_model(input_shape, num_outputs, 52)
    model2 = create_attentionMIC_SGE_model(input_shape, num_outputs, 8, filters=61 - 61 % 8 + 8)
    model3 = create_attentionMIC_TFQ_model(input_shape, num_outputs, False,
                                                       quantize=False, filters=50)
    model4 = create_no_attentionMIC_model(input_shape, num_outputs, 64)
    model5 = create_attentionMIC_ECA_TFQ_hybrid_model(input_shape, num_outputs, 42)
    model6 = create_attentionMIC_model(input_shape, num_outputs, 61)
    model7 = create_attentionMIC_multihead_model(input_shape, num_outputs, 42, 32, 8)

    model8 = build_resnet_model(input_shape, num_outputs, 38)
    model9 = build_combined_conv_model(input_shape, num_outputs, 32, 20)
    model10 = build_crnn_model(input_shape, num_outputs, 32, 14)
    model11 = build_crnn_bidirectional_model(input_shape, num_outputs, 24, 10)
    model12 = build_mobilenet_style_model(input_shape, num_outputs, 112)
    model13 = build_dilated_conv_model(input_shape, num_outputs, 64)
    model14 = build_inception_model(input_shape, num_outputs, 35, 25)
    model15 = build_multi_scale_cnn_model(input_shape, num_outputs, 27)
    model16 = build_se_model(input_shape, num_outputs, 64)
    model17 = build_tcn_model(input_shape, num_outputs, 54)
    model18 = build_mobilenet_resnet_style_model(input_shape, num_outputs, 104)



    print(f"For size {input_shape}, SMALL models have")
    print(f"Model 1 (ECA) parameters: {model1.count_params()}")
    print(f"Model 2 (SGE) parameters: {model2.count_params()}")
    print(f"Model 3 (TFQ) parameters: {model3.count_params()}")
    print(f"Model 4 (Base) parameters: {model4.count_params()}")
    print(f"Model 5 (ECA TFQ Hybrid) parameters: {model5.count_params()}")
    print(f"Model 6 (Attention) parameters: {model6.count_params()}")
    print(f"Model 7 (Multi-head Attention) parameters: {model7.count_params()}")

    print("\n")

    print("Small should be around: 95000")
    print("Big should be around: 165000")

    print("\n")

    print(f"Model 8 (ResNet) parameters: {model8.count_params()}")
    print(f"Model 9 (Combined 1d 2d conv) parameters: {model9.count_params()}")
    print(f"Model 10 (CRNN) parameters: {model10.count_params()}")
    print(f"Model 11 (CRNN bidirectional) parameters: {model11.count_params()}")
    print(f"Model 12 (MobileNet) parameters: {model12.count_params()}")
    print(f"Model 13 (Dilated) parameters: {model13.count_params()}")
    print(f"Model 14 (Inception) parameters: {model14.count_params()}")
    print(f"Model 15 (Multi Scale Conv) parameters: {model15.count_params()}")
    print(f"Model 16 (Squeeze and Excitation) parameters: {model16.count_params()}")
    print(f"Model 17 (Temporal Convolution) parameters: {model17.count_params()}")
    print(f"Model 18 (MobileNet/Resnet) parameters: {model18.count_params()}")


    model1 = create_attentionMIC_ECA_model(input_shape, num_outputs, 73)
    model2 = create_attentionMIC_SGE_model(input_shape, num_outputs, 8, filters=84 - 84 % 8 + 8)
    model3 = create_attentionMIC_TFQ_model(input_shape, num_outputs, False,
                                           quantize=False, filters=74)
    model4 = create_no_attentionMIC_model(input_shape, num_outputs, 90)
    model5 = create_attentionMIC_ECA_TFQ_hybrid_model(input_shape, num_outputs, 64)
    model6 = create_attentionMIC_model(input_shape, num_outputs, 86)
    model7 = create_attentionMIC_multihead_model(input_shape, num_outputs, 68, 32, 8)

    model8 = build_resnet_model(input_shape, num_outputs, 52)
    model9 = build_combined_conv_model(input_shape, num_outputs, 42, 28)
    model10 = build_crnn_model(input_shape, num_outputs, 46, 18)
    model11 = build_crnn_bidirectional_model(input_shape, num_outputs, 32, 14)
    model12 = build_mobilenet_style_model(input_shape, num_outputs, 160)
    model13 = build_dilated_conv_model(input_shape, num_outputs, 90)
    model14 = build_inception_model(input_shape, num_outputs, 52, 32)
    model15 = build_multi_scale_cnn_model(input_shape, num_outputs, 36)
    model16 = build_se_model(input_shape, num_outputs, 88)
    model17 = build_tcn_model(input_shape, num_outputs, 74)
    model18 = build_mobilenet_resnet_style_model(input_shape, num_outputs, 146)

    print(f"For size {input_shape}, BIG models have")
    print(f"Model 1 (ECA) parameters: {model1.count_params()}")
    print(f"Model 2 (SGE) parameters: {model2.count_params()}")
    print(f"Model 3 (TFQ) parameters: {model3.count_params()}")
    print(f"Model 4 (Base) parameters: {model4.count_params()}")
    print(f"Model 5 (ECA TFQ Hybrid) parameters: {model5.count_params()}")
    print(f"Model 6 (Attention) parameters: {model6.count_params()}")
    print(f"Model 7 (Multi-head Attention) parameters: {model7.count_params()}")

    print("\n")

    print("Small should be around: 95000")
    print("Big should be around: 165000")

    print("\n")

    print(f"Model 8 (ResNet) parameters: {model8.count_params()}")
    print(f"Model 9 (Combined 1d 2d conv) parameters: {model9.count_params()}")
    print(f"Model 10 (CRNN) parameters: {model10.count_params()}")
    print(f"Model 11 (CRNN bidirectional) parameters: {model11.count_params()}")
    print(f"Model 12 (MobileNet) parameters: {model12.count_params()}")
    print(f"Model 13 (Dilated) parameters: {model13.count_params()}")
    print(f"Model 14 (Inception) parameters: {model14.count_params()}")
    print(f"Model 15 (Multi Scale Conv) parameters: {model15.count_params()}")
    print(f"Model 16 (Squeeze and Excitation) parameters: {model16.count_params()}")
    print(f"Model 17 (Temporal Convolution) parameters: {model17.count_params()}")
    print(f"Model 18 (MobileNet/Resnet) parameters: {model18.count_params()}")


if __name__ == "__main__":
    main()










