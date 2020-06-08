#!/bin/bash
cd external_data && \
tar -xzf noisy_student_efficientnet-b7.tar.gz && \
python3.6 convert_tf_to_pt.py --model_name efficientnet-b7 --tf_checkpoint noisy-student-efficientnet-b7 --output_file noisy_student_efficientnet-b7.pth && \
rm -rf noisy-student-efficientnet-b7 tmp && \
cd ..