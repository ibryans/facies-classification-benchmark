# SECTION TRAINING 
# nice -n 20 python section_train.py --device cuda:1 --channel_delta 0 --class_weights --aug
# nice -n 20 python section_train.py --device cuda:2 --channel_delta 3 --class_weights --aug
# nice -n 20 python section_train.py --device cuda:3 --channel_delta 5 --class_weights --aug

# python section_train_ts.py --device cuda:1 --channel_delta 0 --class_weights --aug

# BASELINE
python section_test.py --channel_delta  0 --split both --model_path runs/20230212_151356_section_deconvnet_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230212_151356_section_deconvnet_delta=0/output.txt;

# DELTA EVALUATION
python section_test.py --channel_delta  1 --split both --model_path runs/20230215_104429_section_deconvnet_delta=1/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_104429_section_deconvnet_delta=1/output.txt;
python section_test.py --channel_delta  2 --split both --model_path runs/20230215_112010_section_deconvnet_delta=2/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_112010_section_deconvnet_delta=2/output.txt;
python section_test.py --channel_delta  3 --split both --model_path runs/20230215_115513_section_deconvnet_delta=3/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_115513_section_deconvnet_delta=3/output.txt;
python section_test.py --channel_delta  5 --split both --model_path runs/20230215_131009_section_deconvnet_delta=5/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_131009_section_deconvnet_delta=5/output.txt;
python section_test.py --channel_delta  8 --split both --model_path runs/20230215_141339_section_deconvnet_delta=8/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_141339_section_deconvnet_delta=8/output.txt;

# AUGMENTATION EVALUATION
python section_test.py --channel_delta  0 --split both --model_path runs/20230215_150502_section_deconvnet_weighted_delta=0/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_150502_section_deconvnet_weighted_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230215_155638_section_deconvnet_aug=rotate_delta=0/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_155638_section_deconvnet_aug=rotate_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230215_165057_section_deconvnet_aug=Hflip_delta=0/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_165057_section_deconvnet_aug=Hflip_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230215_220406_section_deconvnet_aug=Vflip_delta=0/section_deconvnet_model.pkl --device cuda:0 > runs/20230215_220406_section_deconvnet_aug=Vflip_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230215_223610_section_deconvnet_aug=noise_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230215_223610_section_deconvnet_aug=noise_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230216_101709_section_deconvnet_aug=crop_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230216_101709_section_deconvnet_aug=crop_delta=0/output.txt;

python section_test.py --channel_delta  0 --split both --model_path runs/20230215_231942_section_deconvnet_aug=rotate,noise_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230215_231942_section_deconvnet_aug=rotate,noise_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230216_000802_section_deconvnet_aug=rotate,Hflip,noise_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230216_000802_section_deconvnet_aug=rotate,Hflip,noise_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230216_083502_section_deconvnet_aug=rotate,Vflip,noise_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230216_083502_section_deconvnet_aug=rotate,Vflip,noise_delta=0/output.txt;
python section_test.py --channel_delta  0 --split both --model_path runs/20230216_113551_section_deconvnet_aug=rotate,Vflip_delta=0/section_deconvnet_model.pkl --device cuda:0  > runs/20230216_113551_section_deconvnet_aug=rotate,Vflip_delta=0/output.txt;


# TWO-STREAM EVALUATION
python section_test.py --channel_delta  0 --split both --model_path runs/20230214_235631_section_two_stream_delta=0/section_two_stream_model.pkl --device cuda:0 > runs/20230214_235631_section_two_stream_delta=0/output.txt;