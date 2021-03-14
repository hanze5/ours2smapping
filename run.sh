#!/bin/bash
op=$1
if [ "$op" == "demo" ]; then
  python3 mapper.py --forward_only=True --continue_training=True --use_attention=True \
  --input_vec_len=5 --data_path=data/m2d1.sub.grf.converted --target_graph_path=data/m2d1.grf.converted \
  --max_input_seq_len=4 --max_output_seq_len=6 --max_iteration=50
elif [ "$op" == "demo_200" ]; then
  nohup python3 mapper.py --forward_only=True --continue_training=False --use_attention=True \
  --input_vec_len=6 --data_path=data/si2_r001_m200.A50.out --target_graph_path=data/si2_r001_m200.B50.out \
  --max_input_seq_len=40 --max_output_seq_len=42 ${extra_op}
elif [ "$op" == "test20" ]; then
  python3 mapper.py --forward_only=True --continue_training=False --use_attention=True \
  --input_vec_len=4 --data_path=test/si2_r001_s20.A00.out --target_graph_path=test/si2_r001_s20.B00.out \
  --max_input_seq_len=4 --max_output_seq_len=6 --max_iteration=5000

elif [ "$op" == "test40" ]; then
  python3 mapper.py --forward_only=True --continue_training=False --use_attention=True \
  --input_vec_len=5 --data_path=data/si2_r005_s40.A28.out --target_graph_path=test/si2_r005_s40.B28.out \
  --max_input_seq_len=8 --max_output_seq_len=10 --max_iteration=10000

elif [ "$op" == "test60" ]; then
  python3 mapper.py --forward_only=True --continue_training=False --use_attention=True \
  --input_vec_len=4 --data_path=test/si2_r001_s60.A00.out --target_graph_path=test/si2_r001_s60.B00.out \
  --max_input_seq_len=12 --max_output_seq_len=14 --max_iteration=5000

elif [ "$op" == "test100" ]; then
  python3 mapper.py --forward_only=True --continue_training=False --use_attention=True \
  --input_vec_len=5 --data_path=test/si2_r001_s100.A00.out --target_graph_path=test/si2_r001_s100.B00.out \
  --max_input_seq_len=20 --max_output_seq_len=22

elif [ "$op" == "clean" ]; then
    rm -f out.txt best_reward.txt
    rm -rf ./log/*

elif [ "$op" == "show" ]; then
  tensorboard --logdir=log/train --port=6007
else
  echo "./run (test_case_name | clean)"
fi
