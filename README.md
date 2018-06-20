# MLND-Capstone-Project

main process:
1.main.first_step_training.py
2.main.second_step_training.py

bcnn model:
1.bcnn_model_architecture.bcnn_last_layer.py
2.bcnn_model_architecture.bcnn_last_layer.py

utils:
1.utils.create_test_h5_file.py
2.utils.create_train_val_h5_file.py

Step 1:
after download images, use utils.create_test_h5_file.py and utils.create_train_val_h5_file.py to create .h5 file.

Step 2:
run main.first_step_training.py and save the weights.

Step 3:
run main.second_step_training.py to load the weights from step 2

Step 4:
output the prediction and upload to Kaggle to evaluate