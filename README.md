# vehicle_recognition
neural network for vehicle recognition (based on signals from inductive-loop sensors)

correction.py and correction_service.py - modules used for sensor data correction
detection_change.py - functions responsible for detecting interesting part of signal (car passing the measurement station)
models.py - different models for recognition (accuracy measured with KFold cross-validation)
neural_net.py - final model
model_test - model tested with greater amount of data
