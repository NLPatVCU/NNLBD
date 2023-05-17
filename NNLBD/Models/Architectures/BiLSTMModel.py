#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/07/2020                                                                   #
#    Revised: 01/13/2022                                                                   #
#                                                                                          #
#    Generates A Neural Network Used For LBD.                                              #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Suppress Warnings/FutureWarnings
import warnings
warnings.filterwarnings( 'ignore' )
#warnings.simplefilter( action = 'ignore', category = Warning )
#warnings.simplefilter( action = 'ignore', category = FutureWarning )   # Also Works For Future Warnings

# Standard Modules
import os, re

# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Removes TensorFlow GPU CUDA Checking Error/Warning Messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import tensorflow as tf
#tf.logging.set_verbosity( tf.logging.ERROR )                       # TensorFlow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # TensorFlow v1.x

import numpy as np
from tensorflow import keras

# TensorFlow v2.x Support
if re.search( r'2.\d+', tf.__version__ ):
    import tensorflow.keras.backend as K
    from tensorflow.keras import optimizers
    from tensorflow.keras import regularizers
    # from keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation, Average, BatchNormalization, Bidirectional, Concatenate, Dense, Dropout, Embedding, Flatten, Input, Lambda, LSTM, Multiply
# TensorFlow v1.15.x Support
else:
    import keras.backend as K
    from keras import optimizers
    from keras import regularizers
    # from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from keras.layers import Activation, Average, BatchNormalization, Bidirectional, Concatenate, Dense, Dropout, Embedding, Flatten, Input, Lambda, LSTM, Multiply

# Custom Modules
from NNLBD.Models.Base import BaseModel



############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class BiLSTMModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, network_model = "bilstm", model_type = "open_discovery",
                  optimizer = 'adam', activation_function = 'sigmoid', loss_function = "binary_crossentropy", number_of_embedding_dimensions = 200,
                  bilstm_dimension_size = 64, bilstm_merge_mode = "concat", learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.5,
                  batch_size = 32, prediction_threshold = 0.5, shuffle = True, use_csr_format = True, per_epoch_saving = False, use_gpu = True,
                  device_name = "/gpu:0", verbose = 2, debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False,
                  early_stopping_metric_monitor = "loss", early_stopping_persistence = 3, use_batch_normalization = False, trainable_weights = False,
                  embedding_path = "", final_layer_type = "dense", feature_scale_value = 1.0, learning_rate_decay = 0.004, weight_decay = 0.0001,
                  use_cosine_annealing = False, cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4, margin = 30.0, scale = 0.35,
                  embedding_modification = "concatenate", skip_gpu_init = False ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, network_model = network_model, model_type = model_type,
                          momentum = momentum, bilstm_merge_mode = bilstm_merge_mode, optimizer = optimizer, activation_function = activation_function,
                          batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format,
                          loss_function = loss_function, number_of_embedding_dimensions = number_of_embedding_dimensions, learning_rate = learning_rate,
                          bilstm_dimension_size = bilstm_dimension_size, epochs = epochs, dropout = dropout, per_epoch_saving = per_epoch_saving,
                          use_gpu = use_gpu, device_name = device_name, verbose = verbose, debug_log_file_handle = debug_log_file_handle, margin = margin,
                          enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, final_layer_type = final_layer_type,
                          early_stopping_metric_monitor = early_stopping_metric_monitor, early_stopping_persistence = early_stopping_persistence,
                          use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_path = embedding_path,
                          feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay, weight_decay = weight_decay, scale = scale,
                          use_cosine_annealing = use_cosine_annealing, cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max,
                          embedding_modification = embedding_modification, skip_gpu_init = skip_gpu_init )
        self.version       = 0.15
        self.network_model = "bilstm"   # Force Setting Model To 'Bi-LSTM' Model.


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

    """
        Converts Randomized Batches Of Model Inputs & Outputs From CSR_Matrix Format
          To Numpy Arrays For Model Training

        Inputs:
            X              : Model Concept Inputs (CSR_Matrix)
            Y              : Model Concept Outputs (CSR_Matrix)
            batch_size     : Batch Size (Integer)
            steps_per_batch: Number Of Iterations Per Epoch (Integer)
            shuffle        : Shuffles Data Prior To Conversion (Boolean)

        Outputs:
            X_1_batch      : Numpy 2D Matrix Of Model Concept Inputs (Numpy Array)
            X_2_batch      : Numpy 2D Matrix Of Model Predicate Inputs (Numpy Array)
            Y_batch        : Numpy 2D Matrix Of Model Concept Outputs (Numpy Array)

            Modification Of Code From Source: https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
    """
    def Batch_Generator( self, X, Y, batch_size, steps_per_batch, shuffle ):
        number_of_instances = X.shape[0]      # Should Be The Same As 'self.trained_instances'
        counter             = 0
        sample_index        = np.arange( number_of_instances )

        if shuffle:
            np.random.shuffle( sample_index )

        while True:
            start_index = batch_size * counter
            end_index   = batch_size * ( counter + 1 )

            # Check - Fixes Batch_Generator Training Errors With The Number Of Instances % Batch Sizes != 0
            if end_index > number_of_instances: end_index = number_of_instances

            batch_index = sample_index[start_index:end_index]
            X_batch     = X[batch_index,:]
            Y_input     = Y[batch_index,:].todense()
            Y_output    = Y_input
            counter     += 1

            # CosFace, ArcFace & SphereFae Final Layers
            if self.final_layer_type in ["cosface", "arcface", "sphereface"]:
                yield [X_batch, Y_input], Y_output
            # Dense Final Layer
            else:
                yield X_batch, Y_output

            # Reset The Batch Index After Final Batch Has Been Reached
            if counter == steps_per_batch:
                if shuffle:
                    np.random.shuffle( sample_index )
                counter = 0

    """
        Trains Model Using Training Data, Fits Model To Data

        Inputs:
            training_file_path          : Evaluation File Path (String)
            number_of_hidden_dimensions : Number Of Hidden Dimensions When Building Neural Network Architecture (Integer)
            learning_rate               : Learning Rate (Float)
            epochs                      : Number Of Training Epochs (Integer)
            batch_size                  : Size Of Each Training Batch (Integer)
            momentum                    : Momentum Value (Float)
            dropout                     : Dropout Value (Float)
            verbose                     : Sets Training Verbosity - Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)
            per_epoch_saving            : Toggle To Save Model After Each Training Epoch (Boolean: True, False)
            use_csr_format              : Toggle To Use Compressed Sparse Row (CSR) Formatted Matrices For Storing Training/Evaluation Data (Boolean: True, False)

        Outputs:
            None
    """
    def Fit( self, train_input_1 = None, train_input_2 = None, train_input_3 = None, train_outputs = None,
             epochs = None, batch_size = None, momentum = None, dropout = None, verbose = None, shuffle = None,
             use_csr_format = None, per_epoch_saving = None, use_cosine_annealing = None, cosine_annealing_min = None,
             cosine_annealing_max = None ):
        # Update 'BaseModel' Class Variables
        if epochs               is not None: self.Set_Epochs( epochs )
        if batch_size           is not None: self.Set_Batch_Size( batch_size )
        if momentum             is not None: self.Set_Momentum( momentum )
        if dropout              is not None: self.Set_Dropout( dropout )
        if verbose              is not None: self.Set_Verbose( verbose )
        if shuffle              is not None: self.Set_Shuffle( shuffle )
        if use_csr_format       is not None: self.Set_Use_CSR_Format( use_csr_format )
        if per_epoch_saving     is not None: self.Set_Per_Epoch_Saving( per_epoch_saving )
        if use_cosine_annealing is not None: self.Set_Use_Cosine_Annealing( use_cosine_annealing )
        if cosine_annealing_min is not None: self.Set_Cosine_Annealing_Min( cosine_annealing_min )
        if cosine_annealing_max is not None: self.Set_Cosine_Annealing_Max( cosine_annealing_max )

        # Add Model Callback Functions
        super().Add_Enabled_Model_Callbacks()

        if self.use_csr_format:
            self.trained_instances           = train_input_1.shape[0]
            number_of_train_output_instances = train_outputs.shape[0]
        else:
            self.trained_instances           = len( train_input_1 )
            number_of_train_output_instances = len( train_outputs )

        self.Print_Log( "BiLSTMModel::Fit() - Model Training Settings" )
        self.Print_Log( "                   - Epochs             : " + str( self.epochs            ) )
        self.Print_Log( "                   - Batch Size         : " + str( self.batch_size        ) )
        self.Print_Log( "                   - Verbose            : " + str( self.verbose           ) )
        self.Print_Log( "                   - Shuffle            : " + str( self.shuffle           ) )
        self.Print_Log( "                   - Use CSR Format     : " + str( self.use_csr_format    ) )
        self.Print_Log( "                   - Per Epoch Saving   : " + str( self.per_epoch_saving  ) )
        self.Print_Log( "                   - BiLSTM Merge Mode  : " + str( self.bilstm_merge_mode ) )
        self.Print_Log( "                   - No. of Train Inputs: " + str( self.trained_instances ) )

        # Compute Number Of Steps Per Batch (Use CSR Format == True)
        steps_per_batch = 0

        if self.batch_size >= self.trained_instances:
            steps_per_batch = 1
        else:
            steps_per_batch = self.trained_instances // self.batch_size if self.trained_instances % self.batch_size == 0 else self.trained_instances // self.batch_size + 1

        # Perform Model Training
        self.Print_Log( "BiLSTMModel::Fit() - Executing Model Training", force_print = True )

        with tf.device( self.device_name ):
            if self.use_csr_format:
                self.model_history = self.model.fit_generator( generator = self.Batch_Generator( train_input_1, train_outputs, batch_size = self.batch_size, steps_per_batch = steps_per_batch, shuffle = self.shuffle ),
                                                               epochs = self.epochs, steps_per_epoch = steps_per_batch, verbose = self.verbose, callbacks = self.callback_list )
            else:
                # CosFace, ArcFace & SphereFae Final Layers
                if self.final_layer_type in ["cosface", "arcface", "sphereface"]:
                    self.model_history = self.model.fit( [train_input_1, train_outputs], train_outputs, shuffle = self.shuffle, batch_size = self.batch_size,
                                                         epochs = self.epochs, verbose = self.verbose, callbacks = self.callback_list )
                # Dense Final Layer
                else:
                    self.model_history = self.model.fit( train_input_1, train_outputs, shuffle = self.shuffle, batch_size = self.batch_size,
                                                         epochs = self.epochs, verbose = self.verbose, callbacks = self.callback_list )

        # Print Last Epoch Metrics
        if self.verbose == False:
            final_epoch = self.model_history.epoch[-1]
            history     = self.model_history.history
            self.Print_Log( "", force_print = True )
            self.Print_Log( "BiLSTMModel::Final Training Metric(s) At Epoch: " + str( final_epoch ), force_print = True )

            # Iterate Through Available Metrics And Print Their Formatted Values
            for metric in history.keys():
                self.Print_Log( "BiLSTMModel::  - " + str( metric.capitalize() ) + ":\t{:.4f}" . format( history[metric][-1] ), force_print = True )

        self.Print_Log( "BiLSTMModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "BiLSTMModel::Fit() - Complete" )

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            primary_input_vector   : Vectorized Primary Model Input (Numpy Array)
            secondary_input_vector : Vectorized Secondary Model Input (Numpy Array)

        Outputs:
            prediction              : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict( self, inputs ):
        self.Print_Log( "BiLSTMModel::Predict() - Predicting Using Inputs: " + str( inputs ) )

        with tf.device( self.device_name ):
            if self.final_layer_type in ["cosface", "arcface", "sphereface"]:
                # Give The Network False Output Instance Since They're Not Used For Inference Anyway
                outputs = None

                if inputs.ndim == 2:
                    outputs = np.zeros( ( inputs.shape[0], self.Get_Number_Of_Outputs() ), dtype = np.int32 )
                elif inputs.ndim == 3:
                    outputs = np.zeros( ( inputs.shape[0], inputs.shape[1], self.Get_Number_Of_Outputs() ), dtype = np.int32 )

                return self.model.predict( [inputs, outputs], verbose = self.verbose )
            else:
                return self.model.predict( [inputs], verbose = self.verbose )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs_1, inputs_2, inputs_3, outputs ):
        self.Print_Log( "BiLSTMModel::Evaluate() - Executing Model Evaluation" )

        inputs = np.hstack( ( inputs_1, inputs_2 ) )

        with tf.device( self.device_name ):
            if self.final_layer_type in ["cosface", "arcface", "sphereface"]:
                loss, accuracy, precision, recall, f1_score = self.model.evaluate( [inputs, outputs], outputs, verbose = self.verbose )
            else:
                loss, accuracy, precision, recall, f1_score = self.model.evaluate( inputs, outputs, verbose = self.verbose )

            self.Print_Log( "BiLSTMModel::Evaluate() - Complete" )

            return loss, accuracy, precision, recall, f1_score

    ############################################################################################
    #                                                                                          #
    #    Keras Model(s)                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Build The Keras Model

        Inputs:
            number_of_train_1_inputs    : (Integer)
            number_of_train_2_inputs    : (Integer)
            number_of_hidden_dimensions : (Integer)
            number_of_outputs           : (Integer)

        Outputs:
            None
    """
    def Build_Model( self, number_of_features, number_of_embedding_dimensions, number_of_outputs, bilstm_dimension_size = 64, bilstm_merge_mode = "concat", max_sequence_length = 2,
                     embeddings = [], final_layer_type = None, weight_decay = None ):
        # Update 'BaseModel' Class Variables
        if number_of_features             != self.number_of_features:             self.number_of_features             = number_of_features
        if number_of_embedding_dimensions != self.number_of_embedding_dimensions: self.number_of_embedding_dimensions = number_of_embedding_dimensions
        if bilstm_dimension_size          != 64:                                  self.bilstm_dimension_size          = bilstm_dimension_size
        if bilstm_merge_mode              != "concat":                            self.bilstm_merge_mode              = bilstm_merge_mode
        if number_of_outputs              != self.number_of_outputs:              self.number_of_outputs              = number_of_outputs
        if final_layer_type               is not None:                            self.final_layer_type               = final_layer_type
        if weight_decay                   is not None:                            self.weight_decay                   = weight_decay

        self.Print_Log( "BiLSTMModel::Build_Model() - Model Settings" )
        self.Print_Log( "                           - Network Model              : " + str( self.network_model                  ) )
        self.Print_Log( "                           - Final Layer Type           : " + str( self.final_layer_type               ) )
        self.Print_Log( "                           - Learning Rate              : " + str( self.learning_rate                  ) )
        self.Print_Log( "                           - Dropout                    : " + str( self.dropout                        ) )
        self.Print_Log( "                           - Momentum                   : " + str( self.momentum                       ) )
        self.Print_Log( "                           - Optimizer                  : " + str( self.optimizer                      ) )
        self.Print_Log( "                           - Weight Decay               : " + str( self.weight_decay                   ) )
        self.Print_Log( "                           - Activation Function        : " + str( self.activation_function            ) )
        self.Print_Log( "                           - No. of Features            : " + str( self.number_of_features             ) )
        self.Print_Log( "                           - No. of Embedding Dimensions: " + str( self.number_of_embedding_dimensions ) )
        self.Print_Log( "                           - BiLSTM Dimension Size      : " + str( self.bilstm_dimension_size          ) )
        self.Print_Log( "                           - BiLSTM Merge Mode          : " + str( self.bilstm_merge_mode              ) )
        self.Print_Log( "                           - No. of Outputs             : " + str( self.number_of_outputs              ) )
        self.Print_Log( "                           - Trainable Weights          : " + str( self.trainable_weights              ) )
        self.Print_Log( "                           - Feature Scaling Value      : " + str( self.feature_scale_value            ) )
        self.Print_Log( "                           - Max Sequence Length        : " + str( max_sequence_length                 ) )

        # Check(s)
        if self.final_layer_type not in self.final_layer_type_list:
            self.Print_Log( "MLPModel::Build_Model() - Error: Invalid Final Layer Type", force_print = True )
            self.Print_Log( "                            - Options: " + str( self.final_layer_type_list ), force_print = True )
            self.Print_Log( "                            - Specified Option: " + str( self.final_layer_type ), force_print = True )
            return

        use_regularizer = True if self.final_layer_type in [ "cosface", "arcface", "sphereface" ] else False

        #######################
        #                     #
        #  Build Keras Model  #
        #                     #
        #######################

        input_layer             = Input( shape = ( None, ), name = "Input_Layer" )
        cosface_input_layer     = Input( shape = ( number_of_outputs, ), name = "CosFace_Input" )

        if( len( embeddings ) > 0 ):
            embedding_layer     = Embedding( input_dim = number_of_features, output_dim = number_of_embedding_dimensions, input_length = max_sequence_length, name = 'Embedding_Layer', weights = [embeddings], trainable = self.trainable_weights )( input_layer )
        else:
            embedding_layer     = Embedding( input_dim = number_of_features, output_dim = number_of_embedding_dimensions, input_length = max_sequence_length, name = 'Embedding_Layer', trainable = self.trainable_weights )( input_layer )

        # Perform Feature Scaling Prior To Generating An Embedding Representation
        if self.feature_scale_value != 1.0:
            feature_scale_value = self.feature_scale_value  # Fixes Python Recursion Limit Error (Model Tries To Save All 'self' Variables When Used With Lambda Function)
            embedding_layer     = Lambda( lambda x: x * feature_scale_value )( embedding_layer )

        bilstm_layer            = Bidirectional( LSTM( self.bilstm_dimension_size, return_sequences = True, dropout = self.dropout ), merge_mode = self.bilstm_merge_mode, name = 'BiLSTM_Layer_1' )( embedding_layer )

        # Batch Normalization Model
        if self.use_batch_normalization:
            bilstm_layer        = BatchNormalization( name = "Batch_Norm_Layer_1" )( bilstm_layer )

        bilstm_layer            = Bidirectional( LSTM( self.bilstm_dimension_size, dropout = self.dropout ), merge_mode = self.bilstm_merge_mode, name = 'BiLSTM_Layer_2' )( bilstm_layer )

        if use_regularizer:
            dense_layer         = Dense( units = 256, activation = 'tanh', name = 'Internal_Proposition_Representation',
                                         kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2( self.weight_decay ) )( bilstm_layer )
        else:
            dense_layer         = Dense( units = 256, activation = 'relu', name = 'Internal_Proposition_Representation' )( bilstm_layer )

        if self.use_batch_normalization:
            dense_layer         = BatchNormalization( name = "Batch_Norm_Layer_2" )( dense_layer )

        # Final Model Output Used For Prediction/Classification (Inherited From BaseModel class)
        output_layer            = self.Multi_Option_Final_Layer( number_of_outputs = number_of_outputs, cosface_input_layer = cosface_input_layer, dense_input_layer = dense_layer )

        if self.final_layer_type in ["cosface", "arcface", "sphereface"]:
            self.model = Model( inputs = [input_layer, cosface_input_layer], outputs = output_layer, name = self.network_model + "_model" )
        else:
            self.model = Model( inputs = input_layer, outputs = output_layer, name = self.network_model + "_model" )

        if self.optimizer == "adam":
            adam_opt = optimizers.Adam( lr = self.learning_rate )
            self.model.compile( loss = self.loss_function, optimizer = adam_opt, metrics = [ 'accuracy', super().Precision, super().Recall, super().F1_Score ] )
        elif self.optimizer == "sgd":
            sgd = optimizers.SGD( lr = self.learning_rate, momentum = self.momentum )
            self.model.compile( loss = self.loss_function, optimizer = sgd, metrics = [ 'accuracy', super().Precision, super().Recall, super().F1_Score ] )

        # Print Model Summary
        self.Print_Log( "BiLSTMModel::Build_Model() - =========================================================" )
        self.Print_Log( "BiLSTMModel::Build_Model() - =                     Model Summary                     =" )
        self.Print_Log( "BiLSTMModel::Build_Model() - =========================================================" )

        self.model.summary( print_fn = lambda x:  self.Print_Log( "BiLSTMModel::Build_Model() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

        self.Print_Log( "BiLSTMModel::Build_Model() - =========================================================" )
        self.Print_Log( "BiLSTMModel::Build_Model() - =                                                       =" )
        self.Print_Log( "BiLSTMModel::Build_Model() - =========================================================" )


    ############################################################################################
    #                                                                                          #
    #    Accessor Functions                                                                    #
    #                                                                                          #
    ############################################################################################


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NNLBD.Models import BiLSTMModel\n" )
    print( "     model = BiLSTMModel( network_model = \"bilstm\", print_debug_log = True," )
    print( "                          per_epoch_saving = False, use_csr_format = False )" )
    print( "     model.Fit( \"data/cui_mini\", epochs = 30, batch_size = 4, verbose = 1 )" )
    exit()
