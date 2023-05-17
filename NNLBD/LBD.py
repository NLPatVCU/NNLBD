#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/10/2020                                                                   #
#    Revised: 01/04/2022                                                                   #
#                                                                                          #
#    Main LBD Driver Class For The NNLBD Package.                                          #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import os, re, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# Custom Modules
from NNLBD.DataLoader           import CrichtonDataLoader, StdDataLoader
from NNLBD.Models.Architectures import *
from NNLBD.Misc                 import Utils


############################################################################################
#                                                                                          #
#    LBD Model Interface Class                                                             #
#                                                                                          #
############################################################################################

class LBD:
    """
    """
    def __init__( self, print_debug_log = False, write_log_to_file = False, network_model = "rumelhart", model_type = "open_discovery",
                  optimizer = 'adam', activation_function = 'sigmoid', loss_function = "binary_crossentropy", margin = 30.0, scale = 0.35,
                  bilstm_merge_mode = "concat", bilstm_dimension_size = 64, learning_rate = 0.005, epochs = 30, momentum = 0.05,
                  dropout = 0.5, batch_size = 32, prediction_threshold = 0.5, shuffle = True, skip_out_of_vocabulary_words = True,
                  use_csr_format = True, per_epoch_saving = True, use_gpu = True, device_name = "/gpu:0", verbose = 2,
                  enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_metric_monitor = "loss",
                  early_stopping_persistence = 3, use_batch_normalization = False, checkpoint_directory = "./ckpt_models",
                  trainable_weights = False, embedding_path = "", embedding_modification = "concatenate", final_layer_type = "dense",
                  feature_scale_value = 1.0, learning_rate_decay = 0.004, weight_decay = 0.0001, restrict_output = False,
                  use_cosine_annealing = False, cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4 ):
        self.version                       = 0.22
        self.model                         = None                            # Automatically Set After Calling 'LBD::Build_Model()' Function
        self.debug_log                     = print_debug_log                 # Options: True, False
        self.write_log                     = write_log_to_file               # Options: True, False
        self.debug_log_file_handle         = None                            # Debug Log File Handle
        self.checkpoint_directory          = checkpoint_directory            # Path (String)
        self.model_data_prepared           = False                           # Options: True, False (Default: False)
        self.data_loader                   = None
        self.debug_log_file_name           = "LBD_Log.txt"                   # File Name (String)

        # Create Log File Handle
        if self.write_log and self.debug_log_file_handle is None:
            self.debug_log_file_handle = open( self.debug_log_file_name, "w" )

        # Create New DataLoader Instance With Options
        # Model Specific DataLoader Parameters/Settings
        if network_model in [ "rumelhart", "hinton", "bilstm", "cnn", "mlp", "mlp_similarity" ]:
            self.data_loader = StdDataLoader()
        elif network_model in [ "cd2" ]:
            self.data_loader = CrichtonDataLoader()
        else:
            print( "LBD::Init() - Error Model \"" + str( network_model ) + "\"'s DataLoader Not Implemented" )
            raise NotImplementedError

        # Set Encoding Output Encoding Type To 'Embeddings' (Only For Embedding Similarity Models)
        output_is_embeddings = True if network_model == "mlp_similarity" else False

        # Initialize DataLoader Parameters
        if self.data_loader:
            self.data_loader.__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                       skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, restrict_output = restrict_output,
                                       debug_log_file_handle = self.debug_log_file_handle, output_is_embeddings = output_is_embeddings )

        # Create New Utils Instance
        self.utils = Utils()

        self.Print_Log( "LBD::Init() - Current Working Directory: \"" + str( self.utils.Get_Working_Directory() ) + "\"" )

        # Check(s)
        if network_model not in [ "rumelhart", "hinton", "bilstm", "cnn", "cd2", "mlp", "mlp_similarity" ]:
            self.Print_Log( "LBD::Init() - Warning: Network Model Type Is Not 'rumelhart', 'hinton', 'bilstm', 'cnn', 'cd2', 'mlp'", force_print = True )
            self.Print_Log( "            - Resetting Network Model Type To: 'rumelhart'", force_print = True )
            network_model  = "rumelhart"
            continue_query = input( "Continue? (Y/N)\n" )
            exit() if re.search( r"[Nn]", continue_query ) else None
        else:
            self.Print_Log( "LBD::Init() - Network Model \"" + str( network_model ) + "\"" )

        if model_type != "open_discovery" and model_type != "closed_discovery":
            self.Print_Log( "LBD::Init() - Warning: Model Type Not Equal 'open_discovery' or 'closed_discovery' / Setting To 'open_discovery'", force_print = True )
            model_type = "open_discovery"
            continue_query  = input( "Continue? (Y/N)\n" )
            exit() if re.search( r"[Nn]", continue_query ) else None

        if use_csr_format == False:
            self.Print_Log( "LBD::Init() - Warning: Use CSR Mode = False / High Memory Consumption May Occur When Vectorizing Data-Sets", force_print = True )
        else:
            self.Print_Log( "LBD::Init() - Using CSR Matrix Format" )

        if per_epoch_saving:
            self.Create_Checkpoint_Directory()

        # Initialize LBD Model Architecture
        if network_model == "hinton" or network_model == "rumelhart":
            self.model = RumelhartHintonModel( skip_gpu_init = True )
        elif network_model == "bilstm":
            self.model = BiLSTMModel( skip_gpu_init = True )
        elif network_model == "cnn":
            self.model = CNNModel( skip_gpu_init = True )
        elif network_model == "cd2":
            self.model = CD2Model( skip_gpu_init = True )
        elif network_model == "mlp":
            self.model = MLPModel( skip_gpu_init = True )
        elif network_model == "mlp_similarity":
            self.model          = MLPModel( skip_gpu_init = True )
            activation_function = "tanh"
            loss_function       = "cosine_similarity"

        # Initialize Model Parameters
        if self.model:
            self.model.__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, network_model = network_model, margin = margin,
                                 model_type = model_type, optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                 learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                 prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                 per_epoch_saving = per_epoch_saving, device_name = device_name, debug_log_file_handle = self.debug_log_file_handle,
                                 enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, scale = scale,
                                 early_stopping_metric_monitor = early_stopping_metric_monitor, early_stopping_persistence = early_stopping_persistence,
                                 use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_path = embedding_path,
                                 embedding_modification = embedding_modification, verbose = verbose, final_layer_type = final_layer_type, weight_decay = weight_decay,
                                 feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay, use_cosine_annealing = use_cosine_annealing,
                                 cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max, bilstm_dimension_size = bilstm_dimension_size,
                                 bilstm_merge_mode = bilstm_merge_mode )

        self.Print_Log( "LBD::Init() - Complete" )

    """
       Remove Variables From Memory
    """
    def __del__( self ):
        del self.model
        del self.utils
        del self.data_loader

        if self.write_log and self.debug_log_file_handle is not None: self.debug_log_file_handle.close()

    """
       Updates Current Model Parameters
    """
    def Update_Model_Parameters( self, print_debug_log = None, optimizer = None, activation_function = None, loss_function = None, dropout = None,
                                 bilstm_merge_mode = None, bilstm_dimension_size = None, learning_rate = None, epochs = None, momentum = None,
                                 batch_size = None, prediction_threshold = None, shuffle = None, embedding_path = None, use_csr_format = None,
                                 per_epoch_saving = None, margin = None, scale = None, verbose = None, trainable_weights = None, enable_tensorboard_logs = None,
                                 enable_early_stopping = None, early_stopping_metric_monitor = None, early_stopping_persistence = None, use_batch_normalization = None,
                                 embedding_modification = None, learning_rate_decay = None, feature_scale_value = None, use_cosine_annealing = None,
                                 cosine_annealing_min = None, cosine_annealing_max = None ):
        if self.model is not None:
            if self.model.Is_Model_Loaded():
                self.Print_Log( "LBD::Update_Model_Parameters() - Warning: Model Has Already Been Built / Unable To Update Some Model Parameters", force_print = True )

            if print_debug_log               is not None and self.model.Get_Debug_Log()                      != print_debug_log                 : self.model.Set_Debug_Log( print_debug_log )
            if optimizer                     is not None and self.model.Get_Optimizer()                      != optimizer                       : self.model.Set_Optimizer( optimizer )
            if activation_function           is not None and self.model.Get_Activation_Function()            != activation_function             : self.model.Set_Activation_Function( activation_function )
            if loss_function                 is not None and self.model.Get_Loss_Function()                  != loss_function                   : self.model.Set_Loss_Function( loss_function )
            if bilstm_merge_mode             is not None and self.model.Get_BiLSTM_Merge_Mode()              != bilstm_merge_mode               : self.model.Set_BiLSTM_Merge_Mode( bilstm_merge_mode )
            if bilstm_dimension_size         is not None and self.model.Get_Number_Of_Embedding_Dimensions() != bilstm_dimension_size           : self.model.Set_BiLSTM_Dimension_Size( bilstm_dimension_size )
            if learning_rate                 is not None and self.model.Get_Learning_Rate()                  != learning_rate                   : self.model.Set_Learning_Rate( learning_rate )
            if learning_rate_decay           is not None and self.model.Get_Learning_Rate_Decay()            != learning_rate_decay             : self.model.Set_Learning_Rate_Decay( learning_rate_decay )
            if feature_scale_value           is not None and self.model.Get_Feature_Scaling_Value()          != feature_scale_value             : self.model.Set_Feature_Scaling_Value( feature_scale_value )
            if epochs                        is not None and self.model.Get_Epochs()                         != epochs                          : self.model.Set_Epochs( epochs )
            if momentum                      is not None and self.model.Get_Momentum()                       != momentum                        : self.model.Set_Momentum( momentum )
            if dropout                       is not None and self.model.Get_Dropout()                        != dropout                         : self.model.Set_Dropout( dropout )
            if batch_size                    is not None and self.model.Get_Batch_Size()                     != batch_size                      : self.model.Set_Batch_Size( batch_size )
            if prediction_threshold          is not None and self.model.Get_Prediction_Threshold()           != prediction_threshold            : self.model.Set_Prediction_Threshold( prediction_threshold )
            if shuffle                       is not None and self.model.Get_Shuffle()                        != shuffle                         : self.model.Set_Shuffle( shuffle )
            if use_csr_format                is not None and self.model.Get_Use_CSR_Format()                 != use_csr_format                  : self.model.Set_Use_CSR_Format( use_csr_format )
            if per_epoch_saving              is not None and self.model.Get_Per_Epoch_Saving()               != per_epoch_saving                : self.model.Set_Per_Epoch_Saving( per_epoch_saving )
            if verbose                       is not None and self.model.Get_Verbose()                        != verbose                         : self.model.Set_Verbose( verbose )
            if trainable_weights             is not None and self.model.Get_Trainable_Weights()              != trainable_weights               : self.model.Set_Trainable_Weights( trainable_weights )
            if enable_tensorboard_logs       is not None and self.model.Get_Enable_Tensorboard_Logs()        != enable_tensorboard_logs         : self.model.Set_Enable_Tensorboard_Logs( enable_tensorboard_logs )
            if enable_early_stopping         is not None and self.model.Get_Enable_Early_Stopping()          != enable_early_stopping           : self.model.Set_Enable_Early_Stopping( enable_early_stopping )
            if early_stopping_metric_monitor is not None and self.model.Get_Early_Stopping_Metric_Monitor()  != early_stopping_metric_monitor   : self.model.Set_Early_Stopping_Metric_Monitor( early_stopping_metric_monitor )
            if early_stopping_persistence    is not None and self.model.Get_Early_Stopping_Persistence()     != early_stopping_persistence      : self.model.Set_Early_Stopping_Persistence( early_stopping_persistence )
            if use_cosine_annealing          is not None and self.model.Get_Use_Cosine_Annealing()           != use_cosine_annealing            : self.model.Set_Use_Cosine_Annealing( use_cosine_annealing )
            if cosine_annealing_min          is not None and self.model.Get_Cosine_Annealing_Min()           != cosine_annealing_min            : self.model.Set_Cosine_Annealing_Min( cosine_annealing_min )
            if cosine_annealing_max          is not None and self.model.Get_Cosine_Annealing_Max()           != cosine_annealing_max            : self.model.Set_Cosine_Annealing_Max( cosine_annealing_max )
            if use_batch_normalization       is not None and self.model.Get_Use_Batch_Normalization()        != use_batch_normalization         : self.model.Set_Use_Batch_Normalization( use_batch_normalization )
            if embedding_path                is not None and self.model.Get_Embedding_Path()                 != embedding_path                  : self.model.Set_Embedding_Path( embedding_path )
            if margin                        is not None and self.model.Get_Margin()                         != margin                          : self.model.Set_Margin( margin )
            if scale                         is not None and self.model.Get_Scale()                          != scale                           : self.model.Set_Scale( scale )
            if embedding_modification        is not None and self.model.Get_Embedding_Modification()         != embedding_modification          : self.model.Set_Embedding_Modification( embedding_modification )

            self.Print_Log( "LBD::Update_Model_Parameters() - Updated Model Parameter(s)" )
        else:
            self.Print_Log( "LBD::Update_Model_Parameters() - Error: No Model In Memory / Unable To Update Model Parameter(s)" )

    """
       Prepares Model And Data For Training/Testing
           Current Neural Architecture Implementations: Rumelhart, Hinton & BiLSTM
    """
    def Prepare_Model_Data( self, training_file_path = "", data_instances = [], number_of_hidden_dimensions = 200, force_run = False ):
        if self.Is_Model_Data_Prepared() and force_run == False:
            self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Model Data Has Already Been Prepared" )
            return True

        # CNN Network Model Check (Not Implemented Error)
        if self.model.Get_Network_Model() == "cnn":
            self.Print_Log( "LBD::Prepare_Model_Data() - Network Model Type - CNN" )
            self.Print_Log( "Error: CNN Model Is Not Implemented / Exiting Program", force_print = True )
            exit()

        # Bug Fix: User Enabled 'per_epoch_saving' After Initially Disabled In __init__()
        if self.model.Get_Per_Epoch_Saving(): self.Create_Checkpoint_Directory()

        # Prepare Embeddings & Data-set
        data_loader = self.Get_Data_Loader()

        # Load Embeddings
        if self.model.Get_Embedding_Path() != "" and data_loader.Is_Embeddings_Loaded() == False and force_run == False:
            self.Print_Log( "LBD::Prepare_Model_Data() - Loading Embeddings: " + str( self.model.Get_Embedding_Path() ), force_print = True )
            embeddings = data_loader.Load_Embeddings( self.model.Get_Embedding_Path() )

            # Check
            if isinstance( embeddings, list ) and len( embeddings ) == 0 or isinstance( embeddings, np.ndarray ) and embeddings.shape[0] == 0:
                self.Print_Log( "LBD::Prepare_Model_Data() - Error Loading Embeddings", force_print = True )
                self.Print_Log( "LBD::Prepare_Model_Data() -       Embedding Path: " + str( self.model.Get_Embedding_Path() ), force_print = True )
                return False
            else:
                self.model.Set_Embeddings_Loaded( data_loader.Is_Embeddings_Loaded() )

        # Read Training Data
        if len( data_instances ) == 0:
            self.Print_Log( "LBD::Prepare_Model_Data() - Reading Training Data: " + str( training_file_path ), force_print = True )
            training_data = data_loader.Read_Data( training_file_path )

            # Generate Token IDs
            self.Print_Log( "LBD::Prepare_Model_Data() - Generating Token IDs From Training Data", force_print = True )

            if self.model.Get_Network_Model() == "cd2":
                data_loader.Generate_Token_IDs( skip_association_value = True )
            else:
                data_loader.Generate_Token_IDs()

        # Train Using Passed Data From 'data_instances' Parameter.
        # This Also Assumes Token ID Dictionary Has Been Previously Generated.
        else:
            training_data = data_instances

            # Generate Token IDs
            self.Print_Log( "LBD::Prepare_Model_Data() - Generating Token IDs From Training Data", force_print = True )

            if self.model.Get_Network_Model() == "cd2":
                data_loader.Generate_Token_IDs( data_instances, skip_association_value = True )
            else:
                data_loader.Generate_Token_IDs( data_instances )

        primary_embeddings, secondary_embeddings, tertiary_embeddings, output_embeddings = [], [], [], []

        # If We're Actually Generating Embeddings And Passing To The Model. If We're Refining A Model, It Already Contains The Embedding
        #    Representations, So We Just Set 'simulate_embeddings_loaded' Parameter To 'True' In The DataLoader Class To Avoid Generating Them Again.
        if data_loader.Is_Embeddings_Loaded() and data_loader.Simulate_Embeddings_Loaded_Mode() == False:
            # Use Full Embeddings For All Models As Input
            primary_embeddings   = data_loader.Get_Model_Embeddings( embedding_type = "primary"   )
            secondary_embeddings = data_loader.Get_Model_Embeddings( embedding_type = "secondary" )
            tertiary_embeddings  = data_loader.Get_Model_Embeddings( embedding_type = "tertiary"  )
            output_embeddings    = data_loader.Get_Model_Embeddings( embedding_type = "output"    )

            if len( primary_embeddings   ) == 0: self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Primary Embeddings Data Length == 0"   )
            if len( secondary_embeddings ) == 0: self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Secondary Embeddings Data Length == 0" )
            if len( tertiary_embeddings  ) == 0: self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Tertiary Embeddings Data Length == 0"  )
            if len( output_embeddings    ) == 0: self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Output Embeddings Data Length == 0"    )

            number_of_hidden_dimensions = data_loader.Get_Embedding_Dimension_Size()

            self.Save_Model_Keys( model_name = "last_" + self.model.Get_Network_Model() + "_model" )

        self.Print_Log( "LBD::Prepare_Model_Data() - Network Model Type - " + str( self.model.Get_Network_Model() ) )

        #######################################################################
        #                                                                     #
        #   Prepare Model Input/Output Data                                   #
        #                                                                     #
        #######################################################################

        # Binarize Model Data
        self.Print_Log( "LBD::Prepare_Model_Data() - Binarizing/Vectorizing Model Inputs & Outputs From Training Data", force_print = True )

        train_input_1, train_input_2, train_input_3, train_outputs = None, None, None, None

        # Model Specific DataLoader Parameters/Settings
        pad_inputs   = False
        pad_output   = True if self.model.Get_Network_Model() != "cd2"    else False
        stack_inputs = True if self.model.Get_Network_Model() == "bilstm" else False

        if self.Is_Embeddings_Loaded():
            self.Print_Log( "LBD::Prepare_Model_Data() - Warning: 'pad_inputs == True' When Embeddings Have Been Loaded Into The DataLoader / Setting 'pad_inputs = False'" )
            pad_inputs = False

        # Train On Data Instances Passed By Parameter (Batch Training)
        if len( data_instances ) == 0:
            train_input_1, train_input_2, train_input_3, train_outputs = self.Encode_Model_Data( model_type = self.model.Get_Model_Type(),
                                                                                                 use_csr_format = self.model.Get_Use_CSR_Format(),
                                                                                                 pad_inputs = pad_inputs, pad_output = pad_output,
                                                                                                 stack_inputs = stack_inputs )
        # Train On Data Instances Within The DataLoader Class
        else:
            train_input_1, train_input_2, train_input_3, train_outputs = self.Encode_Model_Data( training_data, model_type = self.model.Get_Model_Type(),
                                                                                                 use_csr_format = self.model.Get_Use_CSR_Format(),
                                                                                                 pad_inputs = pad_inputs, pad_output = pad_output,
                                                                                                 stack_inputs = stack_inputs )

        # Check(s)
        if isinstance( train_input_1, ( list, tuple, np.ndarray ) ) and len( train_input_1 ) == 0: train_input_1 = None
        if isinstance( train_input_2, ( list, tuple, np.ndarray ) ) and len( train_input_2 ) == 0: train_input_2 = None
        if isinstance( train_input_3, ( list, tuple, np.ndarray ) ) and len( train_input_3 ) == 0: train_input_3 = None
        if isinstance( train_outputs, ( list, tuple, np.ndarray ) ) and len( train_outputs ) == 0: train_outputs = None

        if train_input_1 is None or train_input_2 is None or train_outputs is None:
            self.Print_Log( "LBD::Prepare_Model_Data() - Error Occurred During Model Data Vectorization", force_print = True )
            return False
        if self.model.Get_Network_Model() == "cd2" and train_input_3 is None:
            self.Print_Log( "LBD::Prepare_Model_Data() - Error Occurred During Model Data Vectorization", force_print = True )
            return False

        # Concatenate Inputs Across Columns (Only Bi-LSTM Model)
        if self.model.Get_Network_Model() == "bilstm":
            self.Print_Log( "LBD::Prepare_Model_Data() - Primary Input Shape     : (" + str( len( train_input_1 ) ) + ", " + str( len( train_input_1[0] ) ) + ")" )
            self.Print_Log( "LBD::Prepare_Model_Data() - Secondary Input Shape   : (" + str( len( train_input_2 ) ) + ", " + str( len( train_input_2[0] ) ) + ")" )

            self.Print_Log( "LBD::Prepare_Model_Data() - Horizontal Stacking Primary And Secondary Inputs" )
            train_input_1 = np.hstack( ( train_input_1, train_input_2 ) )
            train_input_2 = None

            self.Print_Log( "LBD::Prepare_Model_Data() - Primary Input Shape (H-Stacked): " + str( train_input_1.shape ) )

        # CSR Matrix Format
        if self.model.Get_Use_CSR_Format():
            number_of_train_1_input_instances = train_input_1.shape[0] if train_input_1 is not None else -1
            number_of_train_2_input_instances = train_input_2.shape[0] if train_input_2 is not None else -1
            number_of_train_3_input_instances = train_input_3.shape[0] if train_input_3 is not None else -1
            number_of_train_output_instances  = train_outputs.shape[0] if train_outputs is not None else -1

            if train_input_1 is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Primary Input Shape  : " + str( train_input_1.shape ) )
            if train_input_2 is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Secondary Input Shape: " + str( train_input_2.shape ) )
            if train_input_3 is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Tertiary Input Shape : " + str( train_input_3.shape ) )
            if train_outputs is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Output Shape         : " + str( train_outputs.shape ) )

        # List/Array Format
        else:
            number_of_train_1_input_instances = len( train_input_1 )
            number_of_train_2_input_instances = len( train_input_2 ) if train_input_2 is not None else -1
            number_of_train_3_input_instances = len( train_input_3 ) if train_input_3 is not None else -1
            number_of_train_output_instances  = len( train_outputs ) if train_outputs is not None else -1

            if self.model.Get_Network_Model() == "cd2":
                train_input_1 = train_input_1.reshape( number_of_train_1_input_instances, 1 )
                train_input_2 = train_input_2.reshape( number_of_train_2_input_instances, 1 )
                train_input_3 = train_input_3.reshape( number_of_train_3_input_instances, 1 )

            if train_input_1 is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Primary Input Shape     : (" + str( len( train_input_1 ) ) + ", " + str( len( train_input_1[0] ) ) + ")" )
            if train_input_2 is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Secondary Input Shape   : (" + str( len( train_input_2 ) ) + ", " + str( len( train_input_2[0] ) ) + ")" )
            if train_input_3 is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Tertiary Input Shape    : (" + str( len( train_input_3 ) ) + ", " + str( len( train_input_3[0] ) ) + ")" )
            if train_outputs is not None: self.Print_Log( "LBD::Prepare_Model_Data() - Output Shape            : (" + str( len( train_outputs ) ) + ", " + str( len( train_outputs[0] ) ) + ")" )

        # Check(s)
        if self.model.Get_Network_Model() in ["rumelhart", "hinton", "mlp", "cnn", "mlp_similarity"]:
            if number_of_train_1_input_instances == 0 or number_of_train_2_input_instances == 0 or \
                number_of_train_output_instances == 0:
                self.Print_Log( "LBD::Prepare_Model_Data() - Error Vectorizing Model Input/Output Data", force_print = True )
                return False
        elif self.model.Get_Network_Model() == "bilstm":
            if number_of_train_1_input_instances == 0 or number_of_train_output_instances == 0:
                self.Print_Log( "LBD::Prepare_Model_Data() - Error Vectorizing Model Input/Output Data", force_print = True )
                return False
        elif self.model.Get_Network_Model() == "cd2":
            if number_of_train_1_input_instances == 0 or number_of_train_2_input_instances == 0 or \
            number_of_train_3_input_instances == 0 or number_of_train_output_instances == 0:
                self.Print_Log( "LBD::Prepare_Model_Data() - Error Vectorizing Model Input/Output Data", force_print = True )
                return False

        if data_loader.Is_Embeddings_Loaded():
            number_of_features = data_loader.Get_Number_Of_Embeddings()
        else:
            number_of_features = 0

            if self.model.Get_Network_Model() != "cd2":      # Rumelhart, Hinton, Bi-LSTM, CosFace & CNN Models
                number_of_features = train_input_1[0:].shape[1] + train_input_2[0:].shape[1]
            else:                                               # CD2 Model
                number_of_features = train_input_1[0:].shape[1] + train_input_2[0:].shape[1] + train_input_3[0:].shape[1]

        # More Checks
        #   Check To See If Number Of Instances Is Divisible By Batch Size With No Remainder
        #   (Used For Batch_Generator)
        if self.model.Get_Use_CSR_Format() and number_of_train_1_input_instances % self.model.Get_Batch_Size() != 0:
            self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Number Of Instances Not Divisible By Batch Size" )
            self.Print_Log( "                          - Number Of Instances  : " + str( number_of_train_1_input_instances ) )
            self.Print_Log( "                          - Batch Size           : " + str( self.model.Get_Batch_Size()       ) )
            self.Print_Log( "                          - Batch_Generator Might Not Train Correctly / Change To Another Batch Size" )

            possible_batch_sizes = [ str( i ) if number_of_train_1_input_instances % i == 0 else "" for i in range( 1, number_of_train_1_input_instances ) ]
            possible_batch_sizes = " ".join( possible_batch_sizes )
            possible_batch_sizes = re.sub( r'\s+', ' ', possible_batch_sizes )

            self.Print_Log( "           - Possible Batch Sizes : " + possible_batch_sizes )

        # Get Model Parameters From Training Data
        self.Print_Log( "LBD::Prepare_Model_Data() - Fetching Model Parameters (Input/Output Sizes)" )

        number_of_train_1_inputs = data_loader.Get_Number_Of_Unique_Features()
        number_of_train_2_inputs = data_loader.Get_Number_Of_Unique_Features()
        number_of_train_3_inputs = data_loader.Get_Number_Of_Unique_Features()

        if data_loader.Get_Restrict_Output():
            number_of_outputs = data_loader.Get_Number_Of_Output_Elements()
        else:
            number_of_outputs = data_loader.Get_Number_Of_Unique_Features()

        if self.model.Get_Network_Model() == "bilstm":
            number_of_outputs  = data_loader.Get_Number_Of_Output_Elements()
            number_of_features = data_loader.Get_Number_Of_Unique_Features()

        if self.model.Get_Network_Model() == "cd2": number_of_outputs = 1
        if self.model.Get_Network_Model() == "mlp_similarity": number_of_outputs = data_loader.Get_Embedding_Dimension_Size()

        self.Print_Log( "                          - Number Of Features         : " + str( number_of_features          ) )
        self.Print_Log( "                          - Number Of Primary Inputs   : " + str( number_of_train_1_inputs    ) )
        self.Print_Log( "                          - Number Of Secondary Inputs : " + str( number_of_train_2_inputs    ) )
        self.Print_Log( "                          - Number Of Tertiary Inputs  : " + str( number_of_train_3_inputs    ) )
        self.Print_Log( "                          - Number Of Hidden Dimensions: " + str( number_of_hidden_dimensions ) )
        self.Print_Log( "                          - Number Of Outputs          : " + str( number_of_outputs           ) )

        self.model_data_prepared = True

        # Build Neural Network Model Based On Architecture
        self.Print_Log( "LBD::Prepare_Model_Data() - Building Model" )

        if self.Is_Model_Loaded() == False:
            if self.model.Get_Network_Model() in ["bilstm", "cnn"]:
                self.model.Build_Model( number_of_features, number_of_hidden_dimensions, number_of_outputs, embeddings = data_loader.Get_Embeddings() )
            elif self.model.Get_Network_Model() in ["rumelhart", "hinton", "mlp", "mlp_similarity"]:
                self.model.Build_Model( number_of_train_1_inputs, number_of_train_2_inputs, number_of_hidden_dimensions,
                                        number_of_outputs, number_of_primary_embeddings = number_of_train_1_inputs,
                                        number_of_secondary_embeddings = number_of_train_2_inputs, primary_embeddings = primary_embeddings,
                                        secondary_embeddings = secondary_embeddings )
            elif self.model.Get_Network_Model() == "cd2":
                self.model.Build_Model( number_of_features, number_of_train_1_inputs, number_of_train_2_inputs, number_of_train_3_inputs,
                                        number_of_hidden_dimensions, number_of_outputs, data_loader.Get_Embeddings() )
            else:
                self.Print_Log( "LBD::Prepare_Model_Data() - Error: Specified Network Model Not Supported", force_print = True )
        else:
            self.Print_Log( "LBD::Prepare_Model_Data() - Warning: Model Has Already Been Built And Loaded In Memory" )

        self.Print_Log( "LBD::Prepare_Model_Data() - Complete" )
        return True

    """
       Trains LBD Model
           Current Neural Architecture Implementations: Rumelhart, Hinton, BiLSTM, CNN, CD2 & CosFace
    """
    def Fit( self, training_file_path = "", data_instances = [], learning_rate = None, epochs = None, batch_size = None,
             momentum = None, dropout = None, verbose = None, shuffle = None, per_epoch_saving = None, use_csr_format = None,
             embedding_path = None, trainable_weights = None, margin = None, scale = None, learning_rate_decay = None,
             feature_scale_value = None, use_cosine_annealing = None, cosine_annealing_min = None, cosine_annealing_max = None ):
        # Check(s)
        if training_file_path == "" and len( data_instances ) == 0:
            self.Print_Log( "LBD::Fit() - Error: No Training File Path Specified Or Training Instance List Given", force_print = True )
            return False
        if self.utils.Check_If_File_Exists( training_file_path ) == False and len( data_instances ) == 0:
            self.Print_Log( "LBD::Fit() - Error: Training File Data Path Does Not Exist", force_print = True )
            return False
        if feature_scale_value is not None and feature_scale_value <= 0 or self.Get_Model().Get_Feature_Scaling_Value() <= 0:
            self.Print_Log( "LBD::Fit() - Error: Feature Scaling Value Must Be >= 0", force_print = True )
            return False

        # Check If Default Parameters Have Changed, If So Change Private Variables
        self.Update_Model_Parameters( learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, momentum = momentum,
                                      dropout = dropout, verbose = verbose, shuffle = shuffle, use_csr_format = use_csr_format,
                                      per_epoch_saving = per_epoch_saving, trainable_weights = trainable_weights, margin = margin,
                                      embedding_path = embedding_path, scale = scale, learning_rate_decay = learning_rate_decay,
                                      feature_scale_value = feature_scale_value, use_cosine_annealing = use_cosine_annealing,
                                      cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )

        # Start Elapsed Time Timer
        start_time = time.time()

        # Prepare Model Data
        is_data_prepared = self.Prepare_Model_Data( training_file_path = training_file_path, data_instances = data_instances,
                                                    number_of_hidden_dimensions = self.model.Get_Number_Of_Hidden_Dimensions() )

        # Check If Data Preparation Completed Successfully
        if is_data_prepared == False:
            self.Print_Log( "LBD::Fit() - Error Preparing Data / Exiting Program", force_print = True )
            exit()

        # Check If Model Has Been Loaded/Created Prior To Continuing
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Fit() - Error: Model Has Not Been Created/Loaded", force_print = True )
            return False

        #######################################################################
        #                                                                     #
        #   Execute Model Training                                            #
        #                                                                     #
        #######################################################################

        self.Print_Log( "LBD::Fit() - Network Model Type - " + str( self.model.Get_Network_Model() ) )

        # Fetching Binarized Training Data From DataLoader Class
        self.Print_Log( "LBD::Fit() - Fetching Model Inputs & Output Training Data" )
        train_input_1 = self.Get_Data_Loader().Get_Primary_Inputs()
        train_input_2 = self.Get_Data_Loader().Get_Secondary_Inputs()
        train_input_3 = self.Get_Data_Loader().Get_Tertiary_Inputs()
        train_outputs = self.Get_Data_Loader().Get_Outputs()

        # Concatenate Inputs Across Columns (Bi-LSTM)
        if self.model.Get_Network_Model() == "bilstm":
            self.Print_Log( "LBD::Fit() - Horizontal Stacking Primary And Secondary Inputs" )
            train_input_1 = np.hstack( ( train_input_1, train_input_2 ) )

        # Train Model
        self.model.Fit( train_input_1, train_input_2, train_input_3, train_outputs, epochs = self.model.Get_Epochs(),
                        batch_size = self.model.Get_Batch_Size(), momentum = self.model.Get_Momentum(),
                        dropout = self.model.Get_Dropout(), verbose = self.model.Get_Verbose(), shuffle = self.model.Get_Shuffle(),
                        use_csr_format = self.model.Get_Use_CSR_Format(), per_epoch_saving = self.model.Get_Per_Epoch_Saving(),
                        use_cosine_annealing = self.model.Get_Use_Cosine_Annealing(), cosine_annealing_min = self.model.Get_Cosine_Annealing_Min(),
                        cosine_annealing_max = self.model.Get_Cosine_Annealing_Max() )

        #######################################################################
        #                                                                     #
        #   Post Model Training                                               #
        #                                                                     #
        #######################################################################

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Fit() - Elapsed Time: " + str( elapsed_time ) + " secs", force_print = True )

        self.Print_Log( "LBD::Fit() - Training Metrics:" )
        self.model.Print_Model_Training_Metrics()

        self.Print_Log( "LBD::Fit() - Complete" )

        return True

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            primary_input           : Primary Model Input (String)
            secondary_input         : Secondary Model Input (String)
            tertiary_input          : Tertiary Model Input (String)
            encoded_primary_input   : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            encoded_secondary_input : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            encoded_tertiary_input  : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            encoded_output          : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            return_vector           : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values       : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)
            instance_separator      : String Delimiter Used To Separate Model Data Instances (String)

        Outputs:
            prediction              : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict( self, primary_input = "", secondary_input = "", tertiary_input = "", output = "",
                 encoded_primary_input = None, encoded_secondary_input = None, encoded_tertiary_input = None, encoded_output = None,
                 return_vector = False, return_raw_values = False, instance_separator = '<:>' ):
        # Start Elapsed Time Timer
        start_time        = time.time()

        prediction        = []
        prediction_tokens = ""
        vectorize_data    = True

        # Check For Vectorized Input Matrices Have Been Passed As Parameters
        if encoded_primary_input is not None or encoded_secondary_input is not None \
           or encoded_tertiary_input is not None or encoded_output is not None:
            self.Print_Log( "LBD::Predict() - Vectorized Input Matrices Passed / Skipping Plain Text Vectorization" )
            vectorize_data = False

        self.Print_Log( "LBD::Predict() - Model Prediction Inputs:" )
        self.Print_Log( "               - Primary Input(s)  : " + str( primary_input      ) )
        self.Print_Log( "               - Secondary Input(s): " + str( secondary_input    ) )
        self.Print_Log( "               - Tertiary Input(s) : " + str( tertiary_input     ) )
        self.Print_Log( "               - Output(s)         : " + str( output             ) )
        self.Print_Log( "               - Instance Separator: " + str( instance_separator ) )

        # Check To See If Primary And Secondary ID Key Dictionaries Are Loaded
        self.Print_Log( "LBD::Predict() - Checking For Token Keys" )

        if self.Get_Data_Loader().Get_Number_Of_Unique_Features() == 0:
            self.Print_Log( "LBD::Predict() - Error: Token ID Key Dictionary Is Empty", force_print = True )
            return []

        # Check
        if return_raw_values and not return_vector:
            self.Print_Log( "LBD::Predict() - Warning: Unable To Return Raw Values With 'return_vector = False' / Setting 'return_vector = True'" )
            return_vector = True

        # Check If We're Performing Closed Discovery. If So Swap Secondary Input To Output For Prediction.
        if encoded_primary_input is None and encoded_secondary_input is None and output == "":
            self.Print_Log( "LBD::Predict() - Output == \"\" / Setting 'output = <*>padding<*>'" )
            output = "<*>padding<*>"

        #######################################################################
        #                                                                     #
        #   Prepare Model Input/Output Data                                   #
        #                                                                     #
        #######################################################################

        # Binarize Model Data
        self.Print_Log( "LBD::Predict() - Binarizing/Vectorizing Model Inputs" )

        # Model Specific DataLoader Parameters/Settings (Assumes No Embeddings Are Loaded)
        pad_inputs = False
        pad_output = True if self.model.Get_Network_Model() != "cd2" else False

        if self.Is_Embeddings_Loaded(): pad_inputs = False

        if vectorize_data:
            encoded_primary_input, encoded_secondary_input, encoded_tertiary_input, encoded_output = self.Encode_Model_Instance( primary_input, secondary_input, tertiary_input, output,
                                                                                                                                 model_type = self.Get_Model_Type(), pad_inputs = pad_inputs,
                                                                                                                                 pad_output = pad_output, instance_separator = instance_separator )

        # Check(s)
        if ( isinstance( encoded_primary_input, np.ndarray ) and len( encoded_primary_input ) == 0 ) or ( isinstance( encoded_primary_input, csr_matrix ) and encoded_primary_input.shape[0] == 0 ):
            self.Print_Log( "LBD::Predict() - Error: Encoded Primary Input Is Empty",   force_print = True )
            return []
        if ( isinstance( encoded_secondary_input, np.ndarray ) and len( encoded_secondary_input ) == 0 ) or ( isinstance( encoded_secondary_input, csr_matrix ) and encoded_secondary_input.shape[0] == 0 ):
            self.Print_Log( "LBD::Predict() - Error: Encoded Secondary Input Is Empty", force_print = True )
            return []
        if tertiary_input != "" and ( isinstance( encoded_tertiary_input, np.ndarray ) and len( encoded_tertiary_input ) == 0 ) or ( isinstance( encoded_tertiary_input, csr_matrix ) and encoded_tertiary_input.shape[0] == 0 ):
            self.Print_Log( "LBD::Predict() - Error: Encoded Tertiary Input Is Empty",  force_print = True )
            return []
        if output != "" and ( isinstance( encoded_output, np.ndarray ) and len( encoded_output ) == 0 ) or ( isinstance( encoded_output, csr_matrix ) and encoded_output.shape[0] == 0 ):
            self.Print_Log( "LBD::Predict() - Error: Encoded Output Is Empty", force_print = True )
            return []

        # Perform Model Predictions
        if encoded_primary_input == [] or encoded_secondary_input == []:
            self.Print_Log( "LBD::Predict() - Error: Encoded Primary Or Secondary Input(s) Are Empty", force_print = True )
            return []

        self.Print_Log( "LBD::Predict() - Predicting Outputs" )

        # Ensure Vectorized Model Inputs Are Numpy Arrays Prior To Sending To The Model For Prediction
        if isinstance( encoded_primary_input,   list ): encoded_primary_input   = np.asarray( encoded_primary_input   )
        if isinstance( encoded_secondary_input, list ): encoded_secondary_input = np.asarray( encoded_secondary_input )
        if isinstance( encoded_tertiary_input,  list ): encoded_tertiary_input  = np.asarray( encoded_tertiary_input  )
        if isinstance( encoded_output,          list ): encoded_output          = np.asarray( encoded_output          )

        if self.model.Get_Network_Model() == "hinton" or self.model.Get_Network_Model() == "rumelhart":
            prediction = self.model.Predict( encoded_primary_input, encoded_secondary_input )
        elif self.model.Get_Network_Model() == "bilstm":
            # Concatenate Inputs Across Columns
            train_inputs = np.hstack( ( encoded_primary_input, encoded_secondary_input ) )
            prediction = self.model.Predict( train_inputs )
        elif self.model.Get_Network_Model() == "cd2":
            if isinstance( encoded_primary_input,   csr_matrix ): encoded_primary_input   = encoded_primary_input.todense()
            if isinstance( encoded_secondary_input, csr_matrix ): encoded_secondary_input = encoded_secondary_input.todense()
            if isinstance( encoded_tertiary_input,  csr_matrix ): encoded_tertiary_input  = encoded_tertiary_input.todense()
            prediction = self.model.Predict( encoded_primary_input, encoded_secondary_input, encoded_tertiary_input )
        elif self.model.Get_Network_Model() in ["mlp", "mlp_similarity"]:
            prediction = self.model.Predict( encoded_primary_input, encoded_secondary_input )
        else:
            self.Print_Log( "LBD::Predict() - Error: Unknown Network Type / Functionality Not Implemented", force_print = True )
            return []


        #######################################################################
        #                                                                     #
        #   Post Prediction Processing                                        #
        #                                                                     #
        #######################################################################

        self.Print_Log( "LBD::Predict() - Raw Prediction Output: " + str( prediction ) )
        self.Print_Log( "LBD::Predict() - Applying Prediction Threshold - Value: " + str( self.model.Get_Prediction_Threshold() ) )

        # Check
        if isinstance( prediction, list ) and len( prediction ) == 0:
            self.Print_Log( "LBD::Predict() - Error Occurred During Model Prediction" )
            return []

        # Perform Prediction Thresholding
        if prediction.ndim == 2:
            temp_prediction = []

            if return_raw_values == False:
                for instance_values in prediction:
                    instance_values = [1 if value > self.model.Get_Prediction_Threshold() else 0 for value in instance_values]
                    temp_prediction.append( instance_values )

                prediction = np.asarray( temp_prediction )

        elif prediction.ndim == 1:
            prediction = [1 if value > self.model.Get_Prediction_Threshold() else 0 for value in prediction.squeeze()] if return_raw_values == False else prediction.squeeze()
        else:
            if return_raw_values:
                prediction = prediction.squeeze()
            else:
                prediction = 1 if prediction > self.model.Get_Prediction_Threshold() else 0

        self.Print_Log( "LBD::Predict() - Prediction Output: " + str( prediction ) )

        # Convert Predictions To Tokens
        if return_vector == False:
            self.Print_Log( "LBD::Predict() - Converting Predicted Indices To Word Tokens" )
            token_type = None

            # Ensure Correct Token ID Dictionary Is Used Depending On 'restrict_output' Variable And 'model_type' Settings.
            if self.Get_Data_Loader().Get_Restrict_Output():
                if self.model.Get_Model_Type() == "open_discovery"  : token_type = "output"
                if self.model.Get_Model_Type() == "closed_discovery": token_type = "secondary"

            if prediction.ndim == 2:
                for instance_values in prediction:
                    for index, value in enumerate( instance_values ):
                        if value == 1:
                            prediction_tokens += self.Get_Data_Loader().Get_Token_From_ID( index, token_type = token_type ) + " "
            else:
                for index, value in enumerate( prediction ):
                        if value == 1:
                            prediction_tokens += self.Get_Data_Loader().Get_Token_From_ID( index, token_type = token_type ) + " "

            prediction_tokens = re.sub( r'\s+$', "", prediction_tokens )

            self.Print_Log( "LBD::Predict() - Predictions: \"" + prediction_tokens + "\"" )
        else:
            self.Print_Log( "LBD::Predict() - Prediction Vector: " + str( prediction ) )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Predict() - Elapsed Time: " + str( elapsed_time ) + " secs" )
        self.Print_Log( "LBD::Predict() - Complete" )

        return prediction if return_vector else prediction_tokens

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            primary_input_vector   : Vectorized Primary Model Input (Numpy Array)
            secondary_input_vector : Vectorized Secondary Model Input (Numpy Array)
            return_vector          : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values      : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)

        Outputs:
            prediction            : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict_Vector( self, primary_input_vector, secondary_input_vector, tertiary_input_vector = [], return_vector = True, return_raw_values = False ):
        # Check(s)
        self.Print_Log( "LBD::Predict_Vector() - Error: Primary Input Vector Is Empty",    force_print = True ) if len( primary_input_vector   ) == 0 else None
        self.Print_Log( "LBD::Predict_Vector() - Error: Secondary Input Vector Is Empty",  force_print = True ) if len( secondary_input_vector ) == 0 else None
        self.Print_Log( "LBD::Predict_Vector() - Warning: Tertiary Input Vector Is Empty", force_print = True ) if len( tertiary_input_vector  ) == 0 else None

        if len( primary_input_vector ) == 0 or len( secondary_input_vector ) == 0:
            return []

        # Start Elapsed Time Timer
        start_time = time.time()

        # Ensure Vectorized Model Inputs Are Numpy Arrays Prior To Sending To The Model For Prediction
        if isinstance( primary_input_vector,   list ): primary_input_vector   = np.asarray( primary_input_vector   )
        if isinstance( secondary_input_vector, list ): secondary_input_vector = np.asarray( secondary_input_vector )
        if isinstance( tertiary_input_vector,  list ): tertiary_input_vector  = np.asarray( tertiary_input_vector  )

        self.Print_Log( "LBD::Predict_Vector() - Predicting Outputs" )

        prediction        = []
        prediction_tokens = ""

        #######################################################################
        #                                                                     #
        #   Rumelhart & Hinton Networks                                       #
        #                                                                     #
        #######################################################################
        if self.model.Get_Network_Model() == "hinton" or self.model.Get_Network_Model() == "rumelhart":
            prediction = self.model.Predict( primary_input_vector, secondary_input_vector )
        #######################################################################
        #                                                                     #
        #   BiLSTM Network                                                    #
        #                                                                     #
        #######################################################################
        elif self.model.Get_Network_Model() == "bilstm":
            # Concatenate Inputs Across Columns
            input_vector = np.hstack( ( primary_input_vector, secondary_input_vector ) )

            prediction = self.model.Predict( input_vector )
        #######################################################################
        #                                                                     #
        #   CD2 Network                                                    #
        #                                                                     #
        #######################################################################
        elif self.model.Get_Network_Model() == "cd2":
            prediction = self.model.Predict( primary_input_vector, secondary_input_vector, tertiary_input_vector )
        else:
            self.Print_Log( "LBD::Predict_Vector() - Error: Unknown Network Architecture / Functionality Not Implemented", force_print = True )
            return []

        self.Print_Log( "LBD::Predict_Vector() - Raw Prediction Vector: " + str( prediction ) )

        # Perform Prediction Thresholding
        if prediction.ndim == 2:
            temp_prediction = []

            if return_raw_values == False:
                for instance_values in prediction:
                    instance_values = [1 if value > self.model.Get_Prediction_Threshold() else 0 for value in instance_values]
                    temp_prediction.append( instance_values )

                prediction = np.asarray( temp_prediction )
        elif prediction.ndim == 1:
            prediction = [1 if value > self.model.Get_Prediction_Threshold() else 0 for value in prediction.squeeze()] if return_raw_values == False else prediction.squeeze()
        else:
            if return_raw_values:
                prediction = prediction.squeeze()
            else:
                prediction = 1 if prediction > self.model.Get_Prediction_Threshold() else 0

        # Convert Predictions To Tokens
        prediction_tokens = ""

        if return_vector == False:
            self.Print_Log( "LBD::Predict_Vector() - Converting Predicted Indices To Word Tokens" )

            # Ensure Correct Token ID Dictionary Is Used Depending On 'restrict_output' Variable And 'model_type' Settings.
            if self.Get_Data_Loader().Get_Restrict_Output():
                if self.model.Get_Model_Type() == "open_discovery"  : token_type = "output"
                if self.model.Get_Model_Type() == "closed_discovery": token_type = "secondary"

            if self.model.Get_Model_Type() == "open_discovery":
                for index, value in enumerate( prediction ):
                    if value == 1:
                        prediction_tokens += self.Get_Token_From_ID( index, token_type = token_type ) + " "
            else:
                for index, value in enumerate( prediction ):
                    if value == 1:
                        prediction_tokens += self.Get_Token_From_ID( index, token_type = token_type ) + " "

            prediction_tokens = re.sub( r'\s+$', "", prediction_tokens )

            self.Print_Log( "LBD::Predict_Vector() - Predictions: " + str( prediction_tokens ) )
        else:
            self.Print_Log( "LBD::Predict_Vector() - Prediction Vector: " + str( prediction ) )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Predict_Vector() - Elapsed Time: " + str( elapsed_time ) + " secs" )
        self.Print_Log( "LBD::Predict_Vector() - Complete" )

        return prediction if return_vector else prediction_tokens

    """
        Outputs Model's Prediction Vector Given Inputs And Ranks Outputs Based On Prediction Value

        Inputs:
            primary_input            : Primary Model Input (String)
            secondary_input          : Secondary Model Input (String)

        Outputs:
            ranked_output_dictionary : Contains Dictionary Of Ranked Predictions Per Model Input Instance (Dictionary Of Dictionaries)
    """
    def Predict_Ranking( self, primary_input, secondary_input, number_of_predictions = 0 ):
        # Check(s)
        if self.model.Get_Network_Model() == "cd2":
            self.Print_Log( "LBD::Evaluate() - Error: Evaluation Of CD2 Model Not Supported", force_print = True )
            return {}

        # Start Elapsed Time Timer
        start_time = time.time()

        if self.Get_Data_Loader().Get_Number_Of_Unique_Features() == 0:
            self.Print_Log( "LBD::Predict_Ranking() - Error: Token ID Key Dictionary Is Empty", force_print = True )
            return []

        self.Print_Log( "LBD::Predict_Ranking() - Predicting Outputs" )

        ranked_output_dictionary = {}

        predictions = self.Predict( primary_input, secondary_input, return_vector = True, return_raw_values = True )

        # Convert Predictions To Tokens
        self.Print_Log( "LBD::Predict_Ranking() - Ranking Predictions" )
        self.Print_Log( "LBD::Predict_Ranking() - Converting Predicted Indices To Word Tokens" )

        if self.model.Get_Model_Type() == "open_discovery":
            for index, value in enumerate( predictions ):
                ranked_output_dictionary[ self.Get_Data_Loader().Get_Token_From_ID( index ) ] = value
        else:
            for index, value in enumerate( predictions ):
                ranked_output_dictionary[ self.Get_Data_Loader().Get_Token_From_ID( index ) ] = value

        # Sort Values In Descending Order
        ranked_output_dictionary = dict( sorted( ranked_output_dictionary.items(), key = lambda item: item[1], reverse = True ) )

        # Get Only Top Specific Number Of Predictions
        ranked_output_dictionary = dict( list( ranked_output_dictionary.items() )[0:number_of_predictions] ) if number_of_predictions > 0 else ranked_output_dictionary

        self.Print_Log( "LBD::Predict_Ranking() - Finished Ranking Predictions: " + str( ranked_output_dictionary ) )
        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Predict_Ranking() - Elapsed Time: " + str( elapsed_time ) + " secs" )
        self.Print_Log( "LBD::Predict_Ranking() - Complete" )

        return ranked_output_dictionary

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            encoded_primary_input   : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            encoded_secondary_input : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            encoded_tertiary_input  : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            encoded_output          : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            return_vector           : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values       : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)

        Outputs:
            prediction              : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict_In_Batches( self, encoded_primary_input = None, encoded_secondary_input = None, encoded_tertiary_input = None, encoded_output = None,
                            batch_size = 128, return_vector = False, return_raw_values = False ):
        self.Print_Log( "LBD::Predict_In_Batches() - Predicting In Batches Of " + str( batch_size ) )

        is_csr_matrices = False

        # Assumes Number Of Dimensions For Encoded Inputs (Primary & Secondary) == 2
        if isinstance( encoded_primary_input, np.ndarray ):
            number_of_batches = int( encoded_primary_input.shape[0] / batch_size ) + 1
        if isinstance( encoded_primary_input, csr_matrix ):
            number_of_batches = int( encoded_primary_input.shape[0] / batch_size ) + 1
            is_csr_matrices   = True

        predictions = []

        for i in range( number_of_batches ):
            start_index, end_index = i * batch_size, ( i + 1 ) * batch_size

            # CSR_Matrices
            if is_csr_matrices:
                temp_encoded_primary_inputs   = encoded_primary_input[start_index:end_index,].todense()
                temp_encoded_secondary_inputs = encoded_secondary_input[start_index:end_index,].todense()
                temp_encoded_tertiary_inputs  = encoded_tertiary_input[start_index:end_index,].todense() if encoded_tertiary_input else None
                temp_encoded_outputs          = encoded_output[start_index:end_index,].todense()         if encoded_output         else None
            # Numpy Arrays
            else:
                temp_encoded_primary_inputs   = encoded_primary_input[start_index:end_index]
                temp_encoded_secondary_inputs = encoded_secondary_input[start_index:end_index]
                temp_encoded_tertiary_inputs  = encoded_tertiary_input[start_index:end_index] if encoded_tertiary_input else None
                temp_encoded_outputs          = encoded_output[start_index:end_index]         if encoded_output         else None

            if temp_encoded_primary_inputs.shape[0] == 0: continue

            model_predictions = self.Predict( encoded_primary_input = temp_encoded_primary_inputs, encoded_secondary_input = temp_encoded_secondary_inputs,
                                              encoded_tertiary_input = temp_encoded_tertiary_inputs, encoded_output = temp_encoded_outputs,
                                              return_raw_values = return_raw_values, return_vector = return_vector )

            predictions.extend( model_predictions )

        self.Print_Log( "LBD::Predict_In_Batches() - Complete" )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, test_file_path ):
        # Check(s)
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Evaluate() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
            return -1, -1, -1, -1, -1
        if self.utils.Check_If_File_Exists( test_file_path ) == False:
            self.Print_Log( "LBD::Evaluate() - Error: Training File Data Path Does Not Exist", force_print = True )
            return -1, -1, -1, -1, -1

        # Check(s)
        if self.model.Get_Network_Model() == "cd2":
            self.Print_Log( "LBD::Evaluate() - Error: Evaluation Of CD2 Model Not Supported", force_print = True )
            return -1, -1, -1, -1, -1

        # Start Elapsed Time Timer
        start_time = time.time()

        # Read Training Data
        self.Print_Log( "LBD::Evaluate() - Reading Evaluation Data: " + test_file_path, force_print = True )
        data_loader = self.Get_Data_Loader()
        data_loader.Read_Data( test_file_path )

        # Generate Token IDs
        self.Print_Log( "LBD::Evaluate() - Checking For Token Keys" )

        if data_loader.Is_Dictionary_Loaded() == False:
            self.Print_Log( "LBD::Evaluate() - Error: Token ID Key Dictionary Is Empty", force_print = True )
            return -1, -1, -1, -1, -1

        # Binarize Training Data For Keras Model
        eval_primary_input   = None
        eval_secondary_input = None
        eval_tertiary_input  = None
        eval_outputs         = None

        self.Print_Log( "LBD::Evaluate() - Binarizing/Vectorizing Model Inputs & Outputs From Evaluation Data" )
        if self.model.Get_Network_Model() == "rumelhart" or self.model.Get_Network_Model() == "hinton":
            eval_primary_input, eval_secondary_input, eval_tertiary_input, eval_outputs = self.Encode_Model_Data( model_type     = self.model.Get_Model_Type(),
                                                                                                                  use_csr_format = self.model.Get_Use_CSR_Format(),
                                                                                                                  pad_inputs     = True )
        elif self.model.Get_Network_Model() == "bilstm":
            eval_primary_input, eval_secondary_input, eval_tertiary_input, eval_outputs = self.Encode_Model_Data( model_type     = self.model.Get_Model_Type(),
                                                                                                                  use_csr_format = self.model.Get_Use_CSR_Format(),
                                                                                                                  pad_inputs     = False,
                                                                                                                  stack_inputs   = True )
        # Check(s)
        if eval_primary_input is None or eval_secondary_input is None or eval_tertiary_input is None or eval_outputs is None:
            self.Print_Log( "LBD::Evaluate() - Error Occurred During Model Data Vectorization", force_print = True )
            return False

        if self.model.Get_Use_CSR_Format():
            eval_primary_input_length   = eval_primary_input[0:].shape[1]   if self.model.Get_Network_Model() != "bilstm" else len( eval_primary_input   )
            eval_secondary_input_length = eval_secondary_input[0:].shape[1] if self.model.Get_Network_Model() != "bilstm" else len( eval_secondary_input )
            eval_tertiary_input_length  = eval_tertiary_input[0:].shape[1]  if self.model.Get_Network_Model() != "bilstm" else len( eval_tertiary_input  )
            eval_outputs_length         = eval_outputs[0:].shape[1]
        else:
            eval_primary_input_length   = len( eval_primary_input   )
            eval_secondary_input_length = len( eval_secondary_input )
            eval_tertiary_input_length  = len( eval_tertiary_input  )
            eval_outputs_length         = len( eval_outputs         )

        # Check(s)
        if eval_primary_input_length == 0 or eval_secondary_input_length == 0 or eval_outputs_length == 0:
            self.Print_Log( "LBD::Evaluate() - Error Vectorizing Model Inputs/Output Data", force_print = True )
            return -1, -1, -1, -1, -1

        self.Print_Log( "LBD::Evaluate() - Executing Model Evaluation", force_print = True )
        loss, accuracy, precision, recall, f1_score = self.model.Evaluate( eval_primary_input, eval_secondary_input, eval_tertiary_input, eval_outputs )

        self.Print_Log( "LBD::Evaluate() - Loss     :", loss      )
        self.Print_Log( "LBD::Evaluate() - Accuracy :", accuracy  )
        self.Print_Log( "LBD::Evaluate() - Precision:", precision )
        self.Print_Log( "LBD::Evaluate() - Recall   :", recall    )
        self.Print_Log( "LBD::Evaluate() - F1 Score :", f1_score  )
        self.Print_Log( "LBD::Evaluate() - Finished Model Evaluation" )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Evaluate() - Elapsed Time: " + str( elapsed_time ) + " secs", force_print = True )
        self.Print_Log( "LBD::Evaluate() - Complete" )

        return loss, accuracy, precision, recall, f1_score

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate_Prediction( self, test_file_path ):
        # Check(s)
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Evaluate_Prediction() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
            return -1
        if self.utils.Check_If_File_Exists( test_file_path ) == False:
            self.Print_Log( "LBD::Evaluate_Prediction() - Error: Training File Data Path Does Not Exist", force_print = True )
            return -1

        # Check(s)
        if self.model.Get_Network_Model() == "cd2":
            self.Print_Log( "LBD::Evaluate_Prediction() - Error: Evaluation Of CD2 Model Not Supported", force_print = True )
            return -1

        # Start Elapsed Time Timer
        start_time = time.time()

        self.Print_Log( "LBD::Evaluate_Prediction() - Model Evaluation Type: " + str( self.model.Get_Model_Type() ) )

        # Read Training Data
        self.Print_Log( "LBD::Evaluate_Prediction() - Reading Evaluation Data: " + test_file_path, force_print = True )
        data_loader = self.Get_Data_Loader()
        data_loader.Read_Data( test_file_path )

        # Generate Token IDs
        self.Print_Log( "LBD::Evaluate_Prediction() - Checking For Token Keys" )

        if data_loader.Is_Dictionary_Loaded() == False:
            self.Print_Log( "LBD::Evaluate_Prediction() - Error: Token ID Key Dictionary Is Empty", force_print = True )
            return -1

        # Binarize Training Data For Keras Model
        self.Print_Log( "LBD::Evaluate_Prediction() - Splitting Evaluation Data" )

        eval_primary_input   = []
        eval_secondary_input = []
        eval_outputs         = []
        instance_separator   = " "

        for eval_data in data_loader.Get_Data():
            elements = eval_data.split()

            number_of_instance_outputs = len( elements[2:] )

            input_1_string = " ".join( [elements[0] for i in range( number_of_instance_outputs )] )
            input_2_string = " ".join( [elements[1] for i in range( number_of_instance_outputs )] )

            eval_primary_input.append( input_1_string )
            eval_secondary_input.append( input_2_string )
            eval_outputs.append( " ".join( elements[2:] ) )

        # Check(s)
        if len( eval_primary_input ) == 0 or len( eval_secondary_input ) == 0 or len( eval_outputs ) == 0:
            self.Print_Log( "LBD::Evaluate_Prediction() - Error Splitting Evaluation Inputs/Output Data", force_print = True )
            return -1

        self.Print_Log( "LBD::Evaluate_Prediction() - Executing Model Evaluation", force_print = True )
        correct_instances   = 0
        skipped_input_count = 0
        total_instances     = 0

        # Execute Model Prediction Evaluation
        for index in range( len( eval_primary_input ) ):
            primary_input   = eval_primary_input[index]
            secondary_input = eval_secondary_input[index]
            output          = eval_outputs[index]

            self.Print_Log( "LBD::Evaluate_Prediction() - Inputs: \"" + str( primary_input ) + "\" & \"" + str( secondary_input ) + "\"" )
            self.Print_Log( "LBD::Evaluate_Prediction() - Output(s): \"" + str( output ) + "\"" )

            output_predictions = self.Predict( primary_input, secondary_input, "", output, instance_separator = instance_separator, return_vector = False )
            output_predictions = output_predictions.split() if len( output_predictions ) > 0 else []

            # Skip Checking For Correct Predictions Given Input Instances If Model Is Not Able To Make Predictions
            if len( output_predictions ) == 0:
                skipped_input_count += 1
                total_instances     += 1
                continue

            self.Print_Log( "LBD::Evaluate_Prediction() - Predicted Output(s): \"" + str( output_predictions ) + "\"" )

            # Count Number Of Correct Predictions Versus Evaluation Data Outputs
            output_tokens = secondary_input if self.model.Get_Model_Type() == "closed_discovery" else output

            for output_token in output_tokens.split():
                correct_instances += 1 if output_token in output_predictions else 0
                total_instances   += 1

        accuracy = correct_instances / total_instances if total_instances > 0 else 0

        self.Print_Log( "LBD::Evaluate_Prediction() - Correct Count: " + str( correct_instances   ) )
        self.Print_Log( "LBD::Evaluate_Prediction() - Skipped Count: " + str( skipped_input_count ) )
        self.Print_Log( "LBD::Evaluate_Prediction() - Total Count  : " + str( total_instances     ) )
        self.Print_Log( "LBD::Evaluate_Prediction() - Accuracy     : " + str( accuracy            ) )
        self.Print_Log( "LBD::Evaluate_Prediction() - Finished Model Evaluation" )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Evaluate_Prediction() - Elapsed Time: " + str( elapsed_time ) + " secs", force_print = True )
        self.Print_Log( "LBD::Evaluate_Prediction() - Complete" )

        return accuracy

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate_Ranking( self, test_file_path, number_of_predictions = 0 ):
        # Check(s)
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Evaluate_Ranking() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
            return {}
        if self.utils.Check_If_File_Exists( test_file_path ) == False:
            self.Print_Log( "LBD::Evaluate_Ranking() - Error: Training File Data Path Does Not Exist", force_print = True )
            return {}

        # Check(s)
        if self.model.Get_Network_Model() == "cd2":
            self.Print_Log( "LBD::Evaluate_Ranking() - Error: Evaluation Of CD2 Model Not Supported", force_print = True )
            return -1

        # Start Elapsed Time Timer
        start_time = time.time()

        # Read Training Data
        self.Print_Log( "LBD::Evaluate_Ranking() - Reading Evaluation Data: " + test_file_path, force_print = True )
        data_loader = self.Get_Data_Loader()
        data_loader.Read_Data( test_file_path )

        # Generate Token IDs
        self.Print_Log( "LBD::Evaluate_Ranking() - Checking For Token Keys" )

        if data_loader.Is_Dictionary_Loaded() == False:
            self.Print_Log( "LBD::Evaluate_Ranking() - Error: Token ID Key Dictionary Is Empty", force_print = True )
            return {}

        # Binarize Training Data For Keras Model
        self.Print_Log( "LBD::Evaluate_Ranking() - Splitting Evaluation Data" )
        eval_primary_input   = []
        eval_secondary_input = []

        for eval_data in data_loader.Get_Data():
            elements = eval_data.split()

            if self.model.Get_Model_Type() == "open_discovery":
                eval_primary_input.append( elements[0] )
                eval_secondary_input.append( elements[1] )
            else:
                eval_primary_input.append( elements[0] )
                eval_secondary_input.append( " ".join( elements[2:] ) )

        # Check(s)
        if len( eval_primary_input ) == 0 or len( eval_secondary_input ) == 0:
            self.Print_Log( "LBD::Evaluate_Ranking() - Error Splitting Evaluation Input Data", force_print = True )
            return {}

        self.Print_Log( "LBD::Evaluate_Ranking() - Executing Model Evaluation", force_print = True )

        input_instance_output_rankings = {}

        # Execute Model Prediction Evaluation
        for index in range( len( eval_primary_input ) ):
            primary_input   = eval_primary_input[index]
            secondary_input = eval_secondary_input[index]

            if self.model.Get_Model_Type() == "open_discovery":
                ranked_predictions = self.Predict_Ranking( primary_input, secondary_input, number_of_predictions )
                input_instance_output_rankings[ primary_input + " " + secondary_input ] = ranked_predictions
            else:
                for temp_secondary_input in secondary_input.split():
                    ranked_predictions = self.Predict_Ranking( primary_input, temp_secondary_input, number_of_predictions )
                    input_instance_output_rankings[ primary_input + " " + temp_secondary_input ] = ranked_predictions

        self.Print_Log( "LBD::Evaluate_Ranking() - Listed Ranked Outputs Per Input Instance:" )
        self.Print_Log( str( input_instance_output_rankings ) )
        self.Print_Log( "LBD::Evaluate_Ranking() - Finished Model Evaluation Ranking" )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "LBD::Evaluate_Ranking() - Elapsed Time: " + str( elapsed_time ) + " secs", force_print = True )
        self.Print_Log( "LBD::Evaluate_Ranking() - Complete" )

        return input_instance_output_rankings


    ############################################################################################
    #                                                                                          #
    #    Model Support Functions                                                               #
    #                                                                                          #
    ############################################################################################

    """
        Reads Data From The File
    """
    def Read_Data( self, file_path, convert_to_lower_case = True, keep_in_memory = True, read_all_data = True, number_of_lines_to_read = 32 ):
        return self.Get_Data_Loader().Read_Data( file_path, convert_to_lower_case = convert_to_lower_case, keep_in_memory = keep_in_memory, read_all_data = read_all_data,
                                                 number_of_lines_to_read = number_of_lines_to_read )

    """
        Vectorized/Binarized Model Data - Used For Training/Evaluation Data
    """
    def Encode_Model_Data( self, data_list = [], model_type = None, use_csr_format = None, pad_inputs = True, pad_output = True,
                           stack_inputs = False, keep_in_memory = True, number_of_threads = 4, str_delimiter = '\t'):
        if model_type     == None: model_type     = self.Get_Model_Type()
        if use_csr_format == None: use_csr_format = self.Get_Model().Get_Use_CSR_Format()
        return self.Get_Data_Loader().Encode_Model_Data( data_list, model_type = model_type, use_csr_format = use_csr_format, pad_inputs = pad_inputs,
                                                         pad_output = pad_output, stack_inputs = stack_inputs, number_of_threads = number_of_threads,
                                                         keep_in_memory = keep_in_memory, str_delimiter = str_delimiter )

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            primary_input          = String (ie. C002  )
            secondary_inputs       = String (ie. TREATS)
            outputs                = String (May Contain Multiple Tokens. ie. C001 C003 C004)

        Outputs:
            primary_input_vector   = Numpy Binary Vector
            secondary_input_vector = Numpy Binary Vector
            output_vector          = Numpy Binary Vector
    """
    def Encode_Model_Instance( self, primary_input, secondary_input, tertiary_input = "", outputs = "", model_type = None,
                               pad_inputs = True, pad_output = True, instance_separator = "<:>" ):
        if model_type == None: model_type = self.Get_Model_Type()
        return self.Get_Data_Loader().Encode_Model_Instance( primary_input = primary_input, secondary_input = secondary_input, tertiary_input = tertiary_input, outputs = outputs,
                                                             model_type = model_type, pad_inputs = pad_inputs, pad_output = pad_output, instance_separator = instance_separator )

    """
        Loads The Model From A File

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Load_Model( self, model_path, model_name = "model", load_new_model = True, bypass_gpu_init = False ):
        self.Print_Log( "LBD::Load_Model() - Loading Model From Path - " + str( model_path ), force_print = True )
        self.Print_Log( "LBD::Load_Model() -         Model Name      - " + str( model_name ), force_print = True )

        if not re.search( r"\/$", model_path ): model_path += "/"

        # Check To See The Model Path Exists
        if not self.utils.Check_If_Path_Exists( model_path ):
            self.Print_Log( "LBD::Load_Model() - Error: Specified Model Path Does Not Exist", force_print = True )
            return False

        self.Print_Log( "LBD::Load_Model() - Fetching Network Model Type From Settings File" )
        network_model = self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "NetworkModel" )

        # Get Current Specified Device (GPU/CPU)
        use_gpu     = self.model.Get_Use_GPU()
        device_name = self.model.Get_Device_Name()

        self.Print_Log( "LBD::Load_Model() - Detected Model Type: " + str( network_model ), force_print = True )

        # Load Network Architecture Type
        if network_model != None:
            self.Print_Log( "LBD::Load_Model() - Creating Model Data Loader" )

            # Set Output Encoding Type (Based On Model)
            output_is_embeddings = True if network_model == "mlp_similarity" else False

            # Create New DataLoader Instance With Options
            # Model Specific DataLoader Parameters/Settings
            if network_model in [ "rumelhart", "hinton", "bilstm", "cnn", "mlp", "mlp_similarity" ]:
                self.data_loader = StdDataLoader()
            elif network_model in [ "cd2" ]:
                self.data_loader = CrichtonDataLoader()
            else:
                print( "LBD::Load_Model() - Error Model \"" + str( network_model ) + "\"'s DataLoader Not Implemented" )
                raise NotImplementedError

            # Initialize DataLoader Parameters
            if self.data_loader:
                self.data_loader.__init__( print_debug_log = self.debug_log, write_log_to_file = self.write_log,
                                           debug_log_file_handle = self.debug_log_file_handle, output_is_embeddings = output_is_embeddings )

            # Check If Previous Model Utilized Embeddings To Train
            if self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingsLoaded" ) == "True":
                self.Get_Data_Loader().Set_Simulate_Embeddings_Loaded_Mode( True )

            self.Print_Log( "LBD::Load_Model() - Creating New \"" + str( network_model ) + "\" Model", force_print = True )

            self.model.Set_Debug_Log_File_Handle( None )

            if network_model == "hinton" or network_model == "rumelhart":
                self.model = RumelhartHintonModel( skip_gpu_init = True )
            elif network_model == "bilstm":
                self.model = BiLSTMModel( skip_gpu_init = True )
            elif network_model == "cd2":
                self.model = CD2Model( skip_gpu_init = True )
            elif network_model == "mlp":
                self.model = MLPModel( skip_gpu_init = True )
            elif network_model == "cnn":
                self.model = CNNModel( skip_gpu_init = True )
            else:
                self.Print_Log( "LBD::Load_Model() - Error: Trying To Load Unsupported Model Architecture Or Type", force_print = True )
                self.Print_Log( "LBD::Load_Model() -        Specified Model: " + str( network_model ), force_print = True )
                return False

            # Initialize Model Parameters
            if self.model:
                self.model.__init__( network_model = network_model, debug_log_file_handle = self.debug_log_file_handle,
                                     use_gpu = use_gpu, device_name = device_name )

            # Load The Model From File & Model Settings To Model Object
            self.Print_Log( "LBD::Load_Model() - Loading Model", force_print = True )

            self.model.Load_Model( model_path = model_path + model_name, load_new_model = load_new_model, bypass_gpu_init = bypass_gpu_init )

            # Enforce New GPU Device ID Setting
            self.model.Set_Device_Name( device_name )

            # Load Model Primary And Secondary Keys
            self.Print_Log( "LBD::Load_Model() - Loading Model Token ID Dictionary", force_print = True )

            if self.utils.Check_If_File_Exists( model_path + model_name + "_token_id_key_data" ):
                self.Get_Data_Loader().Load_Token_ID_Key_Data( model_path + model_name + "_token_id_key_data" )
            else:
                self.Print_Log( "LBD::Error: Model Token ID Key File Does Not Exist", force_print = True )

            # Reinitialize Data Loader Primary And Secondary ID Dictionaries Key Variables
            self.Get_Data_Loader().Reinitialize_Token_ID_Values()

            self.Print_Log( "LBD::Load_Model() - Complete", force_print = True )
            return True
        else:
            self.Print_Log( "LBD::Load_Model() - Error: Trying To Load Unsupported Model Architecture Or Type", force_print = True )
            self.Print_Log( "LBD::Load_Model() -        Specified Model: " + str( network_model ), force_print = True )

        self.Print_Log( "LBD::Load_Model() - Error Loading Model \"" + str( model_path + model_name ) + "\"", force_print = True )
        return False

    """
        Saves Model To File

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Save_Model( self, model_path = "./", model_name = "model" ):
        # Check
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Save_Model() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
            return

        if self.utils.Check_If_Path_Exists( model_path ) == False:
            self.Print_Log( "LBD::Save_Model() - Creating Model Save Path: " + str( model_path ) )
            self.utils.Create_Path( model_path )

        if not re.search( r"\/$", model_path ): model_path += "/"

        self.Print_Log( "LBD::Save_Model() - Saving Model To Path: " + str( model_path ), force_print = True )
        self.model.Save_Model( model_path = model_path + model_name )

        # Save Model Keys
        self.Save_Model_Keys( key_path = model_path, model_name = model_name )

        self.Print_Log( "LBD::Save_Model() - Complete" )

    """
        Save Model Token ID Key Data

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Save_Model_Keys( self, key_path = "./", model_name = "model", file_name = "_token_id_key_data" ):
        path_contains_directories = self.utils.Check_If_Path_Contains_Directories( key_path )
        self.Print_Log( "LBD::Save_Model_Keys() - Checking If Path Contains Directories: " + str( path_contains_directories ) )

        if self.utils.Check_If_Path_Exists( key_path ) == False:
            self.Print_Log( "LBD::Save_Model_Keys() - Creating Model Key Save Path: " + str( key_path ) )
            self.utils.Create_Path( key_path )

        if not re.search( r"\/$", key_path ): key_path += "/"

        self.Print_Log( "LBD::Save_Model_Keys() - Saving Model Keys To Path: " + str( key_path ) )
        self.data_loader.Save_Token_ID_Key_Data( key_path + model_name + file_name )

        self.Print_Log( "LBD::Save_Model_Keys() - Complete" )

    """
        Generates Keras Model Depiction
    """
    def Generate_Model_Depiction( self, path = "./" ):
        # Check
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Generate_Model_Depiction() - Error: No Model Object In Memory / Has Model Been Trained Or Loaded Yet?", force_print = True )
            return

        self.Print_Log( "LBD::Generate_Model_Depiction() - Generating Model Depiction" )

        self.model.Generate_Model_Depiction( path )

        self.Print_Log( "LBD::Generate_Model_Depiction() - Complete" )

    """
        Generates Plots (PNG Images) For Reported Metric Values During Each Training Epoch
    """
    def Generate_Model_Metric_Plots( self, path ):
        # Check
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Error: No Model Object In Memory / Has Model Been Trained Or Loaded Yet?", force_print = True )
            return

        self.utils.Create_Path( path )
        if not re.search( r"\/$", path ): path += "/"

        history = self.model.model_history.history

        if "loss" in history:
            self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Plotting Training Set - Loss vs Epoch" )
            figure = plt.figure()
            plt.plot( range( len( self.model.model_history.epoch ) ), history['loss'] )
            plt.title( "Training: Loss vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Loss" )
            plt.savefig( str( path ) + "training_epoch_vs_loss.png" )
            plt.close( figure )
            plt.clf()

        if "Accuracy" in history or "acc" in history:
            self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Plotting Training Set - Accuracy vs Epoch" )
            figure = plt.figure()
            plt.plot( range( len( self.model.model_history.epoch ) ), history['accuracy'] if 'accuracy' in history else history['acc'] )
            plt.title( "Training: Accuracy vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Accuracy" )
            plt.savefig( str( path ) + "training_epoch_vs_accuracy.png" )
            plt.close( figure )
            plt.clf()

        if "Precision" in history:
            self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Plotting Training Set - Precision vs Epoch" )
            figure = plt.figure()
            plt.plot( range( len( self.model.model_history.epoch ) ), history['Precision'] )
            plt.title( "Training: Precision vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Precision" )
            plt.savefig( str( path ) + "training_epoch_vs_precision.png" )
            plt.close( figure )
            plt.clf()

        if "Recall" in history:
            self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Plotting Training Set - Recall vs Epoch" )
            figure = plt.figure()
            plt.plot( range( len( self.model.model_history.epoch ) ), history['Recall'] )
            plt.title( "Training: Recall vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Recall" )
            plt.savefig( str( path ) + "training_epoch_vs_recall.png" )
            plt.close( figure )
            plt.clf()

        if "F1_Score" in history:
            self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Plotting Training Set - F1-Score vs Epoch" )
            figure = plt.figure()
            plt.plot( range( len( self.model.model_history.epoch ) ), history['F1_Score'] )
            plt.title( "Training: F1-Score vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "F1-Score" )
            plt.savefig( str( path ) + "training_epoch_vs_f1.png" )
            plt.close( figure )
            plt.clf()

        self.Print_Log( "LBD::Generate_Model_Metric_Plots() - Complete" )


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Checks If Checkpoint Directory Exists And Creates It If Not Existing
    """
    def Create_Checkpoint_Directory( self ):
        self.Print_Log( "LBD::Create_Checkpoint_Directory() - Checking If Model Save Directory Exists: \"" + str( self.checkpoint_directory ) + "\"", force_print = True )

        if self.utils.Check_If_Path_Exists( self.checkpoint_directory ) == False:
            self.Print_Log( "LBD::Create_Checkpoint_Directory() - Creating Directory", force_print = True )
            os.mkdir( self.checkpoint_directory )
        else:
            self.Print_Log( "LBD::Init() - Directory Already Exists", force_print = True )

    """
        Fetches Neural Model Type From File
    """
    def Get_Setting_Value_From_Model_Settings( self, file_path, setting_name ):
        model_settings_list = self.utils.Read_Data( file_path )

        for model_setting in model_settings_list:
            # Check For Formatting Error i.e. 'setting_name<:>value'
            if "<:>" not in model_setting: continue

            # Check For Setting Name And Value In Model Settings
            if re.match( r'^#', model_setting ) or model_setting == "" or model_setting == "\n": continue
            key, value = model_setting.split( "<:>" )
            if key == setting_name:
                return str( value )

        return None

    """
        Prints Debug Text To Console
    """
    def Print_Log( self, text, print_new_line = True, force_print = False ):
        if self.debug_log or force_print:
            print( text ) if print_new_line else print( text, end = " " )
        if self.write_log:
            self.Write_Log( text, print_new_line )

    """
        Prints Debug Log Text To File
    """
    def Write_Log( self, text, print_new_line = True ):
        if self.write_log and self.debug_log_file_handle is not None:
            self.debug_log_file_handle.write( text + "\n" ) if print_new_line else self.debug_log_file_handle.write( text )

    ############################################################################################
    #                                                                                          #
    #    Accessor Functions                                                                    #
    #                                                                                          #
    ############################################################################################

    def Get_Model( self ):                          return self.model

    def Get_Network_Model( self ):                  return self.model.Get_Network_Model()

    def Get_Model_Type( self ):                     return self.model.Get_Model_Type()

    def Get_Data( self ):                           return self.data_loader.Get_Data()

    def Get_Primary_Inputs( self ):                 return self.data_loader.Get_Primary_Inputs()

    def Get_Secondary_Inputs( self ):               return self.data_loader.Get_Secondary_Inputs()

    def Get_Outputs( self ):                        return self.data_loader.Get_Outputs()

    def Get_Number_Of_Unique_Features( self ):      return self.data_loader.Get_Number_Of_Unique_Features()

    def Get_Number_Of_Primary_Elements( self ):     return self.data_loader.Get_Number_Of_Primary_Elements()

    def Get_Number_Of_Secondary_Elements( self ):   return self.data_loader.Get_Number_Of_Secondary_Elements()

    def Get_Number_Of_Tertiary_Elements( self ):    return self.data_loader.Get_Number_Of_Tertiary_Elements()

    def Get_Data_Loader( self ):                    return self.data_loader

    def Is_Embeddings_Loaded( self ):               return self.data_loader.Is_Embeddings_Loaded()

    def Get_Debug_File_Handle( self ):              return self.debug_log_file_handle

    def Is_Model_Loaded( self ):                    return self.model.Is_Model_Loaded()

    def Reached_End_Of_File( self ):                return self.data_loader.Reached_End_Of_File()

    def Reset_File_Position_Index( self ):          return self.data_loader.Reset_File_Position_Index()

    def Is_Model_Data_Prepared( self ):             return self.model_data_prepared

    def Get_Next_Batch( self, file_path, number_of_elements_to_fetch ):
        return self.data_loader.Get_Next_Batch( file_path, number_of_elements_to_fetch )

    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################

    def Set_Data_Loader( self, new_data_loader ):   self.data_loader = new_data_loader


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    exit()