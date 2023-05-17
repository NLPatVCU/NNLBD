#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/20/2020                                                                   #
#    Revised: 01/03/2022                                                                   #
#                                                                                          #
#    Base Neural Network Architecture Class For NNLBD.                                     #
#                                                                                          #
#    Model Parent Class                                                                    #
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
import math, os, re, time
import subprocess as sp

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

import math as m
import numpy as np
from tensorflow import keras

# TensorFlow v2.x Support
if re.search( r'2.\d+', tf.__version__ ):
    import tensorflow.keras.backend as K
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Dense, Layer
    from tensorflow.keras.metrics import categorical_accuracy
# TensorFlow v1.15.x Support
else:
    import keras.backend as K
    from keras import regularizers
    from keras.layers import Dense, Layer
    from keras.metrics import categorical_accuracy

# Custom Modules
from NNLBD.Misc import Utils


############################################################################################
#                                                                                          #
#    Keras Model Custom Callback Classes                                                   #
#                                                                                          #
############################################################################################

"""
    Keras Model Custom Callback Class - Saves Model After Each Epoch
"""
class Model_Saving_Callback( keras.callbacks.Callback ):
    def on_epoch_end( self, epoch, logs = {} ):
        self.model.save( "ckpt_models/model_{}.hd5".format( epoch ) )

"""
    Keras Model Custom Callback Class - Cosine Annealing Scheduler
"""
class Cosine_Annealing_Scheduler( keras.callbacks.Callback ):
    def __init__( self, T_max, eta_max, eta_min = 0, lr = 0.05, verbose = 0 ):
        super( Cosine_Annealing_Scheduler, self ).__init__()
        self.lr      = lr
        self.T_max   = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin( self, epoch, logs = None ):
        if not hasattr( self.model.optimizer, 'lr' ):
            raise ValueError( 'Optimizer must have a "lr" attribute.' )

        lr = self.eta_min + ( self.eta_max - self.eta_min ) * ( 1 + math.cos( math.pi * epoch / self.T_max ) ) / 2
        K.set_value( self.model.optimizer.lr, lr )

        if self.verbose > 0:
            print( '\nEpoch %05d: CosineAnnealingScheduler setting learning rate to %s.' % ( epoch + 1, lr ) )

    def on_epoch_end( self, epoch, logs = None ):
        logs = logs or {}
        logs['lr'] = K.get_value( self.model.optimizer.lr )


############################################################################################
#                                                                                          #
#    Base Model Class                                                                      #
#                                                                                          #
############################################################################################

class BaseModel( object ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, network_model = "N/A", model_type = "open_discovery",
                  optimizer = 'adam', activation_function = 'sigmoid', loss_function = "binary_crossentropy", number_of_embedding_dimensions = 200,
                  number_of_hidden_dimensions = 200, bilstm_merge_mode = "concat", bilstm_dimension_size = 64, learning_rate = 0.005, epochs = 30,
                  momentum = 0.05, dropout = 1.0, batch_size = 32, prediction_threshold = 0.5, shuffle = True, use_csr_format = True, final_layer_type = "dense",
                  per_epoch_saving = False, use_gpu = True, device_name = "/gpu:0", verbose = 2, debug_log_file_handle = None, embedding_modification = "concatenate",
                  enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_metric_monitor = "loss", early_stopping_persistence = 3,
                  use_batch_normalization = False, checkpoint_directory = "./ckpt_models", trainable_weights = False, embedding_path = "",
                  scale = 30.0, margin = 0.35, feature_scale_value = 1.0, learning_rate_decay = 0.004, weight_decay = 0.0001, use_cosine_annealing = False,
                  cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4, skip_gpu_init = False ):
        self.version                         = 0.19
        self.network_model                   = network_model
        self.model                           = None                            # Automatically Set After Calling 'Build_Model()' Function
        self.epochs                          = epochs                          # Integer Value ie. 10, 32, 64, 200, etc.
        self.verbose                         = verbose                         # Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)
        self.dropout                         = dropout                         # Float Value: 0.05 (Not Currently Used)
        self.debug_log                       = print_debug_log                 # Options: True, False
        self.write_log                       = write_log_to_file               # Options: True, False
        self.batch_size                      = batch_size                      # Integer Values: 10, 32, 64, 200, etc.
        self.shuffle                         = shuffle                         # Options: True, False
        self.momentum                        = momentum                        # Float Value: 0.05 (Not Currently Used)
        self.loss_function                   = loss_function                   # Example Options: "binary_crossentropy", "categorical_crossentropy", sparse_crossentropy", "cosine_similarity"
        self.optimizer                       = optimizer                       # Example Options: "adam", "sgd"
        self.activation_function             = activation_function             # Example Options: "sigmoid", "softplus"
        self.learning_rate                   = learning_rate                   # Known Good: 0.005
        self.feature_scale_value             = feature_scale_value             # Default: 1.0 (Float)
        self.learning_rate_decay             = learning_rate_decay             # Default: 0.004 (Float)
        self.weight_decay                    = weight_decay                    # Default: 0.0001 (Float)
        self.model_type                      = model_type                      # Options: "open_discovery" or "closed_discovery"
        self.prediction_threshold            = prediction_threshold            # Float Value: 0.5 (Default) => Inflection Point Of The Sigmoid Function
        self.trainable_weights               = trainable_weights               # Options: True, False
        self.embeddings_loaded               = False                           # Automatically Set By 'LBD' Class After Embeddings Have Been Loaded In 'DataLoader' Class.
        self.embedding_path                  = embedding_path                  # Path (String)
        self.embedding_modification          = embedding_modification          # Options: 'hadamard', 'average', 'concatenate' (String)
        self.final_layer_type                = final_layer_type                # Options: 'arcface', 'cosface', 'sphereface' (String)
        self.scale                           = scale                           # Float Value: 30.0 - CosFace, ArcFace and SphereFace Default
        self.margin                          = margin                          # Float Value: 0.35 - CosFace, 0.50 - ArcFace, 1.35 - SphereFace (Defaults)
        self.per_epoch_saving                = per_epoch_saving                # Options: True, False
        self.model_history                   = None                            # Automatically Set After Training Data Has Been Parsed
        self.number_of_embedding_dimensions  = number_of_embedding_dimensions  # Integer Value: 200 (Default)
        self.number_of_hidden_dimensions     = number_of_hidden_dimensions     # Integer Value: 200 (Default)
        self.bilstm_dimension_size           = bilstm_dimension_size           # Integer Value: 64 (Default)
        self.bilstm_merge_mode               = bilstm_merge_mode               # BiLSTM Layer Options: 'sum', 'mul', 'ave', 'concat'
        self.use_batch_normalization         = use_batch_normalization         # Options: True (Use BatchNorm), False (No BatchNorm)
        self.number_of_features              = -1                              # Automatically Set While Building Model
        self.number_of_primary_inputs        = -1                              # Automatically Set After Training Data Has Been Parsed
        self.number_of_secondary_inputs      = -1                              # Automatically Set After Training Data Has Been Parsed
        self.number_of_tertiary_inputs       = -1                              # Automatically Set After Training Data Has Been Parsed (Used With CD2Model class)
        self.number_of_outputs               = -1                              # Automatically Set After Training Data Has Been Parsed
        self.debug_log_file_handle           = debug_log_file_handle           # Debug Log File Handle
        self.debug_log_file_name             = "Model_Log.txt"                 # File Name (String)
        self.enable_tensorboard_logs         = enable_tensorboard_logs         # Options: True (Enable), False (Disable)
        self.enable_early_stopping           = enable_early_stopping           # Options: True (Enable), False (Disable)
        self.early_stopping_metric_monitor   = early_stopping_metric_monitor   # String: Default "loss" (Better Choice Might Be "F1_Score")
        self.early_stopping_persistence      = early_stopping_persistence      # Integer Value: 3 (Default)
        self.use_cosine_annealing            = use_cosine_annealing            # Enables Use Of Cosine Annealing Learning Rate Scheduler - Options: True, False
        self.cosine_annealing_min            = cosine_annealing_min            # Minimum Learing Rate Value Used By Cosine Annealing Learning Rate Scheduler
        self.cosine_annealing_max            = cosine_annealing_max            # Maximum Learing Rate Value Used By Cosine Annealing Learning Rate Scheduler
        self.trained_instances               = -1                              # Automatically Set After Training Data Has Been Parsed
        self.evaluated_instances             = -1                              # Automatically Set After Evaluation Data Has Been Parsed
        self.use_csr_format                  = use_csr_format                  # Options: True, False
        self.use_gpu                         = use_gpu                         # Options: True (Use GPU/CUDA), False (Use CPU)
        self.device_name                     = device_name                     # Options: "/cpu:0", "/gpu:0"
        self.printed_gpu_polling_message     = False                           # Debug Statement Printing Which Notifies User The Model Is Polling For Available GPU(s)
        self.checkpoint_directory            = checkpoint_directory            # Path (String)
        self.callback_list                   = []                              # Keras Model Callback List - Set During Model 'Build_Model()' Call.

        # Specify Final Layer Types Available
        self.final_layer_type_list           = ["dense", "cosface", "arcface", "sphereface"]

        # Create New Utils Class
        self.utils                           = Utils()

        # Check(s) - Set Default Parameters If Not Specified
        if self.number_of_embedding_dimensions == None: self.number_of_embedding_dimensions  = 200
        if self.bilstm_dimension_size          == None: self.bilstm_dimension_size           = 64
        if self.number_of_hidden_dimensions    == None: self.number_of_hidden_dimensions     = 200

        # Create Log File Handle
        if self.write_log and self.debug_log_file_handle is None:
            self.debug_log_file_handle = open( self.debug_log_file_name, "w" )

        if self.model_type != "open_discovery" and self.model_type != "closed_discovery":
            self.Print_Log( "BaseModel::Init() - Warning: Model Type Not Equal 'open_discovery' or 'closed_discovery' / Setting To 'open_discovery'", force_print = True )
            self.model_type = "open_discovery"

        if self.use_csr_format == False:
            self.Print_Log( "BaseModel::Init() - Warning: Use CSR Mode = False / High Memory Consumption May Occur When Vectorizing Data-Sets", force_print = True )
        else:
            self.Print_Log( "BaseModel::Init() - Using CSR Matrix Format" )

        if self.per_epoch_saving:
            self.Create_Checkpoint_Directory()

        # Set CosFace, ArcFace and SphereFace Default Settings
        if scale == 0.35:       # Note: 0.35 Is 'CosFace' Default
            if self.final_layer_type == "arcface":
                self.scale = 0.50
            elif self.final_layer_type == "sphereface":
                self.scale = 1.35


        # Checks To See If The User Wants To Utilize The GPU or CPU For Training/Inference.
        if not skip_gpu_init and self.Initialize_GPU() == False:
            self.Print_Log( "BaseModel::Init() - Error Initializing GPU / Exiting", force_print = True )
            exit()

        self.Print_Log( "BaseModel::Init() - Complete" )

    """
       Remove Variables From Memory
    """
    def __del__( self ):
        del self.utils

        if self.write_log and self.debug_log_file_handle is not None: self.debug_log_file_handle.close()


    ############################################################################################
    #                                                                                          #
    #    Model Functions                                                                       #
    #                                                                                          #
    ############################################################################################

    """
        Checks And Initializes Or Deinitialies GPU Devices Depending On User Settings.

          - GPU Polling Algorithm Will Wait Up To Two Weeks While Examining Available GPU Devices By Memory Consumption.
          - Can Initialize Multiple GPUs Depending On 'number_of_desired_gpus'.
            (Only When GPU Polling Algo In Use / 'enable_gpu_polling' == True and 'number_of_desired_gpus' > 1).

        Inputs:
            enable_gpu_polling          : Involks GPU Polling Algorithm (Bool)
            acceptable_available_memory : Number Of Memory Necessary To Determine If A GPU Is Available (MBs) (Integer)
            number_of_desired_gpus      : Number Of Desired GPUs (Integer)
            polling_counter_limit       : GPU Polling Algorithm Threshold / 2 Weeks In Seconds (Integer)

        Outputs:
            True/False                  : True = Success / False = Error (Bool)

    """
    def Initialize_GPU( self, enable_gpu_polling = False, acceptable_available_memory = 4096, number_of_desired_gpus = 1, polling_counter_limit = 1209600 ):
        # Check(s)
        if number_of_desired_gpus <= 0:
            self.Print_Log( "BaseModel::Initialize_GPU() - Error: 'number_of_desired_gpus' <= 0" )
            return False

        # GPU/CUDA Checks
        self.Print_Log( "BaseModel::Initialize_GPU() - Checking For GPU/CUDA Compatibility" )
        self.Print_Log( "BaseModel::Initialize_GPU() - TensorFlow Version: " + str( tf.__version__ ) )

        if re.search( r'2.d+', tf.__version__ ):
            self.Print_Log( "BaseModel::Initialize_GPU() - CUDA Version: " + str( tf.sysconfig.get_build_info()["cuda_version"] ) )
            self.Print_Log( "BaseModel::Initialize_GPU() - CUDNN Version: " + str( tf.sysconfig.get_build_info()["cudnn_version"] ) )

        if self.use_gpu:
            # Check
            if re.search( r'/[Cc][Pp][Uu]:', self.device_name ):
                self.Print_Log( "BaseModel::Initialize_GPU() - Warning: 'use_gpu == True' and 'device_name = /cpu:xx' / Auto-Detecting Available GPU", force_print = True )
                enable_gpu_polling = True

            # Is TensorFlow Built With CUDA / CUDA Supported
            if tf.test.is_built_with_cuda():
                available_gpus     = []
                desired_device_ids = []

                # Get List Of Detected CUDA GPUs
                physical_gpus      = tf.config.experimental.list_physical_devices( device_type = "GPU" )

                if len( physical_gpus ) == 0:
                    self.Print_Log( "BaseModel::Initialize_GPU() - Error: No GPUs Detected By TensorFlow / Using CPU", force_print = True )
                    self.Print_Log( "BaseModel::Initialize_GPU() -        Ensure Your Version Of TensorFlow Supports The Installed CUDA Version", force_print = True )
                    self.Print_Log( "BaseModel::Initialize_GPU() -        TensorFlow Version: " + str( tf.__version__ ), force_print = True )

                    if re.search( r'2.d+', tf.__version__ ):
                        self.Print_Log( "BaseModel::Initialize_GPU() -        CUDA Version: " + str( tf.sysconfig.get_build_info()["cuda_version"] ), force_print = True )
                        self.Print_Log( "BaseModel::Initialize_GPU() -        CUDNN Version: " + str( tf.sysconfig.get_build_info()["cudnn_version"] ), force_print = True )

                    tf.config.experimental.set_visible_devices( [], 'GPU' )
                    self.device_name = "/cpu:0"
                    return True

                self.Print_Log( "BaseModel::Initialize_GPU() - CUDA Supported / GPU Is Available", force_print = True )

                ####################################
                #       GPU Polling (Start)        #
                # Wait For GPU To Become Available #
                ####################################
                if enable_gpu_polling:
                    polling_counter         = 0
                    polling_timer_exceeded  = False
                    silence_warning_message = True

                    # Wait For Number Of Desired GPUs To Become Available
                    while len( desired_device_ids ) != number_of_desired_gpus:
                        # Set Warning Polling (Print Warning Message Every 10 Minutes)
                        silence_warning_message = False if polling_counter == 0 or polling_counter % 600 == 0 else True

                        # Check For GPUs To Become Available
                        desired_device_ids = self.Get_Next_Available_CUDA_GPUs( acceptable_available_memory = acceptable_available_memory,
                                                                                number_of_desired_gpus = number_of_desired_gpus,
                                                                                silence_warning_message = silence_warning_message )

                        # Copy Desired Device IDs
                        available_gpus     = desired_device_ids

                        # Wait For One Second And Then Check Again
                        time.sleep( 1 )

                        # Increment Polling Counter
                        polling_counter += 1

                        # If Polling For Over 2 Weeks, End Polling
                        if polling_counter >= polling_counter_limit:
                            polling_timer_exceeded = True
                            break

                    # Program Has Been Waiting Two Weeks For A GPU To Become Available / Stop Polling And Terminate
                    if polling_timer_exceeded:
                        self.Print_Log( "BaseModel::Initialize_GPU() - Error: Unable To Secure An Available GPU Within A 2 Week Period / Terminating Program" )
                        return False

                ####################################
                #        GPU Polling (End)         #
                ####################################
                else:
                    available_device_ids = self.Get_Next_Available_CUDA_GPUs( acceptable_available_memory = acceptable_available_memory, number_of_desired_gpus = number_of_desired_gpus )
                    available_gpus       = [physical_gpus[id] for id in available_device_ids]

                    # Get Numerical ID Value Of Desired GPU Device
                    desired_device_ids.append( int( self.device_name.split( ":" )[-1] ) if re.search( r'/[Gg][Pp][Uu]:', self.device_name ) else int( self.device_name ) )

                self.Print_Log( "BaseModel::Initialize_GPU() - GPU Device List: " + str( physical_gpus ) )
                self.Print_Log( "BaseModel::Initialize_GPU() - Available GPU Device List: " + str( available_gpus ) )
                self.Print_Log( "BaseModel::Initialize_GPU() - Desired GPU Device IDs: " + str( desired_device_ids ) )

                # Check(s)
                for desired_device_id in desired_device_ids:
                    if isinstance( desired_device_id, int ):
                        if desired_device_id > len( physical_gpus ) - 1:
                            self.Print_Log( "BaseModel::Initialize_GPU() - Error: Device Specified Is Not Detected In GPU List", force_print = True )
                            self.Print_Log( "                            - Total Number GPUs: " + str( len( physical_gpus ) ), force_print = True )
                            self.Print_Log( "                            - Desired Device ID: " + str( desired_device_id    ), force_print = True )
                            return False

                # Limit Memory Growth Among Devices And Set GPU Devices Visible To The Model
                try:
                    # Set Desired GPUs To Make Available To The Model
                    available_gpus = [physical_gpus[desired_device_id] for desired_device_id in desired_device_ids]
                    tf.config.experimental.set_visible_devices( available_gpus, 'GPU' )

                    # Limit Memory Growth Among Devices
                    for available_gpu in available_gpus:
                        tf.config.experimental.set_memory_growth( available_gpu, True )

                    self.Print_Log( "BaseModel::Initialize_GPU() - Using GPU Device(s): " + str( available_gpus ) )
                    self.Print_Log( "BaseModel::Initialize_GPU() - GPU/CUDA Supported And Enabled", force_print = True )

                except RuntimeError as e:
                    self.Print_Log( "BaseModel::Initialize_GPU() - Error: " + str( e ) )
                    return False
            else:
                self.Print_Log( "BaseModel::Initialize_GPU() - Warning: TensorFlow Not Compiled With CUDA Support", force_print = True )
                self.Print_Log( "                            - Using CPU", force_print = True )
                self.device_name = "/cpu:0"
        else:
            self.Print_Log( "BaseModel::Initialize_GPU() - GPU (CUDA) Support Disabled / Using CPU", force_print = True )
            tf.config.experimental.set_visible_devices( [], 'GPU' )
            self.device_name = "/cpu:0"

        self.Print_Log( "BaseModel::Initialize_GPU() - Complete" )

        return True

    """
        Examines The List Of Available CUDA GPUs And Returns Device ID Of A GPU Not Being Utilized.
          Does This By Examining GPU Memory Consumption.

        Inputs:
            acceptable_available_memory : Number Of Memory Necessary To Determine If A GPU Is Available (MBs) (Integer)
            number_of_desired_gpus      : Number Of Desired GPUs (Integer)
            silence_warning_message     : Silences The Warning Message When No Usable GPUs Are Found (Bool)

        Outputs:
            visible_device_ids          : List Of Available GPU Device IDs (Integer)

        Modification Of Source: https://stackoverflow.com/questions/40069883/how-to-set-specific-gpu-in-tensorflow
    """
    def Get_Next_Available_CUDA_GPUs( self, acceptable_available_memory = 4096, number_of_desired_gpus = 1, silence_warning_message = False ):
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

        try:
            _output_to_list    = lambda x: x.decode( 'ascii' ).split( '\n' )[:-1]
            memory_free_info   = _output_to_list( sp.check_output( COMMAND.split() ) )[1:]
            memory_free_values = [int( x.split()[0] ) for i, x in enumerate( memory_free_info )]
            available_gpus     = [i for i, x in enumerate( memory_free_values ) if x > acceptable_available_memory]

            if silence_warning_message == False and len( available_gpus ) < number_of_desired_gpus:
                self.Print_Log( "BaseModel::Get_Next_Available_CUDA_GPU() - Warning: Found " + str( len( available_gpus ) ) + " Usable GPUs In The System", force_print = True )
                self.Print_Log( "BaseModel::Get_Next_Available_CUDA_GPU() -          Desired Available Memory: " + str( acceptable_available_memory ) )
                self.Print_Log( "                                         -          Desired Number Of GPUs: " + str( number_of_desired_gpus ) )

            visible_device_ids = [id for id in available_gpus[:number_of_desired_gpus]]

            if silence_warning_message == False and len( available_gpus ):
                self.Print_Log( "BaseModel::Get_Next_Available_CUDA_GPU() - Available GPU IDs: " + str( visible_device_ids ) )

            return visible_device_ids

        except Exception as e:
            self.Print_Log( "BaseModel::Get_Next_Available_CUDA_GPU() - Warning: 'nvidia-smi' Not Detected In Path. GPUs Are Not Masked" + str( e ), force_print = True )

        return []


    """
        Check To See If A Model Is Loaded
    """
    def Is_Model_Loaded( self ):
        return True if self.model is not None else False

    """
        Loads The Model From A File

        Inputs:
            model_path      : File Path (String)
            load_new_model  : Sets Whether Or Not To Load A New Model Or Use The Model Already In Memory (Bool)
            bypass_gpu_init : Bypassed GPU Initialization (Bool)

        Outputs:
            None
    """
    def Load_Model( self, model_path, load_new_model = True, bypass_gpu_init = False ):
        try:
            if load_new_model:
                self.Print_Log( "BaseModel::Load_Model() - Loading Pretrained Model" )

                # Load Model
                if self.utils.Check_If_File_Exists( model_path + ".h5" ):
                    self.model = keras.models.load_model( model_path + ".h5", custom_objects = { 'F1_Score' : self.F1_Score,
                                                                                                'Precision': self.Precision,
                                                                                                'Recall'   : self.Recall } )
                else:
                    self.Print_Log( "BaseModel::Load_Model() - Error: Model Does Not Exist", force_print = True )
                    return False
            else:
                self.Print_Log( "BaseModel::Load_Model() - Loading Model Configuration / Untrained Model / New Weights Per Layer" )

                # Load Model Configuration
                if self.utils.Check_If_File_Exists( model_path + "_config.json" ):
                    with open( model_path + "_config.json", "r" ) as in_file:
                        model_config = ""
                        model_config = in_file.read()
                        self.model   = keras.models.model_from_json( model_config )
                    in_file.close()
                else:
                    self.Print_Log( "BaseModel::Load_Model() - Error: Model Configuration JSON Files Does Not Exist", force_print = True )
                    return False
        except Exception as e:
            self.Print_Log( "BaseModel::Load_Model() - Error: Unable To Load New Model Or Load Model From Configuration File" )
            self.Print_Log( "                        - Error Message: " + str( e ) )

        # Load Model Settings
        if self.utils.Check_If_File_Exists( model_path + "_settings.cfg" ):
            self.Load_Model_Settings( model_path + "_settings.cfg" )
        else:
            self.Print_Log( "BaseModel::Load_Model() - Error: Model Settings File Does Not Exist", force_print = True )

        # Checks To See If The User Wants To Utilize The GPU or CPU For Training/Inference.
        if bypass_gpu_init == False and self.Initialize_GPU() == False:
            self.Print_Log( "BaseModel::Load_Model() - Error: Unable To Initialize GPU", force_print = True )
            exit()
        else:
            self.Print_Log( "BaseModel::Load_Model() - Bypass GPU Initialization == True" )

        self.Print_Log( "BaseModel::Load_Model() - Complete" )
        return True

    """
        Saves Model To File

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Save_Model( self, model_path ):
        # Check
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "BaseModel::Save_Model() - Error: No Model Object In Memory / Has Model Been Trained Or Loaded?", force_print = True )
            return

        # Save Model Configuration
        self.Print_Log( "BaseModel::Save_Model() - Saving Model Configuration To Path: " + str( model_path + "_config.json" ) )

        try:
            model_configuration = str( self.model.to_json() )

            with open( model_path + "_config.json", 'w' ) as out_file:
                out_file.write( model_configuration )
            out_file.close()

        except Exception as e:
            self.Print_Log( "BaseModel::Save_Model() - Error: Unable To Generate Model Configuration File" )
            self.Print_Log( "                        -   Error Message: " + str( e ) )

        # Save Model
        try:
            self.Print_Log( "BaseModel::Save_Model() - Saving Model To Path: " + str( model_path + ".h5" ) )
            self.model.save( model_path + ".h5")

        except Exception as e:
            self.Print_Log( "BaseModel::Save_Model() - Error: Unable To Save Model" )
            self.Print_Log( "                        -   Error Message: " + str( e ) )

        # Save Model Settings
        self.Print_Log( "BaseModel::Save_Model() - Saving Model Settings To Path: " + str( model_path + "_settings.cfg" ) )
        self.Save_Model_Settings( model_path + "_settings.cfg" )

        self.Print_Log( "BaseModel::Save_Model() - Complete" )

    """
        Loads Model Settings

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Load_Model_Settings( self, file_path ):
        self.Print_Log( "BaseModel::Load_Model_Settings() - Loading Model Settings From File: " + str( file_path ) )

        model_setting_data = self.utils.Read_Data( file_path )

        for model_setting in model_setting_data:
            if re.match( r'^#', model_setting ) or model_setting == "": continue
            key, value = model_setting.split( "<:>" )

            if key == "NetworkModel"               : self.network_model                   = str( value )
            if key == "ModelType"                  : self.model_type                      = str( value )
            if key == "Epochs"                     : self.epochs                          = int( value )
            if key == "Verbose"                    : self.verbose                         = int( value )
            if key == "Dropout"                    : self.dropout                         = float( value )
            if key == "DebugLog"                   : self.debug_log                       = True if value == "True" else False
            if key == "WriteLog"                   : self.write_log                       = True if value == "True" else False
            if key == "BatchSize"                  : self.batch_size                      = int( value )
            if key == "Shuffle"                    : self.shuffle                         = True if value == "True" else False
            if key == "Momentum"                   : self.momentum                        = float( value )
            if key == "Optimizer"                  : self.optimizer                       = str( value )
            if key == "ActivationFunction"         : self.activation_function             = str( value )
            if key == "LossFunction"               : self.loss_function                   = str( value )
            if key == "LearningRate"               : self.learning_rate                   = float( value )
            if key == "LearningRateDecay"          : self.learning_rate_decay             = float( value )
            if key == "WeightDecay"                : self.weight_decay                    = float( value )
            if key == "PredictionThreshold"        : self.prediction_threshold            = float( value )
            if key == "TrainableWeights"           : self.trainable_weights               = True if value == "True" else False
            if key == "EmbeddingsLoaded"           : self.embeddings_loaded               = True if value == "True" else False
            if key == "EmbeddingPath"              : self.embedding_path                  = str( value )
            if key == "EmbeddingModification"      : self.embedding_modification          = str( value )
            if key == "FinalLayerType"             : self.final_layer_type                = str( value )
            if key == "Margin"                     : self.margin                          = float( value )
            if key == "Scale"                      : self.scale                           = float( value )
            if key == "PerEpochSaving"             : self.per_epoch_saving                = True if value == "True" else False
            if key == "LogFileName"                : self.debug_log_file_name             = str( value )
            if key == "UseCSRFormat"               : self.use_csr_format                  = True if value == "True" else False
            if key == "NumberOfEmbeddingDimensions": self.number_of_embedding_dimensions  = int( value )
            if key == "BiLSTMDimensionSize"        : self.bilstm_dimension_size           = int( value )
            if key == "BiLSTMMergeMode"            : self.bilstm_merge_mode               = str( value )
            if key == "UseBatchNormalization"      : self.use_batch_normalization         = str( value )
            if key == "NumberOfFeatures"           : self.number_of_features              = int( value )
            if key == "NumberOfPrimaryInputs"      : self.number_of_primary_inputs        = int( value )
            if key == "NumberOfSecondaryInputs"    : self.number_of_secondary_inputs      = int( value )
            if key == "NumberOfTertiaryInputs"     : self.number_of_tertiary_inputs       = int( value )
            if key == "NumberOfHiddenDimensions"   : self.number_of_hidden_dimensions     = int( value )
            if key == "NumberOfOutputs"            : self.number_of_outputs               = int( value )
            if key == "UseGPU"                     : self.use_gpu                         = True if value == "True" else False
            if key == "DeviceName"                 : self.device_name                     = str( value )
            if key == "EnableTensorboardLogs"      : self.enable_tensorboard_logs         = True if value == "True" else False
            if key == "EnableEarlyStopping"        : self.enable_early_stopping           = True if value == "True" else False
            if key == "EarlyStoppingMetricMonitor" : self.early_stopping_metric_monitor   = str( value )
            if key == "EarlyStoppingPersistence"   : self.early_stopping_persistence      = int( value )
            if key == "UseCosineAnnealing"         : self.use_cosine_annealing            = True if value == "True" else False
            if key == "CosineAnnealingMin"         : self.cosine_annealing_min            = float( value )
            if key == "CosineAnnealingMax"         : self.cosine_annealing_max            = float( value )
            if key == "FeatureScaleValue"          : self.feature_scale_value             = float( value )

        self.Print_Log( "BaseModel::Load_Model_Settings() - Done" )

    """
        Saves Model Settings

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Save_Model_Settings( self, file_path ):
        self.Print_Log( "BaseModel::Save_Model_Settings() - Saving Model Settings To File: " + str( file_path ) )

        # Open File Handle
        fh = open( file_path, "w" )

        fh.write( "NetworkModel<:>"                + str( self.network_model                  ) + "\n" )
        fh.write( "ModelType<:>"                   + str( self.model_type                     ) + "\n" )
        fh.write( "Epochs<:>"                      + str( self.epochs                         ) + "\n" )
        fh.write( "Verbose<:>"                     + str( self.verbose                        ) + "\n" )
        fh.write( "Dropout<:>"                     + str( self.dropout                        ) + "\n" )
        fh.write( "DebugLog<:>"                    + str( self.debug_log                      ) + "\n" )
        fh.write( "WriteLog<:>"                    + str( self.write_log                      ) + "\n" )
        fh.write( "BatchSize<:>"                   + str( self.batch_size                     ) + "\n" )
        fh.write( "Shuffle<:>"                     + str( self.shuffle                        ) + "\n" )
        fh.write( "Momentum<:>"                    + str( self.momentum                       ) + "\n" )
        fh.write( "Optimizer<:>"                   + str( self.optimizer                      ) + "\n" )
        fh.write( "ActivationFunction<:>"          + str( self.activation_function            ) + "\n" )
        fh.write( "LossFunction<:>"                + str( self.loss_function                  ) + "\n" )
        fh.write( "LearningRate<:>"                + str( self.learning_rate                  ) + "\n" )
        fh.write( "LearningRateDecay<:>"           + str( self.learning_rate_decay            ) + "\n" )
        fh.write( "WeightDecay<:>"                 + str( self.weight_decay                   ) + "\n" )
        fh.write( "PredictionThreshold<:>"         + str( self.prediction_threshold           ) + "\n" )
        fh.write( "TrainableWeights<:>"            + str( self.trainable_weights              ) + "\n" )
        fh.write( "EmbeddingPath<:>"               + str( self.embedding_path                 ) + "\n" )
        fh.write( "EmbeddingModification<:>"       + str( self.embedding_modification         ) + "\n" )
        fh.write( "FinalLayerType<:>"              + str( self.final_layer_type               ) + "\n" )
        fh.write( "Margin<:>"                      + str( self.margin                         ) + "\n" )
        fh.write( "Scale<:>"                       + str( self.scale                          ) + "\n" )
        fh.write( "PerEpochSaving<:>"              + str( self.per_epoch_saving               ) + "\n" )
        fh.write( "LogFileName<:>"                 + str( self.debug_log_file_name            ) + "\n" )
        fh.write( "UseCSRFormat<:>"                + str( self.use_csr_format                 ) + "\n" )
        fh.write( "NumberOfEmbeddingDimensions<:>" + str( self.number_of_embedding_dimensions ) + "\n" )
        fh.write( "BiLSTMDimensionSize<:>"         + str( self.bilstm_dimension_size          ) + "\n" )
        fh.write( "BiLSTMMergeMode<:>"             + str( self.bilstm_merge_mode              ) + "\n" )
        fh.write( "UseBatchNormalization<:>"       + str( self.use_batch_normalization        ) + "\n" )
        fh.write( "NumberOfFeatures<:>"            + str( self.number_of_features             ) + "\n" )
        fh.write( "NumberOfPrimaryInputs<:>"       + str( self.number_of_primary_inputs       ) + "\n" )
        fh.write( "NumberOfSecondaryInputs<:>"     + str( self.number_of_secondary_inputs     ) + "\n" )
        fh.write( "NumberOfTertiaryInputs<:>"      + str( self.number_of_tertiary_inputs      ) + "\n" )
        fh.write( "NumberOfHiddenDimensions<:>"    + str( self.number_of_hidden_dimensions    ) + "\n" )
        fh.write( "NumberOfOutputs<:>"             + str( self.number_of_outputs              ) + "\n" )
        fh.write( "UseGPU<:>"                      + str( self.use_gpu                        ) + "\n" )
        fh.write( "DeviceName<:>"                  + str( self.device_name                    ) + "\n" )
        fh.write( "EnableTensorboardLogs<:>"       + str( self.enable_tensorboard_logs        ) + "\n" )
        fh.write( "EnableEarlyStopping<:>"         + str( self.enable_early_stopping          ) + "\n" )
        fh.write( "EarlyStoppingMetricMonitor<:>"  + str( self.early_stopping_metric_monitor  ) + "\n" )
        fh.write( "EarlyStoppingPersistence<:>"    + str( self.early_stopping_persistence     ) + "\n" )
        fh.write( "UseCosineAnnealing<:>"          + str( self.use_cosine_annealing           ) + "\n" )
        fh.write( "CosineAnnealingMin<:>"          + str( self.cosine_annealing_min           ) + "\n" )
        fh.write( "CosineAnnealingMax<:>"          + str( self.cosine_annealing_max           ) + "\n" )
        fh.write( "FeatureScaleValue<:>"           + str( self.feature_scale_value            ) + "\n" )

        fh.write( "\n### DO NOT MODIFY VARIABLE SETTINGS BELOW ###\n"                                  )
        fh.write( "EmbeddingsLoaded<:>"            + str( self.embeddings_loaded              ) + "\n" )

        fh.close()

        self.Print_Log( "BaseModel::Save_Model_Settings() - Done" )

    """
        Prints Model Configuration To Console
    """
    def Print_Configuration( self ):
        self.Print_Log( "BaseModel::========================================================="           , force_print = True )
        self.Print_Log( "BaseModel::~      Neural Network - Literature Based Discovery      ~"           , force_print = True )
        self.Print_Log( "BaseModel::~                 Version " + str( self.version ) + "\t\t\t\t\t\t\t~", force_print = True )
        self.Print_Log( "BaseModel::=========================================================\n"         , force_print = True )

        self.Print_Log( "BaseModel::  Built with TensorFlow v1.14.0", force_print = True )
        self.Print_Log( "BaseModel::  Built with Keras v2.3.1",       force_print = True )
        self.Print_Log( "BaseModel::  Installed TensorFlow v" + str( tf.__version__ ), force_print = True )
        self.Print_Log( "BaseModel::  Installed Keras v"      + str( keras.__version__ ) + "\n", force_print = True )

        # Print Settings To Console
        self.Print_Log( "BaseModel::=========================================================" )
        self.Print_Log( "BaseModel::-   Configuration File Settings                         -" )
        self.Print_Log( "BaseModel::=========================================================" )

        self.Print_Log( "BaseModel::    Neural Network Model          : " + str( self.network_model                  ), force_print = True )
        self.Print_Log( "BaseModel::    Neural Model Type             : " + str( self.model_type                     ), force_print = True )
        self.Print_Log( "BaseModel::    Epochs                        : " + str( self.epochs                         ), force_print = True )
        self.Print_Log( "BaseModel::    Verbose                       : " + str( self.verbose                        ), force_print = True )
        self.Print_Log( "BaseModel::    Dropout                       : " + str( self.dropout                        ), force_print = True )
        self.Print_Log( "BaseModel::    Batch Size                    : " + str( self.batch_size                     ), force_print = True )
        self.Print_Log( "BaseModel::    Shuffle                       : " + str( self.shuffle                        ), force_print = True )
        self.Print_Log( "BaseModel::    Momentum                      : " + str( self.momentum                       ), force_print = True )
        self.Print_Log( "BaseModel::    Optimizer                     : " + str( self.optimizer                      ), force_print = True )
        self.Print_Log( "BaseModel::    Activation Function           : " + str( self.activation_function            ), force_print = True )
        self.Print_Log( "BaseModel::    Learning Rate                 : " + str( self.learning_rate                  ), force_print = True )
        self.Print_Log( "BaseModel::    Learning Rate Decay           : " + str( self.learning_rate_decay            ), force_print = True )
        self.Print_Log( "BaseModel::    Weight Decay                  : " + str( self.weight_decay                   ), force_print = True )
        self.Print_Log( "BaseModel::    Feature Scaling Value         : " + str( self.feature_scale_value            ), force_print = True )
        self.Print_Log( "BaseModel::    Loss Function                 : " + str( self.loss_function                  ), force_print = True )
        self.Print_Log( "BaseModel::    Prediction Threshold          : " + str( self.prediction_threshold           ), force_print = True )
        self.Print_Log( "BaseModel::    Final Layer Type              : " + str( self.final_layer_type               ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Primary Inputs      : " + str( self.number_of_primary_inputs       ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Secondary Inputs    : " + str( self.number_of_secondary_inputs     ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Tertiary Inputs     : " + str( self.number_of_tertiary_inputs      ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Hidden Dimensions   : " + str( self.number_of_hidden_dimensions    ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Outputs             : " + str( self.number_of_outputs              ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Features            : " + str( self.number_of_features             ), force_print = True )
        self.Print_Log( "BaseModel::    Number Of Embedding Dimensions: " + str( self.number_of_embedding_dimensions ), force_print = True )
        self.Print_Log( "BaseModel::    BiLSTM Dimension Size         : " + str( self.bilstm_dimension_size          ), force_print = True )
        self.Print_Log( "BaseModel::    BiLSTM Merge Mode             : " + str( self.bilstm_merge_mode              ), force_print = True )
        self.Print_Log( "BaseModel::    Use Batch Normalization       : " + str( self.use_batch_normalization        ), force_print = True )
        self.Print_Log( "BaseModel::    Use CSR Vector Format         : " + str( self.use_csr_format                 ), force_print = True )
        self.Print_Log( "BaseModel::    Save Model After Each Epoch   : " + str( self.per_epoch_saving               ), force_print = True )
        self.Print_Log( "BaseModel::    Enable Tensorboard Logs       : " + str( self.enable_tensorboard_logs        ), force_print = True )
        self.Print_Log( "BaseModel::    Enable Early Stopping         : " + str( self.enable_early_stopping          ), force_print = True )
        self.Print_Log( "BaseModel::    Early Stopping Monitor Metric : " + str( self.early_stopping_metric_monitor  ), force_print = True )
        self.Print_Log( "BaseModel::    Early Stopping Persistence    : " + str( self.early_stopping_persistence     ), force_print = True )
        self.Print_Log( "BaseModel::    Use Cosine Annealing          : " + str( self.use_cosine_annealing           ), force_print = True )
        self.Print_Log( "BaseModel::    Cosine Annealing Min          : " + str( self.cosine_annealing_min           ), force_print = True )
        self.Print_Log( "BaseModel::    Cosine Annealing Max          : " + str( self.cosine_annealing_max           ), force_print = True )
        self.Print_Log( "BaseModel::    Use GPU                       : " + str( self.use_gpu                        ), force_print = True )
        self.Print_Log( "BaseModel::    Device Name                   : " + str( self.device_name                    ), force_print = True )

        self.Print_Log( "BaseModel::=========================================================",   force_print = True )
        self.Print_Log( "BaseModel::-                                                       -"  , force_print = True )
        self.Print_Log( "BaseModel::=========================================================\n", force_print = True )

        self.Print_Log( "BaseModel::Print_Configuration() - Complete" )

    """
        Generates Model Depiction Images File In Path
    """
    def Generate_Model_Depiction( self, path = "./", show_shapes = True ):
        if self.utils.Check_If_Path_Exists( path ) == False:
            self.utils.Create_Path( path )

        self.Print_Log( "DataLoader::Generate_Model_Depiction() - Generating Model Depictions" )

        keras.utils.plot_model( self.model, path + "model.png", show_shapes = show_shapes )

        self.Print_Log( "DataLoader::Generate_Model_Depiction() - Complete" )

    """
        Prints Model Metrics From History
    """
    def Print_Model_Training_Metrics( self ):
        if not self.model_history:
            self.Print_Log( "BaseModel::Print_Model_Training_Metrics() - Error: No Model History In Memory / Has Model Or Loaded Been Trained Prior To Calling Function?", force_print = True )
            return

        for epoch in self.model_history.epoch:
            accuracy  = 0
            if 'accuracy' in self.model_history.history:
                accuracy  = float( "{:.4f}" . format( self.model_history.history['accuracy' ][epoch] ) )
            elif 'acc' in self.model.history.history:
                accuracy  = float( "{:.4f}" . format( self.model_history.history['acc' ][epoch] ) )

            loss      = float( "{:.6f}" . format( self.model_history.history['loss'     ][epoch] ) )
            precision = float( "{:.4f}" . format( self.model_history.history['Precision'][epoch] ) )
            recall    = float( "{:.4f}" . format( self.model_history.history['Recall'   ][epoch] ) )
            f1_score  = float( "{:.4f}" . format( self.model_history.history['F1_Score' ][epoch] ) )

            self.Print_Log( "\tEpoch: " + str( epoch ) + "\tLoss: " + str( loss ) + "\tAccuracy: " + str( accuracy )
                            + "\tPrecision: " + str( precision ) + "\tRecall: " + str( recall ) + "\tF1-Score: " + str( f1_score ) )

    """
        Adds Custom Keras Callbacks
    """
    def Add_Enabled_Model_Callbacks( self ):
        # Add TensorBoard Callback
        if self.enable_tensorboard_logs:
            tboard_log_dir = os.path.normpath( "./logs" )
            self.callback_list.append( keras.callbacks.TensorBoard( log_dir = tboard_log_dir ) )

        # Add Early Stopping Callback
        if self.enable_early_stopping:
            self.callback_list.append( keras.callbacks.EarlyStopping( monitor = self.early_stopping_metric_monitor, patience = self.early_stopping_persistence ) )

        # Setup Cosine Annealing Learning Rate Scheduler Callback
        if self.use_cosine_annealing:
            self.Print_Log( "BaseModel::Fit() - Adding Cosine Annealing Scheduler To Callback List" )
            self.callback_list.append( Cosine_Annealing_Scheduler( lr = self.learning_rate, T_max = self.epochs,
                                                                   eta_max = self.cosine_annealing_max,
                                                                   eta_min = self.cosine_annealing_min ) )

        # Setup Saving The Model After Each Epoch
        if self.per_epoch_saving:
            self.Print_Log( "BaseModel::Fit() - Adding Per Epoch Model Saving To Callback List" )
            self.callback_list.append( Model_Saving_Callback() )

    """
        Trains Model Using Training Data, Fits Model To Data
    """
    def Fit( self ):
        self.Print_Log( "BaseModel::Fit() - Error: Function Not Implemented / Calling Parent Function", force_print = True )
        raise NotImplementedError

    """
        Outputs Model's Prediction Vector Given Inputs
    """
    def Predict( self ):
        self.Print_Log( "BaseModel::Predict() - Error: Function Not Implemented / Calling Parent Function", force_print = True )
        raise NotImplementedError

    """
        Evaluates Model's Ability To Predict Evaluation Data
    """
    def Evaluate( self ):
        self.Print_Log( "BaseModel::Evaluate() - Error: Function Not Implemented / Calling Parent Function", force_print = True )
        raise NotImplementedError

    ############################################################################################
    #                                                                                          #
    #    Keras Model(s)                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Build The Keras Model
    """
    def Build_Model( self ):
        self.Print_Log( "BaseModel::Build_Model() - Error: Function Not Implemented / Calling Parent Function", force_print = True )
        raise NotImplementedError

    ############################################################################################
    #                                                                                          #
    #                                                                                          #
    #                                                                                          #
    ############################################################################################

    """
        Specifies Many Options Among Final Layers: Dense, CosFace, ArcFace or SphereFace

        Note: Requires Previous Dense Layer To Have An Included Regularizer For Weight Regularization
    """
    def Multi_Option_Final_Layer( self, number_of_outputs = None, cosface_input_layer = None, dense_input_layer = None ):
        output_layer = None

        if self.final_layer_type == "arcface":
            output_layer = ArcFace( number_of_outputs, margin = self.margin, scale = self.scale, activation = "multi_label", regularizer = regularizers.l2( self.weight_decay ) )( [dense_input_layer, cosface_input_layer] )
        elif self.final_layer_type == "sphereface":
            output_layer = SphereFace( number_of_outputs, margin = self.margin, scale = self.scale, activation = "multi_label", regularizer = regularizers.l2( self.weight_decay ) )( [dense_input_layer, cosface_input_layer] )
        elif self.final_layer_type == "cosface":
            output_layer = CosFace( number_of_outputs, margin = self.margin, scale = self.scale, activation = "multi_label", regularizer = regularizers.l2( self.weight_decay ) )( [dense_input_layer, cosface_input_layer] )
        else:
            output_layer = Dense( units = number_of_outputs, activation = self.activation_function, name = 'Localist_Output_Representation', use_bias = True )( dense_input_layer )

        return output_layer


    ############################################################################################
    #                                                                                          #
    #    Keras Model Metrics                                                                   #
    #                                                                                          #
    #        Source: https://gist.github.com/arnaldog12/5f2728f229a8bd3b4673b72786913252       #
    #                                                                                          #
    ############################################################################################

    """
        F1-Score
    """
    def F1_Score( self, y_true, y_pred ):
        precision = self.Precision( y_true, y_pred )
        recall    = self.Recall( y_true, y_pred )
        return 2 * ( ( precision * recall ) / ( precision + recall + K.epsilon() ) )

    """
        Precision

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
    """
    def Precision( self, y_true, y_pred ):
        true_positives      = K.sum( K.round( K.clip( y_true * y_pred, 0, 1 ) ) )
        predicted_positives = K.sum( K.round( K.clip( y_pred, 0, 1 ) ) )
        precision           = true_positives / ( predicted_positives + K.epsilon() )
        return precision

    """
        Recall

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
    """
    def Recall( self, y_true, y_pred ):
        true_positives     = K.sum( K.round( K.clip( y_true * y_pred, 0, 1 ) ) )
        possible_positives = K.sum( K.round( K.clip( y_true, 0, 1 ) ) )
        recall             = true_positives / ( possible_positives + K.epsilon() )
        return recall

    """
        Matthews Correlation

        Source: https://stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras
    """
    def Matthews_Correlation( self, y_true, y_pred ):
        y_pred_pos = K.round( K.clip( y_pred, 0, 1 ) )
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round( K.clip( y_true, 0, 1 ) )
        y_neg = 1 - y_pos

        tp    = K.sum( y_pos * y_pred_pos )
        tn    = K.sum( y_neg * y_pred_neg )

        fp    = K.sum( y_neg * y_pred_pos )
        fn    = K.sum( y_pos * y_pred_neg )

        numerator   = ( tp * tn - fp * fn )
        denominator = K.sqrt( ( tp + fp ) * ( tp + fn ) * ( tn + fp ) * ( tn + fn ) )

        return numerator / ( denominator + K.epsilon() )

    """
        Categorical Crossentropy Loss: Log-Loss + Softmax

        Computes Categorical Loss Between Predictions And Ground-Truth Labels.
        (Used For Debugging Purposes - Do Not Use For Model Training)
    """
    def Categorical_Crossentropy_Loss( self, y_true, y_pred ):
        y_true = K.print_tensor( K.max( y_true, axis = 1 ) )
        y_pred = K.print_tensor( K.max( y_pred, axis = 1 ) )
        return K.categorical_crossentropy( y_true, y_pred )


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Checks If Checkpoint Directory Exists And Creates It If Not Existing
    """
    def Create_Checkpoint_Directory( self ):
        self.Print_Log( "BaseModel::Create_Checkpoint_Directory() - Checking If Model Save Directory Exists: \"" + str( self.checkpoint_directory ) + "\"", force_print = True )

        if self.utils.Check_If_Path_Exists( self.checkpoint_directory ) == False:
            self.Print_Log( "BaseModel::Create_Checkpoint_Directory() - Creating Directory", force_print = True )
            os.mkdir( self.checkpoint_directory )
        else:
            self.Print_Log( "BaseModel::Init() - Directory Already Exists", force_print = True )

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

    def Get_Version( self ):                        return self.version

    def Get_Network_Model( self ):                  return self.network_model

    def Get_Number_Of_Hidden_Dimensions( self ):    return self.number_of_hidden_dimensions

    def Get_Number_Of_Trained_Instances( self ):    return self.trained_instances

    def Get_Number_Of_Evaluated_Instances( self ):  return self.evaluated_instances

    def Get_Epochs( self):                          return self.epochs

    def Get_Verbose( self ):                        return self.verbose

    def Get_Debug_Log( self ):                      return self.debug_log

    def Get_Dropout( self ):                        return self.dropout

    def Get_Write_Log( self ):                      return self.write_log

    def Get_Batch_Size( self ):                     return self.batch_size

    def Get_Shuffle( self ):                        return self.shuffle

    def Get_Momentum( self ):                       return self.momentum

    def Get_Loss_Function( self ):                  return self.loss_function

    def Get_Optimizer( self ):                      return self.optimizer

    def Get_Activation_Function( self ):            return self.activation_function

    def Get_Learning_Rate( self ):                  return self.learning_rate

    def Get_Learning_Rate_Decay( self ):            return self.learning_rate_decay

    def Get_Weight_Decay( self ):                   return self.weight_decay

    def Get_Feature_Scaling_Value( self ):          return self.feature_scale_value

    def Get_Model_Type( self ):                     return self.model_type

    def Get_Prediction_Threshold( self ):           return self.prediction_threshold

    def Get_Trainable_Weights( self ):              return self.trainable_weights

    def Get_Embeddings_Loaded( self ):              return self.embeddings_loaded

    def Get_Embedding_Path( self ):                 return self.embedding_path

    def Get_Embedding_Modification( self ):         return self.embedding_modification

    def Get_Final_Layer_Type( self ):               return self.final_layer_type

    def Get_Scale( self ):                          return self.scale

    def Get_Margin( self ):                         return self.margin

    def Get_Per_Epoch_Saving( self ):               return self.per_epoch_saving

    def Get_Model_History( self ):                  return self.model_history

    def Get_Number_Of_Embedding_Dimensions( self ): return self.number_of_embedding_dimensions

    def Get_BiLSTM_Dimension_Size( self ):          return self.bilstm_dimension_size

    def Get_BiLSTM_Merge_Mode( self ):              return self.bilstm_merge_mode

    def Get_Use_Batch_Normalization( self ):        return self.use_batch_normalization

    def Get_Number_Of_Features( self ):             return self.number_of_features

    def Get_Number_Of_Primary_Inputs( self ):       return self.number_of_primary_inputs

    def Get_Number_Of_Secondary_Inputs( self ):     return self.number_of_secondary_inputs

    def Get_Number_Of_Tertiary_Inputs( self ):      return self.number_of_tertiary_inputs

    def Get_Number_Of_Outputs( self ):              return self.number_of_outputs

    def Get_Debug_Log_File_Handle( self ):          return self.debug_log_file_handle

    def Get_Debug_Log_File_Name( self ):            return self.debug_log_file_name

    def Get_Early_Stopping_Metric_Monitor( self ):  return self.early_stopping_metric_monitor

    def Get_Enable_Early_Stopping( self ):          return self.enable_early_stopping

    def Get_Enable_Tensorboard_Logs( self ):        return self.enable_tensorboard_logs

    def Get_Early_Stopping_Persistence( self ):     return self.early_stopping_persistence

    def Get_Use_Cosine_Annealing( self ):           return self.use_cosine_annealing

    def Get_Cosine_Annealing_Min( self ):           return self.cosine_annealing_min

    def Get_Cosine_Annealing_Max( self ):           return self.cosine_annealing_max

    def Get_Use_CSR_Format( self ):                 return self.use_csr_format

    def Get_Use_GPU( self ):                        return self.use_gpu

    def Get_Device_Name( self ):                    return self.device_name

    def Get_Checkpoint_Directory( self ):           return self.checkpoint_directory

    def Get_Callback_List( self ):                  return self.callback_list

    def Get_Final_Layer_Type_List( self ):          return self.final_layer_type_list

    def Get_Utils( self ):                          return self.utils


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################

    def Set_Number_Of_Trained_Instances( self, value ):        self.trained_instances = value

    def Set_Number_Of_Evaluated_Instances( self, value ):      self.evaluated_instances = value

    def Set_Epochs( self, value ):                             self.epochs = value

    def Set_Verbose( self, value ):                            self.verbose = value

    def Set_Debug_Log( self, value ):                          self.debug_log = value

    def Set_Dropout( self, value ):                            self.dropout = value

    def Set_Batch_Size( self, value):                          self.batch_size = value

    def Set_Shuffle( self, value ):                            self.shuffle = value

    def Set_Momentum( self, value ):                           self.momentum = value

    def Set_Loss_Function( self, value ):                      self.loss_function = value

    def Set_Optimizer( self, value ):                          self.optimizer = value

    def Set_Activation_Function( self, value ):                self.activation_function = value

    def Set_Learning_Rate( self, value ):                      self.learning_rate = value

    def Set_Learning_Rate_Decay( self, value ):                self.learning_rate_decay = value

    def Set_Weight_Decay( self, value ):                       self.weight_decay = value

    def Set_Feature_Scaling_Value( self, value ):              self.feature_scale_value = value

    def Set_Prediction_Threshold( self, value ):               self.prediction_threshold = value

    def Set_Trainable_Weights( self, value ):                  self.trainable_weights = value

    def Set_Embeddings_Loaded( self, value ):                  self.embeddings_loaded = value

    def Set_Embedding_Path( self, value ):                     self.embedding_path = value

    def Set_Embedding_Modification( self, value ):             self.embedding_modification = value

    def Set_Final_Layer_Type( self, value ):                   self.final_layer_type = value

    def Set_Scale( self, value ):                              self.scale = value

    def Set_Margin( self, value ):                             self.margin = value

    def Set_Per_Epoch_Saving( self, value ):                   self.per_epoch_saving = value

    def Set_Number_Of_Hidden_Dimensions( self, value ):        self.number_of_hidden_dimensions = value

    def Set_BiLSTM_Dimension_Size( self, value ):              self.bilstm_dimension_size = value

    def Set_BiLSTM_Merge_Mode( self, value ):                  self.bilstm_merge_mode = value

    def Set_Use_Batch_Normalization( self, value ):            self.use_batch_normalization = value

    def Set_Number_Of_Features( self, value ):                 self.number_of_features = value

    def Set_Number_Of_Primary_Inputs( self, value ):           self.number_of_primary_inputs = value

    def Set_Number_Of_Secondary_Inputs( self, value ):         self.number_of_secondary_inputs = value

    def Set_Number_Of_Tertiary_Inputs( self, value ):          self.number_of_tertiary_inputs = value

    def Set_Number_Of_Outputs( self, value ):                  self.number_of_outputs = value

    def Set_Debug_Log_File_Handle( self, file_handle ):        self.debug_log_file_handle = file_handle

    def Set_Debug_Log_File_Name( self, value ):                self.debug_log_file_name = value

    def Set_Enable_Early_Stopping( self, value ):              self.enable_early_stopping = value

    def Set_Early_Stopping_Metric_Monitor( self, value ):      self.early_stopping_metric_monitor = value

    def Set_Early_Stopping_Persistence( self, value ):         self.early_stopping_persistence = value

    def Set_Use_Cosine_Annealing( self, value ):               self.use_cosine_annealing = value

    def Set_Cosine_Annealing_Min( self, value ):               self.cosine_annealing_min = value

    def Set_Cosine_Annealing_Max( self, value ):               self.cosine_annealing_max = value

    def Set_Enable_Tensorboard_Logs( self, value ):            self.enable_tensorboard_logs = value

    def Set_Use_CSR_Format( self, value ):                     self.use_csr_format = value

    def Set_Use_GPU( self, value ):                            self.use_gpu = value

    def Set_Device_Name( self, value ):                        self.device_name = value

    def Set_Checkpoint_Directory( self, value ):               self.checkpoint_directory = value

    def Set_Callback_List( self, value ):                      self.callback_list = value


############################################################################################
#                                                                                          #
#    ArcFace Class                                                                         #
#                                                                                          #
#        Modified From Source:  https://github.com/4uiiurz1/keras-arcface                  #
#                                                                                          #
############################################################################################

class ArcFace( Layer ):
    def __init__( self, n_classes = 10, scale = 30.0, margin = 0.50, regularizer = None, activation = "softmax", **kwargs ):
        super( ArcFace, self ).__init__( **kwargs )
        self.n_classes   = n_classes
        self.scale       = scale
        self.margin      = margin
        self.activation  = activation
        self.regularizer = regularizers.get( regularizer )

    def build( self, input_shape ):
        super( ArcFace, self ).build( input_shape[0] )
        self.W = self.add_weight( name        = 'ArcFace_Weights',
                                  shape       = ( input_shape[0][-1], self.n_classes ),
                                  initializer = 'glorot_uniform',
                                  trainable   = True,
                                  regularizer = self.regularizer )

    def call( self, inputs ):
        x, y = inputs

        # Used For Debugging Purposes - Prints Ground Truth Labels Per Class
        # y = tf.Print( y, [tf.round(y)], summarize = -1 )

        c = K.shape( x )[-1]
        # normalize feature
        x = tf.nn.l2_normalize( x, axis = 1 )
        # normalize weights
        W = tf.nn.l2_normalize( self.W, axis = 0 )
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos( K.clip( logits, -1.0 + K.epsilon(), 1.0 - K.epsilon() ) )
        target_logits = tf.cos( theta + self.margin )
        # sin = tf.sqrt( 1 - logits ** 2 )
        # cos_m = tf.cos( logits )
        # sin_m = tf.sin( logits )
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * ( 1 - y ) + target_logits * y
        # feature re-scale
        logits *= self.scale

        predictions = None

        # Pass Logits Through Softmax Function
        #   ie. Generate Normalized Distribution Over All Logits Between 0.0 and 1.0
        if self.activation == "softmax":
            predictions = tf.nn.softmax( logits )
        # Pass Each Logit Element Through The Sigmoid Function To Generate A Multi-Class/Multi-Label Distribution Per Element
        else:
            predictions = 1/( 1 + tf.math.exp( ( -logits ) ) )

        # Used For Debugging Purposes - Prints Predicted Labels Per Class
        # predictions = tf.Print( predictions, [tf.round(predictions)], summarize = -1 )

        return predictions

    def compute_output_shape( self, input_shape ):
        return ( None, self.n_classes )


############################################################################################
#                                                                                          #
#    CosFace Class                                                                         #
#                                                                                          #
#        Modified From Source:  https://github.com/4uiiurz1/keras-arcface                  #
#                                                                                          #
############################################################################################

class CosFace( Layer ):
    def __init__( self, n_classes = 10, scale = 30.0, margin = 0.35, regularizer = None, activation = "softmax", **kwargs ):
        super( CosFace, self ).__init__( **kwargs )
        self.n_classes   = n_classes
        self.scale       = scale
        self.margin      = margin
        self.activation  = activation
        self.regularizer = regularizers.get( regularizer )

    def build( self, input_shape ):
        super( CosFace, self ).build( input_shape[0] )
        self.W = self.add_weight( name        = 'CosFace_Weights',
                                  shape       = ( input_shape[0][-1], self.n_classes ),
                                  initializer = 'glorot_uniform',
                                  trainable   = True,
                                  regularizer = self.regularizer )

    def call( self, inputs ):
        x, y = inputs

        # Used For Debugging Purposes - Prints Ground Truth Labels Per Class
        # y = tf.Print( y, [tf.round(y)], summarize = -1 )

        c = K.shape( x )[-1]
        # normalize feature
        x = tf.nn.l2_normalize( x, axis = 1 )
        # normalize weights
        W = tf.nn.l2_normalize( self.W, axis = 0 )
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.margin
        #
        logits = logits * ( 1 - y ) + target_logits * y
        # feature re-scale
        logits *= self.scale

        predictions = None

        # Pass Logits Through Softmax Function
        #   ie. Generate Normalized Distribution Over All Logits Between 0.0 and 1.0
        if self.activation == "softmax":
            predictions = tf.nn.softmax( logits )
        # Pass Each Logit Element Through The Sigmoid Function To Generate A Multi-Class/Multi-Label Distribution Per Element
        else:
            predictions = 1/( 1 + tf.math.exp( ( -logits ) ) )

        # Used For Debugging Purposes - Prints Predicted Labels Per Class
        # predictions = tf.Print( predictions, [tf.round(predictions)], summarize = -1 )

        return predictions

    def compute_output_shape( self, input_shape ):
        return ( None, self.n_classes )


############################################################################################
#                                                                                          #
#    SphereFace Class                                                                      #
#                                                                                          #
#        Modified From Source:  https://github.com/4uiiurz1/keras-arcface                  #
#                                                                                          #
############################################################################################

class SphereFace( Layer ):
    def __init__( self, n_classes = 10, scale = 30.0, margin = 1.35, regularizer = None, activation = "softmax", **kwargs ):
        super( SphereFace, self ).__init__( **kwargs )
        self.n_classes   = n_classes
        self.scale       = scale
        self.margin      = margin
        self.activation  = activation
        self.regularizer = regularizers.get( regularizer )

    def build( self, input_shape ):
        super( SphereFace, self ).build( input_shape[0] )
        self.W = self.add_weight( name        = 'SphereFace_Weights',
                                  shape       = ( input_shape[0][-1], self.n_classes ),
                                  initializer = 'glorot_uniform',
                                  trainable   = True,
                                  regularizer = self.regularizer )

    def call( self, inputs ):
        x, y = inputs

        # Used For Debugging Purposes - Prints Ground Truth Labels Per Class
        # y = tf.Print( y, [tf.round(y)], summarize = -1 )

        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize( x, axis = 1 )
        # normalize weights
        W = tf.nn.l2_normalize( self.W, axis = 0 )
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos( K.clip( logits, -1.0 + K.epsilon(), 1.0 - K.epsilon() ) )
        target_logits = tf.cos( self.margin * theta )
        #
        logits = logits * ( 1 - y ) + target_logits * y
        # feature re-scale
        logits *= self.scale

        predictions = None

        # Pass Logits Through Softmax Function
        #   ie. Generate Normalized Distribution Over All Logits Between 0.0 and 1.0
        if self.activation == "softmax":
            predictions = tf.nn.softmax( logits )
        # Pass Each Logit Element Through The Sigmoid Function To Generate A Multi-Class/Multi-Label Distribution Per Element
        else:
            predictions = 1/( 1 + tf.math.exp( ( -logits ) ) )

        # Used For Debugging Purposes - Prints Predicted Labels Per Class
        # predictions = tf.Print( predictions, [tf.round(predictions)], summarize = -1 )

        return predictions

    def compute_output_shape( self, input_shape ):
        return ( None, self.n_classes )


class ArcFaceAlt( Layer ):
    '''Custom Keras layer implementing ArcFace including:
    1. Generation of embeddings
    2. Loss function
    3. Accuracy function
    '''

    def __init__(self, output_dim, class_num, margin=0.5, scale=64., **kwargs):
        self.output_dim = output_dim
        self.class_num = class_num
        self.margin = margin
        self.s = scale

        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.mm = self.sin_m * margin
        self.threshold = tf.math.cos(tf.constant(m.pi) - margin)
        super(ArcFaceAlt, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.class_num),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(ArcFaceAlt, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        embeddings = tf.nn.l2_normalize(x, axis=1, name='normed_embeddings')
        weights = tf.nn.l2_normalize(self.kernel, axis=0, name='normed_weights')
        cos_t = tf.matmul(embeddings, weights, name='cos_t')
        return cos_t


    def get_logits(self, labels, y_pred):
        cos_t = y_pred
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')
        cond_v = cos_t - self.threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = self.s*(cos_t - self.mm)
        cos_mt_temp = tf.compat.v1.where(cond, cos_mt, keep_val)
        mask = tf.one_hot(labels, depth=self.class_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')
        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
        return output


    def loss(self, y_true, y_pred):
        labels = K.argmax(y_true, axis=-1)
        logits = self.get_logits(labels, y_pred)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return loss


    def accuracy(self, y_true, y_pred):
        labels = K.argmax(y_true, axis=-1)
        logits = self.get_logits(labels, y_pred)
        accuracy = categorical_accuracy(y_true=labels, y_pred=logits)
        return accuracy


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Inherited From A Child Class ****" )
    exit()
