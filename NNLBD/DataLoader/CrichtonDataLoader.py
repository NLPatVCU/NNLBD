#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    05/05/2020                                                                   #
#    Revised: 12/31/2022                                                                   #
#                                                                                          #
#    Crichton Data Loader Class For The NNLBD Package.                                     #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import re, scipy, threading
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

# Custom Modules
from NNLBD.DataLoader import DataLoader


############################################################################################
#                                                                                          #
#    Data Loader Model Class                                                               #
#                                                                                          #
############################################################################################

class CrichtonDataLoader( DataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False,
                  debug_log_file_handle = None, restrict_output = False, output_is_embeddings = False ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, shuffle = shuffle,
                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, debug_log_file_handle = debug_log_file_handle,
                          restrict_output = restrict_output, output_is_embeddings = output_is_embeddings )
        self.version = 0.07

    """
        Performs Checks Against The Specified Data File/Data List To Ensure File Integrity Before Further Processing
            (Only Checks First 10 Lines In Data File)
    """
    def Check_Data_File_Format( self, file_path = "", data_list = [], str_delimiter = '\t' ):
        read_file_handle  = None
        error_flag        = False
        number_of_indices = 10

        # Read Data From File / Favors 'data_list' Versus File Path
        if file_path == "" and len( data_list ) == 0:
            self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Using Internal Data File Stored In Memory" )
            data_list = self.Get_Data()
        elif file_path != "" and len( data_list ) == 0:
            self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Opening Data File From Path" )

            try:
                read_file_handle = open( file_path, "r" )
            except FileNotFoundError:
                self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Error: Unable To Open Data File \"" + str( file_path ) + "\"", force_print = True )

        # Check Data List Again
        if file_path == "" and len( data_list ) == 0:
            self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Warning: No Data In Data List" )
            return True

        if len( data_list ) > 0 and number_of_indices > len( data_list ): number_of_indices = len( data_list )

        self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Verifying File Integrity" )

        # Check First 10 Lines In Data File
        for index in range( number_of_indices ):
            line = read_file_handle.readline() if read_file_handle else data_list[index]
            if not line: break

            line_elements = line.split( str_delimiter )
            if len( line_elements ) != 4: error_flag = True
            if error_flag: break

        # Close File Handle
        if read_file_handle: read_file_handle.close()

        # Notify User Of Data Integrity Error Found
        if error_flag:
            self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Data Integrity Error", force_print = True )
            self.Print_Log( "                                             - Expected Data In The Following Format: 'concept_1\\tconcept_2\\tconcept_3\\tscore'", force_print = True )

        self.Print_Log( "CrichtonDataLoader::Check_Data_File_Format() - Complete" )

        return False if error_flag else True

    """
        Vectorized/Binarized Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list              : String (ie. C002  )
            model_type             : Model Type (String)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            stack_inputs           : True = Stacks Inputs, False = Does Not Stack Inputs - Used For BiLSTM Model (Boolean)
            keep_in_memory         : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            number_of_threads      : Number Of Threads To Deploy For Data Vectorization (Integer)
            str_delimiter          : String Delimiter - Used To Separate Elements Within An Instance (String)

        Outputs:
            primary_input_vector   : CSR Matrix or Numpy Array
            secondary_input_vector : CSR Matrix or Numpy Array
            tertiary_input_vector  : CSR Matrix or Numpy Array
            output_vector          : CSR Matrix or Numpy Array
    """
    def Encode_Model_Data( self, data_list = [], model_type = "open_discovery", use_csr_format = False, pad_inputs = True, pad_output = True,
                           stack_inputs = False, keep_in_memory = True, number_of_threads = 4, str_delimiter = '\t' ):
        # Check(s)
        if model_type == "open_discovery":
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: 'Crichton Data Format' Is Not Currently Supported For 'Open Discovery'", force_print = True )
            return None, None, None, None

        if len( data_list ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None, None, None

        if number_of_threads < 1:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: Number Of Threads < 1 / Setting Number Of Threads = 1", force_print = True )
            number_of_threads = 1

        if self.Check_Data_File_Format() == False:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Data Integrity Violation Found", force_print = True )
            return None, None, None, None

        threads          = []
        primary_inputs   = []
        secondary_inputs = []
        tertiary_inputs  = []
        outputs          = []

        if self.Is_Embeddings_Loaded():
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Detected Loaded Embeddings / Setting 'pad_inputs' = False" )
            pad_inputs = False

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Vectorizing Data Using Settings" )
        self.Print_Log( "                                        - Model Type        : " + str( model_type         ) )
        self.Print_Log( "                                        - Use CSR Format    : " + str( use_csr_format     ) )
        self.Print_Log( "                                        - Pad Inputs        : " + str( pad_inputs         ) )
        self.Print_Log( "                                        - Pad Output        : " + str( pad_output         ) )
        self.Print_Log( "                                        - Stack Inputs      : " + str( stack_inputs       ) )

        if pad_inputs:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: 'Pad Inputs == True' / Setting 'pad_inputs = False", force_print = True )
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: 'Pad Output == True' / Setting 'pad_output = False", force_print = True )
            pad_inputs = False
            pad_output = False

        total_number_of_lines = len( data_list )

        if number_of_threads > total_number_of_lines:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: 'number_of_threads > len( data_list )' / Setting 'number_of_threads = total_number_of_lines'" )
            number_of_threads = total_number_of_lines

        lines_per_thread = int( ( total_number_of_lines + number_of_threads - 1 ) / number_of_threads )

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Number Of Threads: " + str( number_of_threads ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Lines Per Thread : " + str( lines_per_thread  ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Total Lines In File Data: " + str( total_number_of_lines ) )

        ###########################################
        #          Start Worker Threads           #
        ###########################################

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Starting Worker Threads" )

        # Create Storage Locations For Threaded Data Segments
        tmp_thread_data = [None for i in range( number_of_threads )]

        for thread_id in range( number_of_threads ):
            starting_line_index = lines_per_thread * thread_id
            ending_line_index   = starting_line_index + lines_per_thread if starting_line_index + lines_per_thread < total_number_of_lines else total_number_of_lines

            new_thread = threading.Thread( target = self.Worker_Thread_Function, args = ( thread_id, data_list[starting_line_index:ending_line_index], tmp_thread_data,
                                                                                          model_type, use_csr_format, pad_inputs, pad_output, stack_inputs, str_delimiter, ) )
            new_thread.start()
            threads.append( new_thread )

        ###########################################
        #           Join Worker Threads           #
        ###########################################

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Waiting For Worker Threads To Finish" )

        for thread in threads:
            thread.join()

        # Convert To CSR Matrix Format
        if use_csr_format:
            primary_inputs   = csr_matrix( primary_inputs   ) if stack_inputs == False else []
            secondary_inputs = csr_matrix( secondary_inputs ) if stack_inputs == False else []
            tertiary_inputs  = csr_matrix( tertiary_inputs  ) if stack_inputs == False else []
            outputs          = csr_matrix( outputs          )

        if len( tmp_thread_data ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error Vectorizing Model Data / No Data Returned From Worker Threads", force_print = True )
            return None, None, None, None

        # Concatenate Vectorized Model Data Segments From Threads
        for model_data in tmp_thread_data:
            if model_data is None or len( model_data ) < 4:
                self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Expected At Least Four Vectorized Elements / Received None Or < 4", force_print = True )
                continue

            # Vectorized Inputs/Outputs
            primary_input_array   = model_data[0]
            secondary_input_array = model_data[1]
            tertiary_input_array  = model_data[2]
            output_array          = model_data[3]

            ###############################################
            # Convert Primary Inputs To CSR Matrix Format #
            ###############################################
            if use_csr_format and stack_inputs == False:
                # CSR Matrices Are Empty, Overwrite Them With The Appropriate Data With The Correct Shape
                if primary_inputs.shape[1] == 0:
                    primary_inputs = primary_input_array
                # Stack Existing CSR Matrices With New Data By Row
                else:
                    # In-Place Update (Should Be Faster Than The New Copy Replacement)
                    primary_inputs.data    = np.hstack( ( primary_inputs.data, primary_input_array.data ) )
                    primary_inputs.indices = np.hstack( ( primary_inputs.indices, primary_input_array.indices ) )
                    primary_inputs.indptr  = np.hstack( ( primary_inputs.indptr, ( primary_input_array.indptr + primary_inputs.nnz )[1:] ) )
                    primary_inputs._shape  = ( primary_inputs.shape[0] + primary_input_array.shape[0], primary_input_array.shape[1] )
            else:
                for i in range( len( primary_input_array ) ):
                    primary_inputs.append( primary_input_array[i] )

            #################################################
            # Convert Secondary Inputs To CSR Matrix Format #
            #################################################
            if use_csr_format and stack_inputs == False:
                # CSR Matrices Are Empty, Overwrite Them With The Appropriate Data With The Correct Shape
                if secondary_inputs.shape[1] == 0:
                    secondary_inputs = secondary_input_array
                # Stack Existing CSR Matrices With New Data By Row
                else:
                    # In-Place Update (Should Be Faster Than The New Copy Replacement)
                    secondary_inputs.data    = np.hstack( ( secondary_inputs.data, secondary_input_array.data ) )
                    secondary_inputs.indices = np.hstack( ( secondary_inputs.indices, secondary_input_array.indices ) )
                    secondary_inputs.indptr  = np.hstack( ( secondary_inputs.indptr, ( secondary_input_array.indptr + secondary_inputs.nnz )[1:] ) )
                    secondary_inputs._shape  = ( secondary_inputs.shape[0] + secondary_input_array.shape[0], secondary_input_array.shape[1] )
            else:
                for i in range( len( primary_input_array ) ):
                    secondary_inputs.append( secondary_input_array[i] )

            ################################################
            # Convert Tertiary Inputs To CSR Matrix Format #
            ################################################
            if use_csr_format and stack_inputs == False:
                # CSR Matrices Are Empty, Overwrite Them With The Appropriate Data With The Correct Shape
                if tertiary_inputs.shape[1] == 0:
                    tertiary_inputs = tertiary_input_array
                # Stack Existing CSR Matrices With New Data By Row
                else:
                    # In-Place Update (Should Be Faster Than The New Copy Replacement)
                    tertiary_inputs.data    = np.hstack( ( tertiary_inputs.data, tertiary_input_array.data ) )
                    tertiary_inputs.indices = np.hstack( ( tertiary_inputs.indices, tertiary_input_array.indices ) )
                    tertiary_inputs.indptr  = np.hstack( ( tertiary_inputs.indptr, ( tertiary_input_array.indptr + tertiary_inputs.nnz )[1:] ) )
                    tertiary_inputs._shape  = ( tertiary_inputs.shape[0] + tertiary_input_array.shape[0], tertiary_input_array.shape[1] )
            else:
                for i in range( len( tertiary_input_array ) ):
                    tertiary_inputs.append( tertiary_input_array[i] )

            ########################################
            # Convert Outputs To CSR Matrix Format #
            ########################################
            if use_csr_format:
                # CSR Matrices Are Empty, Overwrite Them With The Appropriate Data With The Correct Shape
                if outputs.shape[1] == 0:
                    outputs = output_array
                # Stack Existing CSR Matrices With New Data By Row
                else:
                    # In-Place Update (Should Be Faster Than The New Copy Replacement)
                    outputs.data    = np.hstack( ( outputs.data, output_array.data ) )
                    outputs.indices = np.hstack( ( outputs.indices, output_array.indices ) )
                    outputs.indptr  = np.hstack( ( outputs.indptr, ( output_array.indptr + outputs.nnz )[1:] ) )
                    outputs._shape  = ( outputs.shape[0] + output_array.shape[0], output_array.shape[1] )
            else:
                for i in range( len( output_array ) ):
                    outputs.append( output_array[i] )

        if use_csr_format == False:
            primary_inputs   = np.asarray( primary_inputs   )
            secondary_inputs = np.asarray( secondary_inputs )
            tertiary_inputs  = np.asarray( tertiary_inputs  )
            outputs          = np.asarray( outputs          )

        if isinstance( primary_inputs, list ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Primary Input Length  : " + str( len( primary_inputs ) ) )
        elif isinstance( primary_inputs, csr_matrix ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Primary Input Length  : " + str( primary_inputs.shape  ) )

        if isinstance( secondary_inputs, list ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Secondary Input Length: " + str( len( secondary_inputs ) ) )
        elif isinstance( secondary_inputs, csr_matrix ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Secondary Input Length: " + str( secondary_inputs.shape  ) )

        if isinstance( tertiary_inputs, list ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Tertiary Input Length : " + str( len( tertiary_inputs ) ) )
        elif isinstance( tertiary_inputs, csr_matrix ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Tertiary Input Length : " + str( tertiary_inputs.shape  ) )

        if isinstance( outputs, list ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Output Length         : " + str( len( outputs.shape ) ) )
        elif isinstance( outputs, csr_matrix ):
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Output Length         : " + str( outputs.shape        ) )


        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Vectorized Primary Inputs  :\n" + str( primary_inputs   ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Vectorized Secondary Inputs:\n" + str( secondary_inputs ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Vectorized Tertiary Inputs :\n" + str( tertiary_inputs  ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Vectorized Outputs         :\n" + str( outputs          ) )


        # Clean-Up
        threads         = []
        tmp_thread_data = []

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Complete" )

        #####################
        # List Final Checks #
        #####################
        if isinstance( primary_inputs, list ) and len( primary_inputs ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Primary Input Matrix Is Empty" )
            return None, None, None, None

        if isinstance( secondary_inputs, list ) and len( secondary_inputs ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Secondary Input Matrix Is Empty" )
            return None, None, None, None

        # Only Crichton Data-sets Use A Tertiary Input, So This Matrix Is Not Guaranteed To Be Non-Empty
        if isinstance( tertiary_inputs, list ) and len( tertiary_inputs ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: Tertiary Input Matrix Is Empty" )

        if isinstance( outputs, list ) and len( outputs ) == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None, None, None

        ###########################
        # CSR Matrix Final Checks #
        ###########################
        if isinstance( primary_inputs, csr_matrix ) and primary_inputs.nnz == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Primary Input Matrix Is Empty" )
            return None, None, None, None

        if isinstance( secondary_inputs, csr_matrix ) and secondary_inputs.nnz == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Error: Secondary Input Matrix Is Empty" )
            return None, None, None, None

        # Only Crichton Data-sets Use A Tertiary Input, So This Matrix Is Not Guaranteed To Be Non-Empty
        if isinstance( tertiary_inputs, csr_matrix ) and tertiary_inputs.nnz == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: Tertiary Input Matrix Is Empty" )

        if isinstance( outputs, csr_matrix ) and outputs.nnz == 0:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None, None, None

        if keep_in_memory:
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Data() - Storing In Memory" )
            self.primary_inputs   = primary_inputs
            self.secondary_inputs = secondary_inputs
            self.tertiary_inputs  = tertiary_inputs
            self.outputs          = outputs

        return primary_inputs, secondary_inputs, tertiary_inputs, outputs

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            primary_input          : List Of Strings (ie. C002  )
            secondary_input        : List Of Strings (ie. TREATS)
            tertiary_input         : List Of Strings
            outputs                : List Of Strings (May Be A List Of Lists Tokens. ie. [C001 C003 C004], [C002 C001 C010])
            model_type             : Model Type (String)
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            separate_outputs       : Separates Outputs By Into Individual Vectorized Instances = True, Combine Outputs Per Instance = False. (Only Valid For 'Open Discovery')
            instance_separator     : String Delimiter - Used To Separate Instances (String)

        Outputs:
            primary_input_vector   : Numpy Array
            secondary_input_vector : Numpy Array
            tertiary_input_vector  : Numpy Array
            output_vector          : Numpy Array
    """
    def Encode_Model_Instance( self, primary_input, secondary_input, tertiary_input = "", outputs = "", model_type = "open_discovery",
                               pad_inputs = False, pad_output = False, separate_outputs = False, instance_separator = '<:>' ):
        # Convert Inputs/Outputs To Lowercase
        primary_input   = primary_input.lower()
        secondary_input = secondary_input.lower()
        tertiary_input  = tertiary_input.lower()
        outputs         = outputs.lower()

        # Split Elements Using String Delimiter
        primary_input_instances   = primary_input.split( instance_separator )
        secondary_input_instances = secondary_input.split( instance_separator )
        tertiary_input_instances  = tertiary_input.split( instance_separator )
        output_instances          = outputs.split( instance_separator )

        if len( tertiary_input_instances ) < len( primary_input_instances ): tertiary_input_instances = ["" for i in range( len( primary_input_instances ) )]
        if len( output_instances         ) < len( primary_input_instances ): output_instances         = ["" for i in range( len( primary_input_instances ) )]

        # Placeholders For Vectorized Inputs/Outputs
        primary_input_array   = []
        secondary_input_array = []
        tertiary_input_array  = []
        output_array          = []

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorizing Inputs" )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() -             Primary Input  : " + str( primary_input   ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() -             Secondary Input: " + str( secondary_input ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() -             Tertiary Input : " + str( tertiary_input  ) )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() -             Outputs        : " + str( outputs         ) )

        for primary_input, secondary_input, tertiary_input, outputs in zip( primary_input_instances, secondary_input_instances, tertiary_input_instances, output_instances ):
            primary_input_id   = self.Get_Token_ID( primary_input   )
            secondary_input_id = self.Get_Token_ID( secondary_input )
            tertiary_input_id  = self.Get_Token_ID( tertiary_input  )

            # Check(s)
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Warning: Primary Input  : \"" + str( primary_input   ) + "\" Not In Token ID Dictionary" ) if primary_input_id   == -1 else None
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Warning: Secondary Input: \"" + str( secondary_input ) + "\" Not In Token ID Dictionary" ) if secondary_input_id == -1 else None
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Warning: Teritary Input : \"" + str( tertiary_input  ) + "\" Not In Token ID Dictionary" ) if tertiary_input_id  == -1 else None
            if primary_input_id == -1 or secondary_input_id == -1 or ( tertiary_input != "" and tertiary_input_id == -1 ):
                return [], [], [], []

            self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorizing Inputs \"" + str( primary_input ) + "\" & \"" + str( secondary_input ) + "\" & \"" + str( tertiary_input ) + "\"" )
            self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorizing Outputs \"" + str( outputs ) + "\"" )

            temp_primary_input_array   = []
            temp_secondary_input_array = []
            temp_tertiary_input_array  = []
            temp_output_array          = []

            ##################
            # Open Discovery #
            ##################
            if model_type == "open_discovery":
                #####################
                # Vectorize Outputs #
                #####################
                if outputs != "":
                    temp_output_array = np.array( [0], dtype = int ) if float( outputs ) == 0.0 else np.array( [1] , dtype = int )

                ###########################
                # Vectorize Primary Input #
                ###########################
                if separate_outputs:
                    for i in range( len( temp_output_array ) ):
                        if pad_inputs:
                            # One-Hot Encoding
                            temp_array = np.zeros( ( self.Get_Number_Of_Primary_Elements(), ), dtype = int )
                            temp_array[primary_input_id] = 1
                            temp_primary_input_array.append( temp_array )
                        else:
                            # Single Token ID Per Instance
                            temp_primary_input_array.append( [primary_input_id] )
                else:
                    if pad_inputs:
                        # One-Hot Encoding
                        temp_array = np.zeros( ( self.Get_Number_Of_Primary_Elements(), ), dtype = int )
                        temp_array[primary_input_id] = 1
                        temp_primary_input_array.append( temp_array )
                    else:
                        # Single Token ID Per Instance
                        temp_primary_input_array.append( [primary_input_id] )

                #############################
                # Vectorize Secondary Input #
                #############################
                if separate_outputs:
                    for i in range( len( temp_output_array ) ):
                        if pad_inputs:
                            # One-Hot Encoding
                            array_length = self.Get_Number_Of_Secondary_Elements() if self.Get_Is_CUI_Data() else self.Get_Number_Of_Primary_Elements()
                            temp_array = np.zeros( ( array_length, ), dtype = int )
                            temp_array[secondary_input_id] = 1
                            temp_secondary_input_array.append( temp_array )
                        else:
                            # Single Token ID Per Instance
                            temp_secondary_input_array.append( [secondary_input_id] )
                else:
                    if pad_inputs:
                        # One-Hot Encoding
                        array_length = self.Get_Number_Of_Secondary_Elements() if self.Get_Is_CUI_Data() else self.Get_Number_Of_Primary_Elements()
                        temp_array = np.zeros( ( array_length, ), dtype = int )
                        temp_array[secondary_input_id] = 1
                        temp_secondary_input_array.append( temp_array )
                    else:
                        # Single Token ID Per Instance
                        temp_secondary_input_array.append( [secondary_input_id] )

                ############################
                # Vectorize Tertiary Input #
                ############################
                if tertiary_input_id != -1:
                    if separate_outputs:
                        for i in range( len( temp_output_array ) ):
                            if pad_inputs:
                                # One-Hot Encoding
                                temp_array = np.zeros( ( self.Get_Number_Of_Tertiary_Elements(), ), dtype = int )
                                temp_array[tertiary_input_id] = 1
                                temp_tertiary_input_array.append( temp_array )
                            else:
                                # Single Token ID Per Instance
                                temp_tertiary_input_array.append( [tertiary_input_id] )
                    else:
                        if pad_inputs:
                            # One-Hot Encoding
                            temp_array = np.zeros( ( self.Get_Number_Of_Tertiary_Elements(), ), dtype = int )
                            temp_array[tertiary_input_id] = 1
                            temp_tertiary_input_array.append( temp_array )
                        else:
                            # Single Token ID Per Instance
                            temp_tertiary_input_array.append( [tertiary_input_id] )

            ####################
            # Closed Discovery #
            ####################
            else:
                #############################
                # Vectorize Secondary Input #
                #############################
                if len( outputs.split() ) > 0:
                    # If Crichton Format, Then Use Outputs As Actual Output
                    if outputs != "":
                        temp_output_array.append( 0 ) if float( outputs ) == 0.0 else temp_output_array.append( 1 )

                    #######################################
                    # Vectorize Primary Input And Outputs #
                    #######################################
                    temp_primary_input_array.append( primary_input_id )
                    temp_secondary_input_array.append( secondary_input_id )
                    if tertiary_input_id != -1: temp_tertiary_input_array.append( tertiary_input_id )

                ####################################################
                # Vectorize Primary, Secondary And Tertiary Inputs #
                ####################################################
                # If Outputs Is Empty, We're Just Vectorizing For Prediction
                # We Just Need To Vectorize Primary, Secondary And Tertiary Inputs
                else:
                    # One-Hot Encodings
                    if pad_inputs:
                        temp_array = np.zeros( ( self.Get_Number_Of_Primary_Elements(), ), dtype = int )
                        temp_array[primary_input_id] = 1
                        temp_primary_input_array.append( temp_array )

                        temp_array = np.zeros( ( self.Get_Number_Of_Primary_Elements(), ), dtype = int )
                        temp_array[secondary_input_id] = 1
                        temp_secondary_input_array.append( temp_array )

                        if tertiary_input_id != -1:
                            temp_tertiary_input_array = np.zeros( ( self.Get_Number_Of_Primary_Elements(), ), dtype = int )
                            temp_tertiary_input_array[tertiary_input_id] = 1
                    # Single Token ID Per Instance
                    else:
                        temp_primary_input_array.append( [primary_input_id]     )
                        temp_secondary_input_array.append( [secondary_input_id] )
                        if tertiary_input_id != -1: temp_tertiary_input_array.append( [tertiary_input_id] )

            # Add Vectorized Instances To Instance Lists
            for instance in temp_primary_input_array:
                primary_input_array.append( instance )

            for instance in temp_secondary_input_array:
                secondary_input_array.append( instance )

            for instance in temp_tertiary_input_array:
                tertiary_input_array.append( instance )

            for instance in temp_output_array:
                output_array.append( instance )

        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorized Primary Input   \"" + str( primary_input_array   ) + "\"" )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorized Secondary Input \"" + str( secondary_input_array ) + "\"" )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorized Tertiary Input  \"" + str( tertiary_input_array  ) + "\"" )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Vectorized Outputs         \"" + str( output_array          ) + "\"" )
        self.Print_Log( "CrichtonDataLoader::Encode_Model_Instance() - Complete" )

        return primary_input_array, secondary_input_array, tertiary_input_array, output_array


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

    def Load_Token_ID_Key_Data( self, file_path ):
        if len( self.token_id_dictionary ) != 0:
            self.Print_Log( "CrichtonDataLoader::Load_Token_ID_Key_Data() - Warning: Primary Key Hash Is Not Empty / Saving Existing Data To: \"temp_primary_key_data.txt\"", force_print = True )
            self.Save_Token_ID_Key_Data( "temp_primary_key_data.txt" )

        self.token_id_dictionary = {}

        primary_key_data = self.Read_Data( file_path, keep_in_memory = False )

        if len( primary_key_data ) == 0:
            self.Print_Log( "CrichtonDataLoader::Load_Token_ID_Key_Data() - Error Loading File Data: \"" + str( file_path ) + "\"" )
            return False

        self.Print_Log( "CrichtonDataLoader::Load_Token_ID_Data() - Loading Key Data" )

        for line in primary_key_data:
            key, value = line.split( "<:>" )
            self.Print_Log( "CrichtonDataLoader::Load_Token_ID_Data() - Key: " + str( key ) + " - Value: " + str( value ) )
            self.token_id_dictionary[key] = int( value )

        self.Print_Log( "CrichtonDataLoader::Load_Token_ID_Data() - Complete" )

        return True

    def Save_Token_ID_Key_Data( self, file_path ):
        if len( self.token_id_dictionary ) == 0:
            self.Print_Log( "CrichtonDataLoader::Save_Token_ID_Key_Data() - Warning: Primary Key Data = Empty / No Data To Save" )
            return

        self.Print_Log( "CrichtonDataLoader::Save_Token_ID_Data() - Saving Key Data" )

        # Open File Handle
        fh = open( file_path, "w" )

        for key in self.token_id_dictionary:
            fh.write( str( key ) + "<:>" + str( self.token_id_dictionary[key] ) + "\n" )

        fh.close()

        self.Print_Log( "CrichtonDataLoader::Save_Token_ID_Data() - Complete" )

    """
        Generates IDs For Each Token Given The Following File Format

            Crichton Format:    C001	TREATS	C002	0.01
                                C001	ISA	    C003    0.5
                                C002	TREATS	C005	0.998
                                C002	AFFECTS	C003	0.442
                                C003	TREATS	C002    0.1235
                                C003	AFFECTS	C001	0.642
                                C003	ISA	    C008    0.12
                                C004	ISA	    C010    0.00001
                                C005	TREATS	C003	0.1234
                                C005	AFFECTS	C001	0.0

        Inputs:
            data_list                   : List Of Data By Line (List)
            restrict_output             : Restricts Model Outputs To Only Unique Instances Within The Data (Bool)
            skip_association_value      : True - Skip Final Association Value ie. In Crichton Data-set, False - Keep Value (Bool)
            scale_embedding_weight_value: Scales Embedding Weights By Specified Value ie. embedding_weights *= scale_embedding_weight_value (Float)

        Outputs:
            None
    """
    def Generate_Token_IDs( self, data_list = [], restrict_output = True, skip_association_value = False, scale_embedding_weight_value = 1.0 ):
        # Check(s)
        if self.generated_embedding_ids:
            self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Warning: Already Generated Embedding Token IDs" )
            return

        self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() -           Skip Association Value : " + str( skip_association_value ) )

        # Insert Padding At First Index Of The Token ID Dictionaries
        padding_token = self.Get_Padding_Token()

        if padding_token not in self.token_id_dictionary    : self.token_id_dictionary[padding_token]     = 0
        if padding_token not in self.primary_id_dictionary  : self.primary_id_dictionary[padding_token]   = 0
        if padding_token not in self.secondary_id_dictionary: self.secondary_id_dictionary[padding_token] = 0
        if padding_token not in self.tertiary_id_dictionary : self.tertiary_id_dictionary[padding_token]  = 0
        if padding_token not in self.output_id_dictionary   : self.output_id_dictionary[padding_token]    = 0

        # Generate Embeddings Based On Embeddings (Assumes Word2vec Format)
        if len( self.embeddings ) > 0 and self.generated_embedding_ids == False:
            # Index 0 Of Embeddings Matrix Is Padding
            embeddings = np.zeros( ( len( self.embeddings ) + 1, len( self.embeddings[1].split() ) - 1 ) )

            self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Generating Token IDs Using Embeddings" )

            for embedding in self.embeddings:
                index              = len( self.token_id_dictionary )
                embedding_elements = embedding.split()
                embeddings[index]  = np.asarray( embedding_elements[1:], dtype = 'float32' )

                # Check To See If Element Is Already In Dictionary, If Not Add The Element
                if embedding_elements[0] not in self.token_id_dictionary:
                    self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Adding Token: \"" + str( embedding_elements[0] ) + "\" => Embedding Row Index: " + str( index ) )
                    self.token_id_dictionary[embedding_elements[0]] = index
                else:
                    self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Adding Token - Warning: \"" + str( embedding_elements[0] ) + "\" Already In Dictionary" )

            self.embeddings = []
            self.embeddings = np.asarray( embeddings ) * scale_embedding_weight_value if scale_embedding_weight_value != 1.0 else np.asarray( embeddings )

            self.generated_embedding_ids = True

        # Build Unique Per Input/Output Token ID Dictionaries
        # Iterate Through The Data And Generate The Unique Input/Output Lists
        # Check(s) - If User Does Not Specify Data, Use The Data Stored In Memory
        if len( data_list ) == 0:
            self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Building Token ID Dictionaries" )

        # Process Elements In Data List, Line-By-Line
        self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Building Unique Input/Output Dictionaries" )

        for sequence in data_list:
            self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Processing Sequence: " + str( sequence ) )

            sequence_tokens = sequence.split()

            # IF We're Not Using Embeddings, Build Unique Token ID Dictionary
            if self.generated_embedding_ids == False:
                for token in sequence_tokens:
                    # Check To See If Element Is Already In Dictionary, If Not Add The Element
                    if token not in self.token_id_dictionary:
                        index = len( self.token_id_dictionary )
                        self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Adding Token: \"" + str( token ) + "\" => Embedding Row Index: " + str( index ) )
                        self.token_id_dictionary[token] = index

            # Build Unique Primary Input ID Dictionary
            if sequence_tokens[0] not in self.primary_id_dictionary:
                # If We've Generated Our Unique Token List From Embeddings And The Token Does Not Have An Embedding, Skip It
                if self.generated_embedding_ids == False or \
                    ( self.generated_embedding_ids == True and sequence_tokens[0] in self.token_id_dictionary ):
                    index = len( self.primary_id_dictionary )
                    self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Adding To Primary Dictionary: " + str( sequence_tokens[0] ) + " => " + str( index ) )
                    self.primary_id_dictionary[sequence_tokens[0]] = index

            # Build Unique Secondary Input ID Dictionary
            if sequence_tokens[1] not in self.secondary_id_dictionary:
                # If We've Generated Our Unique Token List From Embeddings And The Token Does Not Have An Embedding, Skip It
                if self.generated_embedding_ids == False or \
                    ( self.generated_embedding_ids == True and sequence_tokens[1] in self.token_id_dictionary ):
                    index = len( self.secondary_id_dictionary )
                    self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Adding To Secondary Dictionary: " + str( sequence_tokens[1] ) + " => " + str( index ) )
                    self.secondary_id_dictionary[sequence_tokens[1]] = index

            # Build Unique Tertiary Input ID Dictionary
            if sequence_tokens[2] not in self.tertiary_id_dictionary:
                # If We've Generated Our Unique Token List From Embeddings And The Token Does Not Have An Embedding, Skip It
                if self.generated_embedding_ids == False or \
                    ( self.generated_embedding_ids == True and sequence_tokens[2] in self.token_id_dictionary ):
                    index = len( self.tertiary_id_dictionary )
                    self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Adding To Tertiary Dictionary: " + str( sequence_tokens[2] ) + " => " + str( index ) )
                    self.tertiary_id_dictionary[sequence_tokens[2]] = index

            self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Dictionaries Built" )

        self.number_of_primary_tokens   = len( self.primary_id_dictionary   )
        self.number_of_secondary_tokens = len( self.secondary_id_dictionary )
        self.number_of_tertiary_tokens  = len( self.tertiary_id_dictionary  )

        self.Print_Log( "CrichtonDataLoader::Generate_Token_IDs() - Complete" )

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Primary Matrix
        self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Primary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "primary_input_matrix.npz" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Primary Matrix" )
            self.primary_inputs = scipy.sparse.load_npz( file_path + "primary_input_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "primary_input_matrix.npy" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Primary Matrix" )
            self.primary_inputs = np.load( file_path + "primary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Primary Input Matrix Not Found" )

        # Secondary Matrix
        self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Secondary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "secondary_input_matrix.npz" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Secondary Matrix" )
            self.secondary_inputs = scipy.sparse.load_npz( file_path + "secondary_input_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "secondary_input_matrix.npy" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Secondary Matrix" )
            self.secondary_inputs = np.load( file_path + "secondary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Secondary Input Matrix Not Found" )

        # Tertiary Matrix
        self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Tertiary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "tertiary_input_matrix.npz" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Tertiary Matrix" )
            self.tertiary_inputs = scipy.sparse.load_npz( file_path + "tertiary_input_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "tertiary_input_matrix.npy" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Tertiary Matrix" )
            self.tertiary_inputs = np.load( file_path + "tertiary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Primary Input Matrix Not Found" )

        # Output Matrix
        self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Output Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "output_matrix.npz" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Output Matrix" )
            self.outputs = scipy.sparse.load_npz( file_path + "output_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "output_matrix.npy" ):
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Output Matrix" )
            self.outputs = np.load( file_path + "output_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Output Matrix Not Found" )

        self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Complete" )

        return False

    """
        Saves Vectorized Model Inputs/Outputs To File.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Save_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "CrichtonDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Primary Matrix
        self.Print_Log( "CrichtonDataLoader::Save_Vectorized_Model_Data() - Saving Primary Matrix" )

        if self.primary_inputs is None:
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Warning: Primary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.primary_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "primary_input_matrix.npz", self.primary_inputs )
        elif isinstance( self.primary_inputs, np.ndarray ):
            np.save( file_path + "primary_input_matrix.npy", self.primary_inputs, allow_pickle = True )

        # Secondary Matrix
        self.Print_Log( "CrichtonDataLoader::Save_Vectorized_Model_Data() - Saving Secondary Matrix" )

        if self.secondary_inputs is None:
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Warning: Secondary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.secondary_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "secondary_input_matrix.npz", self.secondary_inputs )
        elif isinstance( self.secondary_inputs, np.ndarray ):
            np.save( file_path + "secondary_input_matrix.npy", self.secondary_inputs, allow_pickle = True )

        # Tertiary Matrix
        self.Print_Log( "CrichtonDataLoader::Save_Vectorized_Model_Data() - Saving Tertiary Matrix" )

        if self.tertiary_inputs is None:
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Warning: Tertiary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.tertiary_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "tertiary_input_matrix.npz", self.tertiary_inputs )
        elif isinstance( self.tertiary_inputs, np.ndarray ):
            np.save( file_path + "tertiary_input_matrix.npy", self.tertiary_inputs, allow_pickle = True )

        # Output Matrix
        self.Print_Log( "CrichtonDataLoader::Save_Vectorized_Model_Data() - Saving Output Matrix" )

        if self.outputs is None:
            self.Print_Log( "CrichtonDataLoader::Load_Vectorized_Model_Data() - Warning: Output == 'None' / No Data To Save" )
        elif isinstance( self.outputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "output_matrix.npz", self.outputs )
        elif isinstance( self.outputs, np.ndarray ):
            np.save( file_path + "output_matrix.npy", self.outputs, allow_pickle = True )

        self.Print_Log( "CrichtonDataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False

    """
        Fetches Token ID From String.

        Inputs:
            token    : Token (String)

        Outputs:
            token_id : Token ID Value (Integer)
    """
    def Get_Token_ID( self, token ):
        self.Print_Log( "CrichtonDataLoader::Get_Token_ID() - Fetching ID For Token: \"" + str( token ) + "\"" )

        if token in self.token_id_dictionary:
            self.Print_Log( "CrichtonDataLoader::Get_Token_ID() - Token ID Found: \"" + str( token ) + "\" => " + str( self.token_id_dictionary[token] ) )
            return self.token_id_dictionary[token]

        self.Print_Log( "CrichtonDataLoader::Get_Token_ID() - Warning: Key Not Found In Dictionary" )

        return -1

    """
        Fetches Token String From ID Value.

        Inputs:
            index_value  : Token ID Value (Integer)

        Outputs:
            key          : Token String (String)
    """
    def Get_Token_From_ID( self, index_value):
        self.Print_Log( "CrichtonDataLoader::Get_Token_From_ID() - Searching For ID: " + str( index_value ) )

        for key, val in self.token_id_dictionary.items():
            if val == index_value:
                self.Print_Log( "CrichtonDataLoader::Get_Token_From_ID() - Found: \"" + str( key ) + "\"" )
                return key

        self.Print_Log( "CrichtonDataLoader::Get_Token_From_ID() - Warning: Key Not Found In Dictionary" )

        return None

    """
        Fetches Next Set Of Data Instances From File. (Assumes File Is Not Entirely Read Into Memory).

        Inputs:
            number_of_elements_to_fetch  : (Integer)

        Outputs:
            data_list                    : List Of Elements Read From File (String)
    """
    def Get_Next_Batch( self, file_path, number_of_elements_to_fetch ):
        # Load Training File
        if self.utils.Check_If_File_Exists( file_path ) == False:
            self.Print_Log( "CrichtonDataLoader::Get_Next_Batch() - Error: Data File \"" + str( file_path ) + "\" Does Not Exist", force_print = True )
            return [-1]

        self.Print_Log( "CrichtonDataLoader::Get_Next_Batch() - Fetching The Next " + str( number_of_elements_to_fetch ) + " Elements" )

        if self.Reached_End_Of_File():
            self.Print_Log( "CrichtonDataLoader::Get_Next_Batch() - Reached EOF" )
            return []

        data_list = self.Read_Data( file_path, read_all_data = False, keep_in_memory = False, number_of_lines_to_read = number_of_elements_to_fetch )

        self.Print_Log( "CrichtonDataLoader::Get_Next_Batch() - Fetched " + str( len( data_list ) ) + " Elements" )
        self.Print_Log( "CrichtonDataLoader::Get_Next_Batch() - Complete" )

        return data_list

    """
        Reinitialize Token ID Values For Primary And Secondary ID Dictionaries

        Inputs:
            None

        Outputs:
            None
    """
    def Reinitialize_Token_ID_Values( self ):
        self.Print_Log( "CrichtonDataLoader::Reinitialize_Token_ID_Values() - Re-initializing Token ID Values" )

        if self.Is_Embeddings_Loaded() or self.Simulate_Embeddings_Loaded_Mode():
            self.number_of_primary_tokens   = len( self.Get_Token_ID_Dictionary() ) - 1
            self.number_of_secondary_tokens = len( self.Get_Token_ID_Dictionary() ) - 1
            self.number_of_tertiary_tokens  = len( self.Get_Token_ID_Dictionary() ) - 1
        else:
            for token, id_value in self.Get_Token_ID_Dictionary().items():
                # Element Is CUI
                if re.search( r'^[Cc]\d+', token ) and id_value > self.number_of_primary_tokens:
                    self.number_of_primary_tokens = id_value
                # Element Is Not CUI (Relation)
                elif not re.search( r'^[Cc]\d+', token ) and id_value > self.number_of_secondary_tokens:
                    self.number_of_secondary_tokens = id_value

        self.Print_Log( "CrichtonDataLoader::Reinitialize_Token_ID_Values() - New Token ID Values" )
        self.Print_Log( "                                           - # Of Primary Tokens: "   + str( self.number_of_primary_tokens   ) )
        self.Print_Log( "                                           - # Of Secondary Tokens: " + str( self.number_of_secondary_tokens ) )

        self.number_of_primary_tokens   += 1 if self.number_of_primary_tokens   > 0 else self.number_of_primary_tokens
        self.number_of_secondary_tokens += 1 if self.number_of_secondary_tokens > 0 else self.number_of_secondary_tokens
        self.number_of_tertiary_tokens  += 1 if self.number_of_tertiary_tokens  > 0 else self.number_of_tertiary_tokens

        self.Print_Log( "CrichtonDataLoader::Reinitialize_Token_ID_Values() - Complete" )


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################


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
    #    Worker Thread Function                                                                #
    #                                                                                          #
    ############################################################################################

    """
        DataLoader Model Data Vectorization Worker Thread

        Inputs:
            thread_id              : Thread Identification Number (Integer)
            data_list              : List Of String Instances To Vectorize (Data Chunk Determined By CrichtonDataLoader::Encode_Model_Data() Function)
            dest_array             : Placeholder For Threaded Function To Store Outputs (Do Not Modify) (List)
            model_type             : Model Type (String)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            stack_inputs           : True = Stacks Inputs, False = Does Not Stack Inputs - Used For BiLSTM Model (Boolean)
            str_delimiter          : String Delimiter - Used To Separate Elements Within An Instance (String)

        Outputs:
            primary_inputs         : CSR Matrix or Numpy Array
            secondary_inputs       : CSR Matrix or Numpy Array
            tertiary_inputs        : CSR Matrix or Numpy Array
            outputs                : CSR Matrix or Numpy Array

        Note:
            Outputs Are Stored In A List Per Thread Which Is Managed By CrichtonDataLoader::Encode_Model_Data() Function.

    """
    def Worker_Thread_Function( self, thread_id, data_list, dest_array, model_type = "open_discovery", use_csr_format = False,
                                pad_inputs = True, pad_output = True, stack_inputs = False, str_delimiter = " " ):
        if self.Is_Embeddings_Loaded() and pad_inputs == True:
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Detected Embeddings Loaded / Setting 'pad_inputs = False'" )
            pad_inputs = False

        separate_outputs = False

        if model_type == "open_discovery" and use_csr_format and pad_output == False and separate_outputs == False:
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Detected 'model_type == open_discovery', 'use_csr_format == True' and 'pad_outputs == False' / Setting 'separate_outputs = True'" )
            separate_outputs = True

        self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Vectorizing Data Using Settings" )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - Model Type        : " + str( model_type         ) )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - Use CSR Format    : " + str( use_csr_format     ) )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - Pad Inputs        : " + str( pad_inputs         ) )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - Pad Output        : " + str( pad_output         ) )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - Stack Inputs      : " + str( stack_inputs       ) )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - String Delimiter  : " + str( str_delimiter      ) )
        self.Print_Log( "                                             - Thread ID: " + str( thread_id ) + " - Separate Outputs  : " + str( separate_outputs   ) )

        # Reduce Input/Output Space During Vectorization Of Data (Faster Data Vectorization)
        temp_pad_inputs = False if use_csr_format else pad_inputs
        temp_pad_output = False if use_csr_format else pad_output

        # Vectorized Input/Output Placeholder Lists
        primary_inputs   = []
        secondary_inputs = []
        tertiary_inputs  = []
        outputs          = []

        # CSR Matrix Format
        if use_csr_format:
            primary_row_index   = 0
            secondary_row_index = 0
            tertiary_row_index  = 0
            output_row_index    = 0

            primary_input_row,  secondary_input_row,  tertiary_input_row,  output_row  = [], [], [], []
            primary_input_col,  secondary_input_col,  tertiary_input_col,  output_col  = [], [], [], []
            primary_input_data, secondary_input_data, tertiary_input_data, output_data = [], [], [], []

            if stack_inputs == False:
                secondary_inputs = csr_matrix( secondary_inputs )
                tertiary_inputs  = csr_matrix( tertiary_inputs  )
                outputs          = csr_matrix( outputs          )

        # Data Format: node_a\node_b\tnode_c\tjaccard_similarity
        for line in data_list:
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " -> Data Line: " + str( line.strip() ) )

            if not line: break

            line     = line.strip()
            elements = line.split( str_delimiter )

            # Check
            if len( elements ) < 3: continue

            # Vectorize Inputs/Outputs
            primary_input_array   = []
            secondary_input_array = []
            tertiary_input_array  = []
            output_array          = []

            primary_input_array, secondary_input_array, tertiary_input_array, output_array = self.Encode_Model_Instance( elements[0], elements[1], elements[2], " ".join( elements[3:] ),
                                                                                                                         model_type = model_type, pad_inputs = temp_pad_inputs,
                                                                                                                         pad_output = temp_pad_output, separate_outputs = separate_outputs )

            # Check(s)
            if not self.skip_out_of_vocabulary_words and ( len( primary_input_array ) == 0 or len( secondary_input_array ) == 0 or len( output_array ) == 0 ):
                self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Error Vectorizing Input/Output Data", force_print = True )
                self.Print_Log( "                                     - Line: \"" + str( line ) + "\"", force_print = True )
                dest_array[thread_id] = None
                return
            elif self.skip_out_of_vocabulary_words and ( len( primary_input_array ) == 0 or len( secondary_input_array ) == 0 or len( output_array ) == 0 ):
                self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Warning: Vectorizing Input/Output Data - Element Does Not Exist", force_print = True )
                self.Print_Log( "                                     - Line: \"" + str( line ) + "\"", force_print = True )
                continue

            ##################
            # Primary Inputs #
            ##################
            if use_csr_format and stack_inputs == False:
                primary_input_array = [primary_input_array]

                for temp_array in primary_input_array:
                    if len( temp_array ) > 0:
                        for i, value in enumerate( temp_array ):
                            primary_input_row.append( primary_row_index )

                            if pad_inputs:
                                primary_input_col.append( value )
                                primary_input_data.append( 1 )
                            else:
                                primary_input_col.append( i )
                                primary_input_data.append( value )

                            primary_row_index += 1
            else:
                for i in range( len( primary_input_array ) ):
                    primary_inputs.append( primary_input_array[i] )

            ####################
            # Secondary Inputs #
            ####################
            if use_csr_format and stack_inputs == False:
                secondary_input_array = [secondary_input_array]

                for temp_array in secondary_input_array:
                    if len( temp_array ) > 0:
                        for i, value in enumerate( temp_array ):
                            secondary_input_row.append( secondary_row_index )

                            if pad_inputs:
                                secondary_input_col.append( value )
                                secondary_input_data.append( 1 )
                            else:
                                secondary_input_col.append( i )
                                secondary_input_data.append( value )

                            secondary_row_index += 1
            else:
                for i in range( len( secondary_input_array ) ):
                    secondary_inputs.append( secondary_input_array[i] )

            ###################
            # Tertiary Inputs #
            ###################
            if use_csr_format and stack_inputs == False:
                tertiary_input_array = [tertiary_input_array]

                for temp_array in tertiary_input_array:
                    if len( temp_array ) > 0:
                        for i, value in enumerate( temp_array ):
                            tertiary_input_row.append( tertiary_row_index )

                            if pad_inputs:
                                tertiary_input_col.append( value )
                                tertiary_input_data.append( 1 )
                            else:
                                tertiary_input_col.append( i )
                                tertiary_input_data.append( value )

                            tertiary_row_index += 1
            else:
                for i in range( len( tertiary_input_array ) ):
                    tertiary_inputs.append( tertiary_input_array[i] )

            ###########
            # Outputs #
            ###########
            if use_csr_format:
                output_array = [output_array]

                for temp_array in output_array:
                    if len( temp_array ) > 0:
                        for i, value in enumerate( temp_array ):
                            output_row.append( output_row_index )
                            output_col.append( i )
                            output_data.append( value )

                        output_row_index += 1
            else:
                for i in range( len( output_array ) ):
                    outputs.append( np.asarray( output_array[i] ) )

        # Set Variable Lengths For Input/Output Data Matrices/Lists
        # Assumes Inputs Are Embeddings Indices (Length = 1) And Output Is Binary Classification (Length = 1)
        number_of_primary_rows   = 1
        number_of_secondary_rows = 1
        number_of_tertiary_rows  = 1
        number_of_output_rows    = 1

        if model_type == "open_discovery":
            if pad_inputs:
                number_of_primary_rows   = self.Get_Number_Of_Primary_Elements()
                number_of_secondary_rows = self.Get_Number_Of_Secondary_Elements() if self.Get_Is_CUI_Data() else self.Get_Number_Of_Primary_Elements()
                number_of_tertiary_rows  = self.Get_Number_Of_Primary_Elements()

            # Number Of Output Rows Are Not Dependent On Padding Inputs
            if pad_output:
                number_of_output_rows = self.Get_Number_Of_Primary_Elements()

        elif model_type == "closed_discovery":
            if pad_inputs:
                number_of_primary_rows   = self.Get_Number_Of_Primary_Elements()
                number_of_secondary_rows = self.Get_Number_Of_Primary_Elements()
                number_of_tertiary_rows  = self.Get_Number_Of_Primary_Elements()

            # Number Of Output Rows Are Not Dependent On Padding Inputs
            if pad_output:
                number_of_output_rows = self.Get_Number_Of_Secondary_Elements() if self.Get_Is_CUI_Data() else self.Get_Number_Of_Primary_Elements()

        # Outputs Are Binary Classification
        number_of_output_rows = 1

        # Convert Inputs To CSR Matrix Format
        if use_csr_format and stack_inputs == False:
            # Convert Lists To Numpy Arrays
            primary_input_row    = np.asarray( primary_input_row  )
            primary_input_col    = np.asarray( primary_input_col  )
            primary_input_data   = np.asarray( primary_input_data )

            secondary_input_row  = np.asarray( secondary_input_row  )
            secondary_input_col  = np.asarray( secondary_input_col  )
            secondary_input_data = np.asarray( secondary_input_data )

            tertiary_input_row   = np.asarray( tertiary_input_row  )
            tertiary_input_col   = np.asarray( tertiary_input_col  )
            tertiary_input_data  = np.asarray( tertiary_input_data )

            # Check(s)
            if len( primary_input_data ) == 0 and len( secondary_input_data ) == 0 and len( tertiary_input_data ) == 0 and len( output_data ) == 0:
                dest_array[thread_id] = None
                return

            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Input Data:" )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Input Row Size          : " + str( len( primary_input_row  ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Input Col Size          : " + str( len( primary_input_col  ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Input Data Size         : " + str( len( primary_input_data ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Number Of Matrix Columns: " + str( primary_row_index         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Number Of Matrix Rows   : " + str( number_of_primary_rows    ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Matrix Input Row:       \n" + str( primary_input_row         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Matrix Input Column:    \n" + str( primary_input_col         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Matrix Input Data:      \n" + str( primary_input_data        ) )

            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Input Data:" )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Input Row Size          : " + str( len( secondary_input_row  ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Input Col Size          : " + str( len( secondary_input_col  ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Input Data Size         : " + str( len( secondary_input_data ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Number Of Matrix Columns: " + str( secondary_row_index         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Number Of Matrix Rows   : " + str( number_of_secondary_rows    ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Matrix Input Row:       \n" + str( secondary_input_row         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Matrix Input Column:    \n" + str( secondary_input_col         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Matrix Input Data:      \n" + str( secondary_input_data        ) )

            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Input Data:" )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Input Row Size          : " + str( len( tertiary_input_row  ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Input Col Size          : " + str( len( tertiary_input_col  ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Input Data Size         : " + str( len( tertiary_input_data ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Number Of Matrix Columns: " + str( tertiary_row_index         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Number Of Matrix Rows   : " + str( number_of_tertiary_rows    ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Matrix Input Row:       \n" + str( tertiary_input_row         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Matrix Input Column:    \n" + str( tertiary_input_col         ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Matrix Input Data:      \n" + str( tertiary_input_data        ) )

            # Check(s)
            if ( len( primary_input_row ) == 0 or len( primary_input_col ) == 0 or len( primary_input_data ) == 0 ) and primary_row_index == 0 or number_of_primary_rows == 0:
                self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Error: Primary Input Matrix Dimensions Do Not Match" )
                dest_array[thread_id] = None
                return

            if ( len( secondary_input_row ) == 0 or len( secondary_input_col ) == 0 or len( secondary_input_data ) == 0 ) and secondary_row_index == 0 or number_of_secondary_rows == 0:
                self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Error: Secondary Input Matrix Dimensions Do Not Match" )
                dest_array[thread_id] = None
                return

            if ( len( tertiary_input_row ) == 0 or len( tertiary_input_col ) == 0 or len( tertiary_input_data ) == 0 ) and number_of_tertiary_rows == 0:
                self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Error: Tertiary Input Matrix Dimensions Do Not Match" )
                dest_array[thread_id] = None
                return

            # Convert Numpy Arrays To CSR Matrices
            primary_inputs     = csr_matrix( ( primary_input_data,   ( primary_input_row,   primary_input_col   ) ), shape = ( primary_row_index, number_of_primary_rows     ) )
            secondary_inputs   = csr_matrix( ( secondary_input_data, ( secondary_input_row, secondary_input_col ) ), shape = ( secondary_row_index, number_of_secondary_rows ) )
            tertiary_inputs    = csr_matrix( ( tertiary_input_data,  ( tertiary_input_row,  tertiary_input_col  ) ), shape = ( tertiary_row_index, number_of_tertiary_rows   ) )
        else:
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Input Data:  \n" + str( primary_inputs   ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Input Data:\n" + str( secondary_inputs ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Input Data: \n" + str( tertiary_inputs  ) )

            primary_inputs   = np.asarray( primary_inputs   )
            secondary_inputs = np.asarray( secondary_inputs )
            tertiary_inputs  = np.asarray( tertiary_inputs  )

            # Check(s)
            if len( primary_inputs ) == 0 and len( secondary_inputs ) == 0 and len( tertiary_inputs ) == 0 and len( outputs ) == 0:
                dest_array[thread_id] = None
                return

        # Convert Outputs To CSR Matrix Format
        if use_csr_format:
            output_row  = np.asarray( output_row  )
            output_col  = np.asarray( output_col  )
            output_data = np.asarray( output_data )

            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data:" )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Row Size                : " + str( len( output_row     ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Col Size                : " + str( len( output_col     ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data Size               : " + str( len( output_data    ) ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Number Of Matrix Columns: " + str( output_row_index      ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Number Of Matrix Rows   : " + str( number_of_output_rows ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Matrix Row:             \n" + str( output_row            ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Matrix Column:          \n" + str( output_col            ) )
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Matrix Data:            \n" + str( output_data           ) )

            # Check(s)
            if len( output_row ) == 0 or len( output_col ) == 0 or len( output_data ) == 0 and output_row_index == 0 or number_of_output_rows == 0:
                self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Error: Output Matrix Dimensions Do Not Match" )
                dest_array[thread_id] = None
                return

            outputs = csr_matrix( ( output_data, ( output_row, output_col ) ), shape = ( output_row_index, number_of_output_rows ) )
        else:
            self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data:         \n" + str( outputs          ) )

            outputs = np.asarray( outputs )

        # Assign Thread Vectorized Data To Temporary DataLoader Placeholder Array
        dest_array[thread_id] = [primary_inputs, secondary_inputs, tertiary_inputs, outputs]

        self.Print_Log( "CrichtonDataLoader::Worker_Thread_Function() - Complete" )



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NNLBD.Models import DataLoader\n" )
    print( "     data_loader = DataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()
