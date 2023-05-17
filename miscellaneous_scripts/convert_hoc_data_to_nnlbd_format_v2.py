#  Converts HOC Data Model Format:
#      entity_1\tentity_2\tentity_3\tjaccard_index
#
#  To NNLBD Triplet (A-B-C) Data Format:
#      concept_a\tconcept_b\tconcept_b or concept_a\tconcept_c\tconcept_b
#
#  ~ Statler

import time

def Main():
    # User-Specified Variables
    file_path     = "./../data/train_cs1_closed_discovery_without_aggregators.tsv"
    new_file_path = "./../data/train_cs1_closed_discovery_without_aggregators_new.tsv"
    skip_zero_jaccard_similarity_instances = True   # Skips Negative Samples
    closed_discovery_format                = False  # Changes Data From Open Discovery (OD) To Closed Discovery (CD) Format, Otherwise OD Format


    # --------------------------- #
    # Do Not Edit Variables Below #
    # --------------------------- #

    # Start Elapsed Time Timer
    start_time = time.time()

    print( "~Begin" )

    ##################
    # Read File Data #
    ##################
    try:
        file_handle       = open( file_path, "r" )
        write_file_handle = open( new_file_path, "w" )
        number_of_instances        = 0
        number_of_negative_samples = 0

        # Write Header To New NNLBD Formatted File
        if closed_discovery_format:
            write_file_handle.write( "a_concept\tc_concept\tb_concept\n" )
        else:
            write_file_handle.write( "a_concept\tb_concept\tc_concept\n" )

        while True:
            line = file_handle.readline()

            if not line: break

            line_elements = line.split( "\t" )

            # Group A-C Concept Relation And Linking B Concepts
            if len( line_elements ) > 3:
                a_concept   = line_elements[0]
                b_concept   = line_elements[1]
                c_concept   = line_elements[2]

                if a_concept == "node1" and b_concept == "node2" and c_concept == "node3": continue

                jaccard_sim = float( line_elements[3] )

                # Check Jaccard Similarity Value - Skip A-B-C Instances If Jaccard Index Score == 0
                if skip_zero_jaccard_similarity_instances and jaccard_sim == 0:
                    number_of_negative_samples += 1
                    continue

                # Write Data To New File
                if closed_discovery_format:
                    write_file_handle.write( a_concept + "\t" + c_concept + "\t" + b_concept + "\n" )
                else:
                    write_file_handle.write( a_concept + "\t" + b_concept + "\t" + c_concept + "\n" )

                number_of_instances += 1

    except FileNotFoundError:
        print( "Error: Unable To Open Data File \"" + str( file_path ) + "\"" )
        exit()
    finally:
        file_handle.close()
        write_file_handle.close()

    print( "Number Of Negative Samples     : " + str( number_of_negative_samples ) )
    print( "Number Of Instances            : " + str( number_of_instances        ) )

    # Compute Elapsed Time
    elapsed_time = "{:.2f}".format( time.time() - start_time )
    print( "Elapsed Time: " + str( elapsed_time ) + " secs" )

    print( "~Fin" )


if __name__ == '__main__':
    Main()
