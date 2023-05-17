#  Converts NNLBD Closed Discovery Data Format:
#      concept_a\tconcept_c\tconcept_b_1\t...\tconcept_b_n
#
#  To NNLBD Open Discovery Data Format:
#      concept_a\tconcept_b\tconcept_c_1\t...\tconcept_c_n
#
#  NOTE: This script assumes the original data is in 'Closed Discovery' format.
#
#  ~ Statler

import re, time

def Main():
    # User-Specified Variables
    file_path     = "./../data/HOC/old/train_cs1_closed_discovery_without_aggregators_mod"
    new_file_path = "./../data/HOC/train_cs1_closed_discovery_without_aggregators_mod"
    delimiter     = "\t"


    # --------------------------- #
    # Do Not Edit Variables Below #
    # --------------------------- #

    # Start Elapsed Time Timer
    start_time    = time.time()
    a_b_relations = {}
    b_c_relations = []
    temp_a_b_relations = []

    print( "~Begin" )

    ##################
    # Read File Data #
    ##################
    try:
        file_handle         = open( file_path, "r" )
        number_of_instances = 0

        while True:
            line = file_handle.readline()
            line = line.rstrip()

            if 'a_concept' in line and 'b_concept' in line and 'c_concept' in line: continue
            if not line: break

            line_elements = line.split( delimiter )

            # Group A-C Concept Relation And Linking B Concepts
            if len( line_elements ) >= 3:
                a_concept   = re.sub( r'^\s+|\s+$', "", line_elements[0] )
                c_concept   = re.sub( r'^\s+|\s+$', "", line_elements[1] )
                b_concepts  = line_elements[2:]

                for b_concept in b_concepts:
                    b_concept   = re.sub( r'^\s+|\s+$', "", b_concept )
                    a_b_concept = a_concept + "\t" + b_concept
                    b_a_concept = b_concept + "\t" + a_concept
                    c_b_concept = c_concept + "\t" + b_concept

                    if a_b_concept not in a_b_relations and b_a_concept not in a_b_relations and c_b_concept not in a_b_relations:
                        a_b_relations[a_b_concept] = [c_concept]
                    else:
                        if a_b_concept in a_b_relations:
                            a_b_relations[a_b_concept].append( c_concept )
                        elif b_a_concept in a_b_relations:
                            a_b_relations[b_a_concept].append( c_concept )
                        elif c_b_concept in a_b_relations:
                            a_b_relations[c_b_concept].append( a_concept )

        file_handle.close()

        # Write Header To New NNLBD Formatted File
        write_file_handle = open( new_file_path, "w" )
        write_file_handle.write( "a_concept\tb_concept\tc_concept\n" )

        # Write Re-Formatted Data To New File
        for a_b_relation in a_b_relations:
            c_concepts = "\t".join( a_b_relations[a_b_relation] )
            write_file_handle.write( str( a_b_relation ) + "\t" + str( c_concepts ) + "\n" )

            number_of_instances += 1

        write_file_handle.close()

    except FileNotFoundError:
        print( "Error: Unable To Open Data File \"" + str( file_path ) + "\"" )
        exit()

    print( "Number Of Instances: " + str( number_of_instances        ) )

    # Compute Elapsed Time
    elapsed_time = "{:.2f}".format( time.time() - start_time )
    print( "Elapsed Time: " + str( elapsed_time ) + " secs" )

    print( "~Fin" )


if __name__ == '__main__':
    Main()
