#  Converts NNLBD Open Discovery Data Format:
#      concept_a\tconcept_b\tconcept_c_1\t...\tconcept_c_n
#
#  To NNLBD Closed Discovery Data Format:
#      concept_a\tconcept_c\tconcept_b_1\t...\tconcept_b_n
#
#  NOTE: This script assumes the original data is in 'Open Discovery' format.
#
#  ~ Statler

import re, time

def Main():
    # User-Specified Variables
    file_path     = "./../data/test/cui_mini_open_discovery"
    new_file_path = "./../data/test/cui_mini_closed_discovery"
    delimiter     = "\t"


    # --------------------------- #
    # Do Not Edit Variables Below #
    # --------------------------- #

    # Start Elapsed Time Timer
    start_time    = time.time()
    a_c_relations = {}

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
                b_concept   = re.sub( r'^\s+|\s+$', "", line_elements[1] )
                c_concepts  = line_elements[2:]

                for c_concept in c_concepts:
                    c_concept   = re.sub( r'^\s+|\s+$', "", c_concept )
                    a_c_concept = a_concept + "\t" + c_concept
                    c_a_concept = c_concept + "\t" + a_concept

                    if a_c_concept not in a_c_relations and c_a_concept not in a_c_relations:
                        a_c_relations[a_c_concept] = [b_concept]
                    else:
                        if a_c_concept in a_c_relations:
                            a_c_relations[a_c_concept].append( b_concept )
                        elif c_a_concept in a_c_relations:
                            a_c_relations[c_a_concept].append( b_concept )

        file_handle.close()

        # Write Header To New NNLBD Formatted File
        write_file_handle = open( new_file_path, "w" )
        write_file_handle.write( "a_concept\tc_concept\tb_concept\n" )

        # Write Re-Formatted Data To New File
        for a_c_relation in a_c_relations:
            b_concepts = "\t".join( a_c_relations[a_c_relation] )
            write_file_handle.write( str( a_c_relation ) + "\t" + str( b_concepts ) + "\n" )

            number_of_instances += 1

        # Clean-Up
        a_c_relations = {}
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
