# Reads HOC CSV Formatted Data And Retains Specific Columns, Saved Into New File
#
# ~Statler

use strict;
use warnings;

my $edges_csv_file_path    = "data/MESHD008881_MESHD008274/edges.csv";
my $edges_with_scores_path = "data/MESHD008881_MESHD008274/edges_with_scores.csv";
my @header_columns_to_save = ( ":START_ID", ":END_ID", "year:int", "metric_jaccard:float[]" );
my @header_indices_to_save = ();

# Do Not Modify
my $header_parsed_flag     = 0;
my $header_length          = 0;

open( FILE,  "<:", "$edges_csv_file_path"    ) or die "Error: Unable To Open File - $!";
open( wFILE, ">:", "$edges_with_scores_path" ) or die "Error: Unable To Create File - $!";

while( <FILE> )
{
    chomp;
    my @tokens = split( /\,/, $_ );

    # Parse Header To Get Desired Header Column Indices
    if( $header_parsed_flag == 0 )
    {
        print( "Parsing Header Column Info\n" );

        for( 0..$#tokens )
        {
            my $current_header_index = $_;
            my $current_header_info  = $tokens[$current_header_index];

            for my $requested_header_info ( @header_columns_to_save )
            {
                push( @header_indices_to_save, $current_header_index ) if( $requested_header_info eq $current_header_info );
            }
        }

        $header_length = scalar @tokens;

        print( "Obtained Header Columns From File: \"$edges_csv_file_path\"\n" );
        print( "Obtained " . scalar @header_indices_to_save . " Column Elements\n" );

        my $header_info = join( ",", @header_columns_to_save );

        print( wFILE "$header_info\n" );

        $header_parsed_flag = 1;
    }
    # Parse Actual Data And Write To New File
    else
    {
        my $new_data = "";

        for my $token_index ( @header_indices_to_save )
        {
            $new_data .= "$tokens[$token_index]," if $token_index < scalar @tokens;
        }

        $new_data =~ s/\,$//;

        print( wFILE "$new_data\n" ) if scalar @tokens == $header_length;
    }
}

close( FILE  );
close( wFILE );


print( "~Fin\n" );
