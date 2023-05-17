#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/07/2020                                                                   #
#    Revised: 12/23/2022                                                                   #
#                                                                                          #
#    Utilities Class For The NNLBD Package.                                                #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Modules
import os, re, shutil

############################################################################################
#                                                                                          #
#    Utils Model Class                                                                     #
#                                                                                          #
############################################################################################

class Utils:
    def __init__( self ):
        pass

    def __del__( self ):
        pass

    """
        Fetches Current Working Directory/Path
    """
    def Get_Working_Directory( self ):
        return os.path.abspath( os.getcwd() )

    """
        Creates Path Along With All Folders/Directories Within The Specified Path
    """
    def Create_Path( self, file_path ):
        if not file_path or file_path == "":
            return

        file_path = re.sub( r'\\+', "/", file_path )
        folders = file_path.split( "/" )

        # Check Existing Path And Create If It Doesn't Exist
        current_path = ""

        for folder in folders:
            if self.Check_If_Path_Exists( current_path + folder ) == False:
                os.mkdir( current_path + folder )

            current_path += folder + "/"

    """
        Checks If The Specified Path Exists and It Is A Directory (Not A File)
    """
    def Check_If_Path_Exists( self, path ):
        if os.path.exists( path ) and os.path.isdir( path ):
            return True
        return False

    """
        Checks If The Specified File Exists and It Is A File (Not A Directory)
    """
    def Check_If_File_Exists( self, file_path ):
        if os.path.exists( file_path ) and os.path.isfile( file_path ):
            return True
        return False

    """
        Checks If Directory/Path Is Empty
    """
    def Is_Directory_Empty( self, path ):
        if self.Check_If_Path_Exists( path ):
            if not os.listdir( path ):
                return True
            else:
                return False

        print( "Utils::Is_Directory_Empty() - Warning: Path Is Either A File Or Not Valid" )

        return True

    """
        Checks If A Specified Path Contains Directories/Folders
    """
    def Check_If_Path_Contains_Directories( self, file_path ):
        file_path = re.sub( r'\\+', "/", file_path )
        folders   = file_path.split( "/" )
        return True if len( folders ) > 1 else False

    """
        Copies File From Source To Destination Path
    """
    def Copy_File( self, source_path, destination_path ):
        if not self.Check_If_File_Exists( source_path ):
            print( "Utils::Copy_File() - Error: Source File Does Not Exist" )
            return False

        if not self.Check_If_Path_Exists( path = destination_path ):
            print( "Utils::Copy_File() - Warning: Source Path Does Not Exist / Creating Path" )
            self.Create_Path( file_path = destination_path )

        # Copy File To Destination
        shutil.copy2( source_path, destination_path )

        return True

    """
        Checks If The Specified Path Exists and Deletes If True
    """
    def Delete_Path( self, path, delete_all_contents = False ):
        if self.Is_Directory_Empty( path ) == False and delete_all_contents == False:
            print( "Utils::Delete_Path() - Warning: Path Contains Files / Unable To Delete Path" )
            print( "                                Set 'delete_all_contents = True' To Delete Files And Path" )
            return

        if   self.Check_If_Path_Exists( path ) and delete_all_contents         : shutil.rmtree( path )
        elif self.Check_If_Path_Exists( path ) and delete_all_contents == False: os.rmdir( path )

    """
        Checks If The Specified File Exists and Deletes If True
    """
    def Delete_File( self, file_path ):
        if os.path.exists( file_path ): os.remove( file_path )

    """
        Reads Data From File And Stores Each Line In A List

        Inputs:
            file_path : File Path (String)
            lowercase : Lowercases All Text (Bool)

        Outputs:
            data_list : File Data By Line As Each List Element (List)
    """
    def Read_Data( self, file_path, lowercase = False ):
        data_list = []

        # Load Training File
        if self.Check_If_File_Exists( file_path ) == False:
            return data_list

        # Read File Data
        try:
            with open( file_path, "r" ) as in_file:
                data_list = in_file.readlines()
                data_list = [ line.strip() for line in data_list ]                  # Removes Trailing Space Characters From CUI Data Strings
                if lowercase: data_list = [ line.lower() for line in data_list ]    # Lowercase All Text
        except FileNotFoundError:
            print( "Error: Unable To Open Data File \"" + str( file_path ) + "\"", 1 )

        return data_list

    """
        Writes Data To File

        Inputs:
            file_path : File Path (String)
            data      : Data To Write To File (String)

        Outputs:
            None
    """
    def Write_Data_To_File( self, file_path, data ):
        # Check
        if data is None or data == "":
            print( "Error: No Data To Write" )
            return

        # Read File Data
        try:
            with open( file_path, "w" ) as out_file:
                out_file.write( str( data ) )
        except Exception as e:
            print( "Error: Unable To Create File" + e )
            return
        finally:
            out_file.close()


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    exit()