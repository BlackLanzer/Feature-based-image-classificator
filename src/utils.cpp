#include "utils.h"

#include <sstream>
#include <stdio.h>
#include <algorithm>

LogStream logStream;
string    homePath = "";

bool alphabeticalSort( string a, string b )
{
    std::transform( a.begin(), a.end(), a.begin(), ::tolower );
    std::transform( b.begin(), b.end(), b.begin(), ::tolower );
    return a.compare( b ) < 0;
}

int num_dirs( const char* path, vector< string >& dir_list )
{
    int dir_count = 0;
    struct dirent* dent;
    DIR* srcdir = opendir( path );

    if( srcdir == NULL )
        return -1;

    while( ( dent = readdir( srcdir ) ) != NULL )
    {
        struct stat st;

        if( strcmp( dent->d_name, "." ) == 0 || strcmp( dent->d_name, ".." ) == 0 )
            continue;

        if( fstatat( dirfd( srcdir ), dent->d_name, &st, 0 ) < 0 ) // not a directory
            continue;

        if( S_ISDIR( st.st_mode ) )
        {
            dir_count++;
            dir_list.push_back( dent->d_name );
        }
    }
    closedir( srcdir );

    std::sort( dir_list.begin(), dir_list.end(), alphabeticalSort );
    return dir_count;
}

int num_images( const char* path )
{
    vector< string > file_list; // will be discarded

    return num_images( path, file_list );
}

int num_images( const char* path, vector< string >& file_list )
{
    DIR * dirp;
    struct dirent * entry;
    int file_count = 0;

    dirp = opendir( path );

    // Error!
    if( dirp == NULL )
        return 0;

    while( ( entry = readdir( dirp ) ) != NULL )
    {
        if( entry->d_type != DT_REG ) // If the entry is a regular file
            continue;

        string filename = entry->d_name;
        string file_ext = filename.substr( filename.rfind(".") );

        for( unsigned int i = 0; i< file_ext.length(); i++ ) // convert extention to lowercase
            file_ext[ i ] = tolower( file_ext[ i ] );

        if( file_ext == ".png" || file_ext == ".jpg" )
        {
            file_count++;
            file_list.push_back( entry->d_name );
        }
    }

    closedir( dirp );

    std::sort( file_list.begin(), file_list.end(), alphabeticalSort );
    return file_count;
}

float strtofloat( const string& what )
{
    istringstream instr( what );
    float val;
    instr >> val;
    return val;
}

string toGlobalPath( string localPath )
{
    string globalPath = localPath;
    size_t pos        = localPath.find( homePath );
    
    if( !homePath.empty() && pos!=string::npos )
        globalPath = localPath.replace( pos, homePath.length(), "~" );
    
    return globalPath;
}

string toLocalPath( string globalPath )
{
    string localPath = globalPath;
    size_t pos       = globalPath.find( "~" );

    if( !homePath.empty() && pos!=string::npos )
        localPath = globalPath.replace( pos, 1, homePath );
    
    return localPath;
}

QString toGlobalPath( QString localPath  )
{
    return QString( toGlobalPath( localPath.toStdString() ).c_str() );
}

QString toLocalPath( QString globalPath )
{
    return QString( toLocalPath( globalPath.toStdString() ).c_str() );
}

time_t fileLastModification( const char* filename )
{
    struct stat st;

    if( stat( filename, &st )==0 )
        return st.st_mtime;

    return -1;
}

void kMajority(InputArray data, OutputArray centers, int maxIterations = 20)
{
    Mat matCenters = centers.getMat();
    Mat matData = data.getMat();
    vector<int> labels(matData.rows); //the cluster of each keypoint
    vector<int> clusterSizes(matCenters.rows);

    // create random centers
    std::srand(std::time(0));
    for (int i=0;i<centers.rows();i++)
     for (int j=0;j<centers.cols();j++)
         matCenters.at<uchar>(i,j) = std::rand();

    bool centroidsChanged = true;
    while(centroidsChanged && maxIterations-- > 0)
    {
        centroidsChanged = false;
        // assign keypoints to clusters
        for (int keypointIndex = 0; keypointIndex<matData.rows; keypointIndex++)
        {
            int minDistance = INT_MAX;
            int index = -1; // the closer cluster
            for (int clusterIndex = 0; clusterIndex<matCenters.rows; clusterIndex++)
            {
                 int distance = 0;

                 // to calculate the distance of all the bitstream we have to sum every single element
                 // because the bitstream is split in 32 uchars
                 for (int i=0; i<matData.cols; i++)
                 {
                     distance += hamming(matData.at<uchar>(keypointIndex,i),matCenters.at<uchar>(clusterIndex,i));
                 }
                 if (distance < minDistance)
                 {
                     minDistance = distance;
                     index = clusterIndex;
                 }

            }
            if (index > -1)
            {
                labels[keypointIndex] = index;
            }
        }

        // we use a single vector of vectors to accumulate all the votes
        vector < vector < int > > v(matCenters.rows);
        for (int i=0; i<v.size(); i++)
        {
            v[i].resize(matCenters.cols*sizeof(uchar)*8);
        }

        // populate v with the votes and calculate the size of clusters
        for (int i = 0; i<matData.rows; i++)
        {
            int vIndex = 0;
            for (int j=0; j<matData.cols; j++)
            {
                for (int bit = 0; bit<8; bit++)
                {
                    v[labels[i]][vIndex++] += (matData.at<uchar>(i,j) >> bit) & 0x01;
                }
            }
            for (int clusterIndex=0; clusterIndex<matCenters.rows; clusterIndex++)
            {
                if (labels[i] == clusterIndex)
                {
                    clusterSizes[clusterIndex]++;
                    break;
                }
            }
        }
        // calculate new centroids:
        // for each centroid check for each bit if more than half of votes wants to change it
        for (int i=0; i<matCenters.rows; i++)
        {
            int vIndex = 0;
            for (int j=0; j<matCenters.cols; j++)
            {
                for (int bit=0; bit<8; bit++)
                {
                    if (v[i][vIndex++] > clusterSizes[i]/2)
                    {
                        matCenters.at<uchar>(i,j) |= 0x80 >> bit;
                        centroidsChanged = true;
                    }
                }
            }
        }
    }
}


int hamming(uchar a, uchar b)
{
    return __builtin_popcount(a^b);
}
