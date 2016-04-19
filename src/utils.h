#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <cstring>
#include <vector>
#include <qt4/QtCore/qstring.h>
#include "logstream.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include <opencv2/flann.hpp>

using namespace std;
using namespace cv;

int num_dirs( const char* path, vector< string >& dir_list );

int num_images( const char* path );
int num_images( const char* path, vector< string >& file_list );

float strtofloat( const string& what );

extern LogStream logStream;
extern string    homePath;

string toGlobalPath( string localPath  );
string toLocalPath ( string globalPath );

QString toGlobalPath( QString localPath  );
QString toLocalPath ( QString globalPath );

time_t fileLastModification(const char* filename );

void kMajority(InputArray data, OutputArray centers, int maxIterations);
int hamming(uchar a, uchar b);


#endif // UTILS_H
