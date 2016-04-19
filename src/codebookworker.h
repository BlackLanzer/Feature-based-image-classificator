#ifndef CODEBOOKWORKER_H
#define CODEBOOKWORKER_H

#include <QThread>
#include <vector>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

class CodebookWorker : public QObject
{
     Q_OBJECT
     QThread workerThread;
     bool    abortRequested;

 public:
     CodebookWorker();
     template<typename descType>
     void detectAndCluster(QString method, Feature2D* detector, Feature2D* extractor, vector < pair < int, string > > fileList, QString inputFolder, QString outputFile, int splitPercent, int minHessian, int clusterSize );

 public slots:
     void doWork( QString method, QString inputFolder, QString outputFile, int splitPercent, int minHessian, int clusterSize );


 signals:
     void imgReadingDone();
     void clusteringDone();

     // common signals
     void processingDone   ();
     void setupProgressBar ( QString inputFolder, int splitPercent );
     void updateProgressbar();
     void throwError       ( const QString& error );
     void checkAbort       ( bool* abortRequested );
};

#endif // CODEBOOKWORKER_H
