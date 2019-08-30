#include "hogsvm.h"
#include "tinyxml2.h"
#include <iostream>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <iomanip>


using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace tinyxml2;


HOGSVM::HOGSVM(int _bin, 
				Size _cellSize, 
				Size _blockSize, 
				Size _blockStride,
				Size _windowStride,
				bool _multiScale,
				float _scale) {

	hog.nbins = _bin;
	hog.cellSize = _cellSize;
	hog.blockSize = _blockSize;
	hog.blockStride = _blockStride;

	windowStride = _windowStride;
	multiScaleFlag = _multiScale;
	detectorScale = _scale;
	posCount = 0;
	negCount = 0;

	truePos = posPredict = posActual = 0;
}


HOGSVM HOGSVM::createMultiScale(int _bin, 
								Size _cellSize, 
								Size _blockSize, 
								Size _blockStride,
								Size _windowStride,
								float _scale) {

	HOGSVM detector(_bin, 
					_cellSize, 
					_blockSize, 
					_blockStride,
					_windowStride,
					true,
					_scale);

	return detector;
}


HOGSVM HOGSVM::create(int _bin, 
						Size _cellSize, 
						Size _blockSize, 
						Size _blockStride,
						Size _windowStride) {

	HOGSVM detector(_bin, 
					_cellSize, 
					_blockSize, 
					_blockStride,
					_windowStride,
					false);

	return detector;
}


void HOGSVM::loadTrainingSet(const char* annotation,
							const char* neg) {

	
	chooseWindowSize(annotation);
	loadPositiveData(annotation);
	loadNegativeData(neg);
	
}


int HOGSVM::checkWindowSize(const char* annotation) {
	string path(annotation);
	path = path.substr(0, path.find_last_of("/")) + '/';

	XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile(annotation);
 
    if (eResult) {
        cerr << XMLDocument::ErrorIDToName(eResult) << endl;
        exit(1);
    }
 
    XMLElement *root = xmlDoc.RootElement();
 
    XMLElement *image = root->FirstChildElement("images")
  						    ->FirstChildElement("image");
    const char *filename;
 	Mat img;
 	int counter = 0;
    while (image != nullptr) {
    	filename = nullptr;
        XMLError eResult = image->QueryStringAttribute("file", &filename);

        if (eResult) {
        	cerr << XMLDocument::ErrorIDToName(eResult) << endl;
        	continue;
        }


 		img = imread(path + filename);

        Rect bb;
        int ignore;

        XMLElement *box = image->FirstChildElement("box");
        while (box != nullptr) {
        	ignore = 0;
        	box->QueryAttribute("ignore", &ignore);

        	if (!ignore) {
        		counter++;
        	
            	box->QueryAttribute("top", &(bb.y));
            	box->QueryAttribute("left", &(bb.x));
            	box->QueryAttribute("width", &(bb.width));
            	box->QueryAttribute("height", &(bb.height));
 
 				windowSize.width += bb.width;
 				windowSize.height += bb.height;
        	}
        	
            box = box->NextSiblingElement();
        }
        image = image->NextSiblingElement();
    }

    windowSize.width /= counter;
    windowSize.height /= counter;

    return counter;
}


void HOGSVM::chooseWindowSize(const char* annotation) {
	
	checkWindowSize(annotation);
	int factor = 1;

    clog << "avg. window: " << windowSize << endl;

    if (multiScaleFlag)
    	factor = 2;

	float origRatio = float(windowSize.width) / windowSize.height;
	Size smallSize(windowSize.width / (8 * factor) * 8, windowSize.height / (8 * factor) * 8);
	Size bigSize((windowSize.width / (8 * factor) + 1) * 8, (windowSize.height / (8 * factor) + 1) * 8);
	float smallSizeRatio = float(smallSize.width) / smallSize.height;
	float bigSizeRatio = float(bigSize.width) / bigSize.height;

	if (abs(origRatio-smallSizeRatio) > abs(origRatio-bigSizeRatio)) {
		windowSize = bigSize;
	}
	else {
		windowSize = smallSize; 
	}

	clog << "window: " << windowSize << endl;
	hog.winSize = windowSize;
}


void HOGSVM::loadPositiveData(const char* annotation) {
	clog << "Loading positive data...";
	int counter = samplePositiveImages(annotation, true);

	posCount += counter;
	trainLabels.insert(trainLabels.end(), counter, +1);
	clog << "[Done]" << endl;
}


int HOGSVM::samplePositiveImages(const char* annotation, bool sampling) {
	string path(annotation);
	path = path.substr(0, path.find_last_of("/")) + '/';

	XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile(annotation);
 
    if (eResult) {
        cerr << XMLDocument::ErrorIDToName(eResult) << endl;
        exit(1);
    }
 
    XMLElement *root = xmlDoc.RootElement();
 
    XMLElement *image = root->FirstChildElement("images")
  						    ->FirstChildElement("image");
    const char *filename;
 	Mat img;
 	int counter = 0;
    while (image != nullptr) {
    	filename = NULL;
        XMLError eResult = image->QueryStringAttribute("file", &filename);

        if (eResult) {
        	cerr << XMLDocument::ErrorIDToName(eResult) << endl;
        	continue;
        }


 		img = imread(path + filename);

        Rect bb;
        int ignore;

        XMLElement *box = image->FirstChildElement("box");
        while (box != nullptr) {
        	ignore = 0;
        	box->QueryAttribute("ignore", &ignore);

        	if (!ignore) {
        		counter++;
        	
            	box->QueryAttribute("top", &(bb.y));
            	box->QueryAttribute("left", &(bb.x));
            	box->QueryAttribute("width", &(bb.width));
            	box->QueryAttribute("height", &(bb.height));
 
 				if (sampling) {
 					Mat roi = img(bb);
 					computeHOG(roi);
            	}
            	else {
            		vector<Rect> detections = detect(img, 1.15);
            		posPredict += detections.size();
            		truePos += computeIOU(detections, bb);
            	}
        	}
        	
            box = box->NextSiblingElement();
        }
        image = image->NextSiblingElement();
    }

    clog << "Positive size: " << counter << endl;
    clog << "Total: " << gradientList.size() << endl;
    return counter;
}


void HOGSVM::loadNegativeData(const char* path) {
	clog << "Loading negative data...";
	if (path != NULL) {

		int count = sampleNegativeImages(path);
		negCount += count;

		trainLabels.insert(trainLabels.end(), count, -1);
	}
	clog << "[Done]" << endl;
}


int HOGSVM::sampleNegativeImages(const char* dirname) {

	Mat img;
	vector<String> files;
	Rect box;
	int counter = 0;

	glob(dirname, files, true);

	box.width = windowSize.width;
	box.height = windowSize.height;

	const int size_x = box.width;
	const int size_y = box.height;

	for (size_t i = 0; i < files.size(); i++) {
		img = imread(files[i]);
		if (img.empty()) {
			cerr << files[i] << " is invalid!" << endl;
			continue;
		}

		srand((unsigned int)time(NULL));

		
		if (img.cols > box.width && img.rows > box.height) {
			counter++;

			box.x = rand() % (img.cols - size_x);
			box.y = rand() % (img.rows - size_y);

			Mat roi = img(box);
			computeHOG(roi);
		}
	}

	clog << "Negative set size: " << counter << endl;
	clog << "Total: " << gradientList.size() << endl;
	return counter;
}


void HOGSVM::computeHOG(Mat& roi) {

	Mat gray;
	vector<float> descriptors;

	resize(roi, roi, windowSize);
	cvtColor(roi, gray, COLOR_BGR2GRAY);

	hog.compute(gray, descriptors, windowStride, Size(0,0));
	gradientList.push_back(Mat(descriptors).clone());
}


void HOGSVM::train(const char* pathToHardTrainDataSet) {
	if (posCount > 0 && negCount > 0) {
		softTrain(1.0);
		hardTrain(pathToHardTrainDataSet);
	}
	else {
		cerr << "No training data!" << endl;
	}
}


void HOGSVM::softTrain(float C) {
	
	prepareData();

	clog << "Soft training...";

	svm = SVM::create();
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 
    								1000, 
    								1e-3));

	svm->setKernel(SVM::LINEAR);

    svm->setNu(0.5);
    svm->setC(C);

    svm->setType(SVM::NU_SVR);

	svm->train(trainData, ROW_SAMPLE, trainLabels);
    hog.setSVMDetector(getLinearSVC(svm));
    clog << "[Done]" << endl;
}


void HOGSVM::hardTrain(const char* path) {
	hardNegativeMine(path);

	prepareData();

	clog << "Hard training...";
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	clog << "[Done]" << endl;

	hog.setSVMDetector(getLinearSVC(svm));
	cout << "C: " << svm->getC() << " Nu: " << svm->getNu() << endl;
}


void HOGSVM::hardNegativeMine(const char* dirname) {
	clog << "Hard negative mining on negative images...";

	Mat img;
	vector<String> files;
	Rect box;

	vector<Rect> detections;
	vector<double> foundWeights;
	int counter = 0;

	glob(dirname, files, true);

	for (size_t i = 0; i < files.size(); i++) {
		img = imread(files[i]);
		if (img.empty()) {
			cerr << files[i] << " is invalid!" << endl;
			continue;
		}

		detections.clear();
		foundWeights.clear();

		hog.detectMultiScale(img, 
							detections, 
							foundWeights, 
							0, windowStride, Size(0,0), detectorScale);
	
		for (size_t j = 0; j < detections.size(); j++) {
			if (foundWeights[j] < 0.5) {
				continue;
			}

			counter++;

			Mat detection = img(detections[j]).clone();
			resize(detection, detection, windowSize, 0, 0, cv::INTER_CUBIC);

			computeHOG(detection);
		}
	}

	negCount += counter;
	trainLabels.insert(trainLabels.end(), counter, -1);

	clog << "[Done]" << endl;
}


void HOGSVM::prepareData() {

	const int rows = (int)gradientList.size();
	const int cols = (int)gradientList[0].rows;

	Mat tmp(1, cols, CV_32FC1);
	trainData = Mat(rows, cols, CV_32FC1);

	for(size_t i = 0; i < gradientList.size(); i++) {
		CV_Assert(gradientList[i].cols == 1);

		transpose(gradientList[i], tmp);
		tmp.copyTo(trainData.row(int(i)));
	}

}


vector<float> HOGSVM::getLinearSVC(const Ptr<SVM>& svmCls) {
	Mat sv = svmCls->getSupportVectors();
	const int sv_total = sv.rows;

	Mat alpha, svidx;
	double rho = svmCls->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) 
    		  || (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(sv.type() == CV_32F);

    vector<float> hog_detector(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = float(-rho);

    return hog_detector;
}


void HOGSVM::saveModel(const String& path) {
	clog << "Saving model...";
    hog.save(path);
    clog << "[Done]" << endl;
}


void HOGSVM::loadModel(const String& path) {
	clog << "Loading model...";
	hog.load(path);
	clog << "[Done]" << endl;
}


int HOGSVM::testVideo(const char* filename, float scale) {

	clog << "Testing trained detector..." << endl;
	
	VideoCapture cap;
	
	cap.open(filename);
	int width = int(cap.get(CAP_PROP_FRAME_WIDTH) * scale);
	int height = int(cap.get(CAP_PROP_FRAME_HEIGHT) * scale);
	cout << "frame width: " << width << endl;
	cout << "frame height: " << height << endl;
    Mat img;
    size_t i;
    for(i = 0;; i++) {
        if (cap.isOpened()) {
            cap >> img;
        }

        if (img.empty()) {
            break;
        }

        resize(img, img, Size(width, height));
        vector<Rect> detections;

        if (i % 3 == 0) {
        	detections = detect(img, 1.15);

        	for (size_t j = 0; j < detections.size(); j++) {
        		Scalar color(0, 255, 0);
            	rectangle(img, detections[j], color, 2);
        	}
    	}
		
        imshow("frame", img);
        if (waitKey(1) == 27) {
            break;
        }
    }
    return i;
}


void HOGSVM::showInfo() {
	clog << "Pos size: " << posCount << endl;
	clog << "Neg size: " << negCount << endl;
}


vector<Rect> HOGSVM::detect(const Mat& image, float scale) {
	vector<Rect> rawRectDetections;
	vector<Point> rawPointDetections;
	vector<Rect> detections;
	vector<double> rawFoundWeights;
    vector<double> foundWeights;

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    if (multiScaleFlag) {

    	if (scale == 1) {
    		scale = detectorScale;
    	}

    	hog.detectMultiScale(gray, 
							rawRectDetections, 
							rawFoundWeights, 
							0, 
							windowStride,
							Size(0,0),
							scale);

    	for (size_t i = 0; i < rawRectDetections.size(); i++) {
    		if (rawFoundWeights[i] > 0.6) {
    			detections.push_back(rawRectDetections[i]);
    			foundWeights.push_back(rawFoundWeights[i]);
    		}
    	}
    }
    else {
    	hog.detect(gray,
    				rawPointDetections,
    				rawFoundWeights,
    				0,
    				windowStride,
    				Size(0,0));

    	for (size_t i = 0; i < rawPointDetections.size(); i++) {
    		if (rawFoundWeights[i] > 0.6) {
    			detections.push_back(Rect(rawPointDetections[i], hog.winSize));
    			foundWeights.push_back(rawFoundWeights[i]);
    		}
    	}
    }

    detections = nonMaxSuppression(foundWeights, detections);

    return detections;
}


vector<Rect> HOGSVM::nonMaxSuppression(const vector<double>& confidences, 
										const vector<Rect>& boxes,
										float overlapThresh) {

	Mat x1, y1, x2, y2;
	vector<Rect> pick;
	
	for (size_t i = 0; i < boxes.size(); i++) {
		x1.push_back(boxes[i].x);
		y1.push_back(boxes[i].y);
		x2.push_back(boxes[i].x + boxes[i].width);
		y2.push_back(boxes[i].y + boxes[i].height);
	}

	Mat area = (x2 - x1 + 1).mul(y2 - y1 + 1);
	vector<size_t> idxs = argsort(confidences);

	size_t last;

	vector<size_t>::iterator begin = idxs.begin();
	vector<size_t>::iterator end = idxs.end();

	while (begin < end) {

		vector<size_t> idxsSample(begin, end);

		last = end - begin - 1;
		size_t id = idxs[last];
		pick.push_back(boxes[id]);

		vector<size_t> suppress;
		suppress.push_back(last);

		for (size_t k = 0; k < last; k++) {
			size_t j = idxs[k];

			int xx1 = max(x1.at<int>(id), x1.at<int>(j));
			int yy1 = max(y1.at<int>(id), y1.at<int>(j));
			int xx2 = min(x2.at<int>(id), x2.at<int>(j));
			int yy2 = min(y2.at<int>(id), y2.at<int>(j));

			int w = max(0, xx2 - xx1 + 1);
			int h = max(0, yy2 - yy1 + 1);

			double overlap = float(w * h) / area.at<int>(j);

			if (overlap > overlapThresh) {
				suppress.push_back(k);
			}
		}
		for (const auto& e : suppress) {
			end = remove(begin, end, idxsSample[e]);
		}
	}
	return pick;
}


template<typename T>
	vector<size_t> HOGSVM::argsort(const vector<T>& input) {
	vector<size_t> idxs(input.size());
	iota(idxs.begin(), idxs.end(), 0);
	    
	sort(idxs.begin(), 
	  		idxs.end(), 
	  		[&](size_t a, size_t b) {
	            return input[a] < input[b];
	        });
	                
	return idxs;
}


void HOGSVM::evaluate(const char* annotation) {
	posActual = samplePositiveImages(annotation, false);

	float precision = float(truePos) / posPredict;
	float recall = float(truePos) / posActual;
	float fscore = 2*precision*recall / (precision + recall);

	cout << "Precision: " << setprecision(5) << precision << endl;
	cout << "Recall: " << setprecision(5) << recall << endl;
	cout << "F1 score: " << setprecision(5) << fscore << endl;
}


int HOGSVM::computeIOU(const vector<Rect>& detections, const Rect& bb) {
	int overlapArea;
	int unionArea;
	double IoU = 0;

	for (const auto& r : detections) {
		overlapArea = (r & bb).area();
		unionArea = r.area() + bb.area() - overlapArea;

		double temp = float(overlapArea) / unionArea;
		if (IoU < temp) {
			IoU = temp;
		}
	}

	return (IoU > 0.5) ? 1 : 0;
}

