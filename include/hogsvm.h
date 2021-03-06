#ifndef GUARD_HOGSVM_H
#define GUARD_HOGSVM_H

#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"


class HOGSVM {
public:

	static HOGSVM createMultiScale(int = 9, 
									cv::Size = cv::Size(8,8), 
									cv::Size = cv::Size(16,16), 
									cv::Size = cv::Size(8,8),
									cv::Size = cv::Size(8,8),
									float = 1.15);

	static HOGSVM create(int = 9, 
						cv::Size = cv::Size(8,8), 
						cv::Size = cv::Size(16,16), 
						cv::Size = cv::Size(8,8),
						cv::Size = cv::Size(8,8));

	void loadTrainingSet(const char*, 
						const char*);

	void train(const char*);
	void evaluate(const char*);
	std::vector<cv::Rect> detect(const cv::Mat&,
								float = 1);

	int testVideo(const char*, float = 0.5);
	void saveModel(const cv::String&);
	void loadModel(const cv::String&);
	void showInfo();

private:
	bool multiScaleFlag = true;
	cv::Size windowSize;
	cv::Size windowStride;
	float detectorScale;
	int posCount, negCount;
	int truePos, posPredict, posActual;

	cv::Mat trainData;
	std::vector<int> trainLabels;
	std::vector<cv::Mat> gradientList;
	cv::Ptr<cv::ml::SVM> svm;
	cv::HOGDescriptor hog;
	
	HOGSVM(int, cv::Size, cv::Size, cv::Size, cv::Size, bool, float = 1.15);

	void loadPositiveData(const char*);
	void loadNegativeData(const char*);

	int samplePositiveImages(const char*, bool);
	int sampleNegativeImages(const char*);

	void computeHOG(cv::Mat&);

	int checkWindowSize(const char*);
	void chooseWindowSize(const char*);
	void prepareData();
	std::vector<float> getLinearSVC(const cv::Ptr<cv::ml::SVM>&);
	void softTrain(float);
	void hardTrain(const char*);

	void hardNegativeMine(const char*);

	std::vector<cv::Rect> nonMaxSuppression(const std::vector<double>&, 
											const std::vector<cv::Rect>&,
											float = 0.3);

	int computeIOU(const std::vector<cv::Rect>&, 
					const cv::Rect&);

	template<typename T>
	std::vector<std::size_t> argsort(const std::vector<T>&);
};

#endif