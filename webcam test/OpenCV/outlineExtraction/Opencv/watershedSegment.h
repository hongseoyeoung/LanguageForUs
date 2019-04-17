#pragma once
class WatershedSegmenter {

private:

	cv::Mat markers;

public:

	void setMarkers(const cv::Mat& markerImage) {

		markerImage.convertTo(markers, CV_32S); //32비트마커스로자료형변환

	}

	cv::Mat process(const cv::Mat& image) {

		cv::watershed(image, markers);

		//분할결과를markers에저장

		return markers;

	}

	cv::Mat getSegmentation() {

		cv::Mat tmp;

		markers.convertTo(tmp, CV_8U); return tmp;

	}

	cv::Mat getWatersheds() {

		cv::Mat tmp;

		markers.convertTo(tmp, CV_8U, 255, 255); return tmp;

	}

};