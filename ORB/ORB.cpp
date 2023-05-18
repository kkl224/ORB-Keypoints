#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
using namespace cv;

int main () {
	//read image
    const Mat img1 = imread("../1.JPG");
	if (!img1.data) {
		printf("could not load image...\n");
		return -1;
	}
	// imshow("img1", img1);

	const Mat img2 = imread("../2.JPG");
	if (!img2.data) {
		printf("could not load image...\n");
		return -1;
	}
	// imshow("img2", img2);
	
    //Initialization of ORB
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

	//Detect ORB features 
    Ptr<FeatureDetector> detector = ORB::create();
	//Compute descriptors
    Ptr<DescriptorExtractor> descriptor = ORB::create();
	//Match ORB features
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	
	//Detect ORB features 
    detector->detect(img1, keypoints_1);
    auto t1 = Clock::now();
    detector->detect(img2, keypoints_2);
    
    //Compute descriptors
    descriptor->compute(img1, keypoints_1, descriptors_1);
    auto t2 = Clock::now();
    std::cout << "Delta t2-t1: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()/1000000000.0
              << " seconds" << std::endl;
    descriptor->compute(img2, keypoints_2, descriptors_2);

    Mat outimg1;
    Mat outimg2;

	/*
	 * Draw keypoints
	 * @param img		Source image
	 * @param keypoints	Keypoints from the source image
	 * @param outImage	Output image
	 * @param color		Color of keypoints
	 * @param flags		Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags. See details above in drawMatches .
	 */
    drawKeypoints(img1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    drawKeypoints(img2, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    cout<<"object--number of keypoints_1:"<<keypoints_1.size()<<endl;
    cout<<"object--number of keypoints_2:"<<keypoints_2.size()<<endl;
    imshow("ORB1", outimg1);
    imshow("ORB2", outimg2);

    //Match ORB features
    vector<DMatch> matches;
	//Match descriptors of two images using BruteForce-Hamming
    matcher->match (descriptors_1, descriptors_2, matches);

    //Compute the max and min distance between keypoints
    double min_dist=1, max_dist=0;

    //Find the minimum and maximum distances between all matches, 
	//the distance between the most similar and least similar two sets of points
    for ( int i = 0; i < descriptors_1.rows; i++ ) {
        double dist = matches[i].distance;
        if (dist < min_dist) 
			min_dist = dist;
        if (dist > max_dist) 
			max_dist = dist;
    }
    
    // min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    // max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    printf ( "-- Max dist : %f \n", max_dist);
    printf ( "-- Min dist : %f \n", min_dist);

    //When the distance between the descriptors is greater than twice the minimum distance,
	//the match is considered wrong. But sometimes the minimum distance will be very small,
	//and an empirical value of 30 is set as the lower limit.
    std::vector<DMatch> good_matches;

	//Find the matching points that meet the requirements
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance < 0.4*max_dist) {
		//if (matches[i].distance < 30){
            good_matches.push_back (matches[i]);
        }
    }

    //Draw matching points
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img_match);
    drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, img_goodmatch);

    imshow ( "All Match", img_match );
    imshow ( "Good Match", img_goodmatch );
    cout<<"ORB number of all match: "<<matches.size()<<endl;
    cout<<"ORB number of good match: "<<good_matches.size()<<endl;

    imwrite("ORB keypoints_1.jpg", outimg1);
    imwrite("ORB keypoints_2.jpg", outimg2);
    imwrite("ORB all.jpg",img_match);
    imwrite("ORB good.jpg",img_goodmatch);

	waitKey(0);

    return 0;
}
