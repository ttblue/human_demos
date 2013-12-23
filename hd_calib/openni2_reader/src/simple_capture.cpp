#include <OpenNI.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv/cvwimage.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <sensor_msgs/image_encodings.h>
#include <iostream>


int main(int argc, char** argv) {
	openni::Status rc = openni::STATUS_OK;

	openni::Device device;
	openni::VideoStream color;
	const char* deviceURI = openni::ANY_DEVICE;

	if (argc > 1)
		deviceURI = argv[1];

	rc = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());

	rc = device.open(deviceURI);
	if (rc != openni::STATUS_OK) {
		printf("Error: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		return 1;
	}

	rc = color.create(device, openni::SENSOR_COLOR);

	if (rc == openni::STATUS_OK) {

		// turn off auto exposure & auto white balance & mirror the image:
		// to access other features, see the NiViewer example in OpenNI2/Source/Tools/NiViewer/NiViewer.cpp
		color.getCameraSettings()->setAutoExposureEnabled(true);
		color.getCameraSettings()->setAutoWhiteBalanceEnabled(true);
		color.setMirroringEnabled(false);
		//color.getCameraSettings()->setExposure(65);

		const openni::SensorInfo* colorSensorInfo = device.getSensorInfo(openni::SENSOR_COLOR);

		// print out the supported video modes for color stream:
		const openni::Array<openni::VideoMode>& supportedModes = colorSensorInfo->getSupportedVideoModes();
		for (int i = 0; i < supportedModes.getSize(); ++i) {
			const openni::VideoMode* mode = &supportedModes[i];
			std::cout << i <<". "<< mode->getResolutionX() << "x"<<mode->getResolutionY();
			std::cout << " | fps : "<< mode->getFps() << " | format : "<<mode->getPixelFormat()<<std::endl;
		}
		// set the color mode to 1280x1024 @ 30 fps, RGB:
		color.setVideoMode(colorSensorInfo->getSupportedVideoModes()[15]);

		// start streaming:
		rc = color.start();
		if (rc != openni::STATUS_OK) {
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}

	openni::VideoFrameRef colorFrame;
	cv::Mat frame;
	frame.create(1024, 1280, CV_8UC3);
	
	cv::namedWindow("RGB Image", 1);

	// ros-stuff:
	ros::init(argc, argv, "simple_capture");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
  	image_transport::Publisher pub = it.advertise("camera_rgb_hires/image", 1);
	cv_bridge::CvImage cv_img;
	ros::Rate loop_rate(30);
  		
  	int seq = 0;

  	std::cout << "Publishing image on camera_rgb_hires/image"<<std::endl;
  	while (nh.ok()) {
    
		color.readFrame(&colorFrame);
		const openni::RGB888Pixel* imageBuffer = (const openni::RGB888Pixel*)colorFrame.getData();

		memcpy( frame.data, imageBuffer, 3*colorFrame.getHeight()*colorFrame.getWidth()*sizeof(uint8_t) );
		//cv::cvtColor(frame,frame,CV_BGR2RGB); //this will put colors right
		
		cv_img.image       = frame;
		//cv_img.encoding    = sensor_msgs::image_encodings::TYPE_8UC3;
		cv_img.encoding    = sensor_msgs::image_encodings::RGB8;
		cv_img.header.seq  = seq;
		cv_img.header.stamp = ros::Time::now();
		cv_img.header.frame_id = "camera_rgb_hires";

		pub.publish(cv_img.toImageMsg());

		seq += 1;
	
		ros::spinOnce();
  		loop_rate.sleep();

		//cv::imshow("RGB Image", frame);
		//cv::waitKey(25);
	}
	return 0;
}