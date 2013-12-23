#include <OpenNI.h>
#include <iostream>

#include <cv.h>
#include <highgui.h>
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
		color.getCameraSettings()->setAutoExposureEnabled(false);
		color.getCameraSettings()->setAutoWhiteBalanceEnabled(true);
		color.setMirroringEnabled(false);
		color.getCameraSettings()->setExposure(65);

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
	cv::namedWindow("RGB Image", 1);

	while (true) {
		color.readFrame(&colorFrame);
		const openni::RGB888Pixel* imageBuffer = (const openni::RGB888Pixel*)colorFrame.getData();

		frame.create(colorFrame.getHeight(), colorFrame.getWidth(), CV_8UC3);
		memcpy( frame.data, imageBuffer, 3*colorFrame.getHeight()*colorFrame.getWidth()*sizeof(uint8_t) );

		cv::cvtColor(frame,frame,CV_BGR2RGB); //this will put colors right
		cv::imshow("RGB Image", frame);
		cv::waitKey(25);
	}
	return 0;
}
