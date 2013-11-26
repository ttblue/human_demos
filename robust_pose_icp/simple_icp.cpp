#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/median_filter.h>


// Types
typedef pcl::PointXYZRGB PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

void saveCloudAsText(PointCloudT::Ptr cloud, std::string fname) {
  
}


int main (int argc, char** argv) {
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
  PointCloudT::Ptr object_filt (new PointCloudT);
  PointCloudT::Ptr scene_filt (new PointCloudT);
  PointCloudT::Ptr Final(new PointCloudT);

 
  // Get input object and scene
  if (argc != 3)  {
    pcl::console::print_error ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
    return (1);
  }
 
  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointNT> (argv[1], *object) < 0 ||
      pcl::io::loadPCDFile<PointNT> (argv[2], *scene) < 0) {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }
  
  /**std::cout << "Median filtering... "<<std::endl;  // <<<<<<<< can only do median filtering on organized point-clouds
  pcl::MedianFilter<PointNT> medf;
  medf.setWindowSize(5);
  medf.setMaxAllowedMovement(0.001);
  medf.setInputCloud (object);
  medf.filter (*object_filt);
  medf.setInputCloud (scene);
  medf.filter (*scene_filt);*/

  /**pcl::StatisticalOutlierRemoval<PointNT> sor;
  sor.setMeanK (50);
  sor.setStddevMulThresh (3.0);
  sor.setInputCloud (Final);
  sor.filter (*object_filt);
  sor.setInputCloud (scene);
  sor.filter (*scene_filt);*/

  object_filt = object;
  scene_filt  = scene;
  
  pcl::IterativeClosestPoint<PointNT, PointNT> icp;
  icp.setInputSource(object_filt);
  icp.setInputTarget(scene_filt);
  icp.align(*Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

  // save the aligned cloud:
  pcl::io::savePCDFileASCII("c1_icped.pcd", *Final);

  // Show alignment
  pcl::visualization::PCLVisualizer visu("Alignment");

    pcl::visualization::PointCloudColorHandlerRGBField<PointNT> scene_rgb(scene_filt);
    //visu.addPointCloud (scene, scene_rgb, "scene");
    visu.addPointCloud (scene, ColorHandlerT(scene_filt, 0.,0.,255.), "scene");

    pcl::visualization::PointCloudColorHandlerRGBField<PointNT> Final_rgb(Final);
    //visu.addPointCloud (Final, Final_rgb, "final");
    visu.addPointCloud (Final, ColorHandlerT(Final, 255.,0.,0.), "final");

  visu.spin ();

 return (0);
}
