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

// Types
typedef pcl::PointXYZRGB PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;


int main (int argc, char** argv) {
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_aligned (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
  
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

  pcl::IterativeClosestPoint<PointNT, PointNT> icp;
  icp.setInputSource(object);
  icp.setInputTarget(scene);
  PointCloudT::Ptr Final(new PointCloudT);
  icp.align(*Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

  // Show alignment
  pcl::visualization::PCLVisualizer visu("Alignment");

    pcl::visualization::PointCloudColorHandlerRGBField<PointNT> scene_rgb(scene);
    //visu.addPointCloud (scene, scene_rgb, "scene");
    visu.addPointCloud (scene, ColorHandlerT(scene, 0.,0.,255.), "scene");


    pcl::visualization::PointCloudColorHandlerRGBField<PointNT> Final_rgb(Final);
    //visu.addPointCloud (Final, Final_rgb, "final");
    visu.addPointCloud (Final, ColorHandlerT(Final, 255.,0.,0.), "final");

  visu.spin ();

 return (0);
}
