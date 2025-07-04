IsaacLab UR5E Visual Pushing & Grasping

This repository provides an IsaacLab-based implementation of the Visual Pushing and Grasping (VPG) algorithm, adapted from andyzeng/visual-pushing-grasping, and tailored for the UR5E robot.

Branch Structure
This branch contains two key folders:
extensions/omni.isaac.lab_assets
extensions/omni.isaac.lab_tasks

Replace the corresponding folders in your original IsaacLab installation with these two folders to enable the UR5E VPG functionality.

Required Assets
You need to download the UR5E USD asset folder from Google Drive:https://drive.google.com/drive/folders/1J4ON7Ai1yEr5tw8R47X9YsnN7pDK9Usm?usp=drive_link

How to Run
1 Replace Folders:
  Copy the two folders from this branch into your IsaacLab source tree, overwriting the original ones:
2 Download and Place UR5E USD Assets:
  Download from the Google Drive link and place as described above.
3 Modify Asset Paths:
  If necessary, update the USD file paths in your configuration files to point to the new UR5E asset location.
4 Run the Main Script:
  Execute the main training/testing script:   python /home/shi/IsaacLab/source/standalone/sky/fyp_sky/visual_pushing_grasping.py


Adjust any script arguments as needed for your experiment.
Project Background
  This project is an IsaacLab adaptation of the original Visual Pushing and Grasping implementation by Andy Zeng et al.
  It enables training and evaluation of VPG policies on the UR5E robot within the IsaacLab simulation environment.
  If you use this code or assets, please also cite the original VPG project and respect the licenses of all dependencies.
