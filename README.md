# NEAT_CarAI

To run this, you need to install the Unity3d 2020.1.15f1 and open this project with it.
Then do the following:
1. Select the scene Assets/#Scenes/Scene1
2. Check that the object Car>VPP Sport Coup has a script CarAI assigned and enabled.
3. Check that the object Neat Manager exists in the root and has both scripts enabled: NEAT_Manager and GUI_Helper. And also if it has a link to object with the CarAI. 
  *note: use another screipt if you want another implementation of NEAT, HyperNEAT_Manager - for HyperNEAT and NEAT_withNoveltySearch - for NEAT with novelty search
4. Check that the object ConfigManager exists in the  root and has a script enabled: ConfigReader
5. Check that Path in the root exists and is assigned into the CarAI
6. You can now run the project by starting the simulation.
