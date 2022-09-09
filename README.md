# Engagement Estimation with Mediapipe in Python
Engagement estimation with MediaPipe and served on Flask.
MediaPipe documentation in https://google.github.io/mediapipe/

## Setting up environment (on Conda Mac) and run the engagement estimation app
1. Run `python main.py` to start the service on `localhost:5050`
2. Click `Start` button to start the engagement estimation. The first click will create a csv file to record the engagement level.
3. Click `Stop` button to pause the engagement estimation. 
4. Click `Start` button again to continue the estimation. It will continue the record on the same file created on step 3.

<img width="865" alt="EE_Idle" src="https://user-images.githubusercontent.com/49575067/189369604-24232b18-97c3-4f65-b6ac-df21437af891.png">
<img width="867" alt="EE_NormalEngage" src="https://user-images.githubusercontent.com/49575067/189369612-3acdbaf2-95a9-417f-b215-f14385b5aa51.png">
<img width="864" alt="EE_NotEngage" src="https://user-images.githubusercontent.com/49575067/189369613-9c648beb-1b15-4669-b1a3-636cdd828b88.png">
<img width="865" alt="EE_VeryEngage" src="https://user-images.githubusercontent.com/49575067/189369616-087490f5-b942-4281-ac28-f949cd8eba46.png">
