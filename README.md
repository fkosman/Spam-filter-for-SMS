# MiniTorch Module 2

<img src="https://minitorch.github.io/_images/match.png" width="100px">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2.html

This assignment requires the following files from the previous assignments.

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py


# Task #5

## Simple Dataset

![image](https://user-images.githubusercontent.com/89897166/137573896-8c5ea8fd-7a09-4211-b24b-d37b8d89b0aa.png)

* Parameters Used: 
    * Learning Rate = 0.5
    * Hidden Layers = 2
    * Epochs = 500
    
# 

### Data logs:


Epoch: 0/500, loss: 0, correct: 0

Epoch: 10/500, loss: 29.840336450052067, correct: 24

Epoch: 20/500, loss: 25.519717664706096, correct: 24

Epoch: 30/500, loss: 23.767723418009776, correct: 38

Epoch: 40/500, loss: 22.389823850286422, correct: 40

Epoch: 50/500, loss: 21.162539638749596, correct: 42

Epoch: 60/500, loss: 19.933655962955086, correct: 44

Epoch: 70/500, loss: 18.682705427328127, correct: 45

Epoch: 80/500, loss: 17.47180253514649, correct: 48

Epoch: 90/500, loss: 16.235161494971383, correct: 48

Epoch: 100/500, loss: 14.97758192081333, correct: 49

Epoch: 110/500, loss: 13.77949942724105, correct: 49

Epoch: 120/500, loss: 12.662170140799917, correct: 49

Epoch: 130/500, loss: 11.673520673622304, correct: 49

Epoch: 140/500, loss: 10.855381062825845, correct: 49

Epoch: 150/500, loss: 10.153992443378577, correct: 49

Epoch: 160/500, loss: 9.525673757255648, correct: 49

Epoch: 170/500, loss: 8.959578832166658, correct: 50

Epoch: 180/500, loss: 8.44729115412103, correct: 50

Epoch: 190/500, loss: 7.9814387904164, correct: 50

Epoch: 200/500, loss: 7.5580341980226065, correct: 50

Epoch: 210/500, loss: 7.174451401399509, correct: 50

Epoch: 220/500, loss: 6.82908944197127, correct: 50

Epoch: 230/500, loss: 6.516002839898755, correct: 50

Epoch: 240/500, loss: 6.228424179821575, correct: 50

Epoch: 250/500, loss: 5.9620282688193065, correct: 50

Epoch: 260/500, loss: 5.715446805906516, correct: 50

Epoch: 270/500, loss: 5.48740319688396, correct: 50

Epoch: 280/500, loss: 5.27582417417125, correct: 50

Epoch: 290/500, loss: 5.078087876727537, correct: 50

Epoch: 300/500, loss: 4.892072824824457, correct: 50

Epoch: 310/500, loss: 4.719302847039739, correct: 50

Epoch: 320/500, loss: 4.557375239395674, correct: 50

Epoch: 330/500, loss: 4.404963914187302, correct: 50

Epoch: 340/500, loss: 4.261191499728911, correct: 50

Epoch: 350/500, loss: 4.125519399576077, correct: 50

Epoch: 360/500, loss: 3.999428309481115, correct: 50

Epoch: 370/500, loss: 3.8822774699273794, correct: 50

Epoch: 380/500, loss: 3.771995576038435, correct: 50

Epoch: 390/500, loss: 3.6686436694636533, correct: 50

Epoch: 400/500, loss: 3.5705842587838714, correct: 50

Epoch: 410/500, loss: 3.477487963258372, correct: 50

Epoch: 420/500, loss: 3.3890572542977733, correct: 50

Epoch: 430/500, loss: 3.3063580338558554, correct: 50

Epoch: 440/500, loss: 3.2246036893837475, correct: 50

Epoch: 450/500, loss: 3.1480466614078146, correct: 50

Epoch: 460/500, loss: 3.0748584535703203, correct: 50

Epoch: 470/500, loss: 3.0047852785252847, correct: 50

Epoch: 480/500, loss: 2.9376153867983956, correct: 50

Epoch: 490/500, loss: 2.8731611752322976, correct: 50

Epoch: 500/500, loss: 2.811252969938706, correct: 50

* Time per epoch: 0.128s

# 
# 


## Diagonal Dataset

![image](https://user-images.githubusercontent.com/89897166/137573905-5351f318-da64-4f79-830a-dd33fc3d8125.png)

* Parameters Used: 
    * Learning Rate = 0.5
    * Hidden Layers = 2
    * Epochs = 500
    
# 

### Data logs:


Epoch: 0/500, loss: 0, correct: 0

Epoch: 10/500, loss: 14.00689455333873, correct: 44

Epoch: 20/500, loss: 11.02914783572722, correct: 44

Epoch: 30/500, loss: 10.509930153272439, correct: 44

Epoch: 40/500, loss: 10.021078271647331, correct: 44

Epoch: 50/500, loss: 9.558824377607506, correct: 44

Epoch: 60/500, loss: 9.13920300776362, correct: 44

Epoch: 70/500, loss: 8.761170503250952, correct: 44

Epoch: 80/500, loss: 8.415329980102275, correct: 44

Epoch: 90/500, loss: 8.094919785872557, correct: 44

Epoch: 100/500, loss: 7.795518046965109, correct: 45

Epoch: 110/500, loss: 7.513714038420375, correct: 45

Epoch: 120/500, loss: 7.246824444016582, correct: 45

Epoch: 130/500, loss: 6.992687558100127, correct: 45

Epoch: 140/500, loss: 6.749836124147558, correct: 45

Epoch: 150/500, loss: 6.516883605603463, correct: 46

Epoch: 160/500, loss: 6.292713344302111, correct: 46

Epoch: 170/500, loss: 6.076601373136516, correct: 46

Epoch: 180/500, loss: 5.867897116890091, correct: 46

Epoch: 190/500, loss: 5.666051114506661, correct: 47

Epoch: 200/500, loss: 5.470540792159207, correct: 47

Epoch: 210/500, loss: 5.280979846821957, correct: 48

Epoch: 220/500, loss: 5.09709073217739, correct: 48

Epoch: 230/500, loss: 4.918677638150678, correct: 48

Epoch: 240/500, loss: 4.746013199239531, correct: 49

Epoch: 250/500, loss: 4.578835535518591, correct: 49

Epoch: 260/500, loss: 4.416901762468488, correct: 49

Epoch: 270/500, loss: 4.260089416378515, correct: 49

Epoch: 280/500, loss: 4.108408871136677, correct: 49

Epoch: 290/500, loss: 3.9624946601808197, correct: 49

Epoch: 300/500, loss: 3.821759888694956, correct: 49

Epoch: 310/500, loss: 3.686223475700629, correct: 49

Epoch: 320/500, loss: 3.5557296222495585, correct: 49

Epoch: 330/500, loss: 3.4300336270941405, correct: 49

Epoch: 340/500, loss: 3.308962877918085, correct: 49

Epoch: 350/500, loss: 3.1924018339037596, correct: 49

Epoch: 360/500, loss: 3.0802585605440878, correct: 49

Epoch: 370/500, loss: 2.9724484196396204, correct: 49

Epoch: 380/500, loss: 2.868979287102932, correct: 49

Epoch: 390/500, loss: 2.7699770802373638, correct: 49

Epoch: 400/500, loss: 2.675133505582819, correct: 49

Epoch: 410/500, loss: 2.5844579565562618, correct: 50

Epoch: 420/500, loss: 2.4977783376939557, correct: 50

Epoch: 430/500, loss: 2.4148944200367812, correct: 50

Epoch: 440/500, loss: 2.33563612054922, correct: 50

Epoch: 450/500, loss: 2.260042360740172, correct: 50

Epoch: 460/500, loss: 2.1885833946726514, correct: 50

Epoch: 470/500, loss: 2.121575498801305, correct: 50

Epoch: 480/500, loss: 2.058185097401466, correct: 50

Epoch: 490/500, loss: 1.9985254372518002, correct: 50

Epoch: 500/500, loss: 1.9419651541730174, correct: 50

* Time per epoch: 0.137s

# 
# 


## Split Dataset

![image](https://user-images.githubusercontent.com/89897166/137575972-625e8467-5327-42f1-9000-e890f821822b.png)

* Parameters Used: 
    * Learning Rate = 0.1
    * Hidden Layers = 10
    * Epochs = 2000
    
# 

### Data logs:


Epoch: 0/2000, loss: 0, correct: 0

Epoch: 10/2000, loss: 39.0340918178325, correct: 15

Epoch: 20/2000, loss: 32.987026906386355, correct: 35

Epoch: 30/2000, loss: 31.064019502476317, correct: 35

Epoch: 40/2000, loss: 30.410476085246575, correct: 35

Epoch: 50/2000, loss: 30.125956417956655, correct: 35

Epoch: 60/2000, loss: 29.946606834100198, correct: 35

Epoch: 70/2000, loss: 29.804173488329223, correct: 35

Epoch: 80/2000, loss: 29.663725758717998, correct: 35

Epoch: 90/2000, loss: 29.507785955938022, correct: 35

Epoch: 100/2000, loss: 29.369103477884618, correct: 35

Epoch: 110/2000, loss: 29.26697489250638, correct: 35

Epoch: 120/2000, loss: 29.16261936928285, correct: 35

Epoch: 130/2000, loss: 29.054547744930698, correct: 35

Epoch: 140/2000, loss: 28.942696815609825, correct: 35

Epoch: 150/2000, loss: 28.82732601384365, correct: 35

Epoch: 160/2000, loss: 28.708645514668554, correct: 35

Epoch: 170/2000, loss: 28.586966532680595, correct: 35

Epoch: 180/2000, loss: 28.470313910690507, correct: 35

Epoch: 190/2000, loss: 28.353639863871244, correct: 35

Epoch: 200/2000, loss: 28.23630294915915, correct: 35

Epoch: 210/2000, loss: 28.11803047712937, correct: 35

Epoch: 220/2000, loss: 28.000078485440277, correct: 36

Epoch: 230/2000, loss: 27.88417463623142, correct: 36

Epoch: 240/2000, loss: 27.773164700842745, correct: 36

Epoch: 250/2000, loss: 27.664203973485108, correct: 36

Epoch: 260/2000, loss: 27.56239851518819, correct: 37

Epoch: 270/2000, loss: 27.46743277187149, correct: 38

Epoch: 280/2000, loss: 27.373800875852865, correct: 38

Epoch: 290/2000, loss: 27.28159902414038, correct: 38

Epoch: 300/2000, loss: 27.190925504818708, correct: 39

Epoch: 310/2000, loss: 27.101603705459166, correct: 39

Epoch: 320/2000, loss: 27.013698785636777, correct: 39

Epoch: 330/2000, loss: 26.92722596039712, correct: 39

Epoch: 340/2000, loss: 26.84201548014536, correct: 39

Epoch: 350/2000, loss: 26.75808562675594, correct: 40

Epoch: 360/2000, loss: 26.67510389145812, correct: 40

Epoch: 370/2000, loss: 26.592997450859716, correct: 40

Epoch: 380/2000, loss: 26.51166041297342, correct: 40

Epoch: 390/2000, loss: 26.430960367048705, correct: 40

Epoch: 400/2000, loss: 26.350773594487688, correct: 40

Epoch: 410/2000, loss: 26.270985503519917, correct: 40

Epoch: 420/2000, loss: 26.191488543930504, correct: 40

Epoch: 430/2000, loss: 26.113064102252242, correct: 40

Epoch: 440/2000, loss: 26.036162259069233, correct: 40

Epoch: 450/2000, loss: 25.963256293772986, correct: 40

Epoch: 460/2000, loss: 25.891301193614456, correct: 40

Epoch: 470/2000, loss: 25.82057768094594, correct: 40

Epoch: 480/2000, loss: 25.75195818676471, correct: 40

Epoch: 490/2000, loss: 25.683566021685202, correct: 40

Epoch: 500/2000, loss: 25.61551854183288, correct: 40

Epoch: 510/2000, loss: 25.5461276594925, correct: 40

Epoch: 520/2000, loss: 25.47402764670518, correct: 40

Epoch: 530/2000, loss: 25.40216353222189, correct: 40

Epoch: 540/2000, loss: 25.329768962286174, correct: 40

Epoch: 550/2000, loss: 25.257213869416482, correct: 40

Epoch: 560/2000, loss: 25.18086933449127, correct: 40

Epoch: 570/2000, loss: 25.09296527573265, correct: 40

Epoch: 580/2000, loss: 25.015021723431396, correct: 40

Epoch: 590/2000, loss: 24.94089881264394, correct: 40

Epoch: 600/2000, loss: 24.86738755149033, correct: 40

Epoch: 610/2000, loss: 24.796147045078342, correct: 40

Epoch: 620/2000, loss: 24.72424452331929, correct: 40

Epoch: 630/2000, loss: 24.656544405461343, correct: 40

Epoch: 640/2000, loss: 24.583247513015497, correct: 40

Epoch: 650/2000, loss: 24.489539372101046, correct: 40

Epoch: 660/2000, loss: 24.38644834013388, correct: 40

Epoch: 670/2000, loss: 24.28946178021757, correct: 40

Epoch: 680/2000, loss: 24.20106099077967, correct: 40

Epoch: 690/2000, loss: 24.124272618775873, correct: 40

Epoch: 700/2000, loss: 24.047783439012793, correct: 40

Epoch: 710/2000, loss: 23.97229434107638, correct: 40

Epoch: 720/2000, loss: 23.897021454044083, correct: 40

Epoch: 730/2000, loss: 23.822105617490983, correct: 40

Epoch: 740/2000, loss: 23.747358974096954, correct: 40

Epoch: 750/2000, loss: 23.67191258714622, correct: 40

Epoch: 760/2000, loss: 23.593695532203725, correct: 40

Epoch: 770/2000, loss: 23.507963165197438, correct: 40

Epoch: 780/2000, loss: 23.423824691602224, correct: 40

Epoch: 790/2000, loss: 23.350974746557668, correct: 40

Epoch: 800/2000, loss: 23.277849329726454, correct: 40

Epoch: 810/2000, loss: 23.20684340240708, correct: 40

Epoch: 820/2000, loss: 23.1340536991653, correct: 40

Epoch: 830/2000, loss: 23.06328466701597, correct: 41

Epoch: 840/2000, loss: 22.990573511299345, correct: 41

Epoch: 850/2000, loss: 22.921198216568566, correct: 40

Epoch: 860/2000, loss: 22.849142550965045, correct: 41

Epoch: 870/2000, loss: 22.78005301764648, correct: 41

Epoch: 880/2000, loss: 22.71091541873452, correct: 41

Epoch: 890/2000, loss: 22.642893139945773, correct: 41

Epoch: 900/2000, loss: 22.576122310155856, correct: 41

Epoch: 910/2000, loss: 22.51054236311075, correct: 41

Epoch: 920/2000, loss: 22.445524382329967, correct: 41

Epoch: 930/2000, loss: 22.380959059773886, correct: 41

Epoch: 940/2000, loss: 22.316620614196427, correct: 41

Epoch: 950/2000, loss: 22.253234688912908, correct: 41

Epoch: 960/2000, loss: 22.190360365387818, correct: 41

Epoch: 970/2000, loss: 22.131484305861008, correct: 41

Epoch: 980/2000, loss: 22.00195399447838, correct: 41

Epoch: 990/2000, loss: 21.83123907706654, correct: 41

Epoch: 1000/2000, loss: 21.596358376498653, correct: 41

Epoch: 1010/2000, loss: 21.35271709974843, correct: 41

Epoch: 1020/2000, loss: 21.105309081408024, correct: 41

Epoch: 1030/2000, loss: 20.86061803872561, correct: 41

Epoch: 1040/2000, loss: 20.619113504082716, correct: 41

Epoch: 1050/2000, loss: 20.347064439051437, correct: 41

Epoch: 1060/2000, loss: 20.064753914258997, correct: 41

Epoch: 1070/2000, loss: 19.756898072253072, correct: 41

Epoch: 1080/2000, loss: 19.437412718210233, correct: 41

Epoch: 1090/2000, loss: 19.09759606678332, correct: 41

Epoch: 1100/2000, loss: 18.732524985960527, correct: 41

Epoch: 1110/2000, loss: 18.39578051465605, correct: 41

Epoch: 1120/2000, loss: 18.058788520559254, correct: 41

Epoch: 1130/2000, loss: 17.729609068536668, correct: 41

Epoch: 1140/2000, loss: 17.40880045280442, correct: 41

Epoch: 1150/2000, loss: 17.094875406673935, correct: 41

Epoch: 1160/2000, loss: 16.783016087103604, correct: 41

Epoch: 1170/2000, loss: 16.415379737203942, correct: 41

Epoch: 1180/2000, loss: 16.064393502947915, correct: 41

Epoch: 1190/2000, loss: 15.7274225826879, correct: 46

Epoch: 1200/2000, loss: 15.406519025729864, correct: 46

Epoch: 1210/2000, loss: 15.096303253821734, correct: 46

Epoch: 1220/2000, loss: 14.784218720542334, correct: 46

Epoch: 1230/2000, loss: 14.484678282585843, correct: 46

Epoch: 1240/2000, loss: 14.202101237930368, correct: 46

Epoch: 1250/2000, loss: 13.933968576947256, correct: 46

Epoch: 1260/2000, loss: 13.688346009698675, correct: 46

Epoch: 1270/2000, loss: 13.441820841041203, correct: 46

Epoch: 1280/2000, loss: 13.156809591917327, correct: 46

Epoch: 1290/2000, loss: 12.892500429422364, correct: 46

Epoch: 1300/2000, loss: 12.637087992428636, correct: 47

Epoch: 1310/2000, loss: 12.39376996787617, correct: 47

Epoch: 1320/2000, loss: 12.164080811338009, correct: 47

Epoch: 1330/2000, loss: 11.94490468607558, correct: 47

Epoch: 1340/2000, loss: 11.736096881477412, correct: 47

Epoch: 1350/2000, loss: 11.538765819138836, correct: 47

Epoch: 1360/2000, loss: 11.351385247521126, correct: 47

Epoch: 1370/2000, loss: 11.190803268810056, correct: 47

Epoch: 1380/2000, loss: 11.101962081750273, correct: 47

Epoch: 1390/2000, loss: 11.755218154629834, correct: 46

Epoch: 1400/2000, loss: 12.785083158313508, correct: 44

Epoch: 1410/2000, loss: 11.930302467529145, correct: 44

Epoch: 1420/2000, loss: 11.510627478489823, correct: 47

Epoch: 1430/2000, loss: 11.231756377741048, correct: 47

Epoch: 1440/2000, loss: 11.479352635596188, correct: 46

Epoch: 1450/2000, loss: 10.948758268544319, correct: 48

Epoch: 1460/2000, loss: 10.464453522154432, correct: 48

Epoch: 1470/2000, loss: 11.022708275829974, correct: 47

Epoch: 1480/2000, loss: 11.034632926153034, correct: 45

Epoch: 1490/2000, loss: 10.145158171390534, correct: 48

Epoch: 1500/2000, loss: 9.871816436515903, correct: 48

Epoch: 1510/2000, loss: 9.988803290688475, correct: 48

Epoch: 1520/2000, loss: 10.294592763309991, correct: 47

Epoch: 1530/2000, loss: 10.060903406818243, correct: 47

Epoch: 1540/2000, loss: 9.292591601433028, correct: 48

Epoch: 1550/2000, loss: 9.017046986797085, correct: 48

Epoch: 1560/2000, loss: 8.991567838741888, correct: 48

Epoch: 1570/2000, loss: 9.3727174823149, correct: 47

Epoch: 1580/2000, loss: 11.160224348279694, correct: 45

Epoch: 1590/2000, loss: 11.833000848671626, correct: 45

Epoch: 1600/2000, loss: 11.210190957389695, correct: 45

Epoch: 1610/2000, loss: 11.44464325739971, correct: 45

Epoch: 1620/2000, loss: 11.61322808626522, correct: 45

Epoch: 1630/2000, loss: 11.592454743252718, correct: 45

Epoch: 1640/2000, loss: 11.64619957053585, correct: 45

Epoch: 1650/2000, loss: 11.710341146767666, correct: 45

Epoch: 1660/2000, loss: 11.738299413299286, correct: 46

Epoch: 1670/2000, loss: 11.762874589093398, correct: 46

Epoch: 1680/2000, loss: 11.78740059792585, correct: 46

Epoch: 1690/2000, loss: 11.812924194487795, correct: 46

Epoch: 1700/2000, loss: 11.83001327701489, correct: 46

Epoch: 1710/2000, loss: 11.835620498859269, correct: 46

Epoch: 1720/2000, loss: 11.851813605857718, correct: 46

Epoch: 1730/2000, loss: 11.848881813089932, correct: 46

Epoch: 1740/2000, loss: 11.856504874005747, correct: 46

Epoch: 1750/2000, loss: 11.854157903153137, correct: 46

Epoch: 1760/2000, loss: 11.849442364746988, correct: 46

Epoch: 1770/2000, loss: 11.84146665943996, correct: 46

Epoch: 1780/2000, loss: 11.832070196457497, correct: 46

Epoch: 1790/2000, loss: 11.830671367659711, correct: 46

Epoch: 1800/2000, loss: 11.820903778933054, correct: 46

Epoch: 1810/2000, loss: 11.808230471955861, correct: 46

Epoch: 1820/2000, loss: 11.794384947563973, correct: 46

Epoch: 1830/2000, loss: 11.778221320731266, correct: 46

Epoch: 1840/2000, loss: 11.767455722019895, correct: 46

Epoch: 1850/2000, loss: 11.75039816088122, correct: 46

Epoch: 1860/2000, loss: 11.73639214564186, correct: 46

Epoch: 1870/2000, loss: 11.692441849683629, correct: 46

Epoch: 1880/2000, loss: 11.682088834142306, correct: 46

Epoch: 1890/2000, loss: 11.659685022339756, correct: 46

Epoch: 1900/2000, loss: 11.636832262842601, correct: 46

Epoch: 1910/2000, loss: 11.617849273029261, correct: 46

Epoch: 1920/2000, loss: 11.595255488894002, correct: 46

Epoch: 1930/2000, loss: 11.574193675532674, correct: 46

Epoch: 1940/2000, loss: 11.545555537346948, correct: 46

Epoch: 1950/2000, loss: 11.51903845351168, correct: 46

Epoch: 1960/2000, loss: 11.4828346355452, correct: 46

Epoch: 1970/2000, loss: 11.466407500547021, correct: 46

Epoch: 1980/2000, loss: 11.435357595468082, correct: 46

Epoch: 1990/2000, loss: 11.42559068651917, correct: 46

Epoch: 2000/2000, loss: 11.392384168128215, correct: 46

* Time per epoch: 1.117s


# 
# 


## XOR Dataset

![image](https://user-images.githubusercontent.com/89897166/137577169-beadfe39-11a0-4bb7-b15e-3dac38c3529f.png)

* Parameters Used: 
    * Learning Rate = 0.05
    * Hidden Layers = 10
    * Epochs = 2500
    
# 

### Data logs:


Epoch: 0/2500, loss: 0, correct: 0

Epoch: 10/2500, loss: 35.71825552344888, correct: 26

Epoch: 20/2500, loss: 31.088958025253028, correct: 25

Epoch: 30/2500, loss: 29.48678109025283, correct: 37

Epoch: 40/2500, loss: 28.63689688491904, correct: 37

Epoch: 50/2500, loss: 28.087433725530207, correct: 37

Epoch: 60/2500, loss: 27.685490040070757, correct: 37

Epoch: 70/2500, loss: 27.352645985996155, correct: 37

Epoch: 80/2500, loss: 27.038667944526942, correct: 37

Epoch: 90/2500, loss: 26.741975252801407, correct: 37

Epoch: 100/2500, loss: 26.453121396137135, correct: 38

Epoch: 110/2500, loss: 26.18434000134956, correct: 38

Epoch: 120/2500, loss: 25.92467860670798, correct: 38

Epoch: 130/2500, loss: 25.686135008354405, correct: 38

Epoch: 140/2500, loss: 25.43134878154169, correct: 38

Epoch: 150/2500, loss: 25.175321751192225, correct: 38

Epoch: 160/2500, loss: 24.91742825927156, correct: 38

Epoch: 170/2500, loss: 24.668850340780534, correct: 38

Epoch: 180/2500, loss: 24.4327718102575, correct: 38

Epoch: 190/2500, loss: 24.199195154926525, correct: 38

Epoch: 200/2500, loss: 23.96376173528337, correct: 38

Epoch: 210/2500, loss: 23.726299633780705, correct: 38

Epoch: 220/2500, loss: 23.488407671730506, correct: 38

Epoch: 230/2500, loss: 23.250440254356224, correct: 38

Epoch: 240/2500, loss: 23.009898347006935, correct: 38

Epoch: 250/2500, loss: 22.77323145945382, correct: 38

Epoch: 260/2500, loss: 22.53751651603682, correct: 41

Epoch: 270/2500, loss: 22.29911009661528, correct: 41

Epoch: 280/2500, loss: 22.05926471191732, correct: 41

Epoch: 290/2500, loss: 21.816700459059, correct: 41

Epoch: 300/2500, loss: 21.573729729017447, correct: 41

Epoch: 310/2500, loss: 21.33010157534049, correct: 41

Epoch: 320/2500, loss: 21.090664619482027, correct: 43

Epoch: 330/2500, loss: 20.852783874922498, correct: 43

Epoch: 340/2500, loss: 20.612831798181304, correct: 44

Epoch: 350/2500, loss: 20.372954692860215, correct: 45

Epoch: 360/2500, loss: 20.13685317027595, correct: 45

Epoch: 370/2500, loss: 19.90296025197015, correct: 45

Epoch: 380/2500, loss: 19.67027043311853, correct: 45

Epoch: 390/2500, loss: 19.437923104145096, correct: 45

Epoch: 400/2500, loss: 19.206950865632543, correct: 46

Epoch: 410/2500, loss: 18.977873435377674, correct: 46

Epoch: 420/2500, loss: 18.752983762579785, correct: 46

Epoch: 430/2500, loss: 18.531178626675615, correct: 46

Epoch: 440/2500, loss: 18.31154985925812, correct: 46

Epoch: 450/2500, loss: 18.096604511682276, correct: 46

Epoch: 460/2500, loss: 17.886407943420046, correct: 46

Epoch: 470/2500, loss: 17.679695350465163, correct: 46

Epoch: 480/2500, loss: 17.477953742456712, correct: 46

Epoch: 490/2500, loss: 17.280795452490405, correct: 47

Epoch: 500/2500, loss: 17.0862376625196, correct: 47

Epoch: 510/2500, loss: 16.895379319503643, correct: 47

Epoch: 520/2500, loss: 16.709193945586463, correct: 47

Epoch: 530/2500, loss: 16.527372341587323, correct: 47

Epoch: 540/2500, loss: 16.34925152222565, correct: 47

Epoch: 550/2500, loss: 16.174944749766656, correct: 47

Epoch: 560/2500, loss: 16.004047354521845, correct: 47

Epoch: 570/2500, loss: 15.83574383624356, correct: 47

Epoch: 580/2500, loss: 15.670790660742492, correct: 47

Epoch: 590/2500, loss: 15.5089279351961, correct: 47

Epoch: 600/2500, loss: 15.350784199887462, correct: 47

Epoch: 610/2500, loss: 15.19612988555048, correct: 47

Epoch: 620/2500, loss: 15.045287135216002, correct: 47

Epoch: 630/2500, loss: 14.899614123930876, correct: 47

Epoch: 640/2500, loss: 14.758105285795537, correct: 47

Epoch: 650/2500, loss: 14.61979093744786, correct: 47

Epoch: 660/2500, loss: 14.485959991107858, correct: 47

Epoch: 670/2500, loss: 14.354424835342622, correct: 47

Epoch: 680/2500, loss: 14.225231724772376, correct: 47

Epoch: 690/2500, loss: 14.09935138158297, correct: 47

Epoch: 700/2500, loss: 13.975079938527879, correct: 47

Epoch: 710/2500, loss: 13.851972937859772, correct: 47

Epoch: 720/2500, loss: 13.731400766147864, correct: 47

Epoch: 730/2500, loss: 13.613301549986101, correct: 47

Epoch: 740/2500, loss: 13.497330876520287, correct: 47

Epoch: 750/2500, loss: 13.383817465351708, correct: 47

Epoch: 760/2500, loss: 13.27292933963543, correct: 47

Epoch: 770/2500, loss: 13.163653199311204, correct: 48

Epoch: 780/2500, loss: 13.058430843084256, correct: 48

Epoch: 790/2500, loss: 12.954975633122118, correct: 48

Epoch: 800/2500, loss: 12.854263854312272, correct: 48

Epoch: 810/2500, loss: 12.754728029513984, correct: 48

Epoch: 820/2500, loss: 12.658873410281192, correct: 48

Epoch: 830/2500, loss: 12.566654600491002, correct: 48

Epoch: 840/2500, loss: 12.475974756572283, correct: 48

Epoch: 850/2500, loss: 12.387313107465948, correct: 48

Epoch: 860/2500, loss: 12.300010212510232, correct: 48

Epoch: 870/2500, loss: 12.214998355082557, correct: 48

Epoch: 880/2500, loss: 12.132640457607566, correct: 48

Epoch: 890/2500, loss: 12.052466529242798, correct: 48

Epoch: 900/2500, loss: 11.973386471754347, correct: 48

Epoch: 910/2500, loss: 11.8969711571464, correct: 48

Epoch: 920/2500, loss: 11.819403094335906, correct: 48

Epoch: 930/2500, loss: 11.74324488330956, correct: 48

Epoch: 940/2500, loss: 11.668573368128143, correct: 48

Epoch: 950/2500, loss: 11.595342709125529, correct: 48

Epoch: 960/2500, loss: 11.52337032370809, correct: 48

Epoch: 970/2500, loss: 11.452424678580485, correct: 48

Epoch: 980/2500, loss: 11.382493333684335, correct: 48

Epoch: 990/2500, loss: 11.313746627447074, correct: 48

Epoch: 1000/2500, loss: 11.246106297708323, correct: 48

Epoch: 1010/2500, loss: 11.179324282775315, correct: 48

Epoch: 1020/2500, loss: 11.113779008639392, correct: 48

Epoch: 1030/2500, loss: 11.049285578959207, correct: 48

Epoch: 1040/2500, loss: 10.985804286124592, correct: 48

Epoch: 1060/2500, loss: 10.861144832318093, correct: 48

Epoch: 1070/2500, loss: 10.799851929644358, correct: 48

Epoch: 1080/2500, loss: 10.739412530294722, correct: 48

Epoch: 1090/2500, loss: 10.68021897465554, correct: 48

Epoch: 1100/2500, loss: 10.62165929148192, correct: 48

Epoch: 1110/2500, loss: 10.563469619082868, correct: 48

Epoch: 1120/2500, loss: 10.506340464475446, correct: 48

Epoch: 1130/2500, loss: 10.450196298545666, correct: 48

Epoch: 1140/2500, loss: 10.395155609901906, correct: 48

Epoch: 1150/2500, loss: 10.341121633995831, correct: 48

Epoch: 1160/2500, loss: 10.287294271147113, correct: 48

Epoch: 1170/2500, loss: 10.23348066032314, correct: 48

Epoch: 1180/2500, loss: 10.180920701612989, correct: 48

Epoch: 1190/2500, loss: 10.12939609576564, correct: 48

Epoch: 1200/2500, loss: 10.07861690397879, correct: 48

Epoch: 1210/2500, loss: 10.02971974931391, correct: 48

Epoch: 1220/2500, loss: 9.980023969109995, correct: 48

Epoch: 1230/2500, loss: 9.933550008049853, correct: 48

Epoch: 1240/2500, loss: 9.884520424631447, correct: 48

Epoch: 1250/2500, loss: 9.837826462059061, correct: 48

Epoch: 1260/2500, loss: 9.79111271227389, correct: 48

Epoch: 1270/2500, loss: 9.746033257886031, correct: 48

Epoch: 1280/2500, loss: 9.700162569902751, correct: 48

Epoch: 1290/2500, loss: 9.6537413306125, correct: 48

Epoch: 1300/2500, loss: 9.607788711245119, correct: 48

Epoch: 1310/2500, loss: 9.563894708858133, correct: 48

Epoch: 1320/2500, loss: 9.519675684963252, correct: 48

Epoch: 1330/2500, loss: 9.476596645772156, correct: 48

Epoch: 1340/2500, loss: 9.434109485473662, correct: 48

Epoch: 1350/2500, loss: 9.393223072873093, correct: 48

Epoch: 1360/2500, loss: 9.351638601033718, correct: 48

Epoch: 1370/2500, loss: 9.30843905159498, correct: 48

Epoch: 1380/2500, loss: 9.267512982334466, correct: 48

Epoch: 1390/2500, loss: 9.226981446461613, correct: 48

Epoch: 1400/2500, loss: 9.188544040817435, correct: 48

Epoch: 1410/2500, loss: 9.146242056727596, correct: 48

Epoch: 1420/2500, loss: 9.106667801001697, correct: 48

Epoch: 1430/2500, loss: 9.068506706366064, correct: 48

Epoch: 1440/2500, loss: 9.027218508907238, correct: 48

Epoch: 1450/2500, loss: 8.989195557527074, correct: 48

Epoch: 1460/2500, loss: 8.950247387933294, correct: 48

Epoch: 1470/2500, loss: 8.913044122663612, correct: 48

Epoch: 1480/2500, loss: 8.87203512691789, correct: 48

Epoch: 1490/2500, loss: 8.833611287151829, correct: 48

Epoch: 1500/2500, loss: 8.79623468833557, correct: 48

Epoch: 1510/2500, loss: 8.757854159782637, correct: 48

Epoch: 1520/2500, loss: 8.720998631038473, correct: 48

Epoch: 1530/2500, loss: 8.68282413876567, correct: 48

Epoch: 1540/2500, loss: 8.64597106109073, correct: 48

Epoch: 1550/2500, loss: 8.608653265148122, correct: 48

Epoch: 1560/2500, loss: 8.57362045891936, correct: 48

Epoch: 1570/2500, loss: 8.537164264667439, correct: 48

Epoch: 1580/2500, loss: 8.501849214222636, correct: 48

Epoch: 1590/2500, loss: 8.464281338809707, correct: 48

Epoch: 1600/2500, loss: 8.43012630101072, correct: 48

Epoch: 1610/2500, loss: 8.395343168080718, correct: 48

Epoch: 1620/2500, loss: 8.360679011415877, correct: 48

Epoch: 1630/2500, loss: 8.32791149586129, correct: 48

Epoch: 1640/2500, loss: 8.29113590853657, correct: 48

Epoch: 1650/2500, loss: 8.259054953700046, correct: 48

Epoch: 1660/2500, loss: 8.225272515443308, correct: 48

Epoch: 1670/2500, loss: 8.190542544085325, correct: 48

Epoch: 1680/2500, loss: 8.159989224735378, correct: 48

Epoch: 1690/2500, loss: 8.125065762126455, correct: 48

Epoch: 1700/2500, loss: 8.092812357588967, correct: 48

Epoch: 1710/2500, loss: 8.061410122092461, correct: 48

Epoch: 1720/2500, loss: 8.02976572526337, correct: 48

Epoch: 1730/2500, loss: 7.995920080579002, correct: 48

Epoch: 1740/2500, loss: 7.962849044221751, correct: 48

Epoch: 1750/2500, loss: 7.931798211423848, correct: 48

Epoch: 1760/2500, loss: 7.900865339305738, correct: 48

Epoch: 1770/2500, loss: 7.87104812824401, correct: 48

Epoch: 1780/2500, loss: 7.838735439425824, correct: 48

Epoch: 1790/2500, loss: 7.806716854917727, correct: 48

Epoch: 1800/2500, loss: 7.774802640370519, correct: 48

Epoch: 1810/2500, loss: 7.747342778752549, correct: 48

Epoch: 1820/2500, loss: 7.71562722664996, correct: 48

Epoch: 1830/2500, loss: 7.686817089286034, correct: 48

Epoch: 1840/2500, loss: 7.656970090557599, correct: 48

Epoch: 1850/2500, loss: 7.6265831319373865, correct: 48

Epoch: 1860/2500, loss: 7.597536523236531, correct: 48

Epoch: 1870/2500, loss: 7.570200928615304, correct: 48

Epoch: 1880/2500, loss: 7.5400732812161575, correct: 48

Epoch: 1890/2500, loss: 7.5135410901461315, correct: 48

Epoch: 1900/2500, loss: 7.487062208779962, correct: 48

Epoch: 1910/2500, loss: 7.455955355076124, correct: 48

Epoch: 1920/2500, loss: 7.427274870581709, correct: 48

Epoch: 1930/2500, loss: 7.400636794131276, correct: 48

Epoch: 1940/2500, loss: 7.374324277835999, correct: 48

Epoch: 1950/2500, loss: 7.347990982424488, correct: 48

Epoch: 1960/2500, loss: 7.320552718741243, correct: 48

Epoch: 1970/2500, loss: 7.29225288047279, correct: 48

Epoch: 1980/2500, loss: 7.26340137855227, correct: 48

Epoch: 1990/2500, loss: 7.235176139334147, correct: 48

Epoch: 2000/2500, loss: 7.208183226366572, correct: 48

Epoch: 2010/2500, loss: 7.182172808702143, correct: 48

Epoch: 2020/2500, loss: 7.155660227109697, correct: 48

Epoch: 2030/2500, loss: 7.129863164888491, correct: 48

Epoch: 2040/2500, loss: 7.105294043402305, correct: 48

Epoch: 2050/2500, loss: 7.0815759964267135, correct: 48

Epoch: 2060/2500, loss: 7.054760510004119, correct: 48

Epoch: 2070/2500, loss: 7.028161064477803, correct: 48

Epoch: 2080/2500, loss: 7.0018063987717865, correct: 48

Epoch: 2090/2500, loss: 6.9794483614676155, correct: 48

Epoch: 2100/2500, loss: 6.952727620848219, correct: 48

Epoch: 2110/2500, loss: 6.928109142186605, correct: 48

Epoch: 2120/2500, loss: 6.9054454355238075, correct: 48

Epoch: 2130/2500, loss: 6.877300039336494, correct: 48

Epoch: 2140/2500, loss: 6.854623958340449, correct: 48

Epoch: 2150/2500, loss: 6.831359389823065, correct: 48

Epoch: 2160/2500, loss: 6.804702547455162, correct: 48

Epoch: 2170/2500, loss: 6.78412518411209, correct: 48

Epoch: 2180/2500, loss: 6.758203361478025, correct: 48

Epoch: 2190/2500, loss: 6.734597756427129, correct: 48

Epoch: 2200/2500, loss: 6.7111829027126, correct: 48

Epoch: 2210/2500, loss: 6.687337906410506, correct: 48

Epoch: 2220/2500, loss: 6.664040059678181, correct: 48

Epoch: 2230/2500, loss: 6.642905835700404, correct: 48

Epoch: 2240/2500, loss: 6.61641613749935, correct: 48

Epoch: 2250/2500, loss: 6.595163019268552, correct: 48

Epoch: 2260/2500, loss: 6.56936957558918, correct: 48

Epoch: 2270/2500, loss: 6.549787692873676, correct: 48

Epoch: 2280/2500, loss: 6.52354007755139, correct: 48

Epoch: 2290/2500, loss: 6.504264794255185, correct: 48

Epoch: 2300/2500, loss: 6.481369625916688, correct: 48

Epoch: 2310/2500, loss: 6.457679287604611, correct: 48

Epoch: 2320/2500, loss: 6.438957829641753, correct: 48

Epoch: 2330/2500, loss: 6.412412074598308, correct: 48

Epoch: 2340/2500, loss: 6.392080256692065, correct: 48

Epoch: 2350/2500, loss: 6.367666198110026, correct: 48

Epoch: 2360/2500, loss: 6.351119706027587, correct: 48

Epoch: 2370/2500, loss: 6.3244273370603725, correct: 48

Epoch: 2380/2500, loss: 6.305612936246144, correct: 48

Epoch: 2390/2500, loss: 6.286393128382313, correct: 48

Epoch: 2400/2500, loss: 6.259061303508715, correct: 49

Epoch: 2410/2500, loss: 6.242655196244979, correct: 49

Epoch: 2420/2500, loss: 6.21839181236057, correct: 49

Epoch: 2430/2500, loss: 6.196113080498928, correct: 49

Epoch: 2440/2500, loss: 6.1782282834403, correct: 49

Epoch: 2450/2500, loss: 6.152644268327677, correct: 49

Epoch: 2460/2500, loss: 6.137722778233536, correct: 49

Epoch: 2470/2500, loss: 6.111205086238095, correct: 49

Epoch: 2480/2500, loss: 6.095789344359326, correct: 49

Epoch: 2490/2500, loss: 6.072398043466061, correct: 49

Epoch: 2500/2500, loss: 6.0484878887226925, correct: 49

* Time per epoch: 1.153s


# 
# 

