import DCQL_Agent
import DCQL_Pong
import numpy as np
import skimage as skimage
import warnings
warnings.filterwarnings("ignore")

TOTAL_TrainTime = 100000

IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

def ProcessGameImage(RawImage):
    
    GreyImage = skimage.color.rgb2gray(RawImage)
    
    CroppedImage = GreyImage[0:400,0:400]
    
    ReducedImage = skimage.transform.resize(CroppedImage,(IMGHEIGHT,IMGWIDTH))
    
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range = (0,255))
    
    ReducedImage = ReducedImage / 128
    
    return ReducedImage
        
def TrainExperiment():
    
    TrainHistory = []
    
    TheGame = DCQL_Pong.PongGame()
    
    TheGame.InitialDisplay()
    
    TheAgent = DCQL_Agent.Agent()
    
    BestAction = 0
    
    [InitialScore, InitialScreenImage] = TheGame.PlayNextMove(BestAction)
    InitialGameImage = ProcessGameImage(InitialScreenImage)
    
    GameState = np.stack((InitialGameImage,InitialGameImage,InitialGameImage,InitialGameImage),axis = 2)
    
    GameState = GameState.reshape(1, GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    
    for i in range(TOTAL_TrainTime):
        
        BestAction = TheAgent.FindBestAct(GameState)
        [ReturnScore, NewScreenImage] = TheGame.PlayNextMove(BestAction)
        
        NewGameImage = ProcessGameImage(NewScreenImage)
        
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
        
        NextState = np.append(NewGameImage, GameState[:,:,:,:3], axis = 3)
        
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState))
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 250 == 0:
            print("Train time: ",i, " game score: ",TheGame.GScore)
            TrainHistory.append(TheGame.GScore)
            
        
TrainExperiment()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










