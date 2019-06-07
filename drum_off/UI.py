'''
This module provides the User Interface
'''

import kivy
from kivy.app import App
from kivy.uix.checkbox import CheckBox
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.clock import  mainthread
from kivy.config import Config
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
import drum_off.game as game
import drum_off.utils as utils
from os import mkdir, listdir
import threading
from os.path import join, isdir
import pickle
import drum_off.drumsynth as drumsynth
import drum_off.learner as mgu
import numpy as np
kivy.require('1.10.1')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')


class CCheckBox(CheckBox):
    pass

class PicCheckBox(CheckBox):
    pass

class RecButton(ButtonBehavior, Image):
    isActive=False
    def on_press(self):
        self.changeStatus()
        if self.isActive:
            print ('Rec')
            self.source = "./UiImg/StopBtn.png"
        else:
            print ("deactivated")
    def changeStatus(self):
        if self.isActive:
            self.isActive=False
            self.source="./UiImg/RecBtn.png"
        else:
            self.isActive=True
            self.source = "./UiImg/RecBtn.png"
class PlayButton(ButtonBehavior, Image):
    isActive=False
    def on_press(self):
        self.changeStatus()
        if self.isActive:
            print ('Play')
            self.source = "./UiImg/StopBtn.png"
        else:
            print ("deactivated")
    def changeStatus(self):
        if self.isActive:
            self.isActive=False
            self.source="./UiImg/PlayBtn.png"
        else:
            self.isActive=True
            self.source = "./UiImg/PlayBtn.png"
class StopButton(ButtonBehavior, Image):
    isActive=False
    def on_press(self):
        self.changeStatus()
        if self.isActive:
            print ('Stop')
        else:
            print ("deactivated")
    def changeStatus(self):
        if self.isActive:
            self.isActive=False
            self.source="./UiImg/StopBtnDeactivated.png"
        else:
            self.isActive=True
            self.source = "./UiImg/StopBtn.png"

class NumberButton(Image,PicCheckBox,ButtonBehavior):
    activeNumber=None
    pass


Factory.register('CCheckBox', cls=CCheckBox)
#model=mgu.initModel()

drumNames = {'kick': 0,  # only one allowed
             'snare': 1,  # only one allowed
             'closed hi-hat': 2,  # only one allowed
             'open hi-hat': 3,  # only one allowed
             'hi-hat pedal': 4,  # only one allowed
             'rack tom': 5,  # leave space for more
             'floor tom': 9,  # leave space for more
             'ride cymbal': 12,  # only one allowed
             'crash cymbal': 13,  # leave space for more
             'misc': 17
             }
countInPath='./countIn.csv'
countInWavPath='click.wav'
countInLength=2
model_type='multi2multi'

class StartScreen(Screen):
    def getStatus(self):
        app = App.get_running_app()
        string = 'Current Kit: {} (kit size: {} drums)\n'.format(app.KitName, app.NrOfDrums) + 'Kit Ready:{}\n'.format(
            app.KitInit) + 'Model ready:{}'.format(app.Model)
        return string

    def SetText(self):
        app = App.get_running_app()
        if len(app.NrOfDrums):
            nr = sum(app.NrOfDrums)
        else:
            nr = 0
        string = 'Current Kit: {} (kit size: {} drums)\n'.format(app.KitName, nr) + 'Kit Ready:{}\n'.format(
            app.KitInit) + 'Model ready:{}'.format(app.Model)
        self.ids.status.text = string
    def quit(self):
        drumsynth._ImRunning=False
        utils._ImRunning=False
        App.get_running_app().stop()
    pass


class SoundCheckScreen(Screen):

    performMessage = StringProperty()
    btnMessage = StringProperty()
    drmMessage = StringProperty()
    nrMessage = StringProperty()
    finishMessage = StringProperty()
    finishStatus = StringProperty()
    model_type=StringProperty()
    kdTake = NumericProperty()
    snTake = NumericProperty()
    hhTake = NumericProperty()
    ttTake = NumericProperty()
    ftTake = NumericProperty()
    rdTake = NumericProperty()
    crTake = NumericProperty()
    toyTake = NumericProperty()

    def __init__(self, **kwargs):
        super(SoundCheckScreen, self).__init__(**kwargs)
        self.performMessage = 'Next Drum:'
        self.btnMessage = 'GO!'
        self.drmMessage = 'Play 16 hits on each drum. \nPress run when ready!\n (You can re run souncheck as many times as you like)'
        self.nrMessage = '1'
        self.finishMessage = 'Load Soundchecked Kit and Exit'
        self.finishStatus = 'Soundcheck not complete'
        self.model_type = model_type
        self.kdTake = 0
        self.snTake = 0
        self.hhTake = 0
        self.ttTake = 0
        self.ftTake = 0
        self.rdTake = 0
        self.crTake = 0
        self.toyTake = 0

    @mainthread
    def getTxt(self):
        app = App.get_running_app()
        return str(app.NrOfDrums)

    @mainthread
    def setMsg(self, msg):
        self.performMessage = msg

    @mainthread
    def setDrumNro(self, nr):
        app = App.get_running_app()
        if nr <= sum(app.NrOfDrums):
            self.nrMessage = nr
        else:
            self.ids.checkBtn.txt = 'Finished'
            self.nrMessage = ''
    def addDrumToKit(self,nr):
        app = App.get_running_app()
        app.NrOfDrums[nr]=1
    def getDrumChecked(self,nr):
        app = App.get_running_app()
        return app.NrOfDrums[nr]
    # @mainthread
    def set_model_type(self, model_type):
        self.model_type=model_type
    def saveKitTemplate(self, instance):
        app = App.get_running_app()
        app.KitName = self.ids.drumkit_name.text
        app.NrOfDrums = [int(self.ids.kdn.text), int(self.ids.snn.text), int(self.ids.hhn.text),
                         int(self.ids.ttn.text), int(self.ids.rdn.text), int(self.ids.crn.text), int(self.ids.ton.text)]
        # self.manager.get_screen('MainMenu').SetText()
        # print(app.KitName)

    def getDrumNro(self):
        return int(self.ids.drumNro.text)


    def stopSoundCheck(self):

        if(drumsynth._ImRecording):
            #self.ids[name].changeStatus(1)
            drumsynth._ImRecording = False
        elif(drumsynth._ImRunning):
            #self.ids[name].changeStatus(1)
            drumsynth._ImRunning=False

    def playSoundCheck(self, nr,take):
        app = App.get_running_app()
        name='play'+str(nr)
        #self.saveKitTemplate(*args)
        print(app.KitName, int(sum(app.NrOfDrums)))
        fullPath = './Kits/{}'.format(app.KitName)+'/drum'+str(nr+take)+'.wav'
        print(fullPath)

        def callback():
            try:
                drumsynth.playWav(fullPath)
                self.ids[name].changeStatus()
            except Exception as e:
                print('playback SC: ', e)
                self.ids[name].changeStatus()

        t = threading.Thread(target=callback)
        t.start()

    def runSoundCheck(self, nr):
        if nr == '' or nr == '0':
            utils._ImRunning = False
            return
        self.btnMessage = 'NEXT DRUM!'

        app = App.get_running_app()
        if self.ids.drumkit_name.text is not 'none':
            print('oli none')
            app.KitName=self.ids.drumkit_name.text
        print(app.KitName, sum(app.NrOfDrums))
        fullPath = './Kits/{}'.format(app.KitName)
        try:
            mkdir(fullPath)
        except Exception as e:
            pass
        utils._ImRunning = False
        # Should we have a count in?
        #time.sleep(1)
        self.performMessage = 'Start Playing!'

        def callback():
            try:
                #fullPath init??
                game.soundcheckDrum(fullPath, nr)
            except Exception as e:
                print('souncheck ui:', e)

        t = threading.Thread(target=callback)
        t.start()
        #self.nrMessage = str(self.getDrumNro() + 1)
        #self.performMessage = 'Next Drum:'
        #if (int(self.nrMessage) >= sum(app.NrOfDrums) + 1):
        #    self.btnMessage = 'Finish'
         #   self.finishStatus = 'Soundcheck Complete\nPress "Finish Souncheck" to load kit and exit'
         #   self.nrMessage = str(0)

    def finishSoundCheck(self):
        global model_type
        app = App.get_running_app()
        fullPath = './Kits/{}'.format(app.KitName)
        try:
            print(int(sum(app.NrOfDrums)))
            game.initKitBG(fullPath, int(sum(app.NrOfDrums)))
            pickle.dump(self.model_type,open(fullPath+'/model_type.cfg','wb'))
            model_type=self.model_type
            mgu.buildVocabulary(hits=utils.get_possible_notes([i.get_name()[0] for i in game.drumkit]))
            mgu.initModel(kitPath=fullPath+'/',destroy_old=True, model_type=model_type)
            app.KitInit = 'Initialized'
            app.root.current = 'MainMenu'
        except Exception as e:
            print('finish soundcheck: ', e)

    def First_Thread(self, nr):
        self.setMsg('soundcheck drum nr. {}'.format('0'))
        threading.Thread(target=self.runSoundCheck(nr)).start()


class SoundCheckPerformScreen(Screen):
    performMessage = StringProperty()
    btnMessage = StringProperty()
    drmMessage = StringProperty()
    nrMessage = StringProperty()
    finishMessage = StringProperty()
    finishStatus = StringProperty()


    def __init__(self, **kwargs):
        super(SoundCheckPerformScreen, self).__init__(**kwargs)
        self.performMessage = 'Next Drum:'
        self.btnMessage = 'GO!'
        self.drmMessage = 'Play 16 hits on each drum. \nPress run when ready!\n (You can re run souncheck as many times as you like)'
        self.nrMessage = '1'
        self.finishMessage = 'Load Soundchecked Kit and Exit'
        self.finishStatus = 'Soundcheck not complete'


    @mainthread
    def getTxt(self):
        app = App.get_running_app()
        return str(app.NrOfDrums)

    @mainthread
    def setMsg(self, msg):
        self.performMessage = msg

    @mainthread
    def setDrumNro(self, nr):
        app = App.get_running_app()
        if nr <= sum(app.NrOfDrums):
            self.nrMessage = nr
        else:
            self.ids.checkBtn.txt = 'Finished'
            self.nrMessage = ''

    # @mainthread
    def getDrumNro(self):
        return int(self.ids.drumNro.text)
    def stopSoundCheck(self):
        utils._ImRunning = False
        drumsynth._ImRunning = False

    def runSoundCheck(self, nr):
        if nr == '' or nr == '0':
            utils._ImRunning = False
            return
        self.btnMessage = 'NEXT DRUM!'
        app = App.get_running_app()
        print(app.KitName, sum(app.NrOfDrums))
        fullPath = './Kits/{}'.format(app.KitName)
        try:
            mkdir(fullPath)
        except Exception as e:
            pass
        utils._ImRunning = False
        #Should we have a count in?
        #time.sleep(1)
        self.performMessage = 'Start Playing!'

        def callback():
            try:
                game.soundcheckDrum(fullPath, self.getDrumNro()-1)
            except Exception as e:
                print('sounckech ui:',e)

        t = threading.Thread(target=callback)
        t.start()
        self.nrMessage = str(self.getDrumNro() + 1)
        self.performMessage = 'Next Drum:'
        if (int(self.nrMessage) >= sum(app.NrOfDrums) + 1):
            self.btnMessage = 'Finish'
            self.finishStatus = 'Soundcheck Complete\nPress "Finish Souncheck" to load kit and exit'
            self.nrMessage = str(0)

    def finishSoundCheck(self):
        app = App.get_running_app()
        fullPath = './Kits/{}'.format(app.KitName)
        try:
            game.initKitBG(fullPath, sum(app.NrOfDrums))
            app.KitInit = 'Initialized'
            app.root.current = 'MainMenu'
        except Exception as e:
            print('finish soundcheck: ',e)

    def First_Thread(self, nr):
        self.setMsg('soundcheck drum nr. {}'.format('0'))
        threading.Thread(target=self.runSoundCheck(nr)).start()


class PlayScreen(Screen):
    performMessage = StringProperty()
    pBtnMessage = StringProperty()
    cBtnMessage = StringProperty()
    turnMessage = StringProperty()
    lastMessage = StringProperty()
    lastGenPart = StringProperty()
    lastPlayerPart = StringProperty()
    playBackMessage = StringProperty()
    trSize=NumericProperty()
    temperature = NumericProperty()
    threshold = NumericProperty()
    deltaTempo = NumericProperty()
    modify = BooleanProperty()
    step = BooleanProperty()
    halt = BooleanProperty()
    computer_on=BooleanProperty()
    lr=NumericProperty()
    quantize=NumericProperty()
    def __init__(self, **kwargs):
        super(PlayScreen, self).__init__(**kwargs)
        self.performMessage = 'Press play to start'
        self.pBtnMessage = 'Play'
        self.turnMessage = 'Player'
        self.lastMessage = ''
        self.lastPlayerPart = ''
        self.lastGenPart = ''
        self.playBackMessage = ''
        self.trSize=1.33
        self.temperature=0.8
        self.threshold = 0.0
        self.deltaTempo=1.0
        self.modify=True
        self.step = True
        self.halt=True
        self.computer_on=False
        self.lr=0.003
        self.quantize=1.


    def setTrSize(self, *args):
        self.trSize=args[1]

    def setTemp(self, *args):
        self.temperature = args[1]

    def setThresh(self, *args):
        self.threshold = args[1]

    def setModify(self, *args):
        self.modify = args[0]
    def setLr(self,*args):
        self.lr=args[1]

    def setQuantize(self,*args):
        if args[0]:
            self.quantize=1.
        else:
            self.quantize=0.
    def setComputerOn(self,*args):
        self.computer_on=args[0]



    def toggleStep(self):
        """
        Step playing mode toggler
        :return: None
        """
        if self.ids.playBackBtn.disabled:
            self.ids.playBackBtn.disabled = False
            self.step=True
        else:
            self.ids.playBackBtn.disabled = True
            self.step=False

    def toggleSlider(self):
        """
        Slider disabling toggler, (should generalize to disable any slider)
        :return: None
        """
        if self.ids.trs.disabled:
            self.ids.trs.disabled = False
        else:
            self.ids.trs.disabled = True
    def stopAll(self):
        drumsynth._ImRunning=False
        self.halt=True
    def createLast(self, fileName,outFile=None, addCountInAndCountOut=True):
        """
        Calls a wav file to be created to the disk
        :param fileName: Srting, filename of the notation file
        :param outFile: String=None, filename of the resulting wav
        :param addCountInAndCountOut=True: Boolean, if True adds a count in at both ens of result
        :return: None
        """
        fullPath = fileName
        def callback():
            try:
                print('createfile')
                #scale tempos to less abrupt areas to 100-200 bpm
                countTempo=1
                if self.deltaTempo<0.83:
                    countTempo=2
                elif self.deltaTempo>1.66:
                    countTempo=.5
                self.lastMessage=drumsynth.createWav(fullPath,outFile, addCountInAndCountOut=addCountInAndCountOut, deltaTempo=self.deltaTempo, countTempo=countTempo)

            except Exception as e:
                print('createWav ui: ', e)
                self.halt = True
                self.pBtnMessage = 'Play'
        t = threading.Thread(target=callback)
        t.start()

    def playBackLast(self, cue=False):
        """
        Plays back a performance
        :param cue: Boolean, if True then player turn follows playback
        :return: None
        """
        drumsynth._ImRunning = False
        if(self.lastMessage==''):
            return None
        fullPath = self.lastMessage

        if self.playBackMessage=='Stop Playback':
            drumsynth._ImRunning = False
            self.playBackMessage = 'Play Back Last Performance'
            return None
        def callback():
            try:
                self.playBackMessage = 'Stop Playback'
                print('playing:', fullPath)
                drumsynth.playWav(fullPath)
                if (self.lastMessage == ''):
                    self.playBackMessage = ''
                else:
                    self.playBackMessage = 'Play Back Last Performance'
                if (self.step==False or cue==True) and self.pBtnMessage == 'Stop':
                    self.doPlayerTurn()
            except Exception as e:
                print('playback ui: ',e)
                self.halt = True
                self.pBtnMessage = 'Play'
        t = threading.Thread(target=callback)
        t.start()

    def doComputerTurn(self):
        """
        Handles computer turn
        :return: None
        """
        if (self.lastMessage == '')or self.halt==True:
            return None
        fullPath = self.lastPlayerPart
        print(fullPath)
        def callback():
            try:##INIT MODEL!!
                #if mgu.getLr!=self.lr:
                #    try:
                #        mgu.setLr(self.lr)
                #    except Exception as e:
                #        print(e)
                #TODO: make user adjustable length
                self.lastGenPart = mgu.generatePart(
                        mgu.train(fullPath, sampleMul=self.trSize,
                                  forceGen=False, updateModel=self.modify, model_type=model_type),
                    partLength=100, temp=self.temperature, model_type=model_type)
                self.createLast(self.lastGenPart,outFile='./generated.wav',addCountInAndCountOut=(not self.step))
                #Remove hack!!
                while self.lastMessage!='./generated.wav':
                    pass
                if self.step:
                    self.playBackMessage == 'Play Back Computer Performance'
                    return
                elif self.pBtnMessage == 'Stop':
                    self.playBackLast()
            except Exception as e:
                print('virhe! computer turn: ',e)
                self.halt = True
                self.pBtnMessage = 'Play'
        t = threading.Thread(target=callback)
        t.start()


    def playButton(self):
        """
        Play button toggler
        :return: None
        """
        if self.pBtnMessage == 'Stop':
            self.halt = True
            utils._ImRunning = False
            drumsynth._ImRecording=False
            drumsynth._ImRunning =False
            self.pBtnMessage = 'Play'
            return None
        else:
            self.pBtnMessage = 'Stop'
            self.lastMessage=countInWavPath
            self.playBackLast(cue=True)
            return None

    def doPlayerTurn(self):
        """
        Handles player turn thread
        :return: None
        """
        self.halt=False
        app = App.get_running_app()
        fullPath = './Kits/{}'.format(app.KitName)
        utils._ImRunning = False
        try:
            mkdir('{}/takes/'.format(fullPath))
        except Exception as e:
            #throws error if the folder exists
            #print('mkdir:',e)
            pass
        self.performMessage = 'Start Playing!'
        def callback():
            try:
                print('recording turn')
                #TODO:make user adjustable length
                self.lastPlayerPart, self.deltaTempo=game.playLive(fullPath, thresholdAdj=self.threshold,part_length=30, saveAll=True, quantize=self.quantize)
                if self.lastPlayerPart==False:
                    return
                self.createLast(self.lastPlayerPart,outFile='./player_performance.wav',addCountInAndCountOut=(not self.step))
                if self.step:
                    self.playBackMessage = 'Play Back Last Performance'
                    self.pBtnMessage='Play'
                elif self.pBtnMessage == 'Stop':
                    if self.computer_on:
                        self.doComputerTurn()
                    else:
                        while self.lastMessage != './player_performance.wav':
                            pass
                        self.playBackLast()
            except Exception as e:
                print('player playback ui: ',e)
                self.halt = True
                self.pBtnMessage = 'Play'
        t = threading.Thread(target=callback)
        t.start()

class LoadScreen(Screen):
    """
    Load Screen
    """
    def is_dir(self, directory, filename):
        return isdir(join(directory, filename))

    def getKits(self):
        return 'eka\ntoka\nkolmas\njne...'

    def loadKit(self, filename):
        global model_type
        try:
            print(filename[0]+'/')
            game.loadKit(filename[0])
            model_type=pickle.load(open(filename[0]+'/model_type.cfg','rb'))
            mgu.buildVocabulary(hits=utils.get_possible_notes([i.get_name()[0] for i in game.drumkit]))
            mgu.initModel(kitPath=filename[0]+'/', destroy_old=False, model_type=model_type)
            app = App.get_running_app()
            app.KitName = (filename[0].split('/')[-1])
            app.KitInit = 'Initialized'
            app.Model='Loaded'
            #app.NrOfDrums = [0] * len(game.drums)
            #No need for this unless initialization parameters changed
            #game.initKitBG(filename[0],len(game.drums))
            for i in game.drumkit:
                drum = i.get_name()[0]

                # make hihats one drum?? nope
##HÄTÄ PÄÄLLÄ!!!_________________________________________________________________________________________
                app.NrOfDrums[drum] += 1
            print(app.NrOfDrums)
            app.root.current = 'MainMenu'
        except Exception as e:
            print('load kit ui: ',e)

class MyScreenManager(ScreenManager):
    pass

root_widget =Builder.load_file("UI.kv")

class DrumOffApp(App):

    NrOfDrums = np.zeros(24).astype(int)
    KitName = 'none'
    Model = 'model not loaded'
    KitInit = 'Kit not initialized'

    def build(self):
        return root_widget
        sm = ScreenManager()
        return sm
        # All ui views
        #sm.add_widget(StartScreen(name='MainMenu'))
        #sm.add_widget(SoundCheckScreen(name='SoundCheck'))
        #sm.add_widget(SoundCheckPerformScreen(name='SoundCheckPerform'))
        #sm.add_widget(PlayScreen(name='PlayScreen'))
        #sm.add_widget(LoadScreen(name='LoadKit'))
        ###LOAD MODEL HERE???###



if __name__ == "__main__":

    DrumOffApp().run()
