import kivy

kivy.require('1.10.1')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooser
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.slider import Slider
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.clock import Clock, mainthread
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
import drumoff
import utils
from os import mkdir, listdir
import threading
from os.path import join, isdir
import time
#from drumsynth import playFile
import drumsynth
import GRU as mgu
# import GRU


class CCheckBox(CheckBox):
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
             'misc': 20
             }
countInPath='./countIn.csv'
countInWavPath='click.wav'
countInLength=2
class StartScreen(Screen):
    def getStatus(self):
        app = App.get_running_app()
        string = 'Current Kit:{} (kit size:{} drums)\n'.format(app.KitName, app.NrOfDrums) + 'Kit Ready:{}\n'.format(
            app.KitInit) + 'Model ready:{}'.format(app.Model)
        return string

    def SetText(self):
        app = App.get_running_app()
        if len(app.NrOfDrums):
            nr = sum(app.NrOfDrums)
        else:
            nr = 0
        string = 'Current Kit:{} (kit size:{} drums)\n'.format(app.KitName, nr) + 'Kit Ready:{}\n'.format(
            app.KitInit) + 'Model ready:{}'.format(app.Model)
        self.ids.status.text = string
    def quit(self):
        drumsynth._ImRunning=False
        utils._ImRunning=False
        App.get_running_app().stop()
    pass


class SoundCheckScreen(Screen):
    def saveKitTemplate(self, instance):
        app = App.get_running_app()
        app.KitName = self.ids.drumkit_name.text
        app.NrOfDrums = [int(self.ids.kdn.text), int(self.ids.snn.text), int(self.ids.hhn.text),
                         int(self.ids.ttn.text), int(self.ids.rdn.text), int(self.ids.crn.text), int(self.ids.ton.text)]
        # self.manager.get_screen('MainMenu').SetText()
        # print(app.KitName)


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
        self.drmMessage = 'Play 32 hits on each drum. \nPress run when ready!\n (You can re run souncheck as many times as you like)'
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
        time.sleep(1)
        self.performMessage = 'Start Playing!'

        def callback():
            try:
                drumoff.soundcheckDrum(fullPath, self.getDrumNro()-1)
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
            drumoff.initKitBG(fullPath, sum(app.NrOfDrums))
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

    def setTrSize(self, *args):
        self.trSize=args[1]

    def setTemp(self, *args):
        self.temperature = args[1]

    def setThresh(self, *args):
        self.threshold = args[1]

    def setModify(self, *args):
        self.modify = args[0]

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
                if self.deltaTempo<0.83:
                    self.deltaTempo=self.deltaTempo*2
                elif self.deltaTempo>1.66:
                    self.deltaTempo=self.deltaTempo/2
                self.lastMessage=drumsynth.createWav(fullPath,outFile, addCountInAndCountOut=addCountInAndCountOut, deltaTempo=self.deltaTempo)
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
                drumsynth.playWav(fullPath)
                if (self.lastMessage == ''):
                    self.playBackMessage = ''
                else:
                    self.playBackMessage = 'Play Back Last Performance'
                if self.step==False or cue==True:
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
                self.lastGenPart = mgu.generatePart(
                        mgu.train(fullPath, sampleMul=self.trSize,
                                  forceGen=False, updateModel=self.modify), temp=self.temperature)
                self.createLast(self.lastGenPart,outFile='./generated.wav',addCountInAndCountOut=(not self.step))
                #print(self.step)
                if self.step:
                    self.playBackMessage == 'Play Back Computer Performance'
                    return
                else:
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
            print('mkdir:',e)
            pass
        self.performMessage = 'Start Playing!'
        def callback():
            try:
                print('recording turn')
                self.lastPlayerPart, self.deltaTempo=drumoff.playLive(fullPath, self.threshold, saveAll=False)
                if self.lastPlayerPart==False:
                    return
                self.createLast(self.lastPlayerPart,addCountInAndCountOut=(not self.step))
                if self.step:
                    self.playBackMessage = 'Play Back Last Performance'
                    self.pBtnMessage='Play'
                else:
                    self.doComputerTurn()
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
        try:
            drumoff.loadKit(filename[0])
            app = App.get_running_app()
            app.KitName = (filename[0].split('/')[-1])
            app.KitInit = 'Initialized'
            app.NrOfDrums = [0] * len(drumoff.drums)
            drumoff.initKitBG(filename[0],len(drumoff.drums))
            for i in drumoff.drums:
                drum = i.get_name()[0]
                # make hihats one drum?? nope
                app.NrOfDrums[drum] += 1
            app.root.current = 'MainMenu'
        except Exception as e:
            print('load kit ui: ',e)

class MyScreenManager(ScreenManager):
    pass

root_widget =Builder.load_file("UI.kv")

class DrumOffApp(App):

    NrOfDrums = []
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
